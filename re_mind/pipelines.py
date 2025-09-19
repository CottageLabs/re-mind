from typing import TypedDict, List, Tuple, Literal

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from re_mind import components, lc_prompts, retrievers
from re_mind.lc_prompts import DEFAULT_RAG_INSTRUCTION


def create_basic_qa(llm, n_top_result=8):
    def format_docs(documents):
        return "\n\n".join(
            f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in documents
        )

    vectorstore = components.get_vector_store()
    prompt = lc_prompts.get_rag_qa_prompt3()

    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": n_top_result})

    # mmr can handle redundant documents better than similarity
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": n_top_result, "fetch_k": n_top_result + 40, "lambda_mult": 0.5},
    )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


@tool
def get_price(symbol: str) -> str:
    """Return latest cached price string for a ticker."""
    # TOBEREMOVE debug print
    print(f'get_price called with symbol={symbol}')
    return f"{symbol}=123.45"


@tool
def yo_back() -> str:
    """ Use this tool when user says 'yo' """
    print('yo tool called')  # TOBEREMOVE debug print
    return "yooo!!"


@tool
def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    print(f"get_current_temperature called with location={location}, unit={unit}")  # TOBEREMOVE debug print
    return 22.  # Dummy example


def create_demo_graph(llm, n_top_result=8):
    tools = [get_price, yo_back, get_current_temperature]
    try:
        llm = llm.bind_tools(tools, tool_choice='auto')
    except ValueError as e:
        if 'you must provide exactly one tool' in str(e):
            print('Failed to bind tools')

    system_message = SystemMessage(content="Understand user input and use tools if needed to answer.")
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        ("user", "{user_input}")
    ])

    def call_llm(state: MessagesState):
        resp = llm.invoke(state['messages'])
        return {'messages': [resp]}

    def call_llm2(state: MessagesState):
        # Get conversation so far
        messages = state['messages']

        # If you want to just add system_message at the front:
        full_messages = [system_message] + messages

        # Option A: pass messages directly
        # resp = llm.invoke(full_messages)

        # Option B: use prompt template
        resp = prompt | llm
        resp = resp.invoke({"user_input": messages[-1].content})

        return {"messages": [resp]}

    # 2) Router: if the last AI message contains tool_calls, go to tools, else END
    def route(state: MessagesState):
        print(state["messages"][-1])
        print(state["messages"][-1].tool_calls)

        return "tools" if state["messages"][-1].tool_calls else END

    def start_route(state: MessagesState):
        return "yo" if state['messages'] and state['messages'][-1].content == "yo" else "llm"

    def call_yoooo(state: MessagesState):
        return {'messages': state['messages'] + ["yooo"]}

    g = StateGraph(MessagesState)
    # Nodes
    g.add_node('llm', call_llm2)
    g.add_node('tools', ToolNode(tools))
    # g.add_node('yo', call_yoooo)

    # Edges
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")  # after tools, return to llm

    # g.add_conditional_edges(START, start_route, {"llm": "llm", "yo": "yo"})
    # g.add_edge('llm', END)
    # g.add_edge('yo', END)

    app = g.compile()

    return app


def create_demo_graph2(llm, n_top_result=8):
    tools = [get_price, yo_back, get_current_temperature]
    llm_with_tools = llm.bind_tools(tools, tool_choice='auto')

    system_message = SystemMessage(content="Understand user input and use tools if needed to answer."
                                           "Don't use tools if not needed.")
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        ("user", "{user_input}")
    ])

    def call_llm(state: MessagesState):
        resp = llm.invoke(state['messages'])
        return {'messages': [resp]}

    def call_llm2(state: MessagesState):
        # Get conversation so far
        messages = state['messages']

        # # Option A: pass messages directly
        # # If you want to just add system_message at the front:
        # full_messages = [system_message] + messages
        # resp = llm.invoke(full_messages)

        # Option B: use prompt template
        resp = prompt | llm_with_tools
        resp = resp.invoke({"user_input": messages[-1].content})

        return {"messages": [resp]}

    def call_llm_with_tools(state: MessagesState):
        # Get conversation so far
        messages = state['messages']

        # # Option A: pass messages directly
        # # If you want to just add system_message at the front:
        # full_messages = [system_message] + messages
        # resp = llm.invoke(full_messages)

        # Option B: use prompt template
        resp = prompt | llm
        resp = resp.invoke({"user_input": messages[-1].content})

        return {"messages": [resp]}

    # 2) Router: if the last AI message contains tool_calls, go to tools, else END
    def route(state: MessagesState):
        print(state["messages"][-1])
        print(state["messages"][-1].tool_calls)

        return "tools" if state["messages"][-1].tool_calls else 'llm-no-tools'

    def start_route(state: MessagesState):
        return "yo" if state['messages'] and state['messages'][-1].content == "yo" else "llm"

    def call_yoooo(state: MessagesState):
        return {'messages': state['messages'] + ["yooo"]}

    g = StateGraph(MessagesState)
    # Nodes
    g.add_node('llm', call_llm2)
    g.add_node('tools', ToolNode(tools))
    g.add_node('llm-no-tools', call_llm_with_tools)
    # g.add_node('yo', call_yoooo)

    # Edges
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route, {"tools": "tools", 'llm-no-tools': 'llm-no-tools'})
    g.add_edge("tools", "llm-no-tools")  # after tools, return to llm
    g.add_edge("llm-no-tools", END)

    # g.add_conditional_edges(START, start_route, {"llm": "llm", "yo": "yo"})
    # g.add_edge('llm', END)
    # g.add_edge('yo', END)

    app = g.compile()

    return app


class RagState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    query_model: Literal['quick', 'rerank', 'complex']
    extracted_queries: List[str] | None


def build_rag_app(
        llm: BaseChatModel,
        *,
        n_top_result: int = 8,
        cite_metadata_keys: Tuple[str, ...] = ("source", "page"),
        instruction: str = DEFAULT_RAG_INSTRUCTION,
):
    """
    Build a deterministic RAG graph that: question -> quick_retrieve -> synthesize.
    Returns a compiled LangGraph app.

    Args:
        llm: Any tool-less chat model that supports .invoke(prompt).content.
        vectorstore: Optional LangChain vectorstore used to build the retriever.
        n_top_result: Number of top documents to retrieve via MMR.
        cite_metadata_keys: Metadata keys to surface in the 'Sources' footer.
        instruction: Guidance added to the synthesis prompt.

    Returns:
        {"question": ..., "context": List[Document], "answer": str}
    """

    # Prefer not to mutate the incoming retriever; use a k-override if supported.
    def route_retriever(state: RagState):
        query_model = state.get("query_model", 'complex')
        if query_model == 'quick':
            return "quick_retrieve"
        elif query_model == 'rerank':
            return "rerank_retrieve"
        else:
            return "complex_retrieve"

    def quick_retrieve(state: RagState):
        docs: List[Document] = retrievers.quick_retrieve(state["question"], n_top_result)
        return {"context": docs}

    def rerank_retrieve(state: RagState):
        docs = retrievers.rerank_retrieve(state["question"], n_top_result)
        return {"context": docs}

    def complex_retrieve(state: RagState):
        docs, extracted_queries = retrievers.complex_retrieve(state["question"], llm, n_top_result)
        return {"context": docs, "extracted_queries": extracted_queries}

    def synthesize(state: RagState):
        docs: List[Document] = state.get("context", [])
        if docs:
            ctx_blocks = []
            for i, d in enumerate(docs):
                ctx_blocks.append(f"[{i + 1}] {d.page_content}")
            context_text = "\n\n".join(ctx_blocks)
        else:
            context_text = "No relevant context was retrieved."

        prompt = lc_prompts.format_rag_prompt(instruction, context_text, state['question'])

        ai_msg = llm.invoke(prompt)
        answer = ai_msg.content

        # Simple sources footer using chosen metadata keys
        if docs:
            lines = []
            for idx, d in enumerate(docs, start=1):
                meta = [str(d.metadata.get(k)) for k in cite_metadata_keys if d.metadata.get(k) is not None]
                if meta:
                    lines.append(f"[{idx}] " + " – ".join(meta))
            if lines:
                answer += "\n\nSources:\n" + "\n".join(lines)

        return {"answer": answer}

    g = StateGraph(RagState)
    g.add_node("quick_retrieve", quick_retrieve)
    g.add_node("rerank_retrieve", rerank_retrieve)
    g.add_node("complex_retrieve", complex_retrieve)
    g.add_node("synthesize", synthesize)

    # Conditional routing from START based on use_reranker flag
    g.add_conditional_edges(START, route_retriever, {
        "quick_retrieve": "quick_retrieve",
        "rerank_retrieve": "rerank_retrieve",
        "complex_retrieve": "complex_retrieve",
    })

    g.add_edge("quick_retrieve", "synthesize")
    g.add_edge("rerank_retrieve", "synthesize")
    g.add_edge("complex_retrieve", "synthesize")

    g.add_edge("synthesize", END)
    # g.add_conditional_edges("synthesize", route_output, {"print_result": "print_result", END: END})
    # g.add_edge("print_result", END)

    app = g.compile()
    return app
