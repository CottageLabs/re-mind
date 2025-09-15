from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from re_mind import components, lc_prompts


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
    except ValueError as e :
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
