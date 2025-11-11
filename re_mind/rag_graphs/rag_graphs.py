from typing import TypedDict, List, Tuple, Literal

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from re_mind import lc_prompts, retrievers


class RagState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    query_model: Literal['quick', 'rerank', 'complex']  # KTODO rename query_mode
    extracted_queries: List[str] | None
    attached_items: List[str] | None
    # KTODO add user's system instructions


def build_rag_app(
        *,
        n_top_result: int = 8,
        cite_metadata_keys: Tuple[str, ...] = ("source", "page"),
):
    """
    Build a deterministic RAG graph that: question -> quick_retrieve -> synthesize.
    Returns a compiled LangGraph app.

    Args:
        llm: Any tool-less chat model that supports .invoke(prompt).content.
        n_top_result: Number of top documents to retrieve via MMR.
        cite_metadata_keys: Metadata keys to surface in the 'Sources' footer.

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

    def quick_retrieve(state: RagState, config: RunnableConfig):
        vectorstore = config["configurable"]["vectorstore"]
        attached_items = state.get("attached_items")
        docs: List[Document] = retrievers.quick_retrieve(
            state["question"], vectorstore, n_top_result, attached_items=attached_items
        )
        return {"context": docs}

    def rerank_retrieve(state: RagState, config: RunnableConfig):
        vectorstore = config["configurable"]["vectorstore"]
        device = config["configurable"].get("device")
        attached_items = state.get("attached_items")
        docs = retrievers.rerank_retrieve(state["question"], vectorstore, n_top_result, device=device,
                                          attached_items=attached_items)
        return {"context": docs}

    def complex_retrieve(state: RagState, config: RunnableConfig):
        vectorstore = config["configurable"]["vectorstore"]
        llm_instance = config["configurable"]["llm"]
        device = config["configurable"].get('device')
        attached_items = state.get("attached_items")

        docs, extracted_queries = retrievers.complex_retrieve(state["question"], vectorstore, llm_instance,
                                                              n_top_result, device=device,
                                                              attached_items=attached_items)
        return {"context": docs, "extracted_queries": extracted_queries}

    def synthesize(state: RagState, config: RunnableConfig):
        docs: List[Document] = state.get("context", [])
        if docs:
            ctx_blocks = []
            for i, d in enumerate(docs):
                ctx_blocks.append(f"[{i + 1}] {d.page_content}")
            context_text = "\n\n".join(ctx_blocks)
        else:
            context_text = "No relevant context was retrieved."

        sys_prompt = config["configurable"].get("sys_prompt", lc_prompts.DEFAULT_RAG_INSTRUCTION)
        prompt = lc_prompts.format_rag_prompt(sys_prompt, context_text, state['question'])

        llm_instance = config["configurable"]["llm"]
        ai_msg = llm_instance.invoke(prompt)
        answer = ai_msg.content

        # Simple sources footer using chosen metadata keys
        if docs:
            lines = []
            for idx, d in enumerate(docs, start=1):
                meta = [str(d.metadata.get(k)) for k in cite_metadata_keys if d.metadata.get(k) is not None]
                if meta:
                    lines.append(f"- [{idx}] " + " – ".join(meta))
            if lines:
                answer += "\n\nSources:\n" + "\n".join(lines)

        return {"answer": answer}

    # KTODO fix complex_retrieve with subgraph compilation
    """
    child = StateGraph(RagState)
    child.add_node("generate_queries", generate_queries)
    child.add_node("to_json", to_json)
    child.add_node("find_docs", find_docs)
    child.add_edge(START, "generate_queries")
    child.add_edge("generate_queries", "to_json")
    child.add_edge("to_json", "find_docs")
    child.add_edge("find_docs", END)
    complex_retrieve_subgraph = child.compile()

    # ----- Parent graph -----
    def quick_retrieve(state: RagState) -> dict: ...
    def rerank_retrieve(state: RagState) -> dict: ...
    def synthesize(state: RagState) -> dict:
        # ...use state["docs"] to produce final answer
        return {"answer": "..."}

    g = StateGraph(RagState)
    g.add_node("quick_retrieve", quick_retrieve)
    g.add_node("rerank_retrieve", rerank_retrieve)

    # Drop the compiled subgraph in as a single node:
    g.add_node("complex_retrieve", complex_retrieve_subgraph)
    """

    g = StateGraph(RagState)
    g.add_node("quick_retrieve", quick_retrieve)
    g.add_node("rerank_retrieve", rerank_retrieve)
    g.add_node("complex_retrieve", complex_retrieve)
    g.add_node("synthesize", synthesize)

    # Edges
    g.add_conditional_edges(START, route_retriever, {
        "quick_retrieve": "quick_retrieve",
        "rerank_retrieve": "rerank_retrieve",
        "complex_retrieve": "complex_retrieve",
    })

    g.add_edge("quick_retrieve", "synthesize")
    g.add_edge("rerank_retrieve", "synthesize")
    g.add_edge("complex_retrieve", "synthesize")

    g.add_edge("synthesize", END)

    app: CompiledStateGraph = g.compile()

    return app
