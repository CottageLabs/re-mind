from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from .states import RagState
from .nodes import (
    route_retriever,
    quick_retrieve,
    rerank_retrieve,
    complex_retrieve,
    synthesize,
)


def build_rag_app():
    """
    Build a deterministic RAG graph that: question -> quick_retrieve -> synthesize.
    Returns a compiled LangGraph app.

    Returns:
        {"question": ..., "context": List[Document], "answer": str}

    Note:
        The following parameters should be passed via config['configurable'] when invoking the graph:
        - n_top_result: Number of top documents to retrieve (default: "auto")
        - cite_metadata_keys: Metadata keys to surface in the 'Sources' footer (default: ("source", "page"))
    """

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
