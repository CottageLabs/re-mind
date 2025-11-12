from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from re_mind import llm_tasks, lc_prompts
from re_mind.rag_graphs.json_graph import build_json_graph
from re_mind.retrievers import retrieve_by_queries


class ComplexRetrieveState(TypedDict):
    question: str
    extracted_queries_str: str
    extracted_queries: List[str] | None
    documents: List[Document] | None


class ComplexRetrieveGraphConfigurable(TypedDict):
    llm: BaseChatModel
    vectorstore: object
    n_top_result: int | str
    attached_items: List[str] | None
    device: str | None


def extract_queries(state: ComplexRetrieveState, config: RunnableConfig):
    llm_instance = config["configurable"]["llm"]
    content = llm_tasks.get_extracted_queries_content(llm_instance, state["question"])
    return {"extracted_queries_str": content}


class ParseQueriesNode:
    def __init__(self):
        self.json_graph = build_json_graph()

    def __call__(self, state: ComplexRetrieveState, config: RunnableConfig):
        llm_instance = config["configurable"]["llm"]

        result = self.json_graph.invoke(
            {
                "user_input": state["extracted_queries_str"],
                "retry_count": 0
            },
            config={"configurable": {
                "llm": llm_instance,
                'sys_prompt': lc_prompts.POINTS_TO_JSON_LIST_INSTRUCTION,
                'output_type': 'list',
            }}
        )

        # If json_graph failed after retry, use parsed_json or fallback to wrapping json_output in a list
        extracted_queries = result.get("parsed_json") or [result.get("json_output")]

        return {"extracted_queries": extracted_queries}


def retrieve_documents(state: ComplexRetrieveState, config: RunnableConfig):
    configurable = config["configurable"]
    vectorstore = configurable["vectorstore"]
    n_top_result = configurable.get("n_top_result", "auto")
    attached_items = configurable.get("attached_items")
    device = configurable.get("device")

    documents = retrieve_by_queries(
        extracted_queries=state["extracted_queries"],
        vectorstore=vectorstore,
        n_top_result=n_top_result,
        attached_items=attached_items,
        device=device
    )

    return {"documents": documents}


def build_complex_retrieve_graph():
    """
    Build a graph that extracts queries from a question and retrieves documents.

    Returns:
        CompiledStateGraph that takes {"question": str} and returns {"extracted_queries": List[str], "documents": List[Document]}

    Note:
        The following parameters should be passed via config['configurable'] when invoking the graph
        (see ComplexRetrieveGraphConfigurable):
        - llm: LLM instance to use for query extraction
        - vectorstore: Vector store to retrieve documents from
        - n_top_result: (optional) Number of top results to return, defaults to 'auto'
        - attached_items: (optional) List of items to filter by
        - device: (optional) Device to use for ranker model
    """
    g = StateGraph(ComplexRetrieveState)
    g.add_node("extract_queries", extract_queries)
    g.add_node("parse_queries", ParseQueriesNode())
    g.add_node("retrieve_documents", retrieve_documents)

    g.add_edge(START, "extract_queries")
    g.add_edge("extract_queries", "parse_queries")
    g.add_edge("parse_queries", "retrieve_documents")
    g.add_edge("retrieve_documents", END)

    app: CompiledStateGraph = g.compile()

    return app


def main():
    from re_mind.components import get_llm, get_vector_store

    print("Building complex retrieve graph...")
    graph = build_complex_retrieve_graph()

    print("\nInitializing LLM and vector store...")
    llm = get_llm()
    vectorstore = get_vector_store()

    question = "What are the key differences between Python and JavaScript in terms of type systems and async programming?"
    print(f"\nInput question:\n{question}\n")

    print("Extracting queries and retrieving documents...")
    result = graph.invoke(
        {"question": question},
        config={
            "configurable": {
                "llm": llm,
                "vectorstore": vectorstore,
                "n_top_result": "auto"
            }
        }
    )

    print("\nExtracted queries (raw):")
    print(result.get("extracted_queries_str", ""))

    print("\nParsed queries:")
    for i, query in enumerate(result.get("extracted_queries", []), 1):
        print(f"{i}. {query}")

    print("\nRetrieved documents:")
    documents = result.get("documents", [])
    print(f"Total documents: {len(documents)}")
    for i, doc in enumerate(documents[:5], 1):
        print(f"\n{i}. {doc.metadata.get('source', 'Unknown source')}")
        print(f"   Score: {doc.metadata.get('ranker_score', 'N/A')}")
        print(f"   Content preview: {doc.page_content[:200]}...")

    print("\nDemo completed!")


if __name__ == '__main__':
    main()
