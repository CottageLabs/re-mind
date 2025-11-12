from typing import TypedDict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from re_mind import llm_tasks
from re_mind.rag_graphs.json_graph import build_json_graph


class ComplexRetrieveState(TypedDict):
    question: str
    extracted_queries_str: str
    extracted_queries: List[str] | None


class ComplexRetrieveGraphConfigurable(TypedDict):
    llm: BaseChatModel


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
            config={"configurable": {"llm": llm_instance}}
        )

        # If json_graph failed after retry, use parsed_json or fallback to wrapping json_output in a list
        extracted_queries = result.get("parsed_json") or [result.get("json_output")]

        return {"extracted_queries": extracted_queries}


def build_complex_retrieve_graph():
    """
    Build a graph that extracts queries from a given question.

    Returns:
        CompiledStateGraph that takes {"question": str} and returns {"extracted_queries": List[str]}

    Note:
        The following parameters should be passed via config['configurable'] when invoking the graph
        (see ComplexRetrieveGraphConfigurable):
        - llm: LLM instance to use for query extraction
    """
    g = StateGraph(ComplexRetrieveState)
    g.add_node("extract_queries", extract_queries)
    g.add_node("parse_queries", ParseQueriesNode())

    g.add_edge(START, "extract_queries")
    g.add_edge("extract_queries", "parse_queries")
    g.add_edge("parse_queries", END)

    app: CompiledStateGraph = g.compile()

    return app


def main():
    from re_mind.components import get_llm

    print("Building complex retrieve graph...")
    graph = build_complex_retrieve_graph()

    print("\nInitializing LLM...")
    llm = get_llm()

    question = "What are the key differences between Python and JavaScript in terms of type systems and async programming?"
    print(f"\nInput question:\n{question}\n")

    print("Extracting queries...")
    result = graph.invoke(
        {"question": question},
        config={"configurable": {"llm": llm}}
    )

    print("\nExtracted queries (raw):")
    print(result.get("extracted_queries_str", ""))

    print("\nParsed queries:")
    for i, query in enumerate(result.get("extracted_queries", []), 1):
        print(f"{i}. {query}")

    print("\nDemo completed!")


if __name__ == '__main__':
    main()
