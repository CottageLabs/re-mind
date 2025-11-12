from typing import TypedDict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from re_mind import llm_tasks


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
    # KTODO create a node to use build_json_graph

    g.add_edge(START, "extract_queries")
    g.add_edge("extract_queries", END)

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

    print("\nExtracted queries:")
    print(result.get("extracted_queries_str", ""))

    print("\nDemo completed!")


if __name__ == '__main__':
    main()
