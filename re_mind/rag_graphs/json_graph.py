import json
from typing import TypedDict, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from re_mind import lc_prompts


class JsonState(TypedDict):
    user_input: str
    json_output: str | None
    parsed_json: dict | None
    error: str | None
    retry_count: int
    max_retries: int


class JsonGraphConfigurable(TypedDict):
    llm: BaseChatModel
    sys_prompt: NotRequired[str]


def generate_json(state: JsonState, config: RunnableConfig):
    llm_instance = config["configurable"]["llm"]
    sys_prompt = config["configurable"].get("sys_prompt", lc_prompts.DEFAULT_JSON_GENERATION_INSTRUCTION)
    user_input = state["user_input"]

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_input)
    ]
    result = llm_instance.invoke(messages)
    json_output = result.content.strip()

    return {"json_output": json_output}


def validate_json(state: JsonState):
    json_output = state.get("json_output", "")

    if not json_output:
        return {"error": "No JSON output generated", "retry_count": state["retry_count"] + 1}

    try:
        if "```json" in json_output:
            start = json_output.find("```json") + 7
            end = json_output.find("```", start)
            json_output = json_output[start:end].strip()
        elif "```" in json_output:
            start = json_output.find("```") + 3
            end = json_output.find("```", start)
            json_output = json_output[start:end].strip()

        parsed = json.loads(json_output)
        return {"parsed_json": parsed, "error": None}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}", "retry_count": state["retry_count"] + 1}


def should_retry(state: JsonState):
    if state.get("parsed_json") is not None:
        return "success"

    max_retries = state.get("max_retries", 3)
    if state["retry_count"] >= max_retries:
        return "failed"

    return "retry"


def build_json_graph():
    """
    Build a graph that generates and validates JSON output with retry support.

    The graph will attempt to generate valid JSON up to max_retries times before failing.

    Returns:
        CompiledStateGraph that takes {"user_input": str, "retry_count": 0, "max_retries": 3} and returns
        {"parsed_json": dict, "error": str | None}

    Note:
        The following parameters should be passed via config['configurable'] when invoking the graph
        (see JsonGraphConfigurable):
        - llm: LLM instance to use for JSON generation
        - sys_prompt: System instruction for JSON generation (default: DEFAULT_JSON_GENERATION_INSTRUCTION)

        State parameters:
        - max_retries: Maximum number of retry attempts (default: 3)
    """
    g = StateGraph(JsonState)
    g.add_node("generate_json", generate_json)
    g.add_node("validate_json", validate_json)

    g.add_edge(START, "generate_json")
    g.add_edge("generate_json", "validate_json")

    g.add_conditional_edges("validate_json", should_retry, {
        "success": END,
        "retry": "generate_json",
        "failed": END,
    })

    app: CompiledStateGraph = g.compile()

    return app
