from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver


def build_simple_chat_app(llm: BaseChatModel):
    """
    Build a simple chat graph with only a chat node and conversation memory.

    Args:
        llm: Chat model that supports .invoke(messages)

    Returns:
        Compiled LangGraph app with MessagesState for conversation history
    """

    def chat(state: MessagesState, config: RunnableConfig):
        llm_instance = config["configurable"]["llm"]
        ai_msg = llm_instance.invoke(state['messages'])
        return {"messages": [ai_msg]}

    g = StateGraph(MessagesState)
    g.add_node("chat", chat)

    g.add_edge(START, "chat")
    g.add_edge("chat", END)

    memory = MemorySaver()
    app = g.compile(checkpointer=memory)

    configurable = {"llm": llm}
    app = app.with_config({"configurable": configurable})

    return app
