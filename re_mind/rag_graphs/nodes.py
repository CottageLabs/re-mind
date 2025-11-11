from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from re_mind import lc_prompts, retrievers
from .states import RagState


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
    n_top_result = config["configurable"].get("n_top_result", "auto")
    attached_items = state.get("attached_items")
    docs: List[Document] = retrievers.quick_retrieve(
        state["question"], vectorstore, n_top_result, attached_items=attached_items
    )
    return {"context": docs}


def rerank_retrieve(state: RagState, config: RunnableConfig):
    vectorstore = config["configurable"]["vectorstore"]
    n_top_result = config["configurable"].get("n_top_result", "auto")
    device = config["configurable"].get("device")
    attached_items = state.get("attached_items")
    docs = retrievers.rerank_retrieve(
        state["question"], vectorstore, n_top_result, device=device, attached_items=attached_items
    )
    return {"context": docs}


def complex_retrieve(state: RagState, config: RunnableConfig):
    vectorstore = config["configurable"]["vectorstore"]
    n_top_result = config["configurable"].get("n_top_result", "auto")
    llm_instance = config["configurable"]["llm"]
    device = config["configurable"].get('device')
    attached_items = state.get("attached_items")

    docs, extracted_queries = retrievers.complex_retrieve(
        state["question"], vectorstore, llm_instance, n_top_result, device=device, attached_items=attached_items
    )
    return {"context": docs, "extracted_queries": extracted_queries}


def synthesize(state: RagState, config: RunnableConfig):
    docs: List[Document] = state.get("context", [])
    cite_metadata_keys = config["configurable"].get("cite_metadata_keys", ("source", "page"))

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

    if docs:
        lines = []
        for idx, d in enumerate(docs, start=1):
            meta = [str(d.metadata.get(k)) for k in cite_metadata_keys if d.metadata.get(k) is not None]
            if meta:
                lines.append(f"- [{idx}] " + " – ".join(meta))
        if lines:
            answer += "\n\nSources:\n" + "\n".join(lines)

    return {"answer": answer}
