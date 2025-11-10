import logging
from typing import List

import numpy as np
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchText

from re_mind import llm_tasks
from re_mind.llm_tasks import retrieve_and_deduplicate_docs
from re_mind.rankers import rerankers
from re_mind.rankers.rerankers import rerank_with_qa_ranker, BGEQARanker

log = logging.getLogger(__name__)

# KTODO move to librarian

RANKER_SCORE_KEY = "ranker_score"


def build_qdrant_filter(attached_items: list[str] | None) -> Filter | None:
    """
    Build Qdrant filter for attached_items matching.
    Uses OR logic (should=[...]) to match ANY of these conditions:
    - metadata.source matching any item
    - metadata.source_root matching any item
    - metadata.hash_id matching any item

    Returns None if attached_items is empty or None.
    """
    if not attached_items:
        return None

    conditions = []
    for item in attached_items:
        conditions.append(FieldCondition(key="metadata.source", match=MatchText(text=item)))
        conditions.append(FieldCondition(key="metadata.source_root", match=MatchText(text=item)))
        conditions.append(FieldCondition(key="metadata.hash_id", match=MatchText(text=item)))

    return Filter(should=conditions)


def resolve_search_kwargs(k: int, attached_items: list[str] | None = None) -> dict:
    """
    Build search kwargs dict with Qdrant filter if attached_items is provided.

    Args:
        k: Number of results to retrieve
        attached_items: Optional list of items to filter by

    Returns:
        Dict with 'k' and optionally 'filter' keys
    """
    search_kwargs = {"k": k}
    qdrant_filter = build_qdrant_filter(attached_items)
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter
    return search_kwargs


def resolve_n_top(n_top_result: int | str, default: int = 10) -> int:
    if n_top_result == 'auto':
        return 200
    elif isinstance(n_top_result, int):
        return n_top_result
    else:
        log.debug(f"Invalid n_top_result value: {n_top_result}, using default {default}")
        return default


def quick_retrieve(question: str, vectorstore, n_top_result: int | str = 8, attached_items: list[str] | None = None) -> List[Document]:
    n_top = resolve_n_top(n_top_result, default=8)
    search_kwargs = resolve_search_kwargs(n_top, attached_items)
    docs_with_scores = vectorstore.similarity_search_with_score(question, **search_kwargs)

    # Add scores to metadata
    result_docs = []
    for doc, score in docs_with_scores:
        doc.metadata[RANKER_SCORE_KEY] = score
        result_docs.append(doc)

    if n_top_result == 'auto':
        result_docs = auto_select_top_n_doc(result_docs)

    return result_docs


def rerank_retrieve(question: str, vectorstore, n_top_result: int | str = 8, device: str | None = None, attached_items: list[str] | None = None) -> List[Document]:
    n_top = resolve_n_top(n_top_result, default=8)
    search_kwargs = resolve_search_kwargs(n_top + 20, attached_items)

    rerank_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    docs = rerank_retriever.invoke(question)

    ranker = BGEQARanker(device=device)
    result_docs = rerank_with_qa_ranker(question, docs, ranker, top_m=n_top)

    if n_top_result == 'auto':
        result_docs = auto_select_top_n_doc(result_docs)

    return result_docs


def complex_retrieve(question: str, vectorstore, llm, n_top_result: int | str = 'auto', device: str | None = None, attached_items: list[str] | None = None) -> tuple[List[Document], List[str]]:
    n_top = resolve_n_top(n_top_result, default=10)
    search_kwargs = resolve_search_kwargs(n_top + 20, attached_items)

    multi_query_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    extracted_queries = llm_tasks.extract_queries_from_input(llm, question)
    if not extracted_queries:
        log.warning("No queries extracted from input.")
        extracted_queries = [question]

    ranker = BGEQARanker(device=device)

    docs = list(retrieve_and_deduplicate_docs(extracted_queries, multi_query_retriever))

    # KTODO handle empty docs
    scores = rerankers.cal_score_matrix(extracted_queries, docs, ranker=ranker)
    top_docs = rerankers.aggregate_scores(scores, docs, k_final=n_top)

    result_docs = []
    for doc, score in top_docs:
        doc.metadata[RANKER_SCORE_KEY] = score
        result_docs.append(doc)

    if n_top_result == 'auto':
        result_docs = auto_select_top_n_doc(result_docs)

    return result_docs, extracted_queries



def auto_select_top_n_doc(docs: list[Document], n_ref=5):
    if not any(RANKER_SCORE_KEY in doc.metadata for doc in docs):
        log.debug("No documents have ranker_score, returning all documents")
        return docs

    ref_scores = [doc.metadata.get(RANKER_SCORE_KEY) for doc in docs[:n_ref] if RANKER_SCORE_KEY in doc.metadata]

    if len(ref_scores) < 2:
        log.debug(f"Not enough reference scores ({len(ref_scores)}), returning all documents")
        return docs

    ref_scores_array = np.array(ref_scores)
    mean_score = np.mean(ref_scores_array)
    std_score = np.std(ref_scores_array)
    threshold = mean_score - std_score

    filtered_docs = [doc for doc in docs if doc.metadata.get(RANKER_SCORE_KEY, float('-inf')) >= threshold]

    log.debug(f"Filtered {len(docs)} docs to {len(filtered_docs)} docs using threshold {threshold:.4f} (mean={mean_score:.4f}, std={std_score:.4f})")

    return filtered_docs


