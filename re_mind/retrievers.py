import logging
from typing import List

from langchain_core.documents import Document

from re_mind import llm_tasks
from re_mind.llm_tasks import retrieve_and_deduplicate_docs
from re_mind.rankers import rerankers
from re_mind.rankers.rerankers import rerank_with_qa_ranker, BGEQARanker

log = logging.getLogger(__name__)

# KTODO move to librarian

def quick_retrieve(question: str, vectorstore, n_top_result: int = 8) -> List[Document]:
    quick_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": n_top_result, "fetch_k": n_top_result + 40, "lambda_mult": 0.5},
    )
    return quick_retriever.invoke(question)


def rerank_retrieve(question: str, vectorstore, n_top_result: int = 8, device: str | None = None) -> List[Document]:
    rerank_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": n_top_result + 20}
    )
    docs = rerank_retriever.invoke(question)
    ranker = BGEQARanker(device=device)
    return rerank_with_qa_ranker(question, docs, ranker, top_m=n_top_result)


def complex_retrieve(question: str, vectorstore, llm, n_top_result: int = 8, device: str | None = None) -> tuple[List[Document], List[str]]:
    multi_query_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": n_top_result + 20}
    )

    extracted_queries = llm_tasks.extract_queries_from_input(llm, question)
    if not extracted_queries:
        log.warning("No queries extracted from input.")
        extracted_queries = [question]

    ranker = BGEQARanker(device=device)

    docs = list(retrieve_and_deduplicate_docs(extracted_queries, multi_query_retriever))
    scores = rerankers.cal_score_matrix(extracted_queries, docs, ranker=ranker)
    top_docs = rerankers.aggregate_scores(scores, docs, k_final=n_top_result)

    result_docs = []
    for doc, score in top_docs:
        doc.metadata["ranker_score"] = score
        result_docs.append(doc)

    return result_docs, extracted_queries
