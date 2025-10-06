from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document

from re_mind import components
from re_mind.utils import raq_utils, re_mind_utils


class QARanker(ABC):

    @abstractmethod
    def rank(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise NotImplementedError


class CEQARanker(QARanker):
    def __init__(
            self,
            model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
            device: str | None = None,
            batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.device = device or re_mind_utils.get_sys_device()
        self.batch_size = batch_size
        self._encoder = CrossEncoder(model, device=self.device)

    def rank(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._encoder.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]


class BGEQARanker(QARanker):
    def __init__(
            self,
            model: str = "BAAI/bge-reranker-v2-m3",
            device: str | None = None,
            batch_size: int = 32,
    ) -> None:
        from FlagEmbedding import FlagReranker

        self.device = device or re_mind_utils.get_sys_device()
        self.batch_size = batch_size
        self._reranker = FlagReranker(model,
                                      use_fp16=self.device != 'cpu',
                                      device=self.device)

    def rank(self, pairs: list[tuple[str, str]]) -> list[float]:
        formatted_pairs = [[query, passage] for query, passage in pairs]
        scores = self._reranker.compute_score(formatted_pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]


def rerank_with_qa_ranker(query, docs, ranker: QARanker, top_m=8):
    pairs = [(query, d.page_content) for d in docs]
    scores = ranker.rank(pairs)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_m]
    doc_score_pairs = [(docs[i], scores[i]) for i in order]
    for d, s in doc_score_pairs:
        d.metadata["ranker_score"] = s
    result_docs = [d for d, _ in doc_score_pairs]
    return result_docs


def z_norm_per_query(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Z-normalize along axis=0 per *query* row-wise (axis=1 across docs),
    ignoring NaNs. Returns same shape; NaNs preserved.
    Args:
        scores: 2D numpy array with shape (num_queries, num_docs) (Q x N)
    """
    # mean/std per row (query)
    mu = np.nanmean(scores, axis=1, keepdims=True)
    sd = np.nanstd(scores, axis=1, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    z = (scores - mu) / sd
    return z


def cal_score_matrix(queries: list[str], docs: list[Document], ranker: QARanker) -> np.ndarray:
    """Calculate ranking scores for all query-document pairs and return as numpy array.

    Args:
        queries: List of query strings
        docs: List of Document objects
        ranker: QARanker instance to compute scores

    Returns:
        Numpy array with queries as rows, documents as columns, and scores as values
    """

    assert len(queries) > 0, "No queries provided"
    assert len(docs) > 0, "No documents provided"

    # Create all query-document pairs
    all_pairs = []
    for query in queries:
        for doc in docs:
            all_pairs.append((query, doc.page_content))

    # Get scores for all pairs
    scores = ranker.rank(all_pairs)

    # Reshape scores into matrix format
    score_matrix = np.array(scores).reshape(len(queries), len(docs))

    # Apply z-score normalization per query (row-wise)
    normalized_matrix = z_norm_per_query(score_matrix)
    return normalized_matrix


def topk_mean(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    """
    Mean of the top-k values along an axis, ignoring NaNs.
    For NaNs, we treat them as -inf so they don't enter the top-k.
    """
    k = min(k, a.shape[axis])
    a_ = np.where(np.isnan(a), -np.inf, a)
    # np.partition gives kth order stats; we take the last k slices
    part = np.partition(a_, -k, axis=axis)
    topk = np.take(part, indices=range(-k, 0), axis=axis)
    # If fewer than k finite values, -inf will be present; replace with NaN then mean with nanmean
    topk = np.where(np.isneginf(topk), np.nan, topk)
    return np.nanmean(topk, axis=axis)


def aggregate_scores(
        scores: np.ndarray,
        docs: list[str],
        tau_or: float = 0.0,  # gate threshold after z-norm
        coverage_tau: float = 0.0,  # what counts as "covered" after z-norm
        topk: int = 2,  # TOP-k pooling
        w_or: float = 0.15,  # small OR bonus
        w_cov: float = 0.10,  # small coverage bonus
        k_final: int = 12,  # final top-k to return
) -> List[Tuple[str, float]]:
    """
    Implements: OR gate, then rank by TOP-k + small OR + small coverage bonus.
    Returns a list of (doc_id, final_score, components) sorted desc by score.
    """

    Z = scores  # Q x N, normalized per query

    assert Z.shape[1] == len(docs)

    # OR = max over queries (ignore NaNs)
    OR = np.nanmax(Z, axis=0)  # shape (N,)
    # gate: keep docs with OR > tau_or
    keep = OR > tau_or
    if not np.any(keep):
        return []

    Zk = Z[:, keep]  # Q x N_kept
    docs_k = [d for d, m in zip(docs, keep) if m]
    OR_k = OR[keep]

    # TOP-k mean
    TOPK = topk_mean(Zk, k=topk, axis=0)  # shape (N_kept,)

    # Coverage = count of queries where z-score > coverage_tau
    covered = np.sum(Zk > coverage_tau, axis=0)  # integer array (N_kept,)

    # Final score
    final = TOPK + w_or * OR_k + w_cov * covered

    # Package with components for debugging
    ranked_idx = np.argsort(-final)
    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for i in ranked_idx[:k_final]:
        out.append((docs_k[i], float(final[i])))
    return out


def main():
    query = "Tell me about data cleaning and preprocessing."
    vectorstore = components.get_vector_store()
    rerank_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 60, "fetch_k": 180, "lambda_mult": 0.5},
    )
    docs: list[Document] = rerank_retriever.invoke(query)
    # ranker = CEQARanker()
    ranker = BGEQARanker()
    final_docs = rerank_with_qa_ranker(query, docs, ranker, top_m=8)

    # for i, doc in enumerate(final_docs):
    #     print(f"--- Document {i + 1} ---")
    #     print(doc.page_content)
    #     print(doc.metadata)
    #     print()

    raq_utils.print_ref(final_docs)


def main2():
    values = np.random.rand(10, 5)
    x = values.mean(axis=1, keepdims=True)
    print(x.shape)
    print(values.mean(axis=1).shape)


if __name__ == '__main__':
    main2()
