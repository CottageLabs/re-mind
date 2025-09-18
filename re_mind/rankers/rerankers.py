from abc import ABC, abstractmethod

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

        self.device = device or re_mind_utils.get_global_device()
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

        self.device = device or re_mind_utils.get_global_device()
        self.batch_size = batch_size
        self._reranker = FlagReranker(model, use_fp16=True, device=self.device)

    def rank(self, pairs: list[tuple[str, str]]) -> list[float]:
        formatted_pairs = [[query, passage] for query, passage in pairs]
        scores = self._reranker.compute_score(formatted_pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]


def rerank_with_qa_ranker(query, docs, ranker: QARanker, top_m=8):
    pairs = [(query, d.page_content) for d in docs]
    scores = ranker.rank(pairs)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_m]
    result_docs = []
    for i in order:
        doc = docs[i]
        doc.metadata["ranker_score"] = scores[i]
        result_docs.append(doc)
    return result_docs


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


if __name__ == '__main__':
    main()
