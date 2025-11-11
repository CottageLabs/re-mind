from typing import TypedDict, List, Literal

from langchain_core.documents import Document


class RagState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    query_model: Literal['quick', 'rerank', 'complex']  # KTODO rename query_mode
    extracted_queries: List[str] | None
    attached_items: List[str] | None
    # KTODO add user's system instructions
