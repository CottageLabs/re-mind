"""
LLM-based utility functions for various tasks.
"""

from re_mind.lc_prompts import get_query_extraction_prompt
from re_mind.utils import iter_utils


def get_extracted_queries_content(llm, user_input) -> str:
    """Invoke LLM to extract queries and return raw content. Returns empty string if NO_QUERY."""
    prompt = get_query_extraction_prompt(user_input)
    result = llm.invoke(prompt)
    content = result.content

    if "<NO_QUERY>" in content:
        return ''

    return content


def extract_queries_from_input(llm, user_input) -> list[str]:
    """Extract queries from user input using the query extraction prompt."""
    content = get_extracted_queries_content(llm, user_input)

    if not content:
        return []

    queries = [line.strip() for line in content.split('\n') if line.strip()]
    return queries


def retrieve_and_deduplicate_docs(extracted_queries, multi_query_retriever):
    """Retrieve documents for multiple queries and remove duplicates."""

    def doc_generator():
        for query in extracted_queries:
            docs = multi_query_retriever.invoke(query)
            yield from docs

    return iter_utils.remove_duplicate(doc_generator(), id_fn=lambda d: d.metadata['_id'])