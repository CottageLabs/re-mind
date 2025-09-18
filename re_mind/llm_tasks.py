"""
LLM-based utility functions for various tasks.
"""

from re_mind.lc_prompts import get_query_extraction_prompt


def extract_queries_from_input(llm, user_input):
    """Extract queries from user input using the query extraction prompt."""
    prompt = get_query_extraction_prompt(user_input)
    result = llm.invoke(prompt)

    if "<NO_QUERY>" in result.content:
        return []

    # Split content into lines and filter out empty lines
    queries = [line.strip() for line in result.content.split('\n') if line.strip()]
    return queries