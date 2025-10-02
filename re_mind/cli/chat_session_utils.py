"""
Utilities for ChatSession
"""

import typing

from rich.markdown import Markdown
from rich.panel import Panel

if typing.TYPE_CHECKING:
    from re_mind.cli.remind_chat import ChatSession

OUTPUT_MODE_SIMPLE = 'simple'
OUTPUT_MODE_DETAIL = 'detail'
OUTPUT_MODE_DEBUG = 'debug'


def print_response(cs: 'ChatSession', response: dict):
    output_mode = cs.config.get('output_mode', OUTPUT_MODE_SIMPLE)
    if output_mode == OUTPUT_MODE_DEBUG:
        # Debug mode: show full response
        cs.print(response)

        extracted_queries = response.get("extracted_queries")
        context = response.get("context", [])
        if context:
            cs.print(Markdown("# Source Documents"))
            for i, doc in enumerate(context):
                cs.print(f"    [{i + 1}] {doc.metadata}")
                cs.print(Panel(doc.page_content, title=f"Document {i + 1}", expand=False))

        if extracted_queries:
            cs.print(Markdown("# Extracted Queries"))
            for i, query in enumerate(extracted_queries, 1):
                cs.print(f"  {i}. {query}")
            cs.print()


    elif output_mode == OUTPUT_MODE_DETAIL:
        # Detailed mode: show answer and search results
        print_search_results(cs, response.get('query', ''), response.get('context', []),
                             response.get('extracted_queries', []))

    if output_mode in (OUTPUT_MODE_DEBUG, OUTPUT_MODE_DETAIL):
        cs.print(Markdown(f"# {response['question']}"))

    # Standard response output
    output = Markdown(response['answer'])
    output = Panel(output)
    cs.print(output)


def print_search_results(cs: 'ChatSession', query: str, docs: list, extracted_queries: list):
    cs.print(Markdown("## Search Results"))
    cs.print(Markdown(f"**Query:** {query}"))
    cs.print()

    cs.print(Markdown(f"### Documents ({len(docs)} found)"))
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '')
        score = doc.metadata.get('ranker_score', 0)

        cs.print(Markdown(f"**{i}.** Score: {score:.2f}"))
        cs.print(Markdown(f"   **Source:** {source}" + (f" (page {page})" if page else "")))
        cs.print(Markdown(f"   **Content:** {content}"))
        cs.print()

    if extracted_queries:
        queries_text = "### Extracted Queries\n" \
                       + "\n".join(f"{i}. {eq}" for i, eq in enumerate(extracted_queries, 1))
        cs.print(Markdown(queries_text))
        cs.print()
