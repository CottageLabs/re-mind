"""
Utilities for ChatSession
"""

import typing

from rich.markdown import Markdown
from rich.panel import Panel

if typing.TYPE_CHECKING:
    from re_mind.cli.chat_session import ChatSession

OUTPUT_MODE_SIMPLE = 'simple'
OUTPUT_MODE_DETAIL = 'detail'
OUTPUT_MODE_DEBUG = 'debug'


def get_prompt_message(cs: 'ChatSession') -> str:
    from re_mind.cli.components.model_options import HuggingFaceModelOption

    model_name = cs.config.get('model_option_name', 'unknown')
    output_mode = cs.config.get('output_mode', OUTPUT_MODE_SIMPLE)

    if isinstance(cs.model_option, HuggingFaceModelOption):
        device_display = ''
        if device:= cs.config.get('device', None):
            device_display = ' ({})'.format(device)
        return "[{model}{device}][{mode}]>  ".format(model=model_name, device=device_display, mode=output_mode)

    return "[{model}][{mode}]>  ".format(model=model_name, mode=output_mode)


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
                cs.print(Panel(doc.page_content, title=f"Document {i + 1}", expand=True))

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

    if extracted_queries:
        queries_text = "### Extracted Queries\n" \
                       + "\n".join(f"{i}. {eq}" for i, eq in enumerate(extracted_queries, 1))
        cs.print(Markdown(queries_text))
        cs.print()

    cs.print(Markdown(f"### Documents ({len(docs)} found)"))
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '')
        score = doc.metadata.get('ranker_score', 0)

        title = f"{i}. {source}" + (f" (page {page})" if page else "") + f" (Score: {score:.2f})"
        cs.print(Markdown(f"## {title}"))
        cs.print(Panel(content, expand=True))

