from rich.markdown import Markdown
from re_mind import retrievers
from .chat_command_base import ChatCommand
from .command_utils import extract_command_arg


class SearchCommand(ChatCommand):
    def __init__(self, ):
        super().__init__('/search')

    def run(self, user_input: str, cs: 'ChatSession'):
        query = extract_command_arg(self.prefix, user_input)
        if not query:
            cs.print("Please provide a search query after '/search'.")
            cs.print("Example: /search What is the capital of France?")
            return

        docs, extracted_queries = retrievers.complex_retrieve(query, cs.llm, cs.config['n_top_result'])

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

            cs.print(Markdown(f"**{i}.** Score: {score:.2f}"))
            cs.print(Markdown(f"   **Source:** {source}" + (f" (page {page})" if page else "")))
            cs.print(Markdown(f"   **Content:** {content}"))
            cs.print()
