import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import rich
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind import retrievers
from re_mind.rag.rag_chat import RagChat


# KTODO support change model
# KTODO support search only mode
# KTODO support librarian mode (list, add, remove documents)
# KTODO support switch query mode
# KTODO add command librarian
# KTODO add history


class CompleterHelper(ABC):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def is_match(self, user_input: str) -> bool:
        return user_input.startswith(self.prefix)

    def create_nested_dict(self):
        return {self.prefix: None}

    @abstractmethod
    def run(self, user_input: str, cs: 'ChatSession'):
        raise NotImplementedError


class ConfigsCH(CompleterHelper):
    def __init__(self, ):
        super().__init__('/configs')

    def run(self, user_input: str, cs: 'ChatSession'):
        cs.print(Markdown("## Configuration Commands"))
        cs.print(Markdown("""```
Examples:
/configs                  # show configs
/configs n_top_result 8   # change configs
```"""))
        cs.print()
        cs.print(Markdown("## Current Configuration"))
        cs.print(cs.config)


class SearchCH(CompleterHelper):
    def __init__(self, ):
        super().__init__('/search ')

    def run(self, user_input: str, cs: 'ChatSession'):
        query = re.sub(r'^/search\s+', '', user_input).strip()
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


def build_completer(completer_helpers: list[CompleterHelper] | None = None) -> FuzzyCompleter:
    """
    Build a nested completer dynamically so 'use <dataset>' picks up new names.
    """
    # KTODO fix /search not working
    data = {
        "/configs": None,
        "/librarian": {"show": None, }
    }

    for helper in (completer_helpers or []):
        data.update(helper.create_nested_dict())

    nested = NestedCompleter.from_nested_dict(data)
    return FuzzyCompleter(nested)


@dataclass
class ChatSession:
    rag_chat: RagChat
    config: dict
    console: rich.console.Console

    @property
    def llm(self):
        return self.rag_chat.llm

    @property
    def console_width(self):
        return min(self.console.size.width, self.config.get('max_width', 100))

    def print(self, message=None):
        if message is None:
            self.console.print()
        else:
            self.console.print(message, width=self.console_width)


def run_remind_chat():
    config = {
        # Chat
        'max_width': 100,

        # RAG Session
        'temperature': 1.2,
        'n_top_result': 6,

        # Hugging Face LLM
        'device': 'cuda',
        'return_full_text': False,
    }
    console = rich.console.Console()
    prompt_session = PromptSession()
    with console.status("Initializing RAG session..."):
        # rag_chat = RagChat.create_by_huggingface(
        #     temperature=config.get('temperature', 1.2),
        #     n_top_result=config.get('n_top_result', 8),
        #     device=config.get('device'),
        #     return_full_text=config.get('return_full_text', False)
        # )
        rag_chat = RagChat.create_by_openai(n_top_result=config.get('n_top_result', 8))


    cs = ChatSession(
        rag_chat=rag_chat,
        config=config,
        console=console,
    )

    completer_helpers = [
        ConfigsCH(),
        SearchCH(),
    ]
    completer = build_completer(completer_helpers)

    # Chat loop
    while True:
        try:
            user_input = prompt_session.prompt(">  ", completer=completer)
            if not user_input.strip():
                continue

        except (EOFError, KeyboardInterrupt):
            print("Exiting chat...")
            break

        should_chat = True
        for helper in completer_helpers:
            if helper.is_match(user_input):
                helper.run(user_input, cs)
                should_chat = False
                break

        if not should_chat:
            continue

        # with cs.console.status("Generating response..."):
        resp = cs.rag_chat.chat(user_input)

        output = Markdown(resp['answer'])
        output = Panel(output)
        cs.print(output)


def main():
    run_remind_chat()


if __name__ == '__main__':
    main()
