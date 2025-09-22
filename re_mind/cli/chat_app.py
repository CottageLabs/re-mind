from abc import ABC, abstractmethod
from dataclasses import dataclass

import rich
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind.rag.rag_session import RagChat


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


class ConfigsCompleterHelper(CompleterHelper):
    def __init__(self, ):
        super().__init__('/configs')

    def run(self, user_input: str, cs: 'ChatSession'):
        cs.console.print(Markdown("## Current Configuration"))
        cs.console.print(cs.config)


def build_completer():
    """
    Build a nested completer dynamically so 'use <dataset>' picks up new names.
    """
    nested = NestedCompleter.from_nested_dict({
        "/configs": None,
        "/librarian": {"show": None, }
    })
    return FuzzyCompleter(nested)


@dataclass
class ChatSession:
    rag_chat: RagChat
    config: dict
    console: rich.console.Console


def run_chat_app():
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

    completer = build_completer()

    cs = ChatSession(
        rag_chat=rag_chat,
        config=config,
        console=console,
    )

    completer_helpers = [
        ConfigsCompleterHelper(),
    ]

    # Chat loop
    while True:
        try:
            user_input = prompt_session.prompt(">  ", completer=completer)
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

        width = min(cs.console.size.width, cs.config['max_width'])
        output = Markdown(resp['answer'])
        output = Panel(output)
        cs.console.print(output, width=width)


def main():
    run_chat_app()


if __name__ == '__main__':
    main()
