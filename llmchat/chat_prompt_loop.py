import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter

from llmchat.chat_session import ChatSession
from llmchat.chat_session_utils import get_prompt_message
from llmchat.commands import ChatCommand

log = logging.getLogger(__name__)


def validate_command_prefixes(commands: list[ChatCommand]) -> None:
    """
    Validate that no two commands have duplicate prefixes.

    Raises:
        ValueError: If duplicate prefixes are found.
    """
    prefixes = [cmd.prefix for cmd in commands]
    seen = set()
    duplicates = set()

    for prefix in prefixes:
        if prefix in seen:
            duplicates.add(prefix)
        seen.add(prefix)

    if duplicates:
        raise ValueError(f"Duplicate command prefixes found: {', '.join(sorted(duplicates))}")


def build_completer(commands: list[ChatCommand] | None = None) -> FuzzyCompleter:
    """
    Build a nested completer dynamically so 'use <dataset>' picks up new names.
    """
    data = {
        # TOBEREMOVE
        "/librarian": {"show": None, },  # KTODO add librarian commands
    }

    for helper in (commands or []):
        data.update(helper.create_nested_dict())

    nested = NestedCompleter.from_nested_dict(data)
    return FuzzyCompleter(nested)


class ChatPromptLoop:
    def __init__(self, chat_session: ChatSession, commands: list[ChatCommand]):
        self.chat_session = chat_session
        self.commands = commands
        self.prompt_session = PromptSession()
        self.completer = None

    def initialize(self) -> None:
        with self.chat_session.console.status("Initializing RAG session..."):
            self.chat_session.switch_llm(self.chat_session.config['model_option_name'])

        validate_command_prefixes(self.commands)
        self.completer = build_completer(self.commands)

    def run(self) -> None:
        while True:
            try:
                prompt_message = get_prompt_message(self.chat_session)
                user_input = self.prompt_session.prompt(prompt_message, completer=self.completer)
                if not user_input.strip():
                    continue
            except (EOFError, KeyboardInterrupt):
                print("Exiting chat...")
                break

            should_chat = True
            for command in self.commands:
                if command.is_match(user_input):
                    command.run(user_input, self.chat_session)
                    should_chat = False
                    break

            if not should_chat:
                continue

            resp = self.chat_session.chat(user_input)
            self.chat_session.print_response(resp)
