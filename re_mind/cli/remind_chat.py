import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind import cpaths
from re_mind.cli.chat_session import ChatSession
from re_mind.cli.chat_session_utils import (
    get_prompt_message, OUTPUT_MODE_SIMPLE,
)
from re_mind.cli.commands import ChatCommand, ConfigsCommand, SearchCommand, ModelsCommand, ResetConfigCommand
from re_mind.cli.components.model_options import HuggingFaceModelOption, OpenAIModelOption

# KTODO support librarian mode (list, add, remove documents)
# KTODO add command librarian
# KTODO add history
# KTODO cut llm as backend server
# KTODO support add prompt template
# KTODO support sequence messages
# KTODO support context window size management and auto summarization of chat history
# KTODO move device setting to rag_pipline state instead of global system
# KTODO support switch vector store

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


class ChatLoop:
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

    def print_response(self, user_input: str) -> None:
        resp = self.chat_session.chat(user_input)

        # Standard response output
        output = Markdown(resp['answer'])
        output = Panel(output)
        self.chat_session.print(output)

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

            self.print_response(user_input)


class RemindChatLoop(ChatLoop):
    def print_response(self, user_input: str) -> None:
        from re_mind.cli.chat_session_utils import print_response
        resp = self.chat_session.chat(user_input)
        print_response(self.chat_session, resp)


def main():
    DEFAULT_CONFIG = {
        # Chat
        'max_width': 100,

        # RAG Session
        'temperature': 1.2,
        'n_top_result': 6,
        'model_option_name': 'gemma-3-1b',
        'output_mode': OUTPUT_MODE_SIMPLE,

        # Hugging Face LLM
        'device': 'cuda',
        'return_full_text': False,
    }

    model_options = [
        HuggingFaceModelOption(name='gemma-3-1b', model_id="google/gemma-3-1b-it"),
        OpenAIModelOption(name='gpt-5-nano', model='gpt-5-nano-2025-08-07'),
    ]
    commands = [
        ConfigsCommand(),
        SearchCommand(),
        ModelsCommand(),
        ResetConfigCommand(),
    ]
    cs = ChatSession(available_models=model_options, default_config=DEFAULT_CONFIG,
                     config_path=cpaths.CONFIG_PATH)
    chat_loop = ChatLoop(cs, commands)
    chat_loop.initialize()
    chat_loop.run()


if __name__ == '__main__':
    main()
