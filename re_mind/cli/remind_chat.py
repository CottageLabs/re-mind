import logging

import rich
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter

from re_mind.cli.chat_session import ChatSession
from re_mind.cli.chat_session_utils import (
    print_response,
    get_prompt_message,
)
from re_mind.cli.commands import ChatCommand, ConfigsCommand, SearchCommand, ModelsCommand, ResetConfigCommand

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


def build_completer(commands: list[ChatCommand] | None = None) -> FuzzyCompleter:
    """
    Build a nested completer dynamically so 'use <dataset>' picks up new names.
    """
    data = {
        "/librarian": {"show": None, },  # KTODO add librarian commands
    }

    for helper in (commands or []):
        data.update(helper.create_nested_dict())

    nested = NestedCompleter.from_nested_dict(data)
    return FuzzyCompleter(nested)


def run_remind_chat():
    console = rich.console.Console()
    prompt_session = PromptSession()
    with console.status("Initializing RAG session..."):
        cs = ChatSession(console=console)
        cs.switch_llm(cs.config['model_option_name'])

    commands = [
        ConfigsCommand(),
        SearchCommand(),
        ModelsCommand(),
        ResetConfigCommand(),
    ]
    completer = build_completer(commands)
    # KTODO add assert to check command prefix not conflict

    # Chat loop
    while True:
        try:
            prompt_message = get_prompt_message(cs)
            user_input = prompt_session.prompt(prompt_message, completer=completer)
            if not user_input.strip():
                continue

        except (EOFError, KeyboardInterrupt):
            print("Exiting chat...")
            break

        should_chat = True
        for command in commands:
            if command.is_match(user_input):
                command.run(user_input, cs)
                should_chat = False
                break

        if not should_chat:
            continue

        # with cs.console.status("Generating response..."):
        resp = cs.chat(user_input)

        print_response(cs, resp)


def main():
    run_remind_chat()


if __name__ == '__main__':
    main()
