import logging
from dataclasses import dataclass

import rich
import torch
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind import cpaths
from re_mind.cli.commands import ChatCommand, ConfigsCommand, SearchCommand, ModelsCommand
from re_mind.cli.components.model_options import ModelOption
from re_mind.config_manager import ConfigManager
from re_mind.rag.rag_chat import RagChat
from re_mind.utils import re_mind_utils

# KTODO output_model [debugging / detail / simple]
# KTODO support librarian mode (list, add, remove documents)
# KTODO add command librarian
# KTODO add history
# KTODO cut llm as backend server
# KTODO support add prompt template
# KTODO support sequence messages
# KTODO support context window size management and auto summarization of chat history
# KTODO move device setting to rag_pipline state instead of global system

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


@dataclass
class ChatSession:
    config: dict = None
    console: rich.console.Console = None
    rag_chat: RagChat = None
    model_option: ModelOption = None

    def __post_init__(self):
        self.config_manager = ConfigManager(cpaths.CONFIG_PATH)

        if self.config is None:
            self.config = self.config_manager.load()

        if not self.config:
            self.config = {
                # Chat
                'max_width': 100,

                # RAG Session
                'temperature': 1.2,
                'n_top_result': 6,
                'model_option_name': 'gemma-3-1b',

                # Hugging Face LLM
                'device': 'cuda',
                'return_full_text': False,
            }

    @property
    def llm(self):
        return self.rag_chat.llm if self.rag_chat else None

    @property
    def console_width(self):
        return min(self.console.size.width, self.config.get('max_width', 100))

    def print(self, message=None):
        if message is None:
            self.console.print()
        else:
            self.console.print(message, width=self.console_width)

    @staticmethod
    def get_available_models():
        from re_mind.cli.components.model_options import HuggingFaceModelOption, OpenAIModelOption
        return [
            HuggingFaceModelOption(name='gemma-3-1b', model_id="google/gemma-3-1b-it"),
            OpenAIModelOption(name='gpt-5-nano', model='gpt-5-nano-2025-08-07'),
        ]

    def switch_llm(self, model_option_name: str):
        selected_option = None
        model_option_name = model_option_name.lower().strip()
        for model_option in self.get_available_models():
            if model_option.name.lower() == model_option_name:
                selected_option = model_option
                break

        if selected_option is None:
            self.print(f"[red]Model option '{model_option_name}' not found.[/red]")
            raise ValueError(f"Model option '{model_option_name}' not found.")

        selected_option.device = self.config.get('device', 'cuda')

        # delete previous model to free VRAM
        if self.model_option is not None:
            self.model_option.delete()

        # set new model
        self.model_option = selected_option
        try:
            llm = selected_option.create()
        except torch.OutOfMemoryError:
            log.warning("OutOfMemoryError when loading model on GPU, retrying on CPU")
            selected_option.device = 'cpu'
            self.switch_device('cpu')
            llm = selected_option.create()
        self.rag_chat = RagChat(llm=llm, n_top_result=self.config.get('n_top_result', 8))

        # save to config
        self.config['model_option_name'] = selected_option.name
        self.save_config()
        return self

    def switch_device(self, device):
        self.config['device'] = device
        re_mind_utils.set_global_device(device)  # KTODO make it per rag_chat instance instead of global
        self.save_config()

    def save_config(self):
        self.config_manager.save(self.config)

    def load_config(self) -> dict:
        loaded_config = self.config_manager.load()
        if loaded_config:
            self.config.update(loaded_config)
        return self.config


def run_remind_chat():
    console = rich.console.Console()
    prompt_session = PromptSession()
    with console.status("Initializing RAG session..."):
        cs = ChatSession(console=console)
        cs.switch_llm(cs.config['model_option_name'])

    completer_helpers = [
        ConfigsCommand(),
        SearchCommand(),
        ModelsCommand(),
    ]
    completer = build_completer(completer_helpers)
    # KTODO add assert to check command prefix not conflict

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
