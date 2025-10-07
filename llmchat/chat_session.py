import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import rich
import torch
from langchain_core.messages import HumanMessage
from rich.markdown import Markdown
from rich.panel import Panel

from llmchat.config_manager import ConfigManager
from llmchat.cpaths import CONFIG_PATH

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
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


@dataclass
class ChatSession:
    # Configurations
    config: dict = field(default_factory=dict)
    default_config: dict = None
    config_path: str = CONFIG_PATH

    llm: 'Any' = None
    rag_chain: 'CompiledStateGraph' = None
    device: str = None
    model_option: 'ModelOption' = None
    available_models: list['ModelOption'] = None
    vectorstore: 'Any' = None
    thread_id: str = None

    console: rich.console.Console = None

    def __post_init__(self):
        if self.console is None:
            self.console = rich.console.Console()

        if self.available_models is None:
            from llmchat.components.model_options import HuggingFaceModelOption, OpenAIModelOption
            self.available_models = [
                HuggingFaceModelOption(name='gemma-3-1b', model_id="google/gemma-3-1b-it"),
                OpenAIModelOption(name='gpt-5-nano', model='gpt-5-nano-2025-08-07'),
            ]

        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())

        self.config_manager = ConfigManager(self.config_path)

        if not self.config:
            self.config = self.config_manager.load() or (self.default_config or DEFAULT_CONFIG).copy()

    @property
    def console_width(self):
        return min(self.console.size.width, self.config.get('max_width', 100))

    def print(self, message=None):
        if message is None:
            self.console.print()
        else:
            self.console.print(message, width=self.console_width)

    def build_rag_chain(self, llm, vector_store) -> 'CompiledStateGraph':
        from llmchat import pipelines
        return pipelines.build_simple_chat_app(llm)

    def switch_llm(self, model_option_name: str):
        selected_option = None
        model_option_name = model_option_name.lower().strip()
        for model_option in self.available_models:
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

        if self.llm is None:
            self.rag_chain = self.build_rag_chain(llm, self.vectorstore)

        self.llm = llm
        self.device = self.config.get('device', 'cpu')

        # save to config
        self.config['model_option_name'] = selected_option.name
        self.save_config()
        return self

    def switch_device(self, device):
        self.config['device'] = device
        self.save_config()

    def save_config(self):
        self.config_manager.save(self.config)

    def load_config(self) -> dict:
        loaded_config = self.config_manager.load()
        if loaded_config:
            self.config.update(loaded_config)
        return self.config

    def chat(self, user_input,
             state: dict = None,
             configurable: dict = None) -> dict:
        state = state or {}
        configurable = configurable or {}

        final_state = {'messages': [HumanMessage(user_input)]} | state
        final_configurable = {
                                 "thread_id": self.thread_id,
                                 'llm': self.llm,
                                 'device': self.device,
                                 # KTODO handle n_top_result, temperature, etc.
                             } | configurable
        response = self.rag_chain.invoke(
            final_state,
            config={'configurable': final_configurable}
        )
        return response

    def print_response(self, response: dict) -> None:
        output = Markdown(response['messages'][-1].content)
        output = Panel(output)
        self.print(output)
