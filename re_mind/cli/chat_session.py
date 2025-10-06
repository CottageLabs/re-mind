import logging
from dataclasses import dataclass

import rich
import torch

from re_mind import cpaths
from re_mind.cli.chat_session_utils import OUTPUT_MODE_SIMPLE
from re_mind.config_manager import ConfigManager
from re_mind.rag.rag_chat import RagChat

log = logging.getLogger(__name__)

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


@dataclass
class ChatSession:
    config: dict = None
    console: rich.console.Console = None
    rag_chat: RagChat = None
    model_option: 'ModelOption' = None

    def __post_init__(self):
        self.config_manager = ConfigManager(cpaths.CONFIG_PATH)

        if self.config is None:
            self.config = self.config_manager.load()

        if not self.config:
            self.config = DEFAULT_CONFIG.copy()

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
        if self.rag_chat is None:
            self.rag_chat = RagChat(
                llm=llm,
                n_top_result=self.config.get('n_top_result', 8),
                device=self.config.get('device', 'cpu'),
            )
        else:
            self.rag_chat.llm = llm
            self.rag_chat.device = self.config.get('device', 'cpu')

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
