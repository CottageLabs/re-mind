import logging

from llmchat.chat_prompt_loop import ChatPromptLoop
from llmchat.chat_session import ChatSession
from llmchat.commands import ConfigsCommand, ModelsCommand, ResetConfigCommand
from llmchat.components.model_options import HuggingFaceModelOption, OpenAIModelOption
from llmchat.cpaths import CONFIG_PATH

log = logging.getLogger(__name__)


def main():
    DEFAULT_CONFIG = {
        # Chat
        'max_width': 100,

        # Chat Session
        'temperature': 1.2,
        'model_option_name': 'gemma-3-1b',
        'output_mode': 'simple',

        # Hugging Face LLM
        'device': 'cpu',
        'return_full_text': False,
    }

    model_options = [
        HuggingFaceModelOption(name='gemma-3-1b', model_id="google/gemma-3-1b-it"),
        OpenAIModelOption(name='gpt-5-nano', model='gpt-5-nano-2025-08-07'),
    ]
    commands = [
        ConfigsCommand(),
        ModelsCommand(),
        ResetConfigCommand(),
    ]
    cs = ChatSession(
        available_models=model_options,
        default_config=DEFAULT_CONFIG,
        config_path=CONFIG_PATH,
    )
    chat_loop = ChatPromptLoop(cs, commands)
    chat_loop.initialize()
    chat_loop.run()


if __name__ == '__main__':
    main()
