import logging

from llmchat.chat_prompt_loop import ChatPromptLoop
from llmchat.commands import ConfigsCommand, ModelsCommand, ResetConfigCommand
from re_mind import cpaths
from llmchat.chat_session import ChatSession
from re_mind.cli.chat_session_utils import OUTPUT_MODE_SIMPLE, print_response
from re_mind.cli.commands import SearchCommand
from llmchat.components.model_options import HuggingFaceModelOption, OpenAIModelOption

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


class RemindChatPromptLoop(ChatPromptLoop):
    def print_response(self, user_input: str) -> None:
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
    chat_loop = ChatPromptLoop(cs, commands)
    chat_loop.initialize()
    chat_loop.run()


if __name__ == '__main__':
    main()
