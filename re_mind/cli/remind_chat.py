import logging

from llmchat import ChatPromptLoop
from llmchat import ChatSession
from llmchat.commands import ModelsCommand, ResetConfigCommand
from llmchat.components.model_options import HuggingFaceModelOption, OpenAIModelOption
from re_mind import cpaths, components, constants
from re_mind.cli.chat_session_utils import OUTPUT_MODE_SIMPLE, print_response
from re_mind.cli.commands import SearchCommand, SummaryCommand
from re_mind.commands.configs_command import ConfigsExtCommand

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


class ReminChatSession(ChatSession):
    def build_rag_chain(self) -> 'CompiledStateGraph':
        from re_mind import pipelines
        n_top_result = self.config.get('n_top_result', 8)
        return pipelines.build_rag_app(n_top_result=n_top_result)

    def chat(self, user_input,
             state: dict = None,
             configurable: dict = None) -> dict:
        state = state or {}
        configurable = configurable or {}

        collection_name = self.config.get('collection_name')
        vector_store = self.vector_store_factory(collection_name=collection_name)

        final_state = {'question': user_input} | state
        final_configurable = {
                                 'llm': self.llm,
                                 'vectorstore': vector_store,
                                 'device': self.device,
                                 # KTODO handle n_top_result, temperature, etc.
                             } | configurable
        response = self.rag_chain.invoke(
            final_state,
            config={'configurable': final_configurable}
        )
        vector_store.client.close()
        return response

    def print_response(self, response: dict) -> None:
        print_response(self, response)


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

        # Vector Store
        'collection_name': constants.DEFAULT_COLLECTION_NAME,
    }

    model_options = [
        HuggingFaceModelOption(name='gemma-3-1b', model_id="google/gemma-3-1b-it"),
        HuggingFaceModelOption(name='llama-3-3b', model_id='huihui-ai/Llama-3.2-3B-Instruct-abliterated'),
        OpenAIModelOption(name='gpt-5-nano', model='gpt-5-nano-2025-08-07'),
    ]
    commands = [
        ConfigsExtCommand(),
        SearchCommand(),
        SummaryCommand(),
        ModelsCommand(),
        ResetConfigCommand(),
    ]
    cs = ReminChatSession(available_models=model_options, default_config=DEFAULT_CONFIG,
                          config_path=cpaths.CONFIG_PATH,
                          vector_store_factory=components.get_vector_store)
    chat_loop = ChatPromptLoop(cs, commands)
    chat_loop.initialize()
    chat_loop.run()


if __name__ == '__main__':
    main()
