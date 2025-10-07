from llmchat.commands.chat_command_base import ChatCommand
from llmchat.commands.command_utils import extract_command_arg
from re_mind import retrievers, components
from re_mind.cli.chat_session_utils import print_search_results


class SearchCommand(ChatCommand):
    def __init__(self, prefix: str = '/search'):
        super().__init__(prefix)

    def run(self, user_input: str, cs: 'ChatSession'):
        query = extract_command_arg(self.prefix, user_input)
        if not query:
            cs.print("Please provide a search query after '/search'.")
            cs.print("Example: /search What is the capital of France?")
            return

        vectorstore = components.get_vector_store()
        docs, extracted_queries = retrievers.complex_retrieve(query, vectorstore, cs.llm, cs.config['n_top_result'],
                                                              device=cs.config.get('device'))
        print_search_results(cs, query, docs, extracted_queries)
