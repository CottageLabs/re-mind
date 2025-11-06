from llmchat.commands import ChatCommand
from llmchat.commands.command_utils import extract_command_arg
from re_mind import lc_prompts


class SummaryCommand(ChatCommand):
    def __init__(self, prefix: str = '/summary'):
        super().__init__(prefix)

    def run(self, user_input: str, cs: 'ChatSession'):
        query = extract_command_arg(self.prefix, user_input)
        if not query:
            cs.print("Please provide a query after '/summary'.")
            cs.print("Example: /summary What are the main topics discussed in machine learning?")
            return

        response = cs.chat(
            query,
            configurable={'sys_prompt': lc_prompts.QUERY_CONTEXT_SUMMARY_PROMPT}
        )
        cs.print_response(response)
