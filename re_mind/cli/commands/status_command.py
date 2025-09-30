from rich.markdown import Markdown
from .chat_command_base import ChatCommand


class StatusCommand(ChatCommand):
    def __init__(self):
        super().__init__('/status')

    def run(self, user_input: str, cs: 'ChatSession'):
        cs.print(Markdown("## Current Status"))
        model_name = cs.model_option.name if cs.model_option else "No model selected"
        cs.print(Markdown(f"**Selected Model:** {model_name}"))
        cs.print()
