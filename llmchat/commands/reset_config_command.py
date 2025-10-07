from rich.markdown import Markdown

from .chat_command_base import ChatCommand


class ResetConfigCommand(ChatCommand):
    def __init__(self, prefix: str = '/reset_config'):
        super().__init__(prefix)

    def run(self, user_input: str, cs: 'ChatSession'):
        from llmchat.chat_session import DEFAULT_CONFIG
        cs.config = DEFAULT_CONFIG.copy()
        cs.save_config()
        cs.print("[green]Configuration has been reset to default values.[/green]")
        cs.print()
        cs.print(Markdown("## Default Configuration"))
        cs.print(cs.config)
