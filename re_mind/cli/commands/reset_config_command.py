from rich.markdown import Markdown

from .chat_command_base import ChatCommand


class ResetConfigCommand(ChatCommand):
    def __init__(self):
        super().__init__('/reset_config')

    def run(self, user_input: str, cs: 'ChatSession'):
        from re_mind.cli.remind_chat import DEFAULT_CONFIG

        cs.config = DEFAULT_CONFIG.copy()
        cs.save_config()
        cs.print("[green]Configuration has been reset to default values.[/green]")
        cs.print()
        cs.print(Markdown("## Default Configuration"))
        cs.print(cs.config)
