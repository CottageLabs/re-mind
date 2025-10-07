import re

from rich.markdown import Markdown

from . import command_utils
from .chat_command_base import ChatCommand


class ConfigsCommand(ChatCommand):
    def __init__(self, prefix: str = '/configs'):
        super().__init__(prefix)

    def run(self, user_input: str, cs: 'ChatSession'):
        param = command_utils.extract_command_arg(self.prefix, user_input)
        if param:
            params = re.split(r'\s+', param)
            if len(params) != 2:
                cs.print(f"[red]Invalid command format. Use: {self.prefix} <param> <value>[/red]")
            else:
                key, value = params
                if key == 'device':
                    cs.switch_device(value)
                    cs.switch_llm(cs.model_option.name)
                    cs.print(f"[green]Configuration updated: device = {value}[/green]")
                else:
                    if key not in cs.config:
                        cs.print(f"[red]Unknown configuration key: {key}[/red]")
                    cs.config[key] = type(cs.config.get(key, value))(value)
                    cs.save_config()

        else:
            cs.print(Markdown("""```
            Examples:
            /configs n_top_result 8   # change configs
            ```"""))

        cs.print(Markdown("## Configuration Commands"))
        cs.print()
        cs.print(Markdown("## Current Configuration"))
        cs.print(cs.config)
