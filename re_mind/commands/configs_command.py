import re

from rich.markdown import Markdown
from rich.table import Table

from llmchat.commands import command_utils, ChatCommand

import typing

from re_mind import components, envvars

if typing.TYPE_CHECKING:
    from re_mind.cli.remind_chat import ReminChatSession


class ConfigsExtCommand(ChatCommand):
    def __init__(self, prefix: str = '/configs'):
        super().__init__(prefix)

    def run(self, user_input: str, cs: 'ReminChatSession'):
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
                    # KTODO also switch for embedding model
                    cs.print(f"[green]Configuration updated: device = {value}[/green]")
                else:
                    if key not in cs.config:
                        cs.print(f"[red]Unknown configuration key: {key}[/red]")

                    # KTODO maybe use some schema class to type convert
                    try:
                        cs.config[key] = type(cs.config.get(key, value))(value)
                    except ValueError:
                        cs.print(f"[red] convert value to type {type(cs.config.get(key))} failed, use string instead[/red]")
                        cs.config[key] = value
                    cs.save_config()

        else:
            cs.print(Markdown("""```
            Examples:
            /configs n_top_result 8   # change configs
            ```"""))

        table = Table(title="Vectorstore Configuration", show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Qdrant data path", str(envvars.get_qdrant_data_path()))
        table.add_row("Collection name", cs.config.get('collection_name', 'N/A'))
        cs.print(table)

        cs.print(Markdown("## Configuration Commands"))
        cs.print()
        config_table = Table(title="Current Configuration", show_header=False, box=None)
        config_table.add_column("Key", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="white")
        for key, value in sorted(cs.config.items()):
            config_table.add_row(str(key), str(value))
        cs.print(config_table)
