from rich.markdown import Markdown
from .chat_command_base import ChatCommand


class ConfigsCommand(ChatCommand):
    def __init__(self, ):
        super().__init__('/configs')

    def run(self, user_input: str, cs: 'ChatSession'):
        cs.print(Markdown("## Configuration Commands"))
        cs.print(Markdown("""```
Examples:
/configs                  # show configs
/configs n_top_result 8   # change configs
```"""))
        cs.print()
        cs.print(Markdown("## Current Configuration"))
        cs.print(cs.config)
