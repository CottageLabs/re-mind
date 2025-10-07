from rich.markdown import Markdown
from .chat_command_base import ChatCommand
from .command_utils import extract_command_arg


class ModelsCommand(ChatCommand):
    def __init__(self, prefix: str = '/models'):
        super().__init__(prefix)

    # KTODO add available models list to auto completer

    def run(self, user_input: str, cs: 'ChatSession'):
        model_name = extract_command_arg(self.prefix, user_input)
        if model_name:
            # Switch model
            try:
                cs.switch_llm(model_name)
                cs.print(f"[green]Switched to model: {cs.model_option.name}[/green]")
            except ValueError:
                cs.print(f"[red]Model option '{model_name}' not found.[/red]")

        else:
            # List available models
            cs.print(Markdown(f"## Available Model Options"))
            cs.print(Markdown(f"""```
Examples:
{self.prefix} gemma-3-1b       # switch to gemma-3-1b model
```"""))
            available_models = []
            current_model_name = cs.model_option.name if cs.model_option else None
            for m in cs.available_models:
                is_current = current_model_name and m.name == current_model_name
                model_display = f'- **{m.name}** (current)' if is_current else f'- {m.name}'
                available_models.append(model_display)

            cs.print(Markdown('\n'.join(available_models)))
            cs.print()
