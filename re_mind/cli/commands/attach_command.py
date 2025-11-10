from llmchat.commands import ChatCommand
from re_mind.cli.editor_utils import edit_items_in_editor


class AttachCommand(ChatCommand):
    def __init__(self, prefix: str = '/attach'):
        super().__init__(prefix)

    def run(self, user_input: str, cs: 'ChatSession'):
        content = '\n'.join(cs.attached_items)
        edited_content = edit_items_in_editor(content)
        cs.attached_items = [line.strip() for line in edited_content.split('\n') if line.strip()]

        cs.print(f"[green]Updated attached items ({len(cs.attached_items)} items):[/green]")
        for item in cs.attached_items:
            cs.print(f"  - {item}")
