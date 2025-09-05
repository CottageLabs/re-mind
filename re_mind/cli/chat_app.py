import rich
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind.rag.rag_session import RagSession


def run_chat_app():
    max_width = 100


    console = rich.console.Console()
    prompt_session = PromptSession()
    with console.status("Initializing RAG session..."):
        rag_session = RagSession()

    # with patch_stdout():  # ensures prints don't garble the prompt
    while True:
        try:
            user_input = prompt_session.prompt(">  ")
        except (EOFError, KeyboardInterrupt):
            print("Exiting chat...")
            break

        with console.status("Generating response..."):
            resp = rag_session.chat(user_input)

        width = min(console.size.width, max_width)
        output = Markdown(resp)
        output = Panel(output)
        console.print(output, width=width)


def main():
    run_chat_app()


if __name__ == '__main__':
    main()
