import rich
from prompt_toolkit import PromptSession
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind.rag.rag_session import RagSession


def run_chat_app():
    config = {
        'chat': {
            'max_width': 100
        },
        'rag_session': {
            'temperature': 1.2,
            'n_top_result': 8
        },
        'huggingface_llm': {
            'device': 'cuda',
            'return_full_text': False,
        }
    }
    console = rich.console.Console()
    prompt_session = PromptSession()
    with console.status("Initializing RAG session..."):
        rag_config = config.get('rag_session', {})
        rag_session = RagSession(
            temperature=rag_config.get('temperature', 1.2),
            n_top_result=rag_config.get('n_top_result', 8),
            device=config.get('huggingface_llm', {}).get('device'),
            return_full_text=config.get('huggingface_llm', {}).get('return_full_text', False)
        )

    # with patch_stdout():  # ensures prints don't garble the prompt
    while True:
        try:
            user_input = prompt_session.prompt(">  ")
        except (EOFError, KeyboardInterrupt):
            print("Exiting chat...")
            break

        with console.status("Generating response..."):
            resp = rag_session.chat(user_input)

        width = min(console.size.width, config['chat']['max_width'])
        output = Markdown(resp)
        output = Panel(output)
        console.print(output, width=width)


def main():
    run_chat_app()


if __name__ == '__main__':
    main()
