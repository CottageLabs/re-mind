import rich
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind.rag.rag_session import RagSession

def build_completer():
    """
    Build a nested completer dynamically so 'use <dataset>' picks up new names.
    """
    nested = NestedCompleter.from_nested_dict({
        "/configs": None,
        "/librarian": {"show": None,}
    })
    return FuzzyCompleter(nested)


def run_chat_app():
    config = {
        'chat': {
            'max_width': 100
        },
        'rag_session': {
            'temperature': 1.2,
            'n_top_result': 6,
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

    completer = build_completer()

    # Chat loop
    while True:
        try:
            user_input = prompt_session.prompt(">  ", completer=completer)
        except (EOFError, KeyboardInterrupt):
            print("Exiting chat...")
            break

        if user_input == '/configs':
            console.print(Markdown("## Current Configuration"))
            console.print(config)
            continue

        # with console.status("Generating response..."):
        resp = rag_session.chat(user_input)

        width = min(console.size.width, config['chat']['max_width'])
        output = Markdown(resp['answer'])
        output = Panel(output)
        console.print(output, width=width)


def main():
    run_chat_app()


if __name__ == '__main__':
    main()
