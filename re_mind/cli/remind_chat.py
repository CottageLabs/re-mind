import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import rich
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from rich.markdown import Markdown
from rich.panel import Panel

from re_mind import retrievers
from re_mind.cli.components.model_options import ModelOption
from re_mind.rag.rag_chat import RagChat


# KTODO support librarian mode (list, add, remove documents)
# KTODO support switch query mode
# KTODO add command librarian
# KTODO add history
# KTODO add debugging / detail mode
# KTODO output_model [debugging / detail / simple]


def extract_command_arg(command_prefix: str, user_input: str) -> str:
    """Extract argument from command input by removing the command prefix."""
    return re.sub(rf'^{re.escape(command_prefix)}\b\s*', '', user_input).strip()


# KTODO move to new file
# KTODO suggest a better name
class CompleterHelper(ABC):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def is_match(self, user_input: str) -> bool:
        return user_input.startswith(self.prefix)

    def create_nested_dict(self):
        return {self.prefix: None}

    @abstractmethod
    def run(self, user_input: str, cs: 'ChatSession'):
        raise NotImplementedError


class ConfigsCH(CompleterHelper):
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


class SearchCH(CompleterHelper):
    def __init__(self, ):
        super().__init__('/search')

    def run(self, user_input: str, cs: 'ChatSession'):
        query = extract_command_arg(self.prefix, user_input)
        if not query:
            cs.print("Please provide a search query after '/search'.")
            cs.print("Example: /search What is the capital of France?")
            return

        docs, extracted_queries = retrievers.complex_retrieve(query, cs.llm, cs.config['n_top_result'])

        cs.print(Markdown("## Search Results"))
        cs.print(Markdown(f"**Query:** {query}"))
        cs.print()

        if extracted_queries:
            queries_text = "### Extracted Queries\n" \
                           + "\n".join(f"{i}. {eq}" for i, eq in enumerate(extracted_queries, 1))
            cs.print(Markdown(queries_text))
            cs.print()

        cs.print(Markdown(f"### Documents ({len(docs)} found)"))
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            score = doc.metadata.get('ranker_score', 0)

            cs.print(Markdown(f"**{i}.** Score: {score:.2f}"))
            cs.print(Markdown(f"   **Source:** {source}" + (f" (page {page})" if page else "")))
            cs.print(Markdown(f"   **Content:** {content}"))
            cs.print()


class ModelsCH(CompleterHelper):
    def __init__(self):
        super().__init__('/models')

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
            for m in cs.get_available_models():
                is_current = current_model_name and m.name == current_model_name
                model_display = f'- **{m.name}** (current)' if is_current else f'- {m.name}'
                available_models.append(model_display)

            cs.print(Markdown('\n'.join(available_models)))
            cs.print()


class StatusCH(CompleterHelper):
    def __init__(self):
        super().__init__('/status')

    def run(self, user_input: str, cs: 'ChatSession'):
        cs.print(Markdown("## Current Status"))
        model_name = cs.model_option.name if cs.model_option else "No model selected"
        cs.print(Markdown(f"**Selected Model:** {model_name}"))
        cs.print()


def build_completer(completer_helpers: list[CompleterHelper] | None = None) -> FuzzyCompleter:
    """
    Build a nested completer dynamically so 'use <dataset>' picks up new names.
    """
    data = {
        "/librarian": {"show": None, },  # KTODO add librarian commands
    }

    for helper in (completer_helpers or []):
        data.update(helper.create_nested_dict())

    nested = NestedCompleter.from_nested_dict(data)
    return FuzzyCompleter(nested)


@dataclass
class ChatSession:
    config: dict
    console: rich.console.Console
    rag_chat: RagChat = None
    model_option: ModelOption = None

    @property
    def llm(self):
        return self.rag_chat.llm if self.rag_chat else None

    @property
    def console_width(self):
        return min(self.console.size.width, self.config.get('max_width', 100))

    def print(self, message=None):
        if message is None:
            self.console.print()
        else:
            self.console.print(message, width=self.console_width)

    @staticmethod
    def get_available_models():
        from re_mind.cli.components.model_options import HuggingFaceModelOption, OpenAIModelOption
        return [
            HuggingFaceModelOption(name='gemma-3-1b', model_id="google/gemma-3-1b-it"),
            OpenAIModelOption(name='gpt-5-nano', model='gpt-5-nano-2025-08-07'),
        ]

    def switch_llm(self, model_option_name: str):
        selected_option = None
        model_option_name = model_option_name.lower().strip()
        for model_option in self.get_available_models():
            if model_option.name.lower() == model_option_name:
                selected_option = model_option
                break

        if selected_option is None:
            self.print(f"[red]Model option '{model_option_name}' not found.[/red]")
            raise ValueError(f"Model option '{model_option_name}' not found.")

        # delete previous model to free VRAM
        if self.model_option is not None:
            self.model_option.delete()

        # set new model
        self.model_option = selected_option
        llm = selected_option.create()
        self.rag_chat = RagChat(llm=llm, n_top_result=self.config.get('n_top_result', 8))

        return self


def run_remind_chat():
    config = {
        # Chat
        'max_width': 100,

        # RAG Session
        'temperature': 1.2,
        'n_top_result': 6,

        # Hugging Face LLM
        'device': 'cuda',
        'return_full_text': False,
    }
    console = rich.console.Console()
    prompt_session = PromptSession()
    with console.status("Initializing RAG session..."):
        cs = ChatSession(
            config=config,
            console=console,
        ).switch_llm("gemma-3-1b")

    completer_helpers = [
        ConfigsCH(),
        SearchCH(),
        ModelsCH(),
        StatusCH(),
    ]
    completer = build_completer(completer_helpers)
    # KTODO add assert to check command prefix not conflict

    # Chat loop
    while True:
        try:
            user_input = prompt_session.prompt(">  ", completer=completer)
            if not user_input.strip():
                continue

        except (EOFError, KeyboardInterrupt):
            print("Exiting chat...")
            break

        should_chat = True
        for helper in completer_helpers:
            if helper.is_match(user_input):
                helper.run(user_input, cs)
                should_chat = False
                break

        if not should_chat:
            continue

        # with cs.console.status("Generating response..."):
        resp = cs.rag_chat.chat(user_input)

        output = Markdown(resp['answer'])
        output = Panel(output)
        cs.print(output)


def main():
    run_remind_chat()


if __name__ == '__main__':
    main()
