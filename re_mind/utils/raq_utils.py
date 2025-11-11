"""Reusable LangGraph node helpers."""

from typing import TYPE_CHECKING, Optional, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from ..rag_graphs import RagState


def print_ref(context: Sequence["Document"], console: Optional[Console] = None) -> None:
    """Render retrieved documents in Rich panels."""
    if not context:
        return

    console = console or Console()

    console.print(Markdown("# Source Documents"))
    for i, doc in enumerate(context):
        console.print(f"    [{i + 1}] {doc.metadata}")
        console.print(Panel(doc.page_content, title=f"Document {i + 1}", expand=False))




def print_result(state: "RagState", show_ref: bool = False, console: Optional[Console] = None) -> None:
    """Pretty-print the synthesized RAG answer and optionally its references."""
    console = console or Console()

    # Print extracted queries if available
    extracted_queries = state.get("extracted_queries")
    if extracted_queries:
        console.print(Markdown("# Extracted Queries"))
        for i, query in enumerate(extracted_queries, 1):
            console.print(f"  {i}. {query}")
        console.print()

    if show_ref or state.get("print_refs", False):
        print_ref(state.get("context", []), console)
    console.print(Markdown("# Answer"))
    console.print(Markdown(state["answer"]))
