"""

Collection of prompt templates for langchain chains/pipelines.

"""
from langchain_core.prompts import ChatPromptTemplate


def get_rag_qa_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use ONLY the provided context to answer. "
         "If context is insufficient, say you don't know."),
        ("human",
         "Question: {question}\n\nContext:\n{context}")
    ])


def get_rag_qa_prompt2():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. You study question and  context that provided."
         " response summary to answer. "
         "If context is insufficient, say you don't know. "
         ),
        ("human",
         "Question: {question}\n\nContext:\n{context}")
    ])


def get_rag_qa_prompt3():
    return ChatPromptTemplate.from_messages([
        ("system",
         """
You are a helpful assistant. You understand user's request,
reply the request as you can, provided context could be helpful.
         """
         ),
        ("human",
         "Request: {question}\n\nContext:\n{context}")
    ])


DEFAULT_RAG_INSTRUCTION = (
    "Use the provided context to fulfill the user's request, whether that is answering "
    "a question, creating a summary, drafting a report, or producing other content. "
    "If the context is insufficient for any part of the request, clearly state the gap. "
    "Cite sources using [#] indices that map to the context list."
)


def format_rag_prompt(instruction: str, context_text: str, question: str) -> str:
    return (
        f"{instruction}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Request: {question}\n\n"
        "Response:"
    )


def get_query_extraction_prompt(user_input: str) -> str:
    return (
        "Extract search queries from the user's input for vector database retrieval. "
        "Return one query per line that captures the key concepts needed to find "
        "relevant information. If no meaningful queries can be extracted, return "
        "exactly '<NO_QUERY>'. Queries should be concise and focus on searchable "
        "terms, concepts, or topics.\n\n"
        f"User input: {user_input}"
    )
