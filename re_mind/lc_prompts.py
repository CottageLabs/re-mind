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
