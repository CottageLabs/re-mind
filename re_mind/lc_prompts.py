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

QUERY_CONTEXT_SUMMARY_PROMPT = (
    "Analyze the user's query and provide a concise summary of what information is being "
    "requested. Identify the key concepts, topics, and intent behind the query. This will "
    "help in retrieving the most relevant context from the knowledge base."
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


DEFAULT_JSON_GENERATION_INSTRUCTION = (
    "Generate valid JSON output based on the user's request. "
    "Ensure the output is properly formatted and follows JSON syntax rules. "
    "Return only the JSON object without additional explanation or markdown formatting. "
    "If the input is plain text (not a JSON specification), convert it to a JSON array "
    "where each non-empty line becomes a string element in the array."
)

POINTS_TO_JSON_LIST_INSTRUCTION = (
    "Convert the given user input from bullet points or lines into a JSON list (array). "
    "Each point or line should become a string element in the array. "
    "Remove any bullet point markers (-, *, •, numbers, etc.) and trim whitespace. "
    "Skip empty lines. "
    "Return ONLY a valid JSON array without additional explanation or markdown formatting. "
    "The output must be a JSON array/list, NOT a JSON object/dict."
)
