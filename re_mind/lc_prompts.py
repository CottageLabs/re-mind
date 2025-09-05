from langchain_core.prompts import ChatPromptTemplate


def get_rag_qa_prompt():
    """Create a RAG prompt template for question answering with context."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use ONLY the provided context to answer. "
         "If context is insufficient, say you don't know."),
        ("human",
         "Question: {question}\n\nContext:\n{context}")
    ])

def get_rag_qa_prompt2():
    """Create a RAG prompt template for question answering with context."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. You study question and  context that provided."
         " response summary to answer. "
         "If context is insufficient, say you don't know. "
         ),
        ("human",
         "Question: {question}\n\nContext:\n{context}")
    ])
