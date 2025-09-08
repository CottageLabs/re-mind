from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from re_mind import components, lc_prompts


def create_basic_qa(llm, n_top_result=8):
    def format_docs(documents):
        return "\n\n".join(
            f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in documents
        )

    vectorstore = components.get_vector_store()
    prompt = lc_prompts.get_rag_qa_prompt3()

    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": n_top_result})

    # mmr can handle redundant documents better than similarity
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": n_top_result, "fetch_k": n_top_result + 40, "lambda_mult": 0.5},
    )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain
