import time

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

from love_me_tender import llm_utils
from love_me_tender.db.qdrant.qdrant import get_client, get_vector_store
from love_me_tender.llm_utils import get_global_device


def main():
    # device = "auto"
    # device = "cuda"
    llm_utils.set_global_device('cpu')

    # Initialize LLM (OpenAI GPT)
    # llm = OpenAI(temperature=0)
    # llm = OllamaLLM(model="llama2:latest")
    # llm = HuggingFaceEndpoint(
    #     repo_id='google/gemma-3-1b-it'
    # )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype="float16",
    #     bnb_4bit_use_double_quant=True,
    # )
    #
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="openai/gpt-oss-20b",
    #     task="text-generation",
    #     model_kwargs={
    #         "temperature": 0.7, "max_length": 5000,
    #         "quantization_config": quantization_config,
    #     },
    #     pipeline_kwargs={"device_map": device},
    # )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True
    #     # load_in_4bit=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_dtype="float16",
    #     # bnb_4bit_use_double_quant=True,
    # )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,  # Use 8-bit quantization
    #     llm_int8_threshold=6.0,  # Controls mixed precision, lower = more precise
    #     llm_int8_enable_fp32_cpu_offload=True,  # Offload high-precision layers to CPU if needed
    #     llm_int8_has_fp16_weight=False  # Keep some weights in fp32/fp16 to preserve accuracy
    # )

    llm = get_llm()

    # Define prompt template for summarization
    prompt = PromptTemplate(
        input_variables=["document"],
        template="Summarize the following document:\n\n{document}",
    )

    # Create a summarization chain
    summary_chain = prompt | llm
    # summary_chain = LLMChain(llm=llm, prompt=prompt)

    # Example document text
    doc_text = """
    LangChain is a framework to build applications powered by language models. 
    It provides tools for managing prompts, chains, memory, agents, and more.
    """

    # Get summary
    # summary = summary_chain.run(document=doc_text)
    summary = summary_chain.invoke({"document": doc_text})
    print("Summary:", summary)

    time.sleep(1)


def get_llm(device=None, openai_model=None):
    if device is None:
        device = get_global_device()
    if openai_model:

        from openai import OpenAI

        # Tiny adapter so we can use OpenAI client as an LCEL-compatible “callable”
        class OpenAIChatAsRunnable:
            def __init__(self, model="gpt-4o-mini"):
                self.client = OpenAI()
                self.model = model

            def invoke(self, messages):
                # messages is a list of dicts: {"role": "system"/"user", "content": "..."}
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": m.type if hasattr(m, 'type') else m["role"],
                               "content": m.content if hasattr(m, 'content') else m["content"]}
                              for m in messages]
                )
                return completion.choices[0].message.content

        llm = OpenAIChatAsRunnable(model="gpt-4o-mini")

    else:
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/gemma-3-1b-it",
            task="text-generation",
            model_kwargs={
                "temperature": 0.7, "max_length": 10000,

                # "quantization_config": quantization_config,
            },
            pipeline_kwargs={"device_map": device},
        )

    return llm


def main2():
    docs = ["Qdrant supports LangChain", "Qdrant integrates with LlamaIndex"]
    print('hihihi')
    qdrant = get_client(':memory:')

    qdrant.add(
        collection_name='demo_collection',
        documents=docs,

        metadata=[{"source": "LangChain"}, {"source": "LlamaIndex"}],
        ids=[1, 2],
    )
    print('qdrant', qdrant)


def main3():
    docs = ["Qdrant supports LangChain", "Qdrant integrates with LlamaIndex"]

    vstore = get_vector_store(get_client(':memory:'))
    vstore.add_texts(docs)

    retrieved_docs = vstore.search('integrates')
    print("Retrieved documents:", retrieved_docs)


def main4():
    import numpy as np
    from qdrant_client.models import PointStruct

    vectors = np.random.rand(100, 100)
    points = [
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(vectors)
    ]

    client = get_client(':memory:')
    client.upsert(
        collection_name="my_collection",
        points=points
    )


def format_docs(documents):
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in documents
    )


def main5():
    # rag_qdrant_minimal.py

    # # --- Choose ONE LLM backend ---
    # USE_OPENAI = True  # flip to False to use Ollama
    #
    # if USE_OPENAI:
    #     from openai import OpenAI
    #     from langchain_core.language_models.chat_models import BaseChatModel
    #     from langchain_core.messages import HumanMessage, SystemMessage
    #
    #     # Tiny adapter so we can use OpenAI client as an LCEL-compatible “callable”
    #     class OpenAIChatAsRunnable:
    #         def __init__(self, model="gpt-4o-mini"):
    #             self.client = OpenAI()
    #             self.model = model
    #
    #         def invoke(self, messages):
    #             # messages is a list of dicts: {"role": "system"/"user", "content": "..."}
    #             completion = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[{"role": m.type if hasattr(m, 'type') else m["role"],
    #                            "content": m.content if hasattr(m, 'content') else m["content"]}
    #                           for m in messages]
    #             )
    #             return completion.choices[0].message.content
    #
    #     llm = OpenAIChatAsRunnable(model="gpt-4o-mini")
    #
    # else:
    #     # Local LLM via Ollama (requires `ollama` running and a model pulled, e.g. `llama3`)
    #     from langchain_community.chat_models import ChatOllama
    #     llm = ChatOllama(model="llama3")  # or "llama3.1", etc.

    llm_utils.set_global_device('cpu')

    llm = get_llm()

    # ---------------------------
    # 1) Your small document set
    # ---------------------------
    docs_raw = [
        ("intro.txt",
         "LangChain lets you build RAG systems by chaining chunks, retrieval, and LLM prompts."),
        ("qdrant.txt",
         "Qdrant is a vector database for similarity search. It supports HNSW indexes and filtering."),
        ("chunks.txt",
         "Chunking with RecursiveCharacterTextSplitter helps keep semantic units together. "
         "Common sizes: 300-800 chars with 10-100 char overlap."),
        ("eval.txt",
         "Evaluate RAG with precision@k, hit rate, faithfulness, and answer relevancy.")
    ]

    # ---------------------------
    # 2) Split into chunks
    # ---------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for fname, text in docs_raw:
        for chunk in text_splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": fname}))

    # ---------------------------
    # 3) Embeddings (local)
    # ---------------------------
    # embeddings = get_embedding()

    # ---------------------------
    # 4) Qdrant (in-memory) store
    # ---------------------------
    # client = QdrantClient(location=":memory:")  # use path="qdrant_data" for on-disk

    # collection = "rag_demo"
    #
    # # Build the vector store from documents
    # vectorstore = QdrantVectorStore.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     client=client,
    #     collection_name=collection,
    # )

    vectorstore = get_vector_store()
    vectorstore.add_documents(docs)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # ---------------------------
    # 5) RAG prompt + chain (LCEL)
    # ---------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use ONLY the provided context to answer. "
         "If context is insufficient, say you don't know."),
        ("human",
         "Question: {question}\n\nContext:\n{context}")
    ])

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # ---------------------------
    # 6) Ask a question
    # ---------------------------
    for q in [
        "What is Qdrant and what is it used for?",
        "How should I choose chunk size and overlap?",
        "How do I evaluate a RAG system?",
        "Explain LangChain's role in RAG."
    ]:
        print(f"\nQ: {q}")
        print("A:", rag_chain.invoke(q))


if __name__ == "__main__":
    main5()
