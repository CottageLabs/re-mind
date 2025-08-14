import time

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from love_me_tender.db.qdrant.qdrant import get_client, get_vector_store


def main():
    # device = "auto"
    device = "cuda"

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

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-3-1b-it",
        task="text-generation",
        model_kwargs={
            "temperature": 0.7, "max_length": 10000,

            # "quantization_config": quantization_config,
        },
        pipeline_kwargs={"device_map": device},
    )

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


if __name__ == "__main__":
    main4()
