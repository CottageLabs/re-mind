import time
from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from re_mind import components, pipelines
from re_mind.db.qdrant.qdrant import get_client
from re_mind.llm import create_llm_huggingface, create_llama3, create_openai_model, create_vllm_model, \
    create_8bit_quantization_config
from re_mind.text_processing import save_pdf_to_vectorstore
from re_mind.utils import re_mind_utils as llm_utils


def display_qa_session(questions, rag_chain, console=None):
    """Display a formatted Q&A session with visual dividers and rich formatting."""
    if console is None:
        console = Console()

    for i, q in enumerate(questions):
        if i > 0:  # Add divider between questions
            console.print(Rule(style="dim"))

        console.print(f"\n[bold blue]Q:[/bold blue] {q}")
        result = rag_chain.invoke(q)
        console.print("[bold green]A:[/bold green]")
        console.print(Markdown(result))


def display_qa_session_with_tools(questions, rag_chain, console=None):
    """Display a formatted Q&A session with visual dividers and rich formatting."""
    if console is None:
        console = Console()

    for i, q in enumerate(questions):
        if i > 0:  # Add divider between questions
            console.print(Rule(style="dim"))

        console.print(f"\n[bold blue]Q:[/bold blue] {q}")

        messages = [HumanMessage(q)]
        resp1 = rag_chain.invoke(messages)
        messages.append(resp1)
        for tc in resp1.tool_calls:
            console.print(f"[dim]Invoking tool {tc.tool.name}...[/dim]")
            resp2: ToolMessage = tc.tool.invoke(tc)
            messages.append(resp2)

        final_resp = rag_chain.invoke(messages)

        console.print("[bold green]A:[/bold green]")
        console.print(Markdown(final_resp.content))


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

    llm = components.get_llm()

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

    vstore = components.get_vector_store(get_client(':memory:'))
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


def main5__test_rqg_qa():
    device = 'cuda'
    llm_utils.set_global_device(device)

    # llm = components.get_llm()
    # llm = create_llm_huggingface('cuda')
    llm = create_llm_huggingface(
        device=device,
        # model_id="google/gemma-3-1b-it",
        model_id="google/gemma-3-4b-it",
        # model_id="google/gemma-3-12b-it",
        # quantization_config=create_8bit_quantization_config(),
        # temperature=0.3,
        temperature=1.2,
    )

    rag_chain = pipelines.create_basic_qa(llm)

    # questions = [
    #     "What is Qdrant and what is it used for?",
    #     "How should I choose chunk size and overlap?",
    #     "How do I evaluate a RAG system?",
    #     "Explain LangChain's role in RAG."
    # ]

    questions = [
        "tell me about what is machine learning, show me the refs ",
    ]

    display_qa_session(questions, rag_chain=rag_chain)


def main6__inspect_vector_store():
    vectorstore = components.get_vector_store()

    # Get the underlying client for more detailed inspection
    client = vectorstore.client
    collection_name = vectorstore.collection_name

    print(f"Collection name: {collection_name}")

    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection info: {collection_info}")
        print(f"Vector count: {collection_info.points_count}")
        print(f"Vector config: {collection_info.config}")
    except Exception as e:
        print(f"Collection does not exist or error getting info: {e}")
        return

    # Try to scroll through all points to see content
    try:
        points = client.scroll(
            collection_name=collection_name,
            limit=100,  # Adjust limit as needed
            with_payload=True,
            with_vectors=False  # Set to True if you want to see vectors too
        )

        print(f"\nFound {len(points[0])} points:")
        for i, point in enumerate(points[0]):
            print(f"\nPoint {i + 1}:")
            print(f"  ID: {point.id}")
            print(f"  Payload: {point.payload}")
            if hasattr(point, 'vector') and point.vector:
                print(f"  Vector: {point.vector[:5]}..." if len(point.vector) > 5 else f"  Vector: {point.vector}")

    except Exception as e:
        print(f"Error scrolling through points: {e}")

    # Try similarity search to test functionality
    try:
        print("\nTesting similarity search with 'test' query:")
        results = vectorstore.similarity_search("test", k=5)
        print(f"Found {len(results)} similar documents:")
        for i, doc in enumerate(results):
            print(f"  {i + 1}. Content: {doc.page_content[:100]}...")
            print(f"     Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error during similarity search: {e}")

    # Try to get some basic stats
    try:
        print(f"\nVectorstore type: {type(vectorstore)}")
        print(f"Embedding model: {type(vectorstore.embeddings) if hasattr(vectorstore, 'embeddings') else 'Unknown'}")
    except Exception as e:
        print(f"Error getting vectorstore stats: {e}")


def main7__load_pdf():
    pdf_test_path = Path.home() / 'tmp/deep-learning.pdf'

    if not pdf_test_path.exists():
        print(f"PDF file not found at {pdf_test_path}")
        return

    print(f"Loading PDF from {pdf_test_path}")

    docs = save_pdf_to_vectorstore(pdf_test_path)

    print(f"Successfully added {len(docs)} document chunks to vector store")

    # Test retrieval
    print("\nTesting retrieval with 'learning' query:")
    vectorstore = components.get_vector_store()
    results = vectorstore.similarity_search("learning", k=3)
    for i, doc in enumerate(results):
        print(f"  {i + 1}. {doc.page_content[:100]}...")
        print(f"     Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"     Page: {doc.metadata.get('page', 'Unknown')}")
        print()


def main8__test_rqg_qa():
    device = 'cuda'
    llm_utils.set_global_device(device)

    # llm = components.get_llm()
    # llm = create_llm_huggingface('cuda')
    llm = create_llm_huggingface(
        device=device,
        # model_id="google/gemma-3-1b-it",
        model_id="google/gemma-3-4b-it",
        # model_id="google/gemma-3-12b-it",
        # quantization_config=create_8bit_quantization_config(),
        # temperature=0.3,
        temperature=1.2,
    )

    rag_chain = pipelines.create_basic_qa(llm)

    # questions = [
    #     "What is Qdrant and what is it used for?",
    #     "How should I choose chunk size and overlap?",
    #     "How do I evaluate a RAG system?",
    #     "Explain LangChain's role in RAG."
    # ]

    questions = [
        "Search and show me the score that related to machine learning",
    ]

    display_qa_session_with_tools(questions, rag_chain=rag_chain)


def main9__try_demo_graph():
    device = 'cuda'
    llm_utils.set_global_device(device)

    llm = create_llm_huggingface(
        device=device,
        model_id="google/gemma-3-1b-it",
        # model_id="google/gemma-3-4b-it",
        # model_id="google/gemma-3-12b-it",
        # temperature=0.3,
        temperature=1.2,
    )
    # llm = create_llama3(device=device)
    # llm = create_vllm_model()
    # llm = create_openai_model()
    app = pipelines.create_demo_graph(llm)
    # app = pipelines.create_demo_graph2(llm)

    # Run
    # out = app.invoke({"messages": [HumanMessage("Check AAPL and then advise.")]})
    # out = app.invoke({"messages": [HumanMessage("yo")]})
    # question = "get price of AAPL"
    # question = "What's the temperature in Paris right now?"
    question = "Hello, how are you?"
    # question = "Show me some example what can lang-graph do ?"
    out = app.invoke({"messages": [HumanMessage(question)]})


    print('-----------------------------')
    print('Final response:')
    print(out["messages"][-1].content)

def main10__test_tools_call_hugging_face():
    from transformers import pipeline
    # model_id = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    model_id = 'nguyenthanhthuan/Llama_3.2_1B_Intruct_Tool_Calling_V2'
    pipe = pipeline("text-generation",
                    model=model_id,
                    device_map="auto")

    messages = [{"role": "user", "content": "What’s the price of AAPL?"}]
    tools = [{
        "type": "function",
        "function": {
            "name": "GetPrice",
            "description": "Get latest price for a ticker.",
            "parameters": {"type": "object",
                           "properties": {"ticker": {"type": "string"}},
                           "required": ["ticker"]}
        }
    }]
    out = pipe(messages, tools=tools, max_new_tokens=128)
    print(out[0]["generated_text"][-1])  # should include a tool call if supported


def main11__hugging_face_tools():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load your chat model
    checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    # checkpoint = 'nguyenthanhthuan/Llama_3.2_1B_Intruct_Tool_Calling_V2'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")

    # Define your tools as Python functions with Google-style docstrings
    def get_current_temperature(location: str, unit: str):
        """
        Get the current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, Country"
            unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
        """
        return 22.  # Dummy example

    tools = [get_current_temperature]

    # Create the chat history
    messages = [
        {"role": "system", "content": "You are a weather bot."},
        {"role": "user", "content": "What's the temperature in Paris, Celsius right now?"}
    ]

    # Tokenize the chat with tools
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Generate response
    outputs = model.generate(**inputs.to(model.device), max_new_tokens=128)
    generated = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):])
    print(generated)


if __name__ == "__main__":
    # main7__load_pdf()
    # main5__test_rqg_qa()
    # init_test_data()
    # main6__inspect_vector_store()
    # main8__test_rqg_qa()
    main9__try_demo_graph()
    # main11__hugging_face_tools()
