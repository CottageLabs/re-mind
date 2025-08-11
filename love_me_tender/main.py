from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


def main():

    # Initialize LLM (OpenAI GPT)
    # llm = OpenAI(temperature=0)
    llm = OllamaLLM(model="llama2:latest")

    # Define prompt template for summarization
    prompt = PromptTemplate(
        input_variables=["document"],
        template="Summarize the following document:\n\n{document}"
    )

    # Create a summarization chain
    summary_chain = LLMChain(llm=llm, prompt=prompt)

    # Example document text
    doc_text = """
    LangChain is a framework to build applications powered by language models. 
    It provides tools for managing prompts, chains, memory, agents, and more.
    """

    # Get summary
    summary = summary_chain.run(document=doc_text)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
