"""
Tools that convert different source of text (pdf, markdown, raw text) to Document objects
"""
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from re_mind import components
from re_mind.utils import iter_utils


def create_text_splitter():
    # return RecursiveCharacterTextSplitter(
    #     chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n", " ", ""]
    # )

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )


def create_default_text_splitter():
    return create_text_splitter()


def create_docs(sources: Iterable, text_splitter=None) -> Iterable[Document]:
    """

    Args:
        sources: source can be any different type, each different type of source
            yield different metadata by different loader

    """
    if text_splitter is None:
        text_splitter = create_default_text_splitter()

    _obj = iter_utils.peek(sources)
    if isinstance(_obj, tuple):
        for fname, text in sources:
            for chunk in text_splitter.split_text(text):
                yield Document(page_content=chunk, metadata={"source": fname})
    else:
        raise NotImplementedError(f"other type [{_obj}] of source is not implemented yet")


def load_pdf_to_vectorstore(pdf_path, vectorstore=None, text_splitter=None):
    from langchain_community.document_loaders import PyPDFLoader
    from re_mind.text_processing import create_default_text_splitter

    # Load PDF
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    print(f"Loaded {len(pages)} pages from PDF")

    # Split text into chunks
    if text_splitter is None:
        text_splitter = create_default_text_splitter()
    docs = text_splitter.split_documents(pages)

    print(f"Split into {len(docs)} chunks")

    # Get vector store and add documents
    if vectorstore is None:
        vectorstore = components.get_vector_store()
    vectorstore.add_documents(docs)

    return docs
