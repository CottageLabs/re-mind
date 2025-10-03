"""
Tools that convert different source of text (pdf, markdown, raw text) to Document objects
"""
import warnings
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from librarian import components
from librarian import iter_utils


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


def save_pdf_to_vectorstore(pdf_path, vectorstore=None, text_splitter=None, metadata=None):
    # Load PDF
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    print(f"Loaded {len(pages)} pages from PDF")

    # Split text into chunks
    if text_splitter is None:
        text_splitter = create_default_text_splitter()

    docs = text_splitter.split_documents(pages)

    print(f"Split into {len(docs)} chunks")

    # Inject additional metadata if provided
    if metadata:
        docs = list(inject_metadata(docs, metadata))

    # Get vector store and add documents
    vectorstore = vectorstore or components.get_vector_store()
    vectorstore.add_documents(docs)

    return docs


def save_text_to_vectorstore(file_path, vectorstore=None, text_splitter=None, metadata=None):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if text_splitter is None:
        text_splitter = create_default_text_splitter()

    docs = text_splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": file_path.name}) for chunk in docs]

    # Inject additional metadata if provided
    if metadata:
        docs = list(inject_metadata(docs, metadata))

    vectorstore = vectorstore or components.get_vector_store()
    vectorstore.add_documents(docs)

    return docs


def save_any_to_vectorstore(file_path, vectorstore=None, text_splitter=None, metadata=None):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        return save_pdf_to_vectorstore(file_path, vectorstore, text_splitter, metadata)
    else:
        if suffix not in ['.txt']:
            warnings.warn(f"File suffix [{suffix}] is not explicitly supported, treat it as raw text file")

        return save_text_to_vectorstore(file_path, vectorstore=vectorstore,
                                        text_splitter=text_splitter, metadata=metadata)


def inject_metadata(docs: Iterable[Document], metadata: dict = None) -> Iterable[Document]:
    if metadata is None:
        metadata = {}

    for doc in docs:
        if metadata:
            new_metadata = doc.metadata.copy() if doc.metadata else {}
            new_metadata.update(metadata)
            doc.metadata = new_metadata
        yield doc
