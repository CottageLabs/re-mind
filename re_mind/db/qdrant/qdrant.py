from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from re_mind.constants import DEFAULT_COLLECTION_NAME
from re_mind.cpaths import QDRANT_DATA_PATH
from re_mind.rag.embedding import get_embedding


def get_client(location=None, **kwargs):
    final_kwargs = {}
    if location is None:
        final_kwargs['path'] = QDRANT_DATA_PATH
    else:
        final_kwargs['location'] = location
    final_kwargs |= kwargs
    qdrant = QdrantClient(**final_kwargs)
    return qdrant


def get_vector_store(client=None, collection_name=DEFAULT_COLLECTION_NAME):
    if client is None:
        client = get_client()

    embedding = get_embedding()
    init_collection(embedding._client.get_sentence_embedding_dimension(),
                    client=client, collection_name=collection_name)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )

    return vector_store


def init_collection(
        embedding_size,
        client=None,
        collection_name=DEFAULT_COLLECTION_NAME,
):
    if client is None:
        client = get_client()

    # Create the collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )
