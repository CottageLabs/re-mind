from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.grpc import VectorParams
from qdrant_client.http.models import Distance

from love_me_tender.cpaths import QDRANT_DATA_PATH
from love_me_tender.rag.embedding import get_embedding

DEFAULT_COLLECTION_NAME = 'love_me_tender'


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
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )

    return vector_store


def init_collection(
        client=None,
        collection_name=DEFAULT_COLLECTION_NAME,
        **kwargs
):
    if client is None:
        client = get_client()

    # Create the collection if it doesn't exist
    if not client.get_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # KTODO vectors_config need to be update and recreate collection if model changes
                'local': VectorParams(size=100, distance=Distance.COSINE),
            },
            **kwargs
        )
