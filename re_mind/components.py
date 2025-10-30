from librarian.components import (  # noqa
    get_embedding,
)
from llmchat.language_models import get_llm  # noqa


def get_client(**kwargs):
    from librarian.components import get_client as get_client_impl
    from re_mind import cpaths
    return get_client_impl(path=cpaths.QDRANT_DATA_PATH, **kwargs)


def get_vector_store(**kwargs):
    from librarian.components import get_vector_store as get_vector_store_impl
    return get_vector_store_impl(client=get_client(), **kwargs)
