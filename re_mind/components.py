from llmchat.language_models import get_llm  # noqa


def get_client(**kwargs):
    # KTODO we can remove librarian dependency here by copying the code
    # KTODO remove librarian dependency from re-mind
    from librarian.components import get_client as get_client_impl
    from re_mind.envvars import get_qdrant_data_path
    return get_client_impl(path=get_qdrant_data_path(), **kwargs)


def get_vector_store(**kwargs):
    from librarian.components import get_vector_store as get_vector_store_impl
    return get_vector_store_impl(client=get_client(), **kwargs)
