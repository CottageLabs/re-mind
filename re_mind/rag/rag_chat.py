from langchain.chains import LLMChain

from re_mind import pipelines, components


class RagChat:

    def __init__(self, llm, config=None, device=None):
        config = config or {}
        self.llm = llm
        # KTODO rename all vectorstore to vector_store
        self.vectorstore = config.get('vectorstore') or components.get_vector_store()
        n_top_result = config.get('n_top_result', 8)
        self.rag_chain = pipelines.build_rag_app(llm, n_top_result=n_top_result, )
        self.device = device or config.get('device', 'cpu')

    def chat(self, user_input,
             state: dict = None,
             configurable: dict = None):
        state = state or {}
        configurable = configurable or {}

        final_state = {'question': user_input} | state
        final_configurable = {
                                 'llm': self.llm,
                                 'vectorstore': self.vectorstore,
                                 'device': self.device,
                                 # KTODO handle n_top_result, temperature, etc.
                             } | configurable
        response = self.rag_chain.invoke(
            final_state,
            config={'configurable': final_configurable }
        )
        return response
