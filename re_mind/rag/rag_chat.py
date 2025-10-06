from langchain.chains import LLMChain

from re_mind import pipelines, components


class RagChat:

    # KTODO rename all vectorstore to vector_store
    def __init__(self, llm, vectorstore=None, n_top_result=8, device='cpu'):
        self.llm = llm
        self.vectorstore = vectorstore or components.get_vector_store()
        self.rag_chain = pipelines.build_rag_app(llm,  n_top_result=n_top_result, )
        self.device = device

    @classmethod
    def create_by_huggingface(cls, temperature=1.2, n_top_result=8, device=None, return_full_text=False):
        from re_mind.language_models import create_llm_huggingface
        from re_mind.utils import re_mind_utils as llm_utils

        if device is None:
            device = llm_utils.get_global_device()
        llm_utils.set_global_device(device)

        # llm = components.get_llm()
        # llm = create_llm_huggingface('cuda')
        llm = create_llm_huggingface(
            device=device,
            model_id="google/gemma-3-1b-it",
            # model_id="google/gemma-3-4b-it",
            # model_id="google/gemma-3-12b-it",
            # quantization_config=create_8bit_quantization_config(),
            # temperature=0.3,
            temperature=temperature,
            return_full_text=return_full_text,
        )

        return cls(llm, n_top_result=n_top_result)

    @classmethod
    def create_by_openai(cls, model='gpt-5-nano-2025-08-07', n_top_result=8):
        from re_mind.language_models import create_openai_model

        llm = create_openai_model(model=model)
        return cls(llm, n_top_result=n_top_result)

    def chat(self, user_input, **kwargs):
        response = self.rag_chain.invoke(
            {'question': user_input},
            config={'configurable': {
                'llm': self.llm,
                'vectorstore': self.vectorstore,
                'device': self.device,
                # KTODO handle n_top_result, temperature, etc.
            }}

        )
        return response
