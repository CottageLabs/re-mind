from langchain.chains import LLMChain
from langchain.chains import LLMChain

from re_mind import pipelines
from re_mind.language_models import create_llm_huggingface
from re_mind.utils import re_mind_utils as llm_utils


class RagSession:
    def __init__(self, temperature=1.2, n_top_result=8, device=None, return_full_text=False):
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

        self.rag_chain = pipelines.build_rag_app(llm, n_top_result=n_top_result, )

    def chat(self, user_input, **kwargs):
        response = self.rag_chain.invoke(
            {
                'question': user_input
            }
        )
        return response
