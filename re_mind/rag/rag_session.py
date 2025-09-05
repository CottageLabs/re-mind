from langchain.chains import LLMChain

from langchain.chains import LLMChain

from re_mind import pipelines
from re_mind.llm import create_llm_huggingface
from re_mind.utils import re_mind_utils as llm_utils


class RagSession:
    def __init__(self):
        device = 'cuda'
        llm_utils.set_global_device(device)

        # llm = components.get_llm()
        # llm = create_llm_huggingface('cuda')
        llm = create_llm_huggingface(
            device=device,
            # model_id="google/gemma-3-1b-it",
            model_id="google/gemma-3-4b-it",
            # model_id="google/gemma-3-12b-it",
            # quantization_config=create_8bit_quantization_config(),
            # temperature=0.3,
            temperature=1.2,
        )

        self.rag_chain = pipelines.create_basic_qa(llm)

    def chat(self, user_input, **kwargs):
        response = self.rag_chain.invoke(user_input, **kwargs)
        return response
