from langchain_huggingface import HuggingFacePipeline

from re_mind.utils.re_mind_utils import get_global_device


def create_8bit_quantization_config():
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Use 8-bit quantization
        llm_int8_threshold=6.0,  # Controls mixed precision, lower = more precise
        llm_int8_enable_fp32_cpu_offload=True,  # Offload high-precision layers to CPU if needed
        llm_int8_has_fp16_weight=False  # Keep some weights in fp32/fp16 to preserve accuracy
    )
    return quantization_config


def create_llm_huggingface(device=None, model_id="google/gemma-3-1b-it", temperature=0.7, max_length=10000,
                           quantization_config=None
                           ):
    if device is None:
        device = get_global_device()

    model_kwargs = {
        "temperature": temperature, "max_length": max_length,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    return HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        model_kwargs=model_kwargs,
        pipeline_kwargs={"device_map": device},
    )



def get_llm(device=None, openai_model=None):
    if device is None:
        device = get_global_device()
    if openai_model:

        from openai import OpenAI

        # Tiny adapter so we can use OpenAI client as an LCEL-compatible “callable”
        class OpenAIChatAsRunnable:
            def __init__(self, model="gpt-4o-mini"):
                self.client = OpenAI()
                self.model = model

            def invoke(self, messages):
                # messages is a list of dicts: {"role": "system"/"user", "content": "..."}
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": m.type if hasattr(m, 'type') else m["role"],
                               "content": m.content if hasattr(m, 'content') else m["content"]}
                              for m in messages]
                )
                return completion.choices[0].message.content

        llm = OpenAIChatAsRunnable(model="gpt-4o-mini")

    else:
        llm = create_llm_huggingface(device)

    return llm
