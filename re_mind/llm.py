from langchain_huggingface import HuggingFacePipeline

from re_mind.utils.re_mind_utils import get_global_device


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
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/gemma-3-1b-it",
            task="text-generation",
            model_kwargs={
                "temperature": 0.7, "max_length": 10000,

                # "quantization_config": quantization_config,
            },
            pipeline_kwargs={"device_map": device},
        )

    return llm


