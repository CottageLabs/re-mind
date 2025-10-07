import logging

import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline, AutoTokenizer

from llmchat.torch_utils import get_sys_device

log = logging.getLogger(__name__)


def create_8bit_quantization_config():
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Use 8-bit quantization
        llm_int8_threshold=6.0,  # Controls mixed precision, lower = more precise
        llm_int8_enable_fp32_cpu_offload=True,  # Offload high-precision layers to CPU if needed
        llm_int8_has_fp16_weight=False  # Keep some weights in fp32/fp16 to preserve accuracy
    )
    return quantization_config


def create_4bit_quantization_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def create_llm_huggingface(device=None, model_id="google/gemma-3-1b-it", temperature=0.7, max_length=10000,
                           quantization_config=None, return_full_text=False
                           ):

    device = device or get_sys_device()

    model_kwargs = {
        "temperature": temperature, "max_length": max_length,
    }

    if device == 'cpu':
        model_kwargs["torch_dtype"] = torch.float32  # or torch.bfloat16, torch.float16 if your CPU supports it
        # model_kwargs["low_cpu_mem_usage"] = True

    if quantization_config is not None:
        if device == 'cpu':
            log.warning("Quantization is not needed on CPU, ignoring quantization_config")
        else:
            model_kwargs["quantization_config"] = quantization_config

    llm = HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation",
                                            model_kwargs=model_kwargs,
                                            # pipeline_kwargs={"device_map": device},
                                            device=0 if device != 'cpu' else -1,
                                            pipeline_kwargs={
                                                "return_full_text": return_full_text,
                                                "max_new_tokens": 1000,
                                                "do_sample": True,
                                            },
                                            )

    chat = ChatHuggingFace(llm=llm)
    return chat


def create_llama3(device=None, temperature=0.7, max_length=10000,
                  return_full_text=False
                  ):
    device = device or get_sys_device()

    # model_kwargs = {
    #     "temperature": temperature, "max_length": max_length,
    #     "quantization_config": create_8bit_quantization_config(),
    # }

    model_kwargs = {
        "quantization_config": create_8bit_quantization_config(),
        # # "quantization_config": create_4bit_quantization_config(),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "max_memory": {0: "20GiB", "cpu": "64GiB"},
        "offload_folder": "./offload_cache",
        "temperature": temperature,
    }

    # model_id = 'huihui-ai/Llama-3.2-3B-Instruct-abliterated'
    # model_id = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
    # model_id = 'watt-ai/watt-tool-8B'
    model_id = 'NousResearch/Hermes-3-Llama-3.1-8B'
    # llm = HuggingFacePipeline.from_model_id(model_id=model_id,
    #                                         task="text-generation",
    #                                         model_kwargs=model_kwargs,
    #                                         # pipeline_kwargs={"device_map": device},
    #                                         pipeline_kwargs={
    #                                             "return_full_text": return_full_text,
    #                                             # "max_new_tokens": 1000,
    #                                             # "do_sample": True,
    #                                             # "device_map": device,
    #
    #                                             "max_new_tokens": 512,
    #                                             "do_sample": True,
    #                                             "device_map": "auto",
    #                                             "use_cache": False,  # big VRAM saver during generation
    #                                         },
    #                                         )

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=create_4bit_quantization_config(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    )
    # llm = HuggingFacePipeline(pipeline=pipe)
    # chat = ChatHuggingFace(llm=llm)
    chat = ChatHuggingFace(pipeline=pipe)
    return chat

    # return llm

    # chat = ChatHuggingFace(llm=llm)
    # return chat


def create_openai_model(model='gpt-5-nano-2025-08-07'):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model)
    return llm


def create_vllm_model():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",  # vLLM default
        # api_key="dummy",
        model="unsloth/Llama-3.2-3B-Instruct"
    )
    return llm


def get_llm(device=None, openai_model=None):
    device = device or get_sys_device()
    if openai_model:
        llm = create_openai_model()
    else:
        llm = create_llm_huggingface(device)

    return llm
