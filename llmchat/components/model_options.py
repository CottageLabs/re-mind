from abc import abstractmethod, ABC

from langchain_core.language_models import BaseChatModel


class ModelOption(ABC):

    @property
    def name(self):
        """
        identifiable name for the model option
        """
        raise NotImplementedError

    @abstractmethod
    def create(self) -> BaseChatModel:
        """
        Create and return the underlying model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self):
        """
        Free VRAM by detaching and deleting the underlying model, if applicable.
        """
        raise NotImplementedError


class HuggingFaceModelOption(ModelOption):
    def __init__(self, temperature=1.2, n_top_result=8, device=None, return_full_text=False,
                 model_id="google/gemma-3-1b-it", name=None):
        self.temperature = temperature
        self.n_top_result = n_top_result
        self.device = device
        self.return_full_text = return_full_text
        self.model_id = model_id
        self._name = name
        self.chat = None

    @property
    def name(self):
        return self._name or f"HuggingFace ({self.model_id})"

    def create(self) -> BaseChatModel:
        from llmchat.language_models import create_llm_huggingface
        from llmchat.torch_utils import get_sys_device

        device = self.device
        if device is None:
            device = get_sys_device()

        self.chat = create_llm_huggingface(
            device=device,
            model_id=self.model_id,
            temperature=self.temperature,
            return_full_text=self.return_full_text,
        )
        return self.chat

    def delete(self):
        """Free VRAM by detaching and deleting the underlying HF pipeline/model."""
        import torch
        import gc

        # Unwrap: ChatHuggingFace(llm=HuggingFacePipeline(...))
        if self.chat is not None and hasattr(self.chat, "llm"):
            llm = self.chat.llm

        # Get the HF pipeline (LangChain stores it on `pipeline`; some older versions use `client`)
        pipe = None
        if llm is not None:
            pipe = getattr(llm, "pipeline", None) or getattr(llm, "client", None)

        # Try to move model to CPU first (optional but can reduce fragmentation)
        try:
            if pipe is not None and hasattr(pipe, "model") and hasattr(pipe.model, "to"):
                pipe.model.to("cpu")
        except Exception:
            pass  # not fatal if sharded / 4-bit and .to('cpu') is awkward

        # Break references so GC can collect
        try:
            if pipe is not None:
                # Remove big attrs on the pipeline
                if hasattr(pipe, "model"):
                    delattr(pipe, "model")
                if hasattr(pipe, "tokenizer"):
                    delattr(pipe, "tokenizer")
        except Exception:
            pass

        try:
            if llm is not None and hasattr(llm, "pipeline"):
                llm.pipeline = None
            if llm is not None and hasattr(llm, "client"):
                llm.client = None
        except Exception:
            pass

        # Nuke top-level references in caller scope after calling this:
        #   chat = None; llm = None
        gc.collect()

        if torch.cuda.is_available():
            # Clear all device caches (useful when device_map='auto' sharded across GPUs)
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


class OpenAIModelOption(ModelOption):
    def __init__(self, model='gpt-5-nano-2025-08-07', n_top_result=8, name=None):
        self.model = model
        self.n_top_result = n_top_result
        self._name = name

    @property
    def name(self):
        return self._name or f"OpenAI ({self.model})"

    def create(self) -> BaseChatModel:
        from llmchat.language_models import create_openai_model

        llm = create_openai_model(model=self.model)
        return llm

    def delete(self):
        pass
