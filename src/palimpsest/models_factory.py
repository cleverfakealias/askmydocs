"""Build the local chat model and embedder from settings. Heavy imports stay inside functions."""

# transformers ships incomplete type information: its loaders and config attributes are
# untyped. This file is the boundary to it, so strict "unknown type" reports are off here only.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from typing import TYPE_CHECKING, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from palimpsest.config import Device, Settings

if TYPE_CHECKING:
    import torch
    from transformers import GenerationConfig

type ResolvedDevice = Literal["cpu", "cuda", "mps"]


def resolve_device(requested: Device) -> ResolvedDevice:
    """Turn 'auto' into the best device the machine offers.

    Args:
        requested: The configured device. Any value other than 'auto' is returned as is.

    Returns:
        'cuda' when an NVIDIA GPU is usable, then 'mps' on Apple Silicon, then 'cpu'.
    """
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for(device: ResolvedDevice) -> torch.dtype:
    """Choose the weight precision that the device runs fast.

    Args:
        device: The resolved device.

    Returns:
        bfloat16 on GPUs that support it, float16 on other GPUs and on MPS, and
        float32 on CPU, where half precision is slow or unsupported.
    """
    import torch

    match device:
        case "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case "mps":
            return torch.float16
        case "cpu":
            return torch.float32


def apply_generation_settings(
    config: GenerationConfig, settings: Settings, fallback_pad_token_id: int | None
) -> None:
    """Apply the reply length and temperature on top of the checkpoint's tuned defaults.

    The checkpoint's own sampling options (top_p, top_k, repetition penalty) and its
    end-of-sequence tokens are kept.

    Args:
        config: The model's generation config, changed in place.
        settings: The application settings.
        fallback_pad_token_id: Padding token to use when the checkpoint sets none.
    """
    if settings.temperature > 0:
        config.update(do_sample=True, temperature=settings.temperature)
    else:
        # Greedy decoding ignores the sampling options. Clear them so transformers does not warn.
        config.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    config.update(max_new_tokens=settings.max_new_tokens, max_length=None)
    if config.pad_token_id is None:
        config.pad_token_id = fallback_pad_token_id


def build_embeddings(settings: Settings) -> Embeddings:
    """Load the sentence embedder. Vectors are normalized so cosine scores stay in range."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.resolved_embedding_model,
        model_kwargs={"device": resolve_device(settings.device)},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Load the chat model and wrap its text-generation pipeline for chat messages."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = settings.resolved_llm_model
    device = resolve_device(settings.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype_for(device), device_map=device
    )
    apply_generation_settings(model.generation_config, settings, tokenizer.eos_token_id)

    # Byte-level BPE tokenizers such as Qwen's lose spaces before punctuation when cleanup is on.
    text_generation = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )
    # The pipeline fills in max_length=20 when it starts. Clear it, so max_new_tokens alone
    # sets the reply length.
    text_generation.generation_config.max_length = None
    # Pass the tokenizer, so the chat wrapper does not download and load a second copy.
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_generation), tokenizer=tokenizer)
