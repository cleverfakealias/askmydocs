import pytest
import torch
from transformers import GenerationConfig

from palimpsest.config import Settings
from palimpsest.models_factory import (
    ResolvedDevice,
    apply_generation_settings,
    dtype_for,
    resolve_device,
)


def _checkpoint_config() -> GenerationConfig:
    """A generation config shaped like the one Qwen2.5 ships."""
    return GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_explicit_device_is_returned_unchanged(device: ResolvedDevice) -> None:
    assert resolve_device(device) == device


def test_cpu_runs_in_full_precision() -> None:
    assert dtype_for("cpu") == torch.float32


def test_sampling_keeps_the_checkpoint_defaults(settings: Settings) -> None:
    config = _checkpoint_config()

    apply_generation_settings(config, settings.model_copy(update={"temperature": 0.3}), 0)

    values = config.to_dict()
    assert values["do_sample"] is True
    assert values["temperature"] == 0.3
    assert (values["top_p"], values["top_k"], values["repetition_penalty"]) == (0.8, 20, 1.05)
    assert values["eos_token_id"] == [151645, 151643]
    assert values["max_new_tokens"] == settings.max_new_tokens
    assert values["max_length"] is None


def test_zero_temperature_switches_to_greedy_decoding(settings: Settings) -> None:
    config = _checkpoint_config()

    apply_generation_settings(config, settings.model_copy(update={"temperature": 0.0}), 0)

    values = config.to_dict()
    assert values["do_sample"] is False
    assert (values["temperature"], values["top_p"], values["top_k"]) == (None, None, None)
    assert values["repetition_penalty"] == 1.05


def test_missing_pad_token_falls_back_to_eos(settings: Settings) -> None:
    config = GenerationConfig()

    apply_generation_settings(config, settings, fallback_pad_token_id=7)

    assert config.to_dict()["pad_token_id"] == 7
