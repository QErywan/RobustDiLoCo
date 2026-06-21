"""
Model factory for the simulation layer.

Wraps LlamaForCausalLM with the same hparams format used by the upstream
train.py (tokenizer_name is popped; all remaining keys go to LlamaConfig).
The returned model satisfies the interface expected by simulation/workers.py:
    outputs = model(input_ids=..., labels=...)
    loss = outputs.loss
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


def build_model(hparams_path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """
    Instantiate a LlamaForCausalLM from a simulation hparams JSON.

    The JSON must contain 'vocab_size'. 'tokenizer_name' is ignored (present
    for compatibility with the upstream format but not used here — the
    simulation handles tokenisation separately).
    """
    with open(hparams_path) as f:
        hparams = json.load(f)

    hparams.pop("tokenizer_name", None)
    config = LlamaConfig(**hparams)
    model = LlamaForCausalLM(config)
    return model.to(device)


def param_count(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
