"""
Model factory for the simulation layer.

Supports two architecture families, selected via "model_type" in the hparams JSON:
  - "llama"  (default): LlamaForCausalLM — upstream SparseLoCo architecture
  - "gpt2":             GPT2LMHeadModel  — NanoGPT-style as cited in the interim report

Both satisfy the interface expected by simulation/workers.py:
    outputs = model(input_ids=..., labels=...)
    loss = outputs.loss
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    LlamaConfig, LlamaForCausalLM,
)


def build_model(hparams_path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """
    Instantiate a model from a simulation hparams JSON.

    The JSON may contain:
      - "model_type": "gpt2" | "llama" (default: "llama")
      - "tokenizer_name": ignored (stripped before passing to config)
      - all other keys forwarded to the corresponding HuggingFace config
    """
    with open(hparams_path) as f:
        hparams = json.load(f)

    hparams.pop("tokenizer_name", None)
    model_type = hparams.pop("model_type", "llama")

    if model_type == "gpt2":
        config = GPT2Config(**hparams)
        model = GPT2LMHeadModel(config)
    else:
        config = LlamaConfig(**hparams)
        model = LlamaForCausalLM(config)

    return model.to(device)


def param_count(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
