"""
Tests for simulation/model.py.

Covers: factory function, forward pass interface, param counts.
Also verifies the model plugs into Worker correctly (one inner step).
"""

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

from simulation.model import build_model, param_count
from simulation.workers import SimConfig, Worker

SMALL_HPARAMS = "hparams/sim/sim_model_hparams.json"
FULL_HPARAMS = "hparams/sim/sim_model_hparams_full.json"
SEQ_LEN = 16
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    return build_model(SMALL_HPARAMS)


def _synthetic_loader(vocab_size: int, seq_len: int = SEQ_LEN, n: int = 8) -> DataLoader:
    # DataLoader over a raw tensor yields plain tensor batches (not lists),
    # matching the interface workers.py expects: batch.to(device)
    tokens = torch.randint(0, vocab_size, (n, seq_len))
    return DataLoader(tokens, batch_size=BATCH_SIZE)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_returns_llama(small_model):
    assert isinstance(small_model, LlamaForCausalLM)


def test_build_model_on_cpu(small_model):
    p = next(small_model.parameters())
    assert p.device.type == "cpu"


def test_tokenizer_name_not_in_config(small_model):
    # tokenizer_name must be stripped — LlamaConfig would error if it wasn't
    assert not hasattr(small_model.config, "tokenizer_name")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_returns_loss(small_model):
    vocab_size = small_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))
    outputs = small_model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss is not None


def test_loss_is_scalar(small_model):
    vocab_size = small_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))
    outputs = small_model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss.shape == torch.Size([])


def test_loss_is_finite(small_model):
    vocab_size = small_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))
    outputs = small_model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(outputs.loss)


# ---------------------------------------------------------------------------
# param_count
# ---------------------------------------------------------------------------

def test_param_count_keys(small_model):
    counts = param_count(small_model)
    assert "total" in counts and "trainable" in counts


def test_param_count_positive(small_model):
    counts = param_count(small_model)
    assert counts["total"] > 0
    assert counts["trainable"] > 0


def test_param_count_all_trainable(small_model):
    counts = param_count(small_model)
    assert counts["total"] == counts["trainable"]


def test_full_model_param_count_in_range():
    model = build_model(FULL_HPARAMS)
    counts = param_count(model)
    assert 100_000_000 <= counts["total"] <= 150_000_000, (
        f"Expected 100–150M params, got {counts['total'] / 1e6:.1f}M"
    )


# ---------------------------------------------------------------------------
# Worker integration
# ---------------------------------------------------------------------------

def test_model_plugs_into_worker():
    """One inner step completes without error and returns a finite loss."""
    model = build_model(SMALL_HPARAMS)
    vocab_size = model.config.vocab_size
    loader = _synthetic_loader(vocab_size)

    config = SimConfig(H=2, device="cpu")
    worker = Worker(rank=0, model=model, dataloader=loader, config=config)

    metrics = worker.inner_step()
    assert "mean_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["mean_loss"]))


def test_pseudo_grad_shape_matches_param_count():
    """Flat pseudo-gradient length must equal total parameter count."""
    model = build_model(SMALL_HPARAMS)
    vocab_size = model.config.vocab_size
    loader = _synthetic_loader(vocab_size)

    config = SimConfig(H=2, device="cpu")
    worker = Worker(rank=0, model=model, dataloader=loader, config=config)
    worker.inner_step()

    pg = worker.compute_pseudo_grad()
    expected = sum(p.numel() for p in model.parameters())
    assert pg.shape == (expected,)
