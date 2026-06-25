"""
Evaluation utilities for DiLoCo simulation experiments.

Two levels of evaluation:
    1. Language-model metrics (primary, always available):
          held-out loss, perplexity = exp(loss).
          Computed by running the model forward on a separate evaluation
          DataLoader — no side effects on training state.

    2. Downstream task benchmarks (clean configs only):
          HellaSwag, ARC-Easy, PIQA via lm-evaluation-harness.
          Requires:  pip install lm-evaluation-harness
          IMPORTANT: run downstream eval ONLY on clean configurations
          (NoPerturbation + MeanAggregator baseline). Byzantine runs
          should not be evaluated for absolute task scores — they are
          assessed via the LM loss / perplexity degradation relative
          to the clean baseline (effect size).

Usage
-----
# Held-out perplexity after the final outer step
from simulation.evaluate import eval_perplexity, make_eval_loader

loader = make_eval_loader(dataset="c4", seq_len=1024, batch_size=4, n_batches=50)
metrics = eval_perplexity(model, loader, device="cuda")
# → {"eval_loss": 3.21, "perplexity": 24.7, "n_tokens": 204800}

# Downstream benchmarks (clean configs only)
from simulation.evaluate import run_downstream_eval
bench = run_downstream_eval(model, device="cuda",
                            tasks=["hellaswag", "arc_easy", "piqa"])
# → {"hellaswag": 0.42, "arc_easy": 0.58, "piqa": 0.61}
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Held-out data loader
# ---------------------------------------------------------------------------

def make_eval_loader(
    dataset: str = "c4",
    seq_len: int = 1024,
    batch_size: int = 4,
    n_batches: int = 50,
    tokenizer_name: str = "gpt2",
    split: str = "validation",
    vocab_size: int = 50257,
) -> DataLoader:
    """
    Create a held-out evaluation DataLoader.

    Uses the dataset validation split (C4 English validation) so eval tokens
    are never seen during training.  n_batches controls how many batches are
    drawn — 50 × batch_size=4 × seq_len=1024 ≈ 200k tokens, enough for a
    stable perplexity estimate.

    Args:
        dataset:        "c4", "fineweb", or "synthetic"
        seq_len:        token sequence length (must match model n_positions)
        batch_size:     eval batch size (can be larger than training batch)
        n_batches:      how many batches to evaluate over (caps the eval cost)
        tokenizer_name: HuggingFace tokenizer ID (default "gpt2" for GPT2 vocab)
        split:          HuggingFace dataset split (default "validation")
        vocab_size:     vocabulary size for synthetic data generation (must match
                        model.config.vocab_size; default 50257 for GPT2).

    Returns:
        A DataLoader that yields (batch_size, seq_len) long tensors.
    """
    if dataset == "synthetic":
        from tplr.data import SyntheticDataset
        ds = SyntheticDataset(
            vocab_size=vocab_size,
            sequence_length=seq_len,
            num_samples=n_batches * batch_size,
        )
        return DataLoader(ds, batch_size=batch_size)

    from simulation.data import HFStreamingDataset

    DATASET_CONFIGS = {
        "c4":      ("allenai/c4",               "en",      tokenizer_name),
        "fineweb": ("HuggingFaceFW/fineweb",     "default", tokenizer_name),
    }
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {list(DATASET_CONFIGS)}")

    ds_name, ds_config, tok_name = DATASET_CONFIGS[dataset]
    # Use rank=0, world_size=1 — eval always sees the full validation split
    ds = HFStreamingDataset(
        dataset_name=ds_name,
        tokenizer_name=tok_name,
        seq_len=seq_len,
        rank=0,
        world_size=1,
        dataset_config=ds_config,
        split=split,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


# ---------------------------------------------------------------------------
# Held-out perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(
    model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cpu",
    n_batches: Optional[int] = 50,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Compute held-out cross-entropy loss and perplexity.

    Runs the model in eval mode with no_grad so training state is untouched.
    The caller is responsible for passing the right model (e.g. worker 0's
    model after an outer step, which is the global model in sync with all
    workers post-aggregation).

    Args:
        model:       Language model (must have a `.forward(input_ids, labels)`
                     that returns an object with `.loss`).
        eval_loader: DataLoader yielding (batch_size, seq_len) long tensors.
        device:      Torch device string.
        n_batches:   Max batches to evaluate over (None = full loader).
        dtype:       Model computation dtype.

    Returns:
        dict with keys: "eval_loss" (float), "perplexity" (float),
        "n_tokens" (int), "n_batches" (int).
    """
    model.eval()
    dev = torch.device(device)
    model_was_on_cpu = next(model.parameters()).device.type == "cpu"

    # Move to eval device temporarily if needed
    if model_was_on_cpu and device != "cpu":
        model.to(dev)

    total_loss = 0.0
    total_batches = 0
    total_tokens = 0

    for i, batch in enumerate(eval_loader):
        if n_batches is not None and i >= n_batches:
            break
        input_ids = batch.to(dev)
        labels    = input_ids.clone()

        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=(dev.type != "cpu")):
            outputs = model(input_ids=input_ids, labels=labels)

        total_loss    += outputs.loss.item()
        total_tokens  += input_ids.numel()
        total_batches += 1

    # Move back if we temporarily moved it
    if model_was_on_cpu and device != "cpu":
        model.to("cpu")

    model.train()

    if total_batches == 0:
        raise RuntimeError("eval_loader yielded no batches — check n_batches or dataset split.")

    avg_loss   = total_loss / total_batches
    perplexity = math.exp(avg_loss)

    return {
        "eval_loss":  avg_loss,
        "perplexity": perplexity,
        "n_tokens":   total_tokens,
        "n_batches":  total_batches,
    }


# ---------------------------------------------------------------------------
# Downstream task evaluation (clean configs only)
# ---------------------------------------------------------------------------

def run_downstream_eval(
    model: nn.Module,
    device: str = "cuda",
    tasks: list[str] | None = None,
    num_fewshot: int = 0,
) -> dict:
    """
    Evaluate on downstream benchmarks using lm-evaluation-harness.

    IMPORTANT: Call this ONLY for clean (NoPerturbation) configurations.
    Byzantine runs are assessed via perplexity degradation, not absolute
    task accuracy — accuracy scores on perturbed models are not meaningful
    thesis outputs.

    Requires:
        pip install lm-evaluation-harness
        (not included in the base project dependencies — install on Imperial
         before running Tier-2 clean-baseline evaluation)

    Args:
        model:       Trained language model.
        device:      Torch device string.
        tasks:       List of lm_eval task names. Defaults to the three thesis
                     tasks: ["hellaswag", "arc_easy", "piqa"].
        num_fewshot: Number of few-shot examples (0 = zero-shot, thesis default).

    Returns:
        dict mapping task name → accuracy (0–1).

    Raises:
        ImportError: if lm-evaluation-harness is not installed.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError(
            "lm-evaluation-harness is required for downstream eval. "
            "Install it with:\n"
            "    pip install lm-evaluation-harness\n"
            "Then re-run with --downstream-eval."
        ) from exc

    if tasks is None:
        tasks = ["hellaswag", "arc_easy", "piqa"]

    # Wrap the model in an lm_eval-compatible interface
    lm = HFLM(pretrained=model, device=device)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
    )

    # Extract per-task accuracy from the nested results dict
    task_scores = {}
    for task in tasks:
        # lm_eval ≥0.4 stores results under results[task]["acc,none"] or similar
        task_results = results["results"].get(task, {})
        # Try common key patterns
        for key in ["acc,none", "acc_norm,none", "acc"]:
            if key in task_results:
                task_scores[task] = task_results[key]
                break
        else:
            task_scores[task] = None   # task ran but metric key not found

    return task_scores
