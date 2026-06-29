"""
Throughput profiler for the DiLoCo Tier-1 training pipeline.

Measures:
    1. Tokenization cost vs. model forward+backward cost — to confirm the bottleneck.
    2. Steps/sec on the streaming C4 loader (current baseline).
    3. Steps/sec on the pre-tokenized shard loader (after pretokenize.py).
    4. (Alongside) GPU utilisation via nvidia-smi dmon — run separately:
           nvidia-smi dmon -s u -d 1

Expected outcome: tokenization is the dominant cost and GPU util is low on
the streaming path; the shard path shows a clear steps/sec improvement.

Usage
-----
# Profile the streaming loader (run before pretokenize.py)
python experiments/profile_throughput.py --device cuda

# Profile after shards are ready — pass --data-path to compare
python experiments/profile_throughput.py --device cuda \\
    --data-path /vol/bitbucket/qe25/data/c4_gpt2

# Quiet run (just the numbers, no per-step output)
python experiments/profile_throughput.py --device cuda --quiet
"""

from __future__ import annotations

import argparse
import time

import torch

from simulation.model import build_model
from experiments.run_baseline import make_worker_loaders, HF_DATASET_CONFIG  # noqa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hparams_for_profile() -> str:
    """Use the Tier-1 30M GPT2-small model for a realistic measurement."""
    return "hparams/sim/sim_model_hparams_nanogpt.json"


def time_tokenization(n_examples: int = 200) -> float:
    """
    Time how long pure HuggingFace tokenization of C4 takes for n_examples.

    Returns seconds per example (wall time, CPU single-threaded).
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("allenai/c4", "en", streaming=True, split="train",
                      trust_remote_code=True)

    t0 = time.perf_counter()
    for i, ex in enumerate(ds):
        tokenizer.encode(ex["text"])
        if i >= n_examples:
            break
    elapsed = time.perf_counter() - t0

    return elapsed / n_examples


def time_forward_backward(model, device: str, seq_len: int = 512,
                           batch_size: int = 4, n_steps: int = 20) -> float:
    """
    Time the model forward + backward for n_steps batches of random tokens.

    Returns average seconds per step (wall time, GPU synchronised).
    """
    dev = torch.device(device)
    model.train()
    vocab_size = model.config.vocab_size
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    times = []
    for _ in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(device_type=dev.type, enabled=(device != "cpu")):
            out = model(input_ids=x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return sum(times) / len(times)


def time_full_inner_loop(
    model,
    loader,
    device: str,
    seq_len: int,
    batch_size: int,
    n_steps: int = 100,
    quiet: bool = False,
) -> tuple[float, float]:
    """
    Time n_steps of the combined data-fetch + forward + backward loop.

    Returns (seconds_per_step, steps_per_second).
    """
    dev = torch.device(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader_iter = iter(loader)

    t_total = 0.0
    t_data = 0.0
    t_compute = 0.0

    for step in range(n_steps):
        # Time data fetch separately
        if device != "cpu":
            torch.cuda.synchronize()
        t_data_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        t_data_end = time.perf_counter()

        x = batch.to(dev)

        if device != "cpu":
            torch.cuda.synchronize()
        t_compute_start = time.perf_counter()

        with torch.autocast(device_type=dev.type, enabled=(device != "cpu")):
            out = model(input_ids=x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

        if device != "cpu":
            torch.cuda.synchronize()
        t_compute_end = time.perf_counter()

        step_data    = t_data_end - t_data_start
        step_compute = t_compute_end - t_compute_start
        step_total   = step_data + step_compute

        t_data    += step_data
        t_compute += step_compute
        t_total   += step_total

        if not quiet and (step % 10 == 0 or step < 5):
            print(f"  step {step+1:>3}/{n_steps}: "
                  f"data={step_data*1000:.1f}ms  "
                  f"compute={step_compute*1000:.1f}ms  "
                  f"total={step_total*1000:.1f}ms")

    secs_per_step = t_total / n_steps
    steps_per_sec = n_steps / t_total

    pct_data    = 100 * t_data / t_total
    pct_compute = 100 * t_compute / t_total

    print(f"\n  Data fetch    : {t_data/n_steps*1000:.1f}ms/step  ({pct_data:.0f}%)")
    print(f"  Model compute : {t_compute/n_steps*1000:.1f}ms/step  ({pct_compute:.0f}%)")
    print(f"  Total         : {secs_per_step*1000:.1f}ms/step  → {steps_per_sec:.1f} steps/sec")

    return secs_per_step, steps_per_sec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile DiLoCo inner-loop throughput (streaming vs shard loader)."
    )
    p.add_argument("--device",    type=str,  default="cuda")
    p.add_argument("--data-path", type=str,  default=None,
                   help="Path to pre-tokenized shard dir; if set, profiles both "
                        "streaming (skipped if --skip-streaming) AND shard loader.")
    p.add_argument("--skip-streaming", action="store_true",
                   help="Skip the streaming baseline (useful after shards are ready "
                        "and streaming results already known).")
    p.add_argument("--n-steps",   type=int,  default=100,
                   help="Inner steps to time per loader (default 100).")
    p.add_argument("--seq-len",   type=int,  default=512)
    p.add_argument("--batch-size",type=int,  default=4)
    p.add_argument("--quiet",     action="store_true",
                   help="Suppress per-step output.")
    p.add_argument("--skip-tok-profile", action="store_true",
                   help="Skip the standalone tokenization timing (saves ~30s).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  DiLoCo Throughput Profiler")
    print(f"  device={args.device}  seq_len={args.seq_len}  "
          f"batch_size={args.batch_size}  n_steps={args.n_steps}")
    print(f"{'='*60}\n")

    # Build 30M model (Tier-1 config)
    model = build_model(_hparams_for_profile(), device=args.device)
    vocab_size = model.config.vocab_size
    print(f"Model vocab_size={vocab_size}")

    # ------------------------------------------------------------------
    # 1. Standalone tokenization cost
    # ------------------------------------------------------------------
    if not args.skip_tok_profile:
        print("\n--- Tokenization cost (CPU, single-threaded) ---")
        n_examples = 50
        print(f"  Timing tokenizer.encode() on {n_examples} C4 examples...")
        tok_per_example = time_tokenization(n_examples=n_examples)
        print(f"  {tok_per_example*1000:.1f}ms per C4 example")
        # Estimate: at batch_size=4, seq_len=512, a GPT-2 tokenizer produces
        # ~512 tokens per example, so each batch needs ~batch_size * seq_len
        # / avg_tok_per_example C4 examples.
        est_tok_per_step = tok_per_example * args.batch_size * 10  # rough
        print(f"  Rough tokenization overhead per inner step: ~{est_tok_per_step*1000:.0f}ms")
    else:
        print("\n[skip] Standalone tokenization profile.")

    # ------------------------------------------------------------------
    # 2. Forward + backward only (upper bound on GPU speed)
    # ------------------------------------------------------------------
    print("\n--- Forward + Backward only (synthetic random data) ---")
    compute_only = time_forward_backward(
        model, args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_steps=20,
    )
    print(f"  Compute-only: {compute_only*1000:.1f}ms/step  "
          f"→ {1/compute_only:.1f} steps/sec (upper bound)")

    # ------------------------------------------------------------------
    # 3. Streaming loader (HFStreamingDataset)
    # ------------------------------------------------------------------
    results = {}

    if not args.skip_streaming:
        print("\n--- Streaming C4 loader (current pipeline) ---")
        streaming_loaders = make_worker_loaders(
            n_workers=1,   # one worker is enough to measure the bottleneck
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dataset="c4",
            data_path=None,   # force streaming
            device=args.device,
        )
        secs, rate = time_full_inner_loop(
            model, streaming_loaders[0],
            device=args.device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            quiet=args.quiet,
        )
        results["streaming"] = (secs, rate)
    else:
        print("\n[skip] Streaming loader profile.")

    # ------------------------------------------------------------------
    # 4. Shard loader (ShadedDataset) — only if --data-path given
    # ------------------------------------------------------------------
    if args.data_path:
        print(f"\n--- Pre-tokenized shard loader ({args.data_path}) ---")
        shard_loaders = make_worker_loaders(
            n_workers=1,
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dataset="c4",
            data_path=args.data_path,
            device=args.device,
        )
        secs, rate = time_full_inner_loop(
            model, shard_loaders[0],
            device=args.device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            quiet=args.quiet,
        )
        results["shard"] = (secs, rate)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Compute-only ceiling: {1/compute_only:.1f} steps/sec")
    if "streaming" in results:
        _, r = results["streaming"]
        print(f"  Streaming loader    : {r:.1f} steps/sec")
    if "shard" in results:
        _, r = results["shard"]
        print(f"  Shard loader        : {r:.1f} steps/sec")
    if "streaming" in results and "shard" in results:
        speedup = results["shard"][1] / results["streaming"][1]
        print(f"  Shard speedup       : {speedup:.1f}×")

    # Time estimate for Tier-1: 8 workers × H=500 inner steps per outer step × 50 outer
    if "streaming" in results or "shard" in results:
        print()
        for label, key in [("Streaming", "streaming"), ("Shard", "shard")]:
            if key not in results:
                continue
            inner_secs_per_step, _ = results[key]
            # Each outer step = 8 workers × H=500 inner steps (sequential)
            outer_step_secs = 8 * 500 * inner_secs_per_step
            total_cell_secs = 50 * outer_step_secs
            print(f"  Estimated Tier-1 time ({label}):")
            print(f"    per outer step : {outer_step_secs/60:.1f} min")
            print(f"    per cell (50)  : {total_cell_secs/3600:.1f} h")
            print(f"    full grid (90) : {90 * total_cell_secs / 3600:.0f} h  "
                  f"({90 * total_cell_secs / 86400:.1f} days)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
