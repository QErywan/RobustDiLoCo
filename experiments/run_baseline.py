"""
Baseline DiLoCo run: MeanAggregator + NoPerturbation.

Usage:
    # Fast smoke test (small model, synthetic data, few steps)
    python experiments/run_baseline.py --smoke

    # Full baseline (134M model, synthetic data, configurable steps)
    python experiments/run_baseline.py --outer-steps 200

    # Full baseline with real data shards
    python experiments/run_baseline.py --outer-steps 200 --data-path /path/to/shards
"""

import argparse
import json
import time
from pathlib import Path

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from tplr.data import SyntheticDataset, ShadedDataset
from simulation.data import make_hf_loaders
from simulation.model import build_model, param_count
from simulation.workers import SimConfig, Worker, Simulation
from simulation.aggregators import MeanAggregator
from simulation.perturbations import NoPerturbation


def _load_shard_meta(data_path: str) -> dict:
    """
    Load meta.json written by pretokenize.py so we know the exact token
    counts to pass to ShadedDataset.  Falls back to safe defaults if the
    file is absent (e.g. legacy shard dirs).
    """
    meta_path = Path(data_path) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    # Fallback: count tokens by scanning shard file sizes.
    # All shards must be int16 (2 bytes/token) for the size estimate to work.
    train_files = sorted(Path(data_path).glob("train_*.npy"))
    val_files   = sorted(Path(data_path).glob("validation_*.npy"))
    tokens_per_shard = (
        Path(train_files[0]).stat().st_size // 2 if train_files else 100_000_000
    )
    n_train = sum(f.stat().st_size // 2 for f in train_files)
    n_val   = sum(f.stat().st_size // 2 for f in val_files)
    return {
        "n_train_tokens":   n_train,
        "n_val_tokens":     n_val,
        "tokens_per_shard": tokens_per_shard,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMOKE = dict(
    hparams="hparams/sim/sim_model_hparams.json",
    n_workers=8,
    H=2,
    outer_steps=5,
    batch_size=2,
    seq_len=64,
)

FULL = dict(
    hparams="hparams/sim/sim_model_hparams_full.json",
    n_workers=8,
    H=500,
    batch_size=16,
    seq_len=1024,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Fast smoke test with tiny model")
    p.add_argument("--hparams", type=str, default=None, help="Override hparams file")
    p.add_argument("--outer-steps", type=int, default=20)
    p.add_argument("--n-workers", type=int, default=None, help="Override number of workers (default: 8)")
    p.add_argument("--H", type=int, default=None, help="Override inner steps per outer step (default: 500)")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "c4", "fineweb"], help="Data source")
    p.add_argument("--data-path", type=str, default=None, help="Path to pre-tokenized .npy shards (synthetic mode only)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--offload", action="store_true", help="Page workers on/off device one at a time to reduce peak VRAM")
    p.add_argument("--verbose", action="store_true", help="Print per-worker progress during each outer step")
    p.add_argument("--out", type=str, default="experiments/results/baseline.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

HF_DATASET_CONFIG = {
    "c4":      ("allenai/c4", "en",      "gpt2"),
    "fineweb": ("HuggingFaceFW/fineweb", "default", "gpt2"),
}


def make_worker_loaders(
    n_workers: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    dataset: str,
    data_path: str | None,
    device: str,
) -> list[DataLoader]:
    """One DataLoader per worker.

    Routing priority:
        1. --data-path given → pre-tokenized ShadedDataset (fastest; pipeline
           consistency requires ALL cells in a reported sweep to use this path).
        2. dataset in {"c4", "fineweb"} and no data-path → HFStreamingDataset
           (slow: tokenizes on the fly; use only for smoke tests or profiling).
        3. Fallback → SyntheticDataset (random tokens; smoke tests only).
    """
    if data_path is not None:
        # Pre-tokenized shards — fastest path.
        # Read meta.json for exact token counts so ShadedDataset slices correctly.
        meta = _load_shard_meta(data_path)
        n_train_tokens   = meta["n_train_tokens"]
        tokens_per_shard = meta["tokens_per_shard"]
        print(f"[data] Pre-tokenized shards: {data_path}  "
              f"({n_train_tokens/1e6:.0f}M train tokens, shard_size={tokens_per_shard/1e6:.0f}M)")
        loaders = []
        for rank in range(n_workers):
            ds = ShadedDataset(
                shards_path=data_path,
                token_budget=n_train_tokens,
                sequence_length=seq_len,
                rank=rank,
                world_size=n_workers,
                device=torch.device(device),
                shard_token_size=tokens_per_shard,
                split="train",
            )
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0))
        return loaders

    if dataset in HF_DATASET_CONFIG:
        # Streaming from HuggingFace — slow (CPU tokenization in the loop).
        # Retained for smoke tests and profiling; not recommended for full sweeps.
        ds_name, ds_config, tokenizer_name = HF_DATASET_CONFIG[dataset]
        print(f"[data] Streaming {dataset} from HuggingFace ({ds_name})  "
              f"[SLOW — run pretokenize.py and pass --data-path for full sweeps]")
        return make_hf_loaders(
            dataset_name=ds_name,
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
            batch_size=batch_size,
            n_workers=n_workers,
        )

    # Synthetic random data — smoke tests only.
    loaders = []
    for _ in range(n_workers):
        ds = SyntheticDataset(
            vocab_size=vocab_size,
            sequence_length=seq_len,
            num_samples=100_000,
        )
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
    return loaders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    cfg = SMOKE if args.smoke else FULL
    if not args.smoke:
        cfg = {**cfg, "outer_steps": args.outer_steps}
    if args.hparams:
        cfg = {**cfg, "hparams": args.hparams}
    if args.n_workers is not None:
        cfg = {**cfg, "n_workers": args.n_workers}
    if args.H is not None:
        cfg = {**cfg, "H": args.H}
    if args.batch_size is not None:
        cfg = {**cfg, "batch_size": args.batch_size}

    print(f"\n{'='*60}")
    print(f"  DiLoCo Baseline — {'SMOKE TEST' if args.smoke else 'FULL RUN'}")
    print(f"  hparams : {cfg['hparams']}")
    print(f"  workers : {cfg['n_workers']}")
    print(f"  H       : {cfg['H']}")
    print(f"  steps   : {cfg['outer_steps']}")
    print(f"  device  : {args.device}")
    print(f"  offload : {args.offload}")
    print(f"{'='*60}\n")

    # Model
    model = build_model(cfg["hparams"], device=args.device)
    counts = param_count(model)
    vocab_size = model.config.vocab_size
    print(f"Model: {counts['total'] / 1e6:.1f}M params")

    # Per-worker dataloaders
    loaders = make_worker_loaders(
        n_workers=cfg["n_workers"],
        vocab_size=vocab_size,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        dataset=args.dataset,
        data_path=args.data_path,
        device=args.device,
    )

    # Workers
    sim_config = SimConfig(
        H=cfg["H"],
        device=args.device,
        offload_between_steps=args.offload,
        verbose=args.verbose,
    )
    workers = [
        Worker(rank=i, model=model, dataloader=loaders[i], config=sim_config)
        for i in range(cfg["n_workers"])
    ]

    sim = Simulation(
        workers=workers,
        aggregator=MeanAggregator(),
        perturbation=NoPerturbation(),
        config=sim_config,
    )

    # Run
    history = []
    print(f"{'Step':>6}  {'Loss':>8}  {'PG Norm':>10}  {'Time':>7}")
    print("-" * 40)

    for step in tqdm(range(cfg["outer_steps"]), desc="Outer steps"):
        t0 = time.time()
        metrics = sim.run_outer_step()
        elapsed = time.time() - t0

        history.append(metrics)
        print(
            f"{metrics['outer_step']:>6}  "
            f"{metrics['loss/mean']:>8.4f}  "
            f"{metrics['pseudo_grad_norm/mean']:>10.4f}  "
            f"{elapsed:>6.1f}s"
        )

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "history": history}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return history


if __name__ == "__main__":
    args = parse_args()
    run(args)
