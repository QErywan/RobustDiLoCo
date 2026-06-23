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
from simulation.model import build_model, param_count
from simulation.workers import SimConfig, Worker, Simulation
from simulation.aggregators import MeanAggregator
from simulation.perturbations import NoPerturbation


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
    p.add_argument("--data-path", type=str, default=None, help="Path to pre-tokenized .npy shards")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--offload", action="store_true", help="Page workers on/off device one at a time to reduce peak VRAM")
    p.add_argument("--verbose", action="store_true", help="Print per-worker progress during each outer step")
    p.add_argument("--out", type=str, default="experiments/results/baseline.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_worker_loaders(
    n_workers: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    data_path: str | None,
    device: str,
) -> list[DataLoader]:
    """One DataLoader per worker. Synthetic if no data_path, real shards otherwise."""
    if data_path is None:
        # Each worker gets its own SyntheticDataset instance (independent streams)
        loaders = []
        for _ in range(n_workers):
            ds = SyntheticDataset(
                vocab_size=vocab_size,
                sequence_length=seq_len,
                num_samples=100_000,
            )
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
        return loaders
    else:
        loaders = []
        for rank in range(n_workers):
            ds = ShadedDataset(
                shards_path=data_path,
                token_budget=int(5e9),
                sequence_length=seq_len,
                rank=rank,
                world_size=n_workers,
                device=torch.device(device),
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
