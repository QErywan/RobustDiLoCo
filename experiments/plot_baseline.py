"""
Plot training loss curve from a baseline results JSON.

Usage:
    python experiments/plot_baseline.py --results experiments/results/baseline.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="experiments/results/baseline.json")
    p.add_argument("--out", type=str, default=None, help="Save path (e.g. baseline.png). Shows interactively if omitted.")
    return p.parse_args()


def plot(results_path: str, out_path: str | None = None):
    with open(results_path) as f:
        data = json.load(f)

    history = data["history"]
    cfg = data["config"]

    steps = [m["outer_step"] for m in history]
    loss_mean = [m["loss/mean"] for m in history]
    loss_min = [m["loss/min"] for m in history]
    loss_max = [m["loss/max"] for m in history]
    pg_norm = [m["pseudo_grad_norm/mean"] for m in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Loss
    ax1.plot(steps, loss_mean, label="Mean loss", color="steelblue", linewidth=2)
    ax1.fill_between(steps, loss_min, loss_max, alpha=0.2, color="steelblue", label="Min/max across workers")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title(
        f"DiLoCo Baseline — MeanAggregator, NoPerturbation\n"
        f"{cfg.get('n_workers', 8)} workers, H={cfg.get('H', 500)}, "
        f"hparams: {Path(cfg.get('hparams', '')).stem}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pseudo-gradient norm
    ax2.plot(steps, pg_norm, color="darkorange", linewidth=2)
    ax2.set_ylabel("Mean pseudo-grad norm")
    ax2.set_xlabel("Outer step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results, args.out)
