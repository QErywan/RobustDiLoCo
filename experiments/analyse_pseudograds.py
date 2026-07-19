"""
Pseudo-gradient geometry analysis for W2 thesis workstream.

Runs InstrumentedSimulation under MagnitudeAttack (f=2, scale=10) for all
5 aggregators and produces three thesis figures:
  fig1_norms.png          — per-worker norm trajectories (attack vs honest)
  fig2_cohesion_oracle.png — cluster cohesion + cosine-to-oracle over steps
  fig3_pca_snapshots.png  — 2D PCA of pseudo-grads at steps 1 and 25

Usage:
    # Full run on Imperial GPU
    python experiments/analyse_pseudograds.py \\
        --device cuda \\
        --data-path /vol/bitbucket/qe25/data/c4_gpt2

    # Local smoke test (tiny model, 3 outer steps)
    python experiments/analyse_pseudograds.py --smoke
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'simulation' is importable when the
# script is invoked as `python experiments/analyse_pseudograds.py` from any
# working directory.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from simulation.analysis import InstrumentedSimulation
from simulation.aggregators import (
    MeanAggregator, TrimmedMeanAggregator, CoordMedianAggregator,
    GeometricMedianAggregator, MultiKrumAggregator,
)
from simulation.model import build_model, param_count
from simulation.perturbations import MagnitudeAttack
from simulation.workers import SimConfig, Worker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WORKERS = 8
BYZANTINE_F = 2
ATTACK_SCALE = 10.0
PCA_STEPS = {1, 10, 25, 50}

AGGREGATOR_COLORS = {
    "mean": "black",
    "trimmed": "royalblue",
    "median": "seagreen",
    "rfa": "darkorange",
    "krum": "firebrick",
}

AGGREGATORS = {
    "mean":    lambda: MeanAggregator(),
    "trimmed": lambda: TrimmedMeanAggregator(f=BYZANTINE_F, n_workers=N_WORKERS),
    "median":  lambda: CoordMedianAggregator(),
    "rfa":     lambda: GeometricMedianAggregator(),
    "krum":    lambda: MultiKrumAggregator(f=BYZANTINE_F, n_workers=N_WORKERS),
}

SMOKE_CFG = dict(
    hparams="hparams/sim/sim_model_hparams.json",
    H=2,
    outer_steps=3,
    batch_size=2,
    seq_len=64,
    pca_steps={1},
)

FULL_CFG = dict(
    hparams="hparams/sim/sim_model_hparams_nanogpt.json",
    H=500,
    outer_steps=50,
    batch_size=8,
    seq_len=512,
    pca_steps=PCA_STEPS,
)


# ---------------------------------------------------------------------------
# Data loading (inline copy from run_baseline.py — do not import from there)
# ---------------------------------------------------------------------------

def _load_shard_meta(data_path: str) -> dict:
    meta_path = Path(data_path) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    train_files = sorted(Path(data_path).glob("train_*.npy"))
    n_train = sum(f.stat().st_size // 2 for f in train_files)
    tokens_per_shard = Path(train_files[0]).stat().st_size // 2 if train_files else 100_000_000
    return {"n_train_tokens": n_train, "tokens_per_shard": tokens_per_shard}


def make_loaders(
    n_workers: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    data_path: str | None,
    device: str,
) -> list[DataLoader]:
    if data_path is not None:
        from tplr.data import ShadedDataset
        meta = _load_shard_meta(data_path)
        loaders = []
        for rank in range(n_workers):
            ds = ShadedDataset(
                shards_path=data_path,
                token_budget=meta["n_train_tokens"],
                sequence_length=seq_len,
                rank=rank,
                world_size=n_workers,
                device=torch.device(device),
                shard_token_size=meta["tokens_per_shard"],
                split="train",
            )
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0))
        return loaders

    # Fallback: synthetic random tokens for smoke tests
    from tplr.data import SyntheticDataset
    loaders = []
    for _ in range(n_workers):
        ds = SyntheticDataset(
            vocab_size=vocab_size,
            sequence_length=seq_len,
            num_samples=10_000,
        )
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
    return loaders


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_norms(all_metrics: dict[str, list[dict]], out_dir: Path) -> None:
    """Figure 1: 1×5 subplots of per-worker norm trajectories."""
    n_honest = N_WORKERS - BYZANTINE_F
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for ax, (agg_name, metrics) in zip(axes, all_metrics.items()):
        steps = [m["outer_step"] for m in metrics]
        for i in range(n_honest):
            norms = [m["per_worker_norms_before"][i] for m in metrics]
            ax.plot(steps, norms, color="steelblue", alpha=0.7, linewidth=1)
        for i in range(n_honest, N_WORKERS):
            norms = [m["per_worker_norms_before"][i] for m in metrics]
            ax.plot(steps, norms, color="firebrick", linestyle="--", alpha=0.8, linewidth=1)
        ax.set_yscale("log")
        ax.set_title(agg_name, fontsize=11)
        ax.set_xlabel("outer step")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("pseudo-gradient L2 norm (log)")
    handles = [
        Line2D([0], [0], color="steelblue", label=f"honest workers (n={n_honest})"),
        Line2D([0], [0], color="firebrick", linestyle="--",
               label=f"Byzantine workers (f={BYZANTINE_F}, scale={ATTACK_SCALE})"),
    ]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle("Per-worker pseudo-gradient norms — MagnitudeAttack f=2 scale=10", y=1.02)
    fig.tight_layout()
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "plots" / "fig1_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cohesion_oracle(all_metrics: dict[str, list[dict]], out_dir: Path) -> None:
    """Figure 2: honest cohesion (top) + cosine-to-oracle (bottom)."""
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Use RFA for the top panel (representative); fall back to first available
    rep_name = "rfa" if "rfa" in all_metrics else next(iter(all_metrics))
    rep_metrics = all_metrics[rep_name]
    steps = [m["outer_step"] for m in rep_metrics]
    ax_top.plot(
        steps, [m["honest_cosine_sim"] for m in rep_metrics],
        color="steelblue", linewidth=2, label=f"honest worker cosine sim ({rep_name})"
    )
    ax_top.plot(
        steps, [m["byzantine_cosine_sim"] for m in rep_metrics],
        color="firebrick", linestyle="--", linewidth=2,
        label=f"Byzantine cosine to oracle ({rep_name})"
    )
    ax_top.set_ylabel("cosine similarity")
    ax_top.set_title(
        f"Cluster cohesion at H=500 — honest workers converge in direction\n"
        f"(pre-perturbation, {rep_name} run shown as representative)"
    )
    ax_top.legend(fontsize=9)
    ax_top.set_ylim(-0.1, 1.1)
    ax_top.grid(True, alpha=0.3)

    # Bottom panel: cosine-to-oracle per aggregator
    for agg_name, metrics in all_metrics.items():
        steps = [m["outer_step"] for m in metrics]
        ax_bot.plot(
            steps, [m["cosine_to_oracle"] for m in metrics],
            label=agg_name, color=AGGREGATOR_COLORS[agg_name], linewidth=2
        )
    ax_bot.set_xlabel("outer step")
    ax_bot.set_ylabel("cosine similarity to oracle honest mean")
    ax_bot.set_title(
        "Aggregator recovery of the honest signal\n"
        "(cosine between aggregate and oracle = mean of honest pseudo-grads)"
    )
    ax_bot.legend(fontsize=9)
    ax_bot.set_ylim(-0.1, 1.1)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "fig2_cohesion_oracle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_snapshots(
    all_metrics: dict[str, list[dict]],
    out_dir: Path,
    steps: list[int] | None = None,
) -> None:
    """Figure 3: 5×2 PCA scatter (one row per aggregator, columns = step 1 and step 25)."""
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    if steps is None:
        steps = [1, 25]
    agg_names = list(all_metrics.keys())
    n_honest = N_WORKERS - BYZANTINE_F

    fig, axes = plt.subplots(len(agg_names), len(steps), figsize=(5 * len(steps), 4 * len(agg_names)))
    if len(agg_names) == 1:
        axes = [axes]
    if len(steps) == 1:
        axes = [[row] for row in axes]

    for row, agg_name in enumerate(agg_names):
        for col, step in enumerate(steps):
            ax = axes[row][col]
            pca_path = out_dir / f"pca_{agg_name}_step{step}.json"
            if not pca_path.exists():
                ax.text(0.5, 0.5, f"step {step}\nnot found",
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(f"{agg_name} — step {step}")
                continue

            with open(pca_path) as f:
                data = json.load(f)

            worker_proj = data["worker_projections"]
            oracle_proj = data["oracle_projection"]
            agg_proj = data["aggregated_projection"]

            # Honest workers: blue circles
            for i in range(n_honest):
                ax.scatter(worker_proj[i][0], worker_proj[i][1],
                           color="steelblue", marker="o", s=80, zorder=3,
                           label="honest" if i == 0 else None)
            # Byzantine workers: red crosses
            for i in range(n_honest, N_WORKERS):
                ax.scatter(worker_proj[i][0], worker_proj[i][1],
                           color="firebrick", marker="x", s=100, linewidths=2, zorder=3,
                           label="Byzantine" if i == n_honest else None)
            # Oracle: green star
            ax.scatter(oracle_proj[0], oracle_proj[1],
                       color="seagreen", marker="*", s=250, zorder=4, label="oracle")
            # Aggregated: coloured diamond
            ax.scatter(agg_proj[0], agg_proj[1],
                       color=AGGREGATOR_COLORS[agg_name], marker="D", s=120, zorder=4,
                       label="aggregated")

            ax.set_title(f"{agg_name} — step {step}", fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if col == 0 and row == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "2D PCA of pseudo-gradients — MagnitudeAttack f=2 scale=10\n"
        "Blue=honest, Red=Byzantine, Green★=oracle, Diamond=aggregated",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "fig3_pca_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(out_dir: Path) -> None:
    """Load all metrics files and produce all three figures."""
    all_metrics: dict[str, list[dict]] = {}
    for agg_name in AGGREGATORS:
        path = out_dir / f"metrics_{agg_name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            all_metrics[agg_name] = json.load(f)

    print("Plotting fig1_norms.png …")
    plot_norms(all_metrics, out_dir)

    print("Plotting fig2_cohesion_oracle.png …")
    plot_cohesion_oracle(all_metrics, out_dir)

    print("Plotting fig3_pca_snapshots.png …")
    plot_pca_snapshots(all_metrics, out_dir)

    print(f"Figures saved to {out_dir}/plots/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pseudo-gradient analysis driver (W2)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: tiny model, H=2, 3 outer steps, synthetic data")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--data-path", default=None,
                   help="Path to pre-tokenized .npy shards (omit for synthetic fallback)")
    p.add_argument("--offload", action="store_true",
                   help="Page workers on/off device one at a time to reduce peak VRAM")
    p.add_argument("--out-dir", default="experiments/results/analysis/pseudograd_magnitude_f2_s10",
                   help="Output directory for metrics + figures")
    p.add_argument("--only", nargs="*", choices=list(AGGREGATORS.keys()),
                   help="Run only these aggregators (default: all 5)")
    return p.parse_args()


def run(args):
    cfg = SMOKE_CFG if args.smoke else FULL_CFG
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_names = args.only or list(AGGREGATORS.keys())
    pca_steps = cfg["pca_steps"]
    perturbation = MagnitudeAttack(n_workers=N_WORKERS, f=BYZANTINE_F, scale=ATTACK_SCALE)

    print(f"\n{'='*60}")
    print(f"  Pseudo-gradient analysis — {'SMOKE' if args.smoke else 'FULL'}")
    print(f"  hparams   : {cfg['hparams']}")
    print(f"  H         : {cfg['H']}")
    print(f"  steps     : {cfg['outer_steps']}")
    print(f"  f         : {BYZANTINE_F}  scale={ATTACK_SCALE}")
    print(f"  device    : {args.device}")
    print(f"  pca_steps : {sorted(pca_steps)}")
    print(f"  out_dir   : {out_dir}")
    print(f"{'='*60}\n")

    # Build initial model once — all aggregators start from the same weights
    initial_model = build_model(cfg["hparams"], device=args.device)
    counts = param_count(initial_model)
    vocab_size = initial_model.config.vocab_size
    print(f"Model: {counts['total'] / 1e6:.1f}M params\n")

    # Build dataloaders once — workers reuse them across aggregator runs
    loaders = make_loaders(
        n_workers=N_WORKERS,
        vocab_size=vocab_size,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        data_path=args.data_path,
        device=args.device,
    )

    for agg_name in agg_names:
        print(f"\n--- Aggregator: {agg_name} ---")

        # Fresh model copy so each aggregator starts from identical weights
        model_copy = copy.deepcopy(initial_model)
        sim_config = SimConfig(
            H=cfg["H"],
            device=args.device,
            offload_between_steps=args.offload,
            verbose=False,
        )
        workers = [
            Worker(rank=i, model=model_copy, dataloader=loaders[i], config=sim_config)
            for i in range(N_WORKERS)
        ]

        sim = InstrumentedSimulation(
            workers=workers,
            aggregator=AGGREGATORS[agg_name](),
            perturbation=perturbation,
            config=sim_config,
            out_dir=out_dir,
            byzantine_f=BYZANTINE_F,
            pca_steps=pca_steps,
        )

        for _step in tqdm(range(cfg["outer_steps"]), desc=f"  {agg_name}"):
            metrics = sim.run_outer_step()
            tqdm.write(
                f"  step {metrics['outer_step']:3d} | "
                f"loss {metrics['loss/mean']:.4f} | "
                f"cosine_to_oracle {metrics['cosine_to_oracle']:.4f} | "
                f"honest_sim {metrics['honest_cosine_sim']:.4f}"
            )

        sim.write_metrics(agg_name)
        print(f"  Written: {out_dir}/metrics_{agg_name}.json")

    print("\nAll aggregators done. Generating plots …")
    plot_all(out_dir)
    print("Done.")


if __name__ == "__main__":
    run(parse_args())
