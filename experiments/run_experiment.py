"""
General DiLoCo experiment runner.

Runs one cell of the thesis experiment grid: one aggregator × one perturbation
× one Byzantine fraction f × one seed.  Injects aggregator + perturbation at
the Simulation hook point (after compute_pseudo_grad, before aggregate).

Usage examples
--------------
# Smoke test — tiny model, CPU, sanity check
python experiments/run_experiment.py --smoke \\
    --aggregator rfa --perturbation magnitude --byzantine-f 2 --severity 100

# Tier-1 grid cell (30M, seq 512, 50 steps)
python experiments/run_experiment.py \\
    --hparams hparams/sim/sim_model_hparams_gpt2_small.json \\
    --aggregator trimmed --perturbation gaussian --byzantine-f 2 --severity 0.5 \\
    --outer-steps 50 --seed 42 --device cuda

# Tier-2 validation cell (134M, seq 1024, 150 steps, 3 seeds)
python experiments/run_experiment.py \\
    --hparams hparams/sim/sim_model_hparams_full.json \\
    --aggregator rfa --perturbation none --byzantine-f 0 \\
    --outer-steps 150 --seed 1 --device cuda

Output
------
A JSON file at --out (default: experiments/results/<cell_id>.json) with schema:
    {
      "config":  { all hyperparameters + aggregator/perturbation/f/seed },
      "history": [ { outer_step, loss/mean, loss/min, loss/max,
                     pseudo_grad_norm/mean, pseudo_grad_norm/max,
                     aggregated_grad_norm, worker_losses }, ... ]
    }

This schema is identical to run_baseline.py so plot_baseline.py works unchanged.
Checkpoint files are saved every --save-every steps to <out>.ckpt.pt so the run
can be resumed after a crash or timeout.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from simulation.model import build_model, param_count
from simulation.workers import SimConfig, Worker, Simulation
from simulation.evaluate import make_eval_loader, eval_perplexity, run_downstream_eval
from simulation.aggregators import (
    MeanAggregator,
    TrimmedMeanAggregator,
    CoordMedianAggregator,
    GeometricMedianAggregator,
    MultiKrumAggregator,
)
from simulation.perturbations import (
    NoPerturbation,
    WorkerDropout,
    GaussianNoise,
    MagnitudeAttack,
)

# Re-use data-loading helpers from the baseline runner
from experiments.run_baseline import make_worker_loaders, HF_DATASET_CONFIG  # noqa: F401


# ---------------------------------------------------------------------------
# Config presets (identical to run_baseline.py so smoke tests match)
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


# ---------------------------------------------------------------------------
# Aggregator + perturbation factories
# ---------------------------------------------------------------------------

AGGREGATOR_CHOICES = ["mean", "trimmed", "median", "rfa", "krum"]
PERTURBATION_CHOICES = ["none", "dropout", "gaussian", "magnitude"]


def build_aggregator(name: str, f: int, n_workers: int):
    """
    Map aggregator name to an Aggregator object.

    Args:
        name:      one of AGGREGATOR_CHOICES
        f:         Byzantine worker count (used by Trimmed Mean and Krum)
        n_workers: total workers (used to validate f)
    """
    if name == "mean":
        return MeanAggregator()
    elif name == "trimmed":
        return TrimmedMeanAggregator(f=f, n_workers=n_workers)
    elif name == "median":
        return CoordMedianAggregator()
    elif name == "rfa":
        return GeometricMedianAggregator()
    elif name == "krum":
        return MultiKrumAggregator(f=f, n_workers=n_workers)
    else:
        raise ValueError(f"Unknown aggregator: {name!r}. Choose from {AGGREGATOR_CHOICES}")


def build_perturbation(name: str, f: int, severity: float, n_workers: int):
    """
    Map perturbation name to a Perturbation object.

    Args:
        name:      one of PERTURBATION_CHOICES
        f:         Byzantine worker count (fraction of n_workers that are corrupted)
        severity:  float whose meaning depends on perturbation type:
                     gaussian  → sigma_scale (e.g. 0.1, 0.5, 1.0)
                     magnitude → scale multiplier (e.g. 10, 100, 1000)
                     dropout   → unused (severity ignored)
                     none      → unused
        n_workers: total workers
    """
    if name == "none":
        return NoPerturbation()
    elif name == "dropout":
        return WorkerDropout(n_workers=n_workers, f=f)
    elif name == "gaussian":
        return GaussianNoise(n_workers=n_workers, f=f, sigma_scale=severity)
    elif name == "magnitude":
        return MagnitudeAttack(n_workers=n_workers, f=f, scale=severity)
    else:
        raise ValueError(f"Unknown perturbation: {name!r}. Choose from {PERTURBATION_CHOICES}")


def cell_id(args) -> str:
    """Stable string ID for this experiment cell — used in filenames."""
    return (
        f"{args.aggregator}"
        f"_p{args.perturbation}"
        f"_f{args.byzantine_f}"
        f"_s{args.severity}"
        f"_seed{args.seed}"
    )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA) so that worker construction
    (deepcopy + random init) and dataloader iteration are deterministic.

    MUST be called before build_model() and make_worker_loaders() so that
    within a cell the *only* difference between aggregators is the aggregator
    itself.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, outer_step: int, workers, sim, history: list) -> None:
    """Save full simulation state so a crashed run can resume from here."""
    state = {
        "outer_step": outer_step,
        "history": history,
        "worker_model_states": [w.model.state_dict() for w in workers],
        "worker_inner_opt_states": [w.inner_optimizer.state_dict() for w in workers],
        "worker_outer_opt_states": [w.outer_optimizer.state_dict() for w in workers],
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }
    torch.save(state, path)


def load_checkpoint(path: Path, workers, sim) -> tuple[int, list]:
    """
    Restore simulation state from a checkpoint.

    Returns (outer_step_resumed_from, history_so_far).
    """
    state = torch.load(path, weights_only=False)
    for w, ms, ios, oos in zip(
        workers,
        state["worker_model_states"],
        state["worker_inner_opt_states"],
        state["worker_outer_opt_states"],
    ):
        w.model.load_state_dict(ms)
        w.inner_optimizer.load_state_dict(ios)
        w.outer_optimizer.load_state_dict(oos)
    torch.set_rng_state(state["rng_torch"])
    np.random.set_state(state["rng_numpy"])
    sim.outer_step_count = state["outer_step"]
    return state["outer_step"], state["history"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run one DiLoCo experiment cell (aggregator × perturbation × f × seed)."
    )
    # Preset
    p.add_argument("--smoke", action="store_true", help="Fast smoke test (tiny model, few steps)")

    # Model / training
    p.add_argument("--hparams",      type=str, default=None)
    p.add_argument("--outer-steps",  type=int, default=50,
                   help="Total outer steps (Tier-1 default: 50)")
    p.add_argument("--H",            type=int, default=None,
                   help="Inner steps per outer step (default: 500)")
    p.add_argument("--n-workers",    type=int, default=None)
    p.add_argument("--batch-size",   type=int, default=None)

    # Data
    p.add_argument("--dataset",    type=str, default="c4",
                   choices=["synthetic", "c4", "fineweb"])
    p.add_argument("--data-path",  type=str, default=None,
                   help="Path to pre-tokenized .npy shards (synthetic mode only)")

    # Aggregator + perturbation (the thesis variables)
    p.add_argument("--aggregator",    type=str, default="mean",
                   choices=AGGREGATOR_CHOICES,
                   help="Aggregation rule to use")
    p.add_argument("--perturbation",  type=str, default="none",
                   choices=PERTURBATION_CHOICES,
                   help="Perturbation type to apply before aggregation")
    p.add_argument("--byzantine-f",   type=int, default=0,
                   help="Number of Byzantine workers (thesis: 0, 1, 2, 4)")
    p.add_argument("--severity",      type=float, default=0.0,
                   help=(
                       "Perturbation severity: sigma_scale for gaussian "
                       "(thesis: 0.1, 0.5, 1.0), scale for magnitude "
                       "(thesis: 10, 100, 1000), ignored for dropout/none"
                   ))

    # Reproducibility
    p.add_argument("--seed", type=int, default=42,
                   help="Global RNG seed — must be identical across aggregators "
                        "within a cell for fair comparison")

    # Hardware
    p.add_argument("--device",   type=str, default="cpu")
    p.add_argument("--offload",  action="store_true",
                   help="Page workers on/off device one at a time (CUDA only)")

    # Output
    p.add_argument("--out",       type=str, default=None,
                   help="Output JSON path (default: experiments/results/<cell_id>.json)")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save checkpoint every N outer steps (0 to disable)")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from checkpoint if one exists for this cell")
    p.add_argument("--verbose",   action="store_true",
                   help="Print per-worker progress during each outer step")
    p.add_argument("--eval-batches", type=int, default=50,
                   help="Held-out eval batches at the end of training (0 to skip)")
    p.add_argument("--downstream-eval", action="store_true",
                   help="Run HellaSwag/ARC-Easy/PIQA after training "
                        "(clean configs only; requires lm-evaluation-harness)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # Apply preset overrides
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

    # Resolve output path
    cid = cell_id(args)
    out_path = Path(args.out) if args.out else Path(f"experiments/results/{cid}.json")
    ckpt_path = out_path.with_suffix(".ckpt.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  DiLoCo Experiment — {'SMOKE' if args.smoke else 'FULL'}")
    print(f"  cell       : {cid}")
    print(f"  aggregator : {args.aggregator}")
    print(f"  perturbation: {args.perturbation}  f={args.byzantine_f}  severity={args.severity}")
    print(f"  seed       : {args.seed}")
    print(f"  hparams    : {cfg['hparams']}")
    print(f"  workers    : {cfg['n_workers']}  H={cfg['H']}  steps={cfg['outer_steps']}")
    print(f"  device     : {args.device}  offload={args.offload}")
    print(f"  out        : {out_path}")
    print(f"{'='*64}\n")

    # ------------------------------------------------------------------
    # Seed BEFORE model construction and dataloader creation so that worker
    # deepcopies and data ordering are identical across aggregators for the
    # same seed.  This is the hard comparability guarantee.
    # ------------------------------------------------------------------
    seed_everything(args.seed)

    # Model
    model = build_model(cfg["hparams"], device=args.device)
    counts = param_count(model)
    vocab_size = model.config.vocab_size
    print(f"Model: {counts['total'] / 1e6:.1f}M params  vocab={vocab_size}")

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

    # Aggregator + perturbation — never hardcoded, always from args
    aggregator = build_aggregator(
        args.aggregator, f=args.byzantine_f, n_workers=cfg["n_workers"]
    )
    perturbation = build_perturbation(
        args.perturbation,
        f=args.byzantine_f,
        severity=args.severity,
        n_workers=cfg["n_workers"],
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
        aggregator=aggregator,
        perturbation=perturbation,
        config=sim_config,
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint if requested and one exists
    # ------------------------------------------------------------------
    history = []
    start_step = 0
    if args.resume and ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        start_step, history = load_checkpoint(ckpt_path, workers, sim)
        print(f"  Resumed at outer step {start_step}/{cfg['outer_steps']}")

    # ------------------------------------------------------------------
    # Serialisable config for the results JSON
    # ------------------------------------------------------------------
    result_config = {
        "hparams":      cfg["hparams"],
        "n_workers":    cfg["n_workers"],
        "H":            cfg["H"],
        "batch_size":   cfg["batch_size"],
        "seq_len":      cfg["seq_len"],
        "outer_steps":  cfg["outer_steps"],
        "aggregator":   args.aggregator,
        "perturbation": args.perturbation,
        "byzantine_f":  args.byzantine_f,
        "severity":     args.severity,
        "seed":         args.seed,
        "dataset":      args.dataset,
        "device":       args.device,
    }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    remaining = cfg["outer_steps"] - start_step
    print(f"{'Step':>6}  {'Loss':>8}  {'PGNorm':>9}  {'Time':>7}")
    print("-" * 42)

    for _i in tqdm(range(remaining), desc="Outer steps", initial=start_step,
                   total=cfg["outer_steps"]):
        t0 = time.time()
        metrics = sim.run_outer_step()
        elapsed = time.time() - t0

        history.append(metrics)
        print(
            f"{metrics['outer_step']:>6}  "
            f"{metrics['loss/mean']:>8.4f}  "
            f"{metrics['pseudo_grad_norm/mean']:>9.3f}  "
            f"{elapsed:>6.1f}s"
        )

        current_step = metrics["outer_step"]

        # Periodic JSON flush — results file is always up-to-date
        with open(out_path, "w") as fh:
            json.dump({"config": result_config, "history": history}, fh, indent=2)

        # Checkpoint
        if args.save_every > 0 and (current_step % args.save_every == 0):
            save_checkpoint(ckpt_path, current_step, workers, sim, history)
            print(f"  [ckpt saved at step {current_step}]")

    # ------------------------------------------------------------------
    # Held-out evaluation
    # ------------------------------------------------------------------
    eval_metrics = {}

    if args.eval_batches > 0:
        print("\nRunning held-out evaluation...")
        eval_loader = make_eval_loader(
            dataset=args.dataset,
            seq_len=cfg["seq_len"],
            batch_size=cfg["batch_size"],
            n_batches=args.eval_batches,
            vocab_size=vocab_size,
        )
        eval_metrics = eval_perplexity(
            sim.global_model,
            eval_loader,
            device=args.device,
            n_batches=args.eval_batches,
        )
        print(
            f"  eval_loss  : {eval_metrics['eval_loss']:.4f}\n"
            f"  perplexity : {eval_metrics['perplexity']:.2f}\n"
            f"  n_tokens   : {eval_metrics['n_tokens']}"
        )

    # Downstream benchmarks — clean configs only
    downstream_metrics = {}
    if args.downstream_eval:
        if args.perturbation != "none" or args.byzantine_f != 0:
            print(
                "\n[WARNING] --downstream-eval requested on a non-clean config "
                f"(perturbation={args.perturbation}, f={args.byzantine_f}). "
                "Downstream eval is only meaningful for clean baselines. Skipping."
            )
        else:
            print("\nRunning downstream evaluation (HellaSwag / ARC-Easy / PIQA)...")
            try:
                downstream_metrics = run_downstream_eval(
                    sim.global_model, device=args.device
                )
                print("  " + "  ".join(f"{k}: {v:.3f}" for k, v in downstream_metrics.items()))
            except ImportError as e:
                print(f"  [SKIP] {e}")

    # Final flush — include eval metrics in the JSON
    with open(out_path, "w") as fh:
        json.dump(
            {
                "config":   result_config,
                "history":  history,
                "eval":     eval_metrics,
                "downstream": downstream_metrics,
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")

    return history


if __name__ == "__main__":
    args = parse_args()
    run(args)
