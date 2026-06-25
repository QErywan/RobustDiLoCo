"""
Experiment sweep driver — Tier-1 comparative grid and Tier-2 validation subset.

Defines the full experiment cell list for each tier, then runs them sequentially
by calling run_experiment.py for each cell.  Cells whose results JSON already
exists are skipped (cell-level resume), so the sweep is safe to interrupt and
restart at any time.

Tier-1 (comparative grid, cheap)
---------------------------------
Model:        30M GPT2 (hparams/sim/sim_model_hparams_gpt2_small.json)
seq_len:      512
outer_steps:  50
Seeds:        1 primary seed across all cells; +3 seeds on decisive cells
                (decisive = run after Tier-1 analysis)
Purpose:      Establish aggregator ranking across all perturbation × f conditions.
              Absolute perplexity NOT reported as headline — only rankings.

Tier-2 (validation, expensive)
--------------------------------
Model:        ~134M (hparams/sim/sim_model_hparams_full.json)
seq_len:      1024
outer_steps:  150
Seeds:        3 seeds per cell
Purpose:      Validate Tier-1 ranking at the committed report scale; produce
              headline perplexity + clean downstream evaluation metrics.
              Subset chosen after Tier-1 analysis — decisive cells only.

Usage
-----
# Dry-run: print cells without running them
python experiments/sweep.py --tier 1 --dry-run

# Run Tier-1 on the 4080
python experiments/sweep.py --tier 1 --device cuda --dataset c4

# Run Tier-2 subset (define after Tier-1 analysis)
python experiments/sweep.py --tier 2 --device cuda --dataset c4

# Resume a partially-completed sweep (cells already done are skipped)
python experiments/sweep.py --tier 1 --device cuda --resume

# Run a single aggregator only (useful for debugging one line of the grid)
python experiments/sweep.py --tier 1 --only-aggregator rfa --device cuda
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Cell definition
# ---------------------------------------------------------------------------

class Cell(NamedTuple):
    """One row of the experiment grid."""
    aggregator:   str
    perturbation: str
    byzantine_f:  int
    severity:     float
    seed:         int
    tier:         int

    @property
    def cell_id(self) -> str:
        return (
            f"{self.aggregator}"
            f"_p{self.perturbation}"
            f"_f{self.byzantine_f}"
            f"_s{self.severity}"
            f"_seed{self.seed}"
        )

    @property
    def result_path(self) -> Path:
        return Path(f"experiments/results/tier{self.tier}/{self.cell_id}.json")


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

AGGREGATORS = ["mean", "trimmed", "median", "rfa", "krum"]
PRIMARY_SEED = 42

# Perturbation severity levels per type (thesis-committed values)
DROPOUT_F_VALUES  = [1, 2, 4]           # f ∈ {1, 2, 4}
GAUSSIAN_SIGMA    = [0.1, 0.5, 1.0]     # sigma_scale
GAUSSIAN_F        = 2                    # fix f=2 for sigma sweep; sweep f separately
MAGNITUDE_SCALES  = [10.0, 100.0, 1000.0]
MAGNITUDE_F       = [1, 2, 4]
HETERO_ALPHAS     = [0.1, 0.5, 1.0]     # Dirichlet α (lower = more heterogeneous)


def _tier1_cells(seed: int = PRIMARY_SEED) -> list[Cell]:
    """~105 cells for the full Tier-1 comparative grid."""
    cells = []

    # 1. Clean baseline — no perturbation, no Byzantine workers
    for agg in AGGREGATORS:
        cells.append(Cell(agg, "none", 0, 0.0, seed, 1))

    # 2. Worker dropout — f ∈ {1, 2, 4}
    for agg in AGGREGATORS:
        for f in DROPOUT_F_VALUES:
            cells.append(Cell(agg, "dropout", f, 0.0, seed, 1))

    # 3. Gaussian noise — (a) vary sigma at fixed f=2; (b) vary f at fixed sigma=0.5
    for agg in AGGREGATORS:
        # (a) sigma sweep at f=2
        for sigma in GAUSSIAN_SIGMA:
            cells.append(Cell(agg, "gaussian", GAUSSIAN_F, sigma, seed, 1))
        # (b) f sweep at sigma=0.5 (avoid duplicating the f=2, sigma=0.5 cell)
        for f in DROPOUT_F_VALUES:
            if f != GAUSSIAN_F:  # f=2 already covered above
                cells.append(Cell(agg, "gaussian", f, 0.5, seed, 1))

    # 4. Magnitude attack — scale ∈ {10, 100, 1000} × f ∈ {1, 2, 4}
    for agg in AGGREGATORS:
        for scale in MAGNITUDE_SCALES:
            for f in MAGNITUDE_F:
                cells.append(Cell(agg, "magnitude", f, scale, seed, 1))

    # 5. Heterogeneous data — Dirichlet α ∈ {0.1, 0.5, 1.0}, no Byzantine workers
    #    NOTE: hetero is a data-level perturbation implemented via the dataset loader,
    #    not via the perturbations.py hook.  The runner handles this via --perturbation
    #    hetero (TODO: implement in run_experiment.py once data.py has HeterogeneousData).
    #    Cells are listed here for completeness; they will be skipped in --dry-run
    #    and flagged as pending until the loader is ready.
    for agg in AGGREGATORS:
        for alpha in HETERO_ALPHAS:
            cells.append(Cell(agg, "hetero", 0, alpha, seed, 1))

    return cells


def _tier2_cells() -> list[Cell]:
    """
    Tier-2 validation cells — chosen after Tier-1 analysis.

    Populate this list after reviewing Tier-1 results and identifying:
      - Which aggregators are clearly superior/inferior
      - Which perturbation × f conditions are the most decisive

    Placeholder: currently includes the five clean-baseline cells × 3 seeds
    plus one representative perturbed cell × 3 seeds per perturbation type.
    Update after Tier-1 analysis.
    """
    cells = []
    seeds = [42, 1, 2]   # 3 seeds for effect-size error bars

    # Clean baseline — all 5 aggregators × 3 seeds
    for agg in AGGREGATORS:
        for seed in seeds:
            cells.append(Cell(agg, "none", 0, 0.0, seed, 2))

    # Representative perturbed cells — update after Tier-1 analysis
    # Magnitude attack f=2 scale=100 (mid-range, expected decisive)
    for agg in AGGREGATORS:
        for seed in seeds:
            cells.append(Cell(agg, "magnitude", 2, 100.0, seed, 2))

    # Gaussian noise f=2 sigma=0.5 (mid-range)
    for agg in AGGREGATORS:
        for seed in seeds:
            cells.append(Cell(agg, "gaussian", 2, 0.5, seed, 2))

    # Worker dropout f=2
    for agg in AGGREGATORS:
        for seed in seeds:
            cells.append(Cell(agg, "dropout", 2, 0.0, seed, 2))

    return cells


# ---------------------------------------------------------------------------
# Tier-specific hparams
# ---------------------------------------------------------------------------

TIER_HPARAMS = {
    1: "hparams/sim/sim_model_hparams_gpt2_small.json",
    2: "hparams/sim/sim_model_hparams_full.json",
}

TIER_OUTER_STEPS = {
    1: 50,
    2: 150,
}

TIER_BATCH_SIZE = {
    1: 4,    # 30M model, seq 512, A100/4080
    2: 8,    # 134M model, seq 1024, A100 only — reduce if OOM
}

TIER_SEQ_LEN = {
    1: 512,
    2: 1024,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def build_command(cell: Cell, args: argparse.Namespace) -> list[str]:
    """Assemble the subprocess command to run one cell via run_experiment.py."""
    hparams     = args.hparams     or TIER_HPARAMS[cell.tier]
    outer_steps = args.outer_steps or TIER_OUTER_STEPS[cell.tier]
    batch_size  = args.batch_size  or TIER_BATCH_SIZE[cell.tier]

    cmd = [
        sys.executable, "-m", "experiments.run_experiment",
        "--hparams",       hparams,
        "--outer-steps",   str(outer_steps),
        "--batch-size",    str(batch_size),
        "--aggregator",    cell.aggregator,
        "--perturbation",  cell.perturbation,
        "--byzantine-f",   str(cell.byzantine_f),
        "--severity",      str(cell.severity),
        "--seed",          str(cell.seed),
        "--device",        args.device,
        "--dataset",       args.dataset,
        "--out",           str(cell.result_path),
        "--save-every",    str(args.save_every),
    ]
    if args.offload:
        cmd.append("--offload")
    if args.verbose:
        cmd.append("--verbose")
    if args.resume:
        cmd.append("--resume")

    return cmd


def run_sweep(args: argparse.Namespace) -> None:
    cells = _tier1_cells() if args.tier == 1 else _tier2_cells()

    # Filter by aggregator if requested
    if args.only_aggregator:
        cells = [c for c in cells if c.aggregator == args.only_aggregator]

    # Filter out hetero cells if the loader is not yet ready (pending TODO)
    hetero_pending = [c for c in cells if c.perturbation == "hetero"]
    cells = [c for c in cells if c.perturbation != "hetero"]
    if hetero_pending:
        print(f"[sweep] NOTE: {len(hetero_pending)} hetero cells skipped — "
              "HeterogeneousData loader not yet wired into run_experiment.py.")

    n_total = len(cells)
    n_skip = sum(1 for c in cells if c.result_path.exists())
    print(f"\n{'='*60}")
    print(f"  Sweep Tier-{args.tier}  |  {n_total} cells  ({n_skip} already done)")
    print(f"  device: {args.device}  dataset: {args.dataset}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN — cells that would run:\n")
        for i, cell in enumerate(cells):
            status = "SKIP" if cell.result_path.exists() else "RUN "
            print(f"  [{status}] {i+1:>3}/{n_total}  {cell.cell_id}")
        print(f"\n{n_total - n_skip} cells to run, {n_skip} to skip.")
        return

    done = 0
    failed = []
    for i, cell in enumerate(cells, 1):
        if cell.result_path.exists():
            print(f"[{i:>3}/{n_total}] SKIP (exists): {cell.cell_id}")
            done += 1
            continue

        cell.result_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_command(cell, args)
        print(f"\n[{i:>3}/{n_total}] RUN: {cell.cell_id}")
        print(f"  cmd: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ERROR: cell {cell.cell_id} exited with code {result.returncode}")
            failed.append(cell.cell_id)
        else:
            done += 1

    print(f"\n{'='*60}")
    print(f"  Sweep complete: {done}/{n_total} succeeded, {len(failed)} failed.")
    if failed:
        print("  Failed cells:")
        for cid in failed:
            print(f"    {cid}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Tier-1 or Tier-2 experiment sweep."
    )
    p.add_argument("--tier", type=int, choices=[1, 2], default=1,
                   help="Which experiment tier to run (1=comparative grid, 2=validation)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cell list without running anything")
    p.add_argument("--only-aggregator", type=str, default=None,
                   choices=["mean", "trimmed", "median", "rfa", "krum"],
                   help="Run only cells for this aggregator (debugging / partial sweep)")

    # Overrides (defaults from TIER_* dicts above)
    p.add_argument("--hparams",       type=str, default=None,
                   help="Override hparams file for all cells")
    p.add_argument("--outer-steps",   type=int, default=None,
                   help="Override outer step count for all cells")
    p.add_argument("--batch-size",    type=int, default=None,
                   help="Override batch size for all cells")

    # Hardware + data
    p.add_argument("--device",   type=str,  default="cuda")
    p.add_argument("--dataset",  type=str,  default="c4",
                   choices=["synthetic", "c4", "fineweb"])
    p.add_argument("--offload",  action="store_true")
    p.add_argument("--verbose",  action="store_true")

    # Resume / checkpointing
    p.add_argument("--resume",     action="store_true",
                   help="Pass --resume to each cell runner (resume from checkpoint)")
    p.add_argument("--save-every", type=int, default=10)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
