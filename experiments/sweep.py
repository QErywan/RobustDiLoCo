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
Model:        124M GPT-2 (hparams/sim/sim_model_hparams_nanogpt.json, vocab 50257)
seq_len:      1024
outer_steps:  150 (default; use --outer-steps 50 for the minimal-budget run)
Seeds:        3 on the headline cell, 1 on supporting decisive cells
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
import datetime
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
    dropout_mode: str = "zero"   # dropout only: "zero" or "exclude"

    @property
    def cell_id(self) -> str:
        # Only exclude-mode dropout is tagged, so zero-mode filenames are unchanged
        # and zero vs exclude never collide under --resume.
        mode_tag = "_excl" if (self.perturbation == "dropout"
                               and self.dropout_mode == "exclude") else ""
        return (
            f"{self.aggregator}"
            f"_p{self.perturbation}"
            f"_f{self.byzantine_f}"
            f"{mode_tag}"
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
    """~90 active cells for the Tier-1 comparative grid.

    Cell ORDER is thesis-priority:
        1. Clean baseline       — primary sanity check and clean reference loss
        2. Magnitude attack     — the thesis's central Byzantine result (expected
                                  to show the largest aggregator gap); runs first
                                  so the headline result lands in hours not days
        3. Gaussian noise       — secondary Byzantine result
        4. Worker dropout       — expected smaller gap (H=500 smooths dropouts)
        5. Heterogeneous data   — data-level perturbation; skipped until loader
                                  is wired in (see sweep.py run_sweep)

    Cell-level resume (skipping existing JSONs) makes the ordering safe to change
    between runs — already-completed cells are just skipped.
    """
    cells = []

    # 1. Clean baseline — no perturbation, no Byzantine workers
    #    ALL 5 aggregators × seed; these must overlap (sanity check).
    for agg in AGGREGATORS:
        cells.append(Cell(agg, "none", 0, 0.0, seed, 1))

    # 2. Magnitude attack — the decisive Byzantine condition.
    #    scale ∈ {10, 100, 1000} × f ∈ {1, 2, 4}: 3 × 3 × 5 = 45 cells.
    #    Run mid-range f=2 first so the key thesis plot appears early.
    for agg in AGGREGATORS:
        for scale in MAGNITUDE_SCALES:
            cells.append(Cell(agg, "magnitude", 2, scale, seed, 1))   # f=2 first
    for agg in AGGREGATORS:
        for scale in MAGNITUDE_SCALES:
            for f in [1, 4]:   # f=1 and f=4 after f=2 is complete
                cells.append(Cell(agg, "magnitude", f, scale, seed, 1))

    # 3. Gaussian noise — (a) vary sigma at fixed f=2; (b) vary f at fixed sigma=0.5
    for agg in AGGREGATORS:
        # (a) sigma sweep at f=2
        for sigma in GAUSSIAN_SIGMA:
            cells.append(Cell(agg, "gaussian", GAUSSIAN_F, sigma, seed, 1))
        # (b) f sweep at sigma=0.5 (avoid duplicating f=2, sigma=0.5)
        for f in DROPOUT_F_VALUES:
            if f != GAUSSIAN_F:  # f=2 already covered above
                cells.append(Cell(agg, "gaussian", f, 0.5, seed, 1))

    # 4. Worker dropout — f ∈ {1, 2, 4}
    for agg in AGGREGATORS:
        for f in DROPOUT_F_VALUES:
            cells.append(Cell(agg, "dropout", f, 0.0, seed, 1))

    # 5. Heterogeneous data — Dirichlet α ∈ {0.1, 0.5, 1.0}, no Byzantine workers
    #    NOTE: hetero is a data-level perturbation implemented via the dataset loader,
    #    not via the perturbations.py hook.  The runner handles this via --perturbation
    #    hetero (TODO: implement in run_experiment.py once data.py has HeterogeneousData).
    #    Cells are listed here for completeness; they will be skipped in run_sweep
    #    and flagged as pending until the loader is ready.
    for agg in AGGREGATORS:
        for alpha in HETERO_ALPHAS:
            cells.append(Cell(agg, "hetero", 0, alpha, seed, 1))

    return cells


def _tier2_cells(seed: int = PRIMARY_SEED) -> list[Cell]:
    """
    Tier-2 validation cells — the DECISIVE conditions from the Tier-1 analysis.

    Tier-2 (134M) exists to show the Tier-1 ranking HOLDS AT SCALE, not to reproduce
    the whole grid. Cells cover the three scientific claims once each, so Tier-2
    validates the full perturbation-DEPENDENT ranking, not just the headline:

      - clean f=0           : sanity — all aggregators converge to ~same loss
      - magnitude f=2 s=10  : headline — mean diverges, robust hold (RFA best)
      - dropout  f=4        : the mirror — RFA is *worst* (perturbation-dependence)
      - gaussian f=2 s=0.5  : natural perturbation — mean fails without an adversary

    Single seed throughout (consistent with Tier-1, which is also single-seed) — the
    minimal-Tier-2 budget on one L4 does not stretch to seed error bars. Cells are
    ordered by PRIORITY: clean + magnitude first (the headline scale result, wanted for
    the report draft), then dropout + gaussian. If a run is cut short, the most
    important cells are already done. Krum at f=4 is auto-skipped by the 2f+2>=n guard.

    Compute: 19 cells after the guard. Pilot measured ~2200s/outer-step at 124M/batch-4
    on an L4, so at TIER_OUTER_STEPS[2]=25 each cell is ~15h (~$13 on-demand); the set
    is ~$250 / ~12 days serial on one GPU. Run with --batch-size 4 (batch 8 OOMs the
    24GB L4 on the seq-1024 attention tensor) and
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
    """
    cells = []
    SEED = seed   # single seed throughout (matches Tier-1); --seed overrides for a 2nd seed

    # 1. Clean baseline (sanity at 134M) — all 5 aggregators.
    for agg in AGGREGATORS:
        cells.append(Cell(agg, "none", 0, 0.0, SEED, 2))

    # 2. HEADLINE — magnitude f=2 s=10: mean diverges, robust hold (RFA best).
    for agg in AGGREGATORS:
        cells.append(Cell(agg, "magnitude", 2, 10.0, SEED, 2))

    # 3. Dropout f=4 — the mirror: RFA *worst* (krum auto-skipped, undefined at f=4).
    for agg in AGGREGATORS:
        cells.append(Cell(agg, "dropout", 4, 0.0, SEED, 2))

    # 4. Gaussian f=2 s=0.5 — natural perturbation: mean fails without an adversary.
    for agg in AGGREGATORS:
        cells.append(Cell(agg, "gaussian", 2, 0.5, SEED, 2))

    return cells


def _dropout_comparison_cells(seed: int = PRIMARY_SEED, tier: int = 1) -> list[Cell]:
    """
    The dropout zeroing-vs-exclusion comparison.

    At f=4 a dropped worker modelled as a zero vector sits at the origin; with four of
    eight inputs there, the geometric median returns the origin and RFA collapses. Under
    exclusion (the survivors only) it does not. Running both models settles whether the
    "RFA worst under dropout" result is real or an artefact of the zeroing choice.

    f=4, the four f-relevant aggregators, both models = 8 cells. Krum is skipped at f=4
    by the 2f+2>=n guard, and mean/median/RFA ignore f, so exclusion is well-defined for
    all four (trimmed caps its trim width to the survivor count).
    """
    cells = []
    for agg in ["mean", "trimmed", "median", "rfa"]:
        for mode in ("zero", "exclude"):
            cells.append(Cell(agg, "dropout", 4, 0.0, seed, tier, mode))
    return cells


# ---------------------------------------------------------------------------
# Tier-specific hparams
# ---------------------------------------------------------------------------

TIER_HPARAMS = {
    1: "hparams/sim/sim_model_hparams_gpt2_small.json",   # 30M GPT-2, vocab 50257
    # Tier-2 must be a GPT-2 (vocab 50257) to match the gpt2-tokenized C4 shards and
    # Tier-1; the old full.json is LLaMA (vocab 32000) and would device-assert on the
    # data. nanogpt = 124M GPT-2, the model the pseudo-grad analysis also used.
    2: "hparams/sim/sim_model_hparams_nanogpt.json",      # 124M GPT-2, vocab 50257
}

TIER_OUTER_STEPS = {
    1: 50,
    2: 25,   # minimal-Tier-2: 25 steps shows the ranking (Tier-1 diverged by ~5-30);
             # the full 150 is only for converged perplexity, out of the 1-GPU budget.
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
        "--dropout-mode",  cell.dropout_mode,
        "--severity",      str(cell.severity),
        "--seed",          str(cell.seed),
        "--device",        args.device,
        "--dataset",       args.dataset,
        "--out",           str(cell.result_path),
        "--save-every",    str(args.save_every),
    ]
    if args.data_path:
        cmd += ["--data-path", args.data_path]
    if args.offload:
        cmd.append("--offload")
    if args.verbose:
        cmd.append("--verbose")
    if args.resume:
        cmd.append("--resume")

    return cmd


def run_sweep(args: argparse.Namespace) -> None:
    if args.dropout_compare:
        seed = args.seed if args.seed is not None else PRIMARY_SEED
        cells = _dropout_comparison_cells(seed=seed, tier=args.tier)
    elif args.seed is not None:
        cells = _tier1_cells(args.seed) if args.tier == 1 else _tier2_cells(args.seed)
    else:
        cells = _tier1_cells() if args.tier == 1 else _tier2_cells()

    # Filter by aggregator if requested
    if args.only_aggregator:
        cells = [c for c in cells if c.aggregator == args.only_aggregator]

    # Multi-Krum is mathematically undefined when 2f+2 >= n (it needs 2f+2 < n to have
    # enough non-Byzantine neighbours to score). At n=8 that rules out f=4. These cells
    # can't be computed — attempting them only wastes compute (they OOM). Skip them here
    # so `--only-aggregator krum` never re-dispatches them. Documented as a limitation.
    n_workers_default = 8
    def _krum_undefined(c):
        return c.aggregator == "krum" and 2 * c.byzantine_f + 2 >= n_workers_default
    undefined_fs = sorted({c.byzantine_f for c in cells if _krum_undefined(c)})
    if undefined_fs:
        before = len(cells)
        cells = [c for c in cells if not _krum_undefined(c)]
        print(f"[sweep] NOTE: {before - len(cells)} krum cell(s) skipped — undefined at "
              f"f={undefined_fs} (2f+2 >= n={n_workers_default}).")

    # Hetero cells are now supported (data-level Dirichlet-alpha split; see
    # simulation/hetero_data.py + run_experiment.py). They require a cluster_ids.npy
    # cache from experiments/cluster_shards.py alongside the shards; without it the
    # runner falls back to random cluster ids (smoke only) and warns.
    n_hetero = sum(1 for c in cells if c.perturbation == "hetero")
    if n_hetero:
        print(f"[sweep] NOTE: {n_hetero} hetero cells included — ensure "
              "cluster_ids.npy exists next to the shards (run cluster_shards.py first).")

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

    # Persistent failures log — survives sweep restarts and terminal scroll.
    failures_log = Path(cells[0].result_path.parent) / "failures.log"

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
            # Append to persistent log so failures survive restarts.
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(failures_log, "a") as fh:
                fh.write(f"{ts}  exit={result.returncode}  {cell.cell_id}\n")
        else:
            done += 1

    print(f"\n{'='*60}")
    print(f"  Sweep complete: {done}/{n_total} succeeded, {len(failed)} failed.")
    if failed:
        print("  Failed cells:")
        for cid in failed:
            print(f"    {cid}")
        print(f"  Failures log: {failures_log}")
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
    p.add_argument("--seed", type=int, default=None,
                   help="Override the per-cell seed (default 42). Use a different value "
                        "(e.g. 43) to run a second seed on separate hardware without "
                        "colliding with the seed42 result files.")
    p.add_argument("--dropout-compare", action="store_true",
                   help="Run only the dropout zeroing-vs-exclusion comparison: f=4, the "
                        "four f-relevant aggregators, both models (8 cells). Overrides "
                        "the tier grid.")

    # Overrides (defaults from TIER_* dicts above)
    p.add_argument("--hparams",       type=str, default=None,
                   help="Override hparams file for all cells")
    p.add_argument("--outer-steps",   type=int, default=None,
                   help="Override outer step count for all cells")
    p.add_argument("--batch-size",    type=int, default=None,
                   help="Override batch size for all cells")

    # Hardware + data
    p.add_argument("--device",    type=str,  default="cuda")
    p.add_argument("--dataset",   type=str,  default="c4",
                   choices=["synthetic", "c4", "fineweb"])
    p.add_argument("--data-path", type=str,  default=None,
                   help="Path to pre-tokenized .npy shards written by pretokenize.py. "
                        "If set, ALL cells use the fast shard loader instead of streaming. "
                        "REQUIRED for the reported sweep (ensure pipeline consistency). "
                        "Example: /vol/bitbucket/qe25/data/c4_gpt2")
    p.add_argument("--offload",   action="store_true")
    p.add_argument("--verbose",   action="store_true")

    # Resume / checkpointing
    p.add_argument("--resume",     action="store_true",
                   help="Pass --resume to each cell runner (resume from checkpoint)")
    p.add_argument("--save-every", type=int, default=10)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
