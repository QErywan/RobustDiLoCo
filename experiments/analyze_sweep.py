"""
Multi-cell sweep analysis for DiLoCo Tier-1 and Tier-2 results.

Reads all *.json result files from a results directory and produces:
    - summary.csv          — one row per cell with final loss, perplexity, n_steps
    - effect_size table    — printed to stdout: final_loss[agg] - final_loss[mean]
                             per condition (negative = robust beats mean)
    - clean_baseline.png   — PRIMARY SANITY CHECK: all 5 aggregators on the clean
                             (none, f=0) condition should track each other closely
    - one PNG per decisive condition present in the data (magnitude, gaussian, dropout)
    - coverage report      — which conditions have all 5 aggregators (complete) vs
                             partial (important to flag at 32/90 cells)

Usage
-----
# Run on the partial Tier-1 set (Mac, after rsync from Imperial)
python experiments/analyze_sweep.py \\
    --results-dir experiments/results/tier1 \\
    --out-dir experiments/results/tier1/analysis

# Only produce the table (no plots)
python experiments/analyze_sweep.py \\
    --results-dir experiments/results/tier1 \\
    --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Aggregator display names and colours (thesis palette)
# ---------------------------------------------------------------------------

AGG_LABELS = {
    "mean":    "Mean (baseline)",
    "trimmed": "Trimmed Mean",
    "median":  "Coord Median",
    "rfa":     "RFA / Geo Median",
    "krum":    "Multi-Krum",
}

AGG_COLOURS = {
    "mean":    "#1f77b4",   # blue
    "trimmed": "#ff7f0e",   # orange
    "median":  "#2ca02c",   # green
    "rfa":     "#d62728",   # red
    "krum":    "#9467bd",   # purple
}

AGG_ORDER = ["mean", "trimmed", "median", "rfa", "krum"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cells(results_dir: str | Path) -> list[dict[str, Any]]:
    """
    Glob all *.json files in results_dir, parse each, and return a list of
    enriched dicts.  Corrupt / empty files are skipped with a warning.

    Each returned dict has:
        - all keys from the original JSON  (config, history, eval, downstream)
        - "file"       → Path to the JSON
        - "cell_id"    → stem of the filename
        - "n_steps"    → number of outer steps recorded in history
        - "final_loss" → loss/mean at the last history entry (None if empty)
        - "perplexity" → eval.perplexity if present, else None
    """
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        print(f"[analyze] No *.json files found in {results_dir}", file=sys.stderr)
        return []

    cells = []
    for p in paths:
        try:
            with open(p) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[analyze] WARNING: skipping {p.name} — {exc}", file=sys.stderr)
            continue

        history = data.get("history", [])
        eval_block = data.get("eval", {})
        config = data.get("config", {})

        # Tolerate files with no history entries (in-progress / crashed at step 0)
        if history:
            last = history[-1]
            final_loss = last.get("loss/mean")
        else:
            final_loss = None

        perplexity = eval_block.get("perplexity") if eval_block else None

        cells.append({
            **data,
            "file":       p,
            "cell_id":    p.stem,
            "n_steps":    len(history),
            "final_loss": final_loss,
            "perplexity": perplexity,
            # Shortcut config fields for convenience
            "aggregator":   config.get("aggregator",   p.stem.split("_p")[0]),
            "perturbation": config.get("perturbation", "unknown"),
            "byzantine_f":  config.get("byzantine_f",  0),
            "severity":     config.get("severity",     0.0),
            "seed":         config.get("seed",         42),
        })

    print(f"[analyze] Loaded {len(cells)} cells from {results_dir}")
    return cells


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def final_metrics(cell: dict) -> dict:
    """Extract scalar summary metrics from a loaded cell dict."""
    return {
        "cell_id":     cell["cell_id"],
        "aggregator":  cell["aggregator"],
        "perturbation":cell["perturbation"],
        "byzantine_f": cell["byzantine_f"],
        "severity":    cell["severity"],
        "seed":        cell["seed"],
        "n_steps":     cell["n_steps"],
        "final_loss":  cell["final_loss"],
        "perplexity":  cell["perplexity"],
    }


def build_table(cells: list[dict], out_path: Path) -> list[dict]:
    """
    Write summary.csv and return the rows as a list of dicts.

    Columns: cell_id, aggregator, perturbation, byzantine_f, severity, seed,
             n_steps, final_loss, perplexity.
    """
    rows = [final_metrics(c) for c in cells]
    rows.sort(key=lambda r: (r["perturbation"], r["byzantine_f"],
                              r["severity"], r["aggregator"]))

    fieldnames = ["cell_id", "aggregator", "perturbation", "byzantine_f",
                  "severity", "seed", "n_steps", "final_loss", "perplexity"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[analyze] summary.csv → {out_path}")
    return rows


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(cells: list[dict]) -> None:
    """
    Print which (perturbation, f, severity) conditions have all 5 aggregators
    present (COMPLETE) vs. partial.  Important to flag at 32/90 cells — do not
    over-interpret partial conditions.
    """
    from collections import defaultdict

    condition_aggs: dict[tuple, set] = defaultdict(set)
    for c in cells:
        key = (c["perturbation"], c["byzantine_f"], c["severity"])
        condition_aggs[key].add(c["aggregator"])

    all_aggs = set(AGG_ORDER)
    print("\n" + "=" * 64)
    print("COVERAGE REPORT")
    print("=" * 64)
    complete = []
    partial = []
    for key in sorted(condition_aggs):
        pert, f, sev = key
        present = condition_aggs[key]
        missing = all_aggs - present
        tag = "COMPLETE" if not missing else f"PARTIAL  (missing: {', '.join(sorted(missing))})"
        label = f"  pert={pert:<10} f={f}  sev={sev:<8.1f}  → {len(present)}/5  {tag}"
        if not missing:
            complete.append(label)
        else:
            partial.append(label)

    for l in complete:
        print(l)
    if partial:
        print()
        for l in partial:
            print(l)

    n_complete = len(complete)
    n_partial = len(partial)
    print(f"\n  {n_complete} complete conditions,  {n_partial} partial conditions")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Effect-size table
# ---------------------------------------------------------------------------

def effect_size_table(rows: list[dict]) -> None:
    """
    Print a table of final_loss[agg] − final_loss[mean] per condition.

    Negative values = robust aggregator outperforms mean (lower loss).
    This is the wiki's headline metric: pairwise comparison vs. plain mean.
    """
    from collections import defaultdict

    # Index: (pert, f, sev) → agg → final_loss
    index: dict[tuple, dict[str, float | None]] = defaultdict(dict)
    for r in rows:
        if r["final_loss"] is None:
            continue
        key = (r["perturbation"], r["byzantine_f"], r["severity"])
        index[key][r["aggregator"]] = r["final_loss"]

    robust_aggs = [a for a in AGG_ORDER if a != "mean"]
    header_aggs = ["mean"] + robust_aggs
    col_w = 14

    print("\n" + "=" * (32 + col_w * len(header_aggs)))
    print("EFFECT SIZE TABLE  (final_loss[agg] − final_loss[mean])")
    print("Negative = robust aggregator outperforms mean")
    print("=" * (32 + col_w * len(header_aggs)))
    header = f"{'Condition':<32}" + "".join(f"{a:>{col_w}}" for a in header_aggs)
    print(header)
    print("-" * len(header))

    for key in sorted(index):
        pert, f, sev = key
        losses = index[key]
        mean_loss = losses.get("mean")
        cond_str = f"{pert:<10} f={f}  sev={sev:<6.1f}"

        # mean column
        if mean_loss is not None:
            mean_str = f"{mean_loss:.4f}"
        else:
            mean_str = "  N/A"

        row_str = f"{cond_str:<32}{mean_str:>{col_w}}"
        for agg in robust_aggs:
            agg_loss = losses.get(agg)
            if agg_loss is None or mean_loss is None:
                row_str += f"{'N/A':>{col_w}}"
            else:
                delta = agg_loss - mean_loss
                row_str += f"{delta:>+{col_w}.4f}"
        print(row_str)

    print("=" * (32 + col_w * len(header_aggs)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_matplotlib():
    """Lazy import so the script is importable on headless machines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[analyze] WARNING: matplotlib not found; skipping plots.", file=sys.stderr)
        return None


def plot_condition(
    cells: list[dict],
    perturbation: str,
    byzantine_f: int,
    severity: float,
    out_path: Path,
    title: str | None = None,
) -> bool:
    """
    Plot loss-vs-step curves for all aggregators present in a given condition,
    one coloured line per aggregator.

    Returns True if the plot was written, False if no matching cells were found.
    """
    plt = _get_matplotlib()
    if plt is None:
        return False

    matching = [
        c for c in cells
        if c["perturbation"] == perturbation
        and c["byzantine_f"] == byzantine_f
        and abs(c["severity"] - severity) < 1e-6
    ]
    if not matching:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))

    for cell in sorted(matching, key=lambda c: AGG_ORDER.index(c["aggregator"])
                        if c["aggregator"] in AGG_ORDER else 99):
        history = cell.get("history", [])
        if not history:
            continue
        steps = [h["outer_step"] for h in history]
        losses = [h["loss/mean"] for h in history]
        agg = cell["aggregator"]
        ax.plot(
            steps, losses,
            label=AGG_LABELS.get(agg, agg),
            color=AGG_COLOURS.get(agg, "grey"),
            linewidth=1.8,
        )

    ax.set_xlabel("Outer step")
    ax.set_ylabel("Loss (mean across workers)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"pert={perturbation}  f={byzantine_f}  sev={severity}"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyze] plot → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Analyse DiLoCo sweep results across multiple cells."
    )
    p.add_argument("--results-dir", type=str, default="experiments/results/tier1",
                   help="Directory containing per-cell *.json result files.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write analysis outputs (default: <results-dir>/analysis).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation (useful on headless machines).")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    cells = load_cells(results_dir)
    if not cells:
        print("[analyze] No cells loaded — nothing to analyse.", file=sys.stderr)
        sys.exit(1)

    # Flag cells with fewer steps than expected (interrupted or in-progress)
    max_steps = max(c["n_steps"] for c in cells)
    incomplete = [c for c in cells if c["n_steps"] < max_steps and c["n_steps"] > 0]
    if incomplete:
        print(f"[analyze] NOTE: {len(incomplete)} cells have fewer than {max_steps} steps "
              f"(may be in-progress or interrupted):")
        for c in incomplete:
            print(f"  {c['cell_id']}  n_steps={c['n_steps']}")

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    coverage_report(cells)

    # ------------------------------------------------------------------
    # Summary CSV + effect-size table
    # ------------------------------------------------------------------
    rows = build_table(cells, out_dir / "summary.csv")
    effect_size_table(rows)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if args.no_plots:
        print("[analyze] --no-plots: skipping all plots.")
        return

    # Primary sanity check: clean baseline (no perturbation, f=0)
    # All 5 aggregators should produce nearly identical loss curves.
    # Divergence here = aggregator implementation bug.
    written = plot_condition(
        cells,
        perturbation="none",
        byzantine_f=0,
        severity=0.0,
        out_path=out_dir / "clean_baseline.png",
        title="Clean baseline (NoPerturbation, f=0) — aggregators must overlap",
    )
    if not written:
        print("[analyze] NOTE: no clean-baseline cells found.")

    # Decisive perturbed conditions — plot whatever is present.
    # Magnitude attack is the thesis's primary positive result.
    perturbed_conditions = []
    seen = set()
    for c in cells:
        if c["perturbation"] == "none":
            continue
        key = (c["perturbation"], c["byzantine_f"], c["severity"])
        if key not in seen:
            seen.add(key)
            perturbed_conditions.append(key)

    # Sort: magnitude first (decisive), then gaussian, then dropout
    def _pert_sort(k):
        pert_rank = {"magnitude": 0, "gaussian": 1, "dropout": 2, "hetero": 3}
        return (pert_rank.get(k[0], 9), k[1], k[2])

    perturbed_conditions.sort(key=_pert_sort)

    for (pert, f, sev) in perturbed_conditions:
        sev_str = f"{sev:.0f}" if sev == int(sev) else f"{sev}"
        fname = f"{pert}_f{f}_s{sev_str}.png"
        plot_condition(
            cells,
            perturbation=pert,
            byzantine_f=f,
            severity=sev,
            out_path=out_dir / fname,
            title=f"{pert}  f={f}  severity={sev}",
        )

    print(f"\n[analyze] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
