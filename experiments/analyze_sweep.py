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
import math
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

# Linestyle and marker per aggregator — used in both absolute and residual modes so
# curves remain distinguishable even when colours overlap.
AGG_LINESTYLES = {
    "mean":    "-",
    "trimmed": "--",
    "median":  "-.",
    "rfa":     ":",
    "krum":    (0, (3, 1, 1, 1)),   # densely dashdotted
}

AGG_MARKERS = {
    "mean":    None,
    "trimmed": "o",
    "median":  "s",
    "rfa":     "^",
    "krum":    "D",
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
        - "status"     → "completed" | "crashed" | "running" (back-compat: inferred
                          from eval presence for legacy JSONs without an explicit field)
        - "error"      → error dict if status=="crashed", else None
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

        # Determine status — explicit field takes priority; infer for legacy JSONs.
        if "status" in data:
            status = data["status"]
        elif eval_block:
            status = "completed"   # legacy: has eval → finished
        else:
            status = "running"     # legacy: no eval, no explicit status → interrupted

        error = data.get("error")  # None for non-crashed cells

        cells.append({
            **data,
            "file":       p,
            "cell_id":    p.stem,
            "n_steps":    len(history),
            "final_loss": final_loss,
            "perplexity": perplexity,
            "status":     status,
            "error":      error,
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
        "status":      cell.get("status", "running"),
        "final_loss":  cell["final_loss"],
        "perplexity":  cell["perplexity"],
    }


def build_table(cells: list[dict], out_path: Path) -> list[dict]:
    """
    Write summary.csv and return the rows as a list of dicts.

    Columns: cell_id, aggregator, perturbation, byzantine_f, severity, seed,
             n_steps, status, final_loss, perplexity.
    """
    rows = [final_metrics(c) for c in cells]
    rows.sort(key=lambda r: (r["perturbation"], r["byzantine_f"],
                              r["severity"], r["aggregator"]))

    fieldnames = ["cell_id", "aggregator", "perturbation", "byzantine_f",
                  "severity", "seed", "n_steps", "status", "final_loss", "perplexity"]
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


def _cell_diverged(cell: dict) -> bool:
    """Return True if any history entry has a non-finite (or None) loss/mean."""
    for h in cell.get("history", []):
        v = h.get("loss/mean")
        if v is None or not math.isfinite(v):
            return True
    return False


def plot_condition(
    cells: list[dict],
    perturbation: str,
    byzantine_f: int,
    severity: float,
    out_path: Path,
    title: str | None = None,
    mode: str = "absolute",
    exclude_diverged: bool = False,
) -> bool:
    """
    Plot loss-vs-step curves for all aggregators present in a given condition,
    one coloured line per aggregator.

    Parameters
    ----------
    mode : "absolute" | "residual"
        "absolute"  — raw loss/mean per step (default, good when gaps are large).
        "residual"  — loss[agg] − loss[mean] per step so that mean becomes a flat
                      zero reference and overlapping curves spread into a readable
                      band.  Alignment is by outer_step, not by index, so partial
                      runs (different lengths) are handled correctly.  Returns False
                      if no mean cell is present or if mean losses are all NaN.
    exclude_diverged : bool
        When True (absolute mode only), filter out any cell whose history contains
        a non-finite loss/mean before plotting.  If *no* cell in the condition
        diverged, returns False without writing a file — so ``_robust.png`` twins
        only appear where they add value (i.e. where mean blew up).  A note listing
        the excluded aggregator names is appended to the plot title so the output is
        self-documenting.

    In both modes each aggregator gets a distinct linestyle + sparse marker
    (AGG_LINESTYLES / AGG_MARKERS) so lines remain distinguishable even where
    colours overlap.

    Returns True if the plot was written, False otherwise.
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

    # ------------------------------------------------------------------
    # Robust-only filter: drop cells whose history contains any NaN/inf.
    # Only applies in absolute mode; residual mode already excludes mean.
    # ------------------------------------------------------------------
    if exclude_diverged and mode == "absolute":
        diverged_aggs = [c["aggregator"] for c in matching if _cell_diverged(c)]
        if not diverged_aggs:
            # Nothing diverged in this condition — the robust plot is identical
            # to the normal plot, so skip it to avoid a redundant duplicate.
            return False
        matching = [c for c in matching if not _cell_diverged(c)]
        if not matching:
            return False

    # ------------------------------------------------------------------
    # Residual mode: build a step→loss lookup from the mean cell
    # ------------------------------------------------------------------
    mean_lookup: dict[int, float] = {}
    if mode == "residual":
        mean_cells = [c for c in matching if c["aggregator"] == "mean"]
        if not mean_cells:
            return False
        mean_hist = mean_cells[0].get("history", [])
        mean_lookup = {
            h["outer_step"]: h["loss/mean"]
            for h in mean_hist
            if h["loss/mean"] is not None and not (h["loss/mean"] != h["loss/mean"])  # not NaN
        }
        if not mean_lookup:
            return False  # mean is all-NaN; residual undefined

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    # Collect all finite loss values in absolute mode to decide on log-scale.
    all_finite_losses: list[float] = []

    for cell in sorted(matching, key=lambda c: AGG_ORDER.index(c["aggregator"])
                        if c["aggregator"] in AGG_ORDER else 99):
        history = cell.get("history", [])
        if not history:
            continue
        agg = cell["aggregator"]

        if mode == "residual":
            # Align by outer_step; only include steps where mean is defined
            pairs = [
                (h["outer_step"], h["loss/mean"] - mean_lookup[h["outer_step"]])
                for h in history
                if h["outer_step"] in mean_lookup
                and h["loss/mean"] is not None
                and not (h["loss/mean"] != h["loss/mean"])  # not NaN
            ]
            if not pairs:
                continue
            steps, losses = zip(*pairs)
        else:
            # Filter to finite (step, loss) pairs only; record if this cell diverged.
            finite_pairs = [
                (h["outer_step"], h["loss/mean"])
                for h in history
                if h["loss/mean"] is not None and math.isfinite(h["loss/mean"])
            ]
            diverged = any(
                h["loss/mean"] is None or not math.isfinite(h["loss/mean"])
                for h in history
            )
            if not finite_pairs:
                continue
            steps, losses = zip(*finite_pairs)

            # Track global finite range for log-scale decision below.
            all_finite_losses.extend(losses)

        markevery = max(1, len(steps) // 8)
        line_colour = AGG_COLOURS.get(agg, "grey")
        ax.plot(
            steps, losses,
            label=AGG_LABELS.get(agg, agg),
            color=line_colour,
            linestyle=AGG_LINESTYLES.get(agg, "-"),
            marker=AGG_MARKERS.get(agg),
            markevery=markevery,
            markersize=4,
            linewidth=1.8,
        )

        # For diverged cells, mark the last finite point with a red X and annotate.
        if mode == "absolute" and diverged and steps:
            last_step, last_loss = steps[-1], losses[-1]
            ax.plot(last_step, last_loss, marker="x", markersize=10,
                    markeredgewidth=2.5, color="red", zorder=5, linestyle="none",
                    label="_nolegend_")
            ax.annotate(
                "diverged (NaN)",
                xy=(last_step, last_loss),
                xytext=(8, 4), textcoords="offset points",
                fontsize=7, color="red",
            )

    if mode == "residual":
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("Loss − mean (Δ)")
    else:
        # Auto log-scale when the finite range spans more than 20×
        # (e.g. robust ~5 vs diverging mean ~2730 on magnitude attack).
        if all_finite_losses:
            lo, hi = min(all_finite_losses), max(all_finite_losses)
            if lo > 0 and hi / lo > 20:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (mean across workers, log scale)")
            else:
                ax.set_ylabel("Loss (mean across workers)")
        else:
            ax.set_ylabel("Loss (mean across workers)")

    ax.set_xlabel("Outer step")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"pert={perturbation}  f={byzantine_f}  sev={severity}"
    if mode == "residual":
        title += "  [residual vs mean]"
    if exclude_diverged and mode == "absolute":
        # diverged_aggs is always defined here (we returned False if empty above)
        dropped = ", ".join(diverged_aggs)  # noqa: F821 — set in filter block above
        title += f"\n[robust only — excluded: {dropped}]"
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

    # ------------------------------------------------------------------
    # Crashed cells — show before anything else so errors are visible
    # ------------------------------------------------------------------
    crashed = [c for c in cells if c.get("status") == "crashed"]
    if crashed:
        print(f"\n{'='*64}")
        print(f"CRASHED CELLS  ({len(crashed)} cells failed with a recorded exception)")
        print(f"{'='*64}")
        for c in crashed:
            err = c.get("error") or {}
            oom_tag = "  [CUDA OOM]" if err.get("oom") else ""
            print(
                f"  {c['cell_id']}\n"
                f"    failed_at_step={err.get('failed_at_step', '?')}  "
                f"error={err.get('type', '?')}: {err.get('message', '')[:120]}"
                f"{oom_tag}"
            )
        print(f"{'='*64}")
    else:
        print("[analyze] No crashed cells (status=crashed) found.")

    # Flag cells with fewer steps than expected (interrupted or in-progress)
    max_steps = max(c["n_steps"] for c in cells)
    incomplete = [c for c in cells if c["n_steps"] < max_steps and c["n_steps"] > 0
                  and c.get("status") != "crashed"]
    if incomplete:
        print(f"[analyze] NOTE: {len(incomplete)} cells have fewer than {max_steps} steps "
              f"(may be in-progress or interrupted):")
        for c in incomplete:
            print(f"  {c['cell_id']}  n_steps={c['n_steps']}  status={c.get('status','?')}")

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

    # Residual twin: mean is the flat-zero reference; other aggregators spread
    # into a readable band.  A curve flying off the band = implementation bug.
    plot_condition(
        cells,
        perturbation="none",
        byzantine_f=0,
        severity=0.0,
        out_path=out_dir / "clean_baseline_residual.png",
        title="Clean baseline — residual vs mean (0 = mean reference)",
        mode="residual",
    )

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

    # Perturbation types where curves are expected to stay close (small gaps) and
    # benefit most from the residual view.  Magnitude/gaussian are excluded because
    # mean diverges/NaN there, making the residual undefined.
    RESIDUAL_PERTS = {"dropout", "hetero"}

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
        # Residual twin for conditions where aggregators are expected to overlap
        if pert in RESIDUAL_PERTS:
            plot_condition(
                cells,
                perturbation=pert,
                byzantine_f=f,
                severity=sev,
                out_path=out_dir / fname.replace(".png", "_residual.png"),
                title=f"{pert}  f={f}  severity={sev}  [residual vs mean]",
                mode="residual",
            )
        # Robust-only twin: drops any diverged aggregator so the surviving curves
        # are readable on a linear axis.  plot_condition returns False (no file
        # written) if nothing in this condition diverged, so this is a no-op for
        # clean / dropout conditions — only magnitude / gaussian get the extra plot.
        plot_condition(
            cells,
            perturbation=pert,
            byzantine_f=f,
            severity=sev,
            out_path=out_dir / fname.replace(".png", "_robust.png"),
            title=f"{pert}  f={f}  severity={sev}",
            exclude_diverged=True,
        )

    print(f"\n[analyze] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
