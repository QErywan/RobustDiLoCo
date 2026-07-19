# Pseudo-Gradient Analysis — Design Spec

**Date:** 2026-07-19
**Workstream:** W2 — Pseudo-gradient geometry analysis
**Status:** Approved, pending implementation

---

## Context

The supervisor asked us to systematically study DiLoCo's pseudo-gradients and understand
why combining robust aggregators with the H=500 regime still works. The core thesis claim
is that the breakdown-point ordering (mean < trimmed < RFA) holds empirically — but *why*
it holds is the analytical contribution. This spec defines an instrumented simulation that
captures the geometric properties of pseudo-gradients during training and produces three
figures that together answer that question.

---

## Goal

Run a dedicated analysis on the most decisive Tier-1 condition (magnitude attack, f=2,
severity=10) across all 5 aggregators. Produce three figures that show:
1. What the attack does to pseudo-gradient norms
2. Why the honest cluster is identifiable at H=500 (and how well each aggregator recovers the honest signal)
3. The geometric intuition of Byzantine outliers vs honest cluster in 2D

Production code (`workers.py`, `run_experiment.py`, sweep harness) is **not modified**.

---

## Architecture

Three components:

```
simulation/analysis.py              ← InstrumentedSimulation subclass
experiments/analyse_pseudograds.py  ← driver script
experiments/results/analysis/
  pseudograd_magnitude_f2_s10/
    metrics_<aggregator>.json       ← per-step rich metrics (one file per aggregator)
    pca_<aggregator>_step<N>.json   ← 8×2 PCA projections at key steps
    plots/
      fig1_norms.png
      fig2_cohesion_oracle.png
      fig3_pca_snapshots.png
```

---

## Component 1 — `simulation/analysis.py`

### `InstrumentedSimulation(Simulation)`

Subclasses `Simulation` and overrides `run_outer_step` fully (the parent method is ~40
lines). Production code is untouched. Adds three measurement points per outer step:

**Point 1 — after `compute_pseudo_grad()`, before perturbation:**

| Metric | Description |
|---|---|
| `per_worker_norms_before` | List of n L2 norms — honest workers' natural norm distribution |
| `honest_cosine_sim` | Mean cosine similarity among the first `n - byzantine_f` workers — how tight the honest cluster is |
| `byzantine_cosine_sim` | Mean cosine similarity between Byzantine and honest workers — how detectable the attack is |
| `oracle_honest_mean` | `mean(pseudo_grads[:n - byzantine_f])` — ground-truth target each aggregator should approximate |

Byzantine workers are always the last `f` workers in the list (consistent with how
perturbations are applied across the codebase).

**Point 2 — after perturbation, before aggregation:**

| Metric | Description |
|---|---|
| `per_worker_norms_after` | List of n norms post-perturbation — shows attack effect (inflated for magnitude, erratic for Gaussian) |

**Point 3 — after aggregation:**

| Metric | Description |
|---|---|
| `cosine_to_oracle` | Cosine similarity between aggregated gradient and `oracle_honest_mean` — how well the aggregator recovered the honest signal |

**At steps `{1, 10, 25, 50}` (PCA):**
- Compute PCA of the `(n, d)` pre-perturbation pseudo-grad matrix → `(n, 2)` projections
- Project `oracle_honest_mean` and `aggregated` into same PCA space
- Store as `pca_<aggregator>_step<N>.json` — **not** full tensors, just 2D coordinates
- Disk cost: negligible (< 10 MB total across all runs)

All per-step metrics appended to `analysis_history` list and written to
`metrics_<aggregator>.json` at end of run.

### Constructor

```python
InstrumentedSimulation(
    config: SimConfig,
    aggregator: BaseAggregator,
    perturbation: BasePerturbation,
    out_dir: Path,           # where to write metrics + PCA files
    pca_steps: set[int] = {1, 10, 25, 50},
)
```

---

## Component 2 — `experiments/analyse_pseudograds.py`

Driver script. Does not use the sweep harness — standalone.

**Config (hardcoded for the decisive condition):**
- Perturbation: `MagnitudeAttack`, severity=10, f=2
- Aggregators: all 5 (mean, trimmed, median, rfa, krum)
- Model: NanoGPT hparams (`hparams/sim/sim_model_hparams_nanogpt.json`)
- Outer steps: 50 (matches Tier-1)
- Device: `--device cuda` (CLI flag)
- Data path: `--data-path` (CLI flag, passed to C4 loader)
- Output: `experiments/results/analysis/pseudograd_magnitude_f2_s10/`

**Execution:** runs each aggregator sequentially, writing its metrics file, then calls
the plotting functions once all 5 are complete.

---

## Component 3 — Plots

### Figure 1 — Per-worker norm trajectories (`fig1_norms.png`)

- Layout: 1×5 subplots (one per aggregator)
- X: outer step 1–50, Y: L2 norm (log scale)
- Lines: `n - f` honest workers (blue), `f` Byzantine workers (red dashed)
- Shows: Byzantine norm inflation vs tight honest cluster; mean has no filtering mechanism

### Figure 2 — Cluster cohesion + cosine to oracle (`fig2_cohesion_oracle.png`)

- Layout: 2 panels stacked
- Top panel: `honest_cosine_sim` (solid) and `byzantine_cosine_sim` (dashed) over steps
  — one line each, shared across aggregators since pre-perturbation pseudo-grads diverge
  per aggregator after step 1; show one representative aggregator (RFA) in top panel
- Bottom panel: `cosine_to_oracle` over steps, one coloured line per aggregator
- Shows: honest cluster cohesion exists at H=500 (top); robust aggregators maintain
  high cosine to oracle while mean drifts away (bottom)

### Figure 3 — PCA snapshots (`fig3_pca_snapshots.png`)

- Layout: 2 columns (step 1, step 25) × 5 rows (one per aggregator)
- Each panel: scatter of 8 pseudo-grad vectors in 2D PCA space
  - Honest workers: blue circles
  - Byzantine workers: red crosses
  - Oracle honest mean: green star
  - Aggregated output: coloured diamond (colour = aggregator)
- Step 1: all aggregators share the same pseudo-grads (same model state) → shows
  initial attack geometry, Byzantine outliers immediately visible
- Step 25: models have diverged under each aggregator → shows cumulative drift effect;
  under mean the honest cluster starts to break apart; under RFA it stays tight

---

## What the figures show

**Figure 1** shows the attack mechanism — Byzantine norms are outliers, honest cluster
is tight. The tight honest cluster is what robust aggregators exploit; mean averages
the inflated norms straight in.

**Figure 2** answers "why does combining robust aggregation with DiLoCo still work":
H=500 averaging creates high inter-honest cosine similarity (workers agree on direction
after 500 steps of gradient descent on similar data). Byzantine workers are geometrically
separable. The bottom panel shows the consequence: RFA/trimmed/median maintain cosine
to oracle ~0.9+; mean drifts toward 0 as Byzantine pull compounds.

**Figure 3** is the geometric intuition — the thesis figure for examiners. Step 1 shows
the initial geometry (Byzantine outliers visible immediately). Step 25 shows cumulative
drift: under mean the honest cluster breaks apart; under RFA it remains tight because
outer updates have consistently approximated the oracle.

---

## Constraints

- Production code untouched (`workers.py`, `run_experiment.py`, sweep harness)
- No full pseudo-grad tensors written to disk — PCA computed in memory, only 2D
  projections stored
- Total disk cost: < 10 MB
- Runtime: ~50h on a single GPU (5 aggregators × ~10h per run) — overnight run
- Byzantine workers assumed to be last `f` workers in the list (consistent with
  perturbation implementations in `simulation/perturbations.py`)

---

## Not in scope

- SparseLoCo compression analysis (separate spec, separate branch)
- Gaussian noise or dropout conditions (magnitude f=2 s=10 is sufficient for the
  mechanism argument; other conditions can reuse the same script with CLI flags later)
- Hetero data condition (pending user's loader implementation)
