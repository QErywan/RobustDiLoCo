# Pseudo-Gradient Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an instrumented simulation that captures geometric properties of pseudo-gradients under MagnitudeAttack (f=2, scale=10) across all 5 aggregators and produces three thesis figures.

**Architecture:** `InstrumentedSimulation` subclasses `Simulation` (from `simulation/workers.py`) and fully overrides `run_outer_step` to capture metrics at three measurement points per step; `analyse_pseudograds.py` wires up all 5 aggregators sequentially and calls plotting helpers that read the per-aggregator JSON output files.

**Tech Stack:** Python 3.11, PyTorch (`torch.pca_lowrank` for in-memory PCA), matplotlib, json, pathlib. No new dependencies.

## Global Constraints

- Production code (`workers.py`, `run_experiment.py`, `sweep.py`) must NOT be modified
- Byzantine workers = last `f` workers in the list (indices `n-f` to `n-1`) — matches `simulation/perturbations.py` convention
- No full pseudo-grad tensors written to disk — PCA computed in-memory; only `(n, 2)` projections stored per snapshot
- Total disk cost: < 10 MB across all 5 aggregator runs
- Single-process only — no `dist.*`, `torchrun`, `gloo`, or `NCCL`
- All 5 aggregator names used for file naming: `mean`, `trimmed`, `median`, `rfa`, `krum`
- Output directory: `experiments/results/analysis/pseudograd_magnitude_f2_s10/`

---

### Task 1: `simulation/analysis.py` — InstrumentedSimulation

**Files:**
- Create: `simulation/analysis.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `Simulation`, `Worker`, `SimConfig` from `simulation/workers.py`; `Tensor` from `torch`
- Produces:
  - `InstrumentedSimulation(workers, aggregator, perturbation, config, out_dir, byzantine_f=0, pca_steps=None)`
  - `run_outer_step() -> dict` — superset of `Simulation.run_outer_step` output; adds keys `per_worker_norms_before`, `per_worker_norms_after`, `honest_cosine_sim`, `byzantine_cosine_sim`, `cosine_to_oracle`; also appends to `self.analysis_history` and buffers PCA at designated steps
  - `write_metrics(agg_name: str) -> None` — writes `metrics_{agg_name}.json` and `pca_{agg_name}_step{N}.json` files to `out_dir`
  - `InstrumentedSimulation._cosine(a: Tensor, b: Tensor) -> float` — static method

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis.py`:

```python
"""
Tests for simulation/analysis.py — InstrumentedSimulation metric helpers.

Strategy: test all analysis methods with synthetic tensors (no model run)
by constructing InstrumentedSimulation via __new__ and setting attributes
directly. This avoids slow training loops and keeps tests in milliseconds.
"""

import json
import torch
import pytest
from pathlib import Path

D = 64   # small synthetic dimension for speed
N = 8
F = 2
N_HONEST = N - F


def _skeleton(tmp_path, byzantine_f=F, pca_steps=None):
    """Build InstrumentedSimulation bypassing Simulation.__init__."""
    from simulation.analysis import InstrumentedSimulation
    inst = InstrumentedSimulation.__new__(InstrumentedSimulation)
    inst.workers = []
    inst.aggregator = None
    inst.perturbation = None
    inst.config = None
    inst.outer_step_count = 0
    inst.byzantine_f = byzantine_f
    inst.out_dir = Path(tmp_path)
    inst.pca_steps = pca_steps or set()
    inst.analysis_history = []
    inst._pca_cache = {}
    return inst


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        from simulation.analysis import InstrumentedSimulation
        a = torch.randn(D)
        assert abs(InstrumentedSimulation._cosine(a, a.clone()) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        from simulation.analysis import InstrumentedSimulation
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        assert abs(InstrumentedSimulation._cosine(a, b)) < 1e-6

    def test_opposite_vectors(self):
        from simulation.analysis import InstrumentedSimulation
        a = torch.randn(D)
        assert abs(InstrumentedSimulation._cosine(a, -a) + 1.0) < 1e-5

    def test_zero_vector_returns_zero(self):
        from simulation.analysis import InstrumentedSimulation
        a = torch.randn(D)
        assert InstrumentedSimulation._cosine(a, torch.zeros(D)) == 0.0

    def test_scale_invariant(self):
        from simulation.analysis import InstrumentedSimulation
        a = torch.randn(D)
        b = torch.randn(D)
        c1 = InstrumentedSimulation._cosine(a, b)
        c2 = InstrumentedSimulation._cosine(a * 100.0, b * 0.01)
        assert abs(c1 - c2) < 1e-4


# ---------------------------------------------------------------------------
# _save_pca
# ---------------------------------------------------------------------------

class TestSavePCA:
    def test_pca_cache_populated(self, tmp_path):
        sim = _skeleton(tmp_path, pca_steps={1})
        pseudo_grads = [torch.randn(D) for _ in range(N_HONEST)] + \
                       [torch.randn(D) * 10 for _ in range(F)]
        oracle = torch.stack(pseudo_grads[:N_HONEST]).mean(dim=0)
        aggregated = oracle.clone()
        sim._save_pca(pseudo_grads, oracle, aggregated, step=1)
        assert 1 in sim._pca_cache
        data = sim._pca_cache[1]
        assert len(data["worker_projections"]) == N
        assert len(data["worker_projections"][0]) == 2
        assert len(data["oracle_projection"]) == 2
        assert len(data["aggregated_projection"]) == 2

    def test_pca_projections_are_finite(self, tmp_path):
        sim = _skeleton(tmp_path)
        pseudo_grads = [torch.randn(D) for _ in range(N)]
        oracle = torch.stack(pseudo_grads[:N_HONEST]).mean(dim=0)
        aggregated = oracle.clone()
        sim._save_pca(pseudo_grads, oracle, aggregated, step=5)
        data = sim._pca_cache[5]
        for proj in data["worker_projections"]:
            assert all(abs(v) < 1e9 for v in proj)

    def test_pca_step_field_matches(self, tmp_path):
        sim = _skeleton(tmp_path)
        pseudo_grads = [torch.randn(D) for _ in range(N)]
        oracle = torch.stack(pseudo_grads[:N_HONEST]).mean(dim=0)
        sim._save_pca(pseudo_grads, oracle, oracle.clone(), step=25)
        assert sim._pca_cache[25]["step"] == 25


# ---------------------------------------------------------------------------
# write_metrics
# ---------------------------------------------------------------------------

class TestWriteMetrics:
    def test_writes_metrics_json(self, tmp_path):
        sim = _skeleton(tmp_path)
        sim.analysis_history = [
            {"outer_step": 1, "loss/mean": 4.0,
             "per_worker_norms_before": [1.0] * N,
             "per_worker_norms_after": [10.0] * N,
             "honest_cosine_sim": 0.9, "byzantine_cosine_sim": 0.8,
             "cosine_to_oracle": 0.5},
            {"outer_step": 2, "loss/mean": 3.5,
             "per_worker_norms_before": [0.9] * N,
             "per_worker_norms_after": [9.0] * N,
             "honest_cosine_sim": 0.92, "byzantine_cosine_sim": 0.82,
             "cosine_to_oracle": 0.55},
        ]
        sim.write_metrics("mean")
        path = tmp_path / "metrics_mean.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["cosine_to_oracle"] == 0.5
        assert data[1]["outer_step"] == 2

    def test_writes_pca_json_files(self, tmp_path):
        sim = _skeleton(tmp_path)
        sim._pca_cache = {
            1: {"step": 1,
                "worker_projections": [[0.1, 0.2]] * N,
                "oracle_projection": [0.05, 0.15],
                "aggregated_projection": [0.08, 0.18]},
        }
        sim.write_metrics("rfa")
        pca_path = tmp_path / "pca_rfa_step1.json"
        assert pca_path.exists()
        data = json.loads(pca_path.read_text())
        assert data["step"] == 1
        assert len(data["worker_projections"]) == N

    def test_writes_multiple_pca_steps(self, tmp_path):
        sim = _skeleton(tmp_path)
        for step in [1, 10, 25]:
            sim._pca_cache[step] = {
                "step": step,
                "worker_projections": [[float(step)] * 2] * N,
                "oracle_projection": [0.0, 0.0],
                "aggregated_projection": [0.0, 0.0],
            }
        sim.write_metrics("krum")
        for step in [1, 10, 25]:
            assert (tmp_path / f"pca_krum_step{step}.json").exists()


# ---------------------------------------------------------------------------
# Metric correctness — test metric computation with known synthetic inputs
# ---------------------------------------------------------------------------

class TestMetricComputation:
    def _run_one_step_metrics(self, tmp_path, pseudo_grads_before, byzantine_f):
        """
        Directly call _collect_metrics (factored helper) with synthetic tensors
        so we can assert exact metric values without running a training loop.
        """
        sim = _skeleton(tmp_path, byzantine_f=byzantine_f)
        oracle = torch.stack(pseudo_grads_before[:N - byzantine_f]).mean(dim=0)
        aggregated = oracle.clone()  # perfect aggregator for this test
        # Magnitude attack scales last f workers by 10x
        pseudo_grads_after = list(pseudo_grads_before)
        for i in range(N - byzantine_f, N):
            pseudo_grads_after[i] = pseudo_grads_before[i] * 10.0
        return sim._collect_metrics(
            pseudo_grads_before, pseudo_grads_after, aggregated, step=1
        )

    def test_perfect_aggregator_cosine_to_oracle_is_one(self, tmp_path):
        grads = [torch.ones(D) for _ in range(N)]
        metrics = self._run_one_step_metrics(tmp_path, grads, byzantine_f=F)
        assert abs(metrics["cosine_to_oracle"] - 1.0) < 1e-5

    def test_norms_before_length(self, tmp_path):
        grads = [torch.randn(D) for _ in range(N)]
        metrics = self._run_one_step_metrics(tmp_path, grads, byzantine_f=F)
        assert len(metrics["per_worker_norms_before"]) == N

    def test_norms_after_inflated_for_byzantine(self, tmp_path):
        grads = [torch.ones(D) for _ in range(N)]
        metrics = self._run_one_step_metrics(tmp_path, grads, byzantine_f=F)
        # Last f workers inflated 10x
        for i in range(N - F, N):
            assert abs(metrics["per_worker_norms_after"][i] /
                       metrics["per_worker_norms_before"][i] - 10.0) < 1e-3

    def test_honest_cosine_sim_uniform_grads_is_one(self, tmp_path):
        grads = [torch.ones(D) for _ in range(N)]
        metrics = self._run_one_step_metrics(tmp_path, grads, byzantine_f=F)
        assert abs(metrics["honest_cosine_sim"] - 1.0) < 1e-5

    def test_no_byzantine_workers_skips_byz_metrics(self, tmp_path):
        grads = [torch.randn(D) for _ in range(N)]
        sim = _skeleton(tmp_path, byzantine_f=0)
        aggregated = torch.stack(grads).mean(dim=0)
        metrics = sim._collect_metrics(grads, grads, aggregated, step=1)
        assert metrics["byzantine_cosine_sim"] == 0.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/qerywan/Developer/Imperial/SparseLoCo
python -m pytest tests/test_analysis.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'simulation.analysis'`

- [ ] **Step 3: Write `simulation/analysis.py`**

```python
"""
Instrumented DiLoCo simulation for pseudo-gradient geometry analysis.

InstrumentedSimulation is a subclass of Simulation that fully overrides
run_outer_step to capture three measurement points per step:
  1. Before perturbation: per-worker norms, honest cluster cosine sim,
     Byzantine cosine sim relative to oracle, oracle honest mean.
  2. After perturbation: per-worker norms (shows attack effect).
  3. After aggregation: cosine similarity between aggregate and oracle.

At designated PCA steps, projects all 8 pseudo-grad vectors into a 2D
PCA space computed in-memory (no full tensors written to disk).

Production code (workers.py, run_experiment.py, sweep.py) is not modified.
"""

import json
from pathlib import Path

import torch
from torch import Tensor

from simulation.workers import Simulation, SimConfig, Worker


class InstrumentedSimulation(Simulation):
    """
    Subclass of Simulation that captures geometric metrics of pseudo-gradients
    at three measurement points per outer step.

    Production code is untouched — this overrides run_outer_step only.
    Byzantine workers are the last `byzantine_f` workers in the list,
    consistent with simulation/perturbations.py.
    """

    def __init__(
        self,
        workers: list[Worker],
        aggregator,
        perturbation,
        config: SimConfig,
        out_dir: Path,
        byzantine_f: int = 0,
        pca_steps: set[int] | None = None,
    ):
        """
        Args:
            workers: list of Worker objects (same as Simulation).
            aggregator: aggregation rule (same as Simulation).
            perturbation: perturbation to inject (same as Simulation).
            config: SimConfig (same as Simulation).
            out_dir: directory for output JSON files.
            byzantine_f: number of Byzantine workers (last f in the list).
            pca_steps: outer steps at which to compute and store PCA snapshots.
                       Defaults to {1, 10, 25, 50}.
        """
        super().__init__(workers, aggregator, perturbation, config)
        self.out_dir = Path(out_dir)
        self.byzantine_f = byzantine_f
        self.pca_steps = pca_steps if pca_steps is not None else {1, 10, 25, 50}
        self.analysis_history: list[dict] = []
        self._pca_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def run_outer_step(self) -> dict:
        """
        Full override of Simulation.run_outer_step. Adds three measurement
        points and PCA snapshots at designated steps. The return dict is a
        superset of the parent's return dict — all downstream logging code
        that reads parent keys continues to work unchanged.
        """
        # --- Inner steps (replicate parent's offload logic) ---
        worker_metrics = []
        pseudo_grads_before: list[Tensor] = []

        for w in self.workers:
            if self.config.verbose:
                print(
                    f"  outer {self.outer_step_count + 1} | "
                    f"worker {w.rank}/{len(self.workers) - 1}",
                    flush=True,
                )
            if self.config.offload_between_steps:
                w.model.to(w.device)
            m = w.inner_step(steps=self.config.H)
            worker_metrics.append(m)
            pseudo_grads_before.append(w.compute_pseudo_grad())
            if self.config.offload_between_steps:
                w.model.to("cpu")

        # --- Perturbation ---
        pseudo_grads_after = self.perturbation.apply(pseudo_grads_before)

        # --- Aggregation ---
        aggregated = self.aggregator.aggregate(pseudo_grads_after)

        # --- PCA snapshot at designated steps (before perturbation) ---
        current_step = self.outer_step_count + 1
        if current_step in self.pca_steps:
            # oracle for PCA: honest workers only (pre-perturbation)
            n = len(self.workers)
            n_honest = n - self.byzantine_f
            oracle = torch.stack(pseudo_grads_before[:n_honest]).mean(dim=0)
            self._save_pca(pseudo_grads_before, oracle, aggregated, current_step)

        # --- Analysis metrics ---
        n = len(self.workers)
        n_honest = n - self.byzantine_f
        oracle = torch.stack(pseudo_grads_before[:n_honest]).mean(dim=0)
        step_metrics = self._collect_metrics(
            pseudo_grads_before, pseudo_grads_after, aggregated, current_step
        )

        # --- Outer updates (replicate parent's offload logic) ---
        for w in self.workers:
            if self.config.offload_between_steps:
                w.model.to(w.device)
            w.apply_outer_update(aggregated)
            if self.config.offload_between_steps:
                w.model.to("cpu")

        self.outer_step_count += 1

        # --- Build full metrics dict (superset of parent's keys) ---
        mean_losses = [m["mean_loss"] for m in worker_metrics]
        per_worker_norms_before = step_metrics["per_worker_norms_before"]
        metrics = {
            "outer_step": self.outer_step_count,
            "loss/mean": sum(mean_losses) / len(mean_losses),
            "loss/min": min(mean_losses),
            "loss/max": max(mean_losses),
            "pseudo_grad_norm/mean": sum(per_worker_norms_before) / n,
            "pseudo_grad_norm/max": max(per_worker_norms_before),
            "aggregated_grad_norm": aggregated.norm().item(),
            "worker_losses": mean_losses,
            **step_metrics,
        }
        self.analysis_history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Metric helpers (factored out so tests can call without a training loop)
    # ------------------------------------------------------------------

    def _collect_metrics(
        self,
        pseudo_grads_before: list[Tensor],
        pseudo_grads_after: list[Tensor],
        aggregated: Tensor,
        step: int,
    ) -> dict:
        """
        Compute analysis metrics from synthetic or real pseudo-grad lists.
        Called inside run_outer_step; also directly callable from tests.
        """
        n = len(pseudo_grads_before)
        n_honest = n - self.byzantine_f

        # Point 1: before perturbation
        per_worker_norms_before = [g.norm().item() for g in pseudo_grads_before]
        oracle = torch.stack(pseudo_grads_before[:n_honest]).mean(dim=0)

        honest_grads = pseudo_grads_before[:n_honest]
        if n_honest >= 2:
            pairs = [
                (i, j)
                for i in range(n_honest)
                for j in range(i + 1, n_honest)
            ]
            honest_cosine_sim = sum(
                self._cosine(honest_grads[i], honest_grads[j]) for i, j in pairs
            ) / len(pairs)
        else:
            honest_cosine_sim = 1.0

        byz_grads = pseudo_grads_before[n_honest:]
        byzantine_cosine_sim = (
            sum(self._cosine(g, oracle) for g in byz_grads) / len(byz_grads)
            if byz_grads
            else 0.0
        )

        # Point 2: after perturbation
        per_worker_norms_after = [g.norm().item() for g in pseudo_grads_after]

        # Point 3: after aggregation
        cosine_to_oracle = self._cosine(aggregated, oracle)

        return {
            "per_worker_norms_before": per_worker_norms_before,
            "per_worker_norms_after": per_worker_norms_after,
            "honest_cosine_sim": honest_cosine_sim,
            "byzantine_cosine_sim": byzantine_cosine_sim,
            "cosine_to_oracle": cosine_to_oracle,
        }

    def _save_pca(
        self,
        pseudo_grads_before: list[Tensor],
        oracle: Tensor,
        aggregated: Tensor,
        step: int,
    ) -> None:
        """
        Compute 2D PCA of the (n, d) pre-perturbation pseudo-grad matrix
        in-memory and buffer the (n, 2) projections in self._pca_cache.
        Full tensors are never written to disk.
        """
        stacked = torch.stack(pseudo_grads_before)   # (n, d)
        center = stacked.mean(dim=0)                  # (d,)
        centered = stacked - center                   # (n, d)

        # pca_lowrank: (U, S, V) where V is (d, 2)
        _, _, V = torch.pca_lowrank(centered, q=2, center=False)

        worker_projections = (centered @ V).tolist()                        # (n, 2)
        oracle_projection = ((oracle - center) @ V).tolist()               # (2,)
        aggregated_projection = ((aggregated - center) @ V).tolist()       # (2,)

        self._pca_cache[step] = {
            "step": step,
            "worker_projections": worker_projections,
            "oracle_projection": oracle_projection,
            "aggregated_projection": aggregated_projection,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_metrics(self, agg_name: str) -> None:
        """
        Write analysis_history to metrics_{agg_name}.json and all buffered
        PCA snapshots to pca_{agg_name}_step{N}.json in out_dir.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = self.out_dir / f"metrics_{agg_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(self.analysis_history, f, indent=2)

        for step, pca_data in self._pca_cache.items():
            pca_path = self.out_dir / f"pca_{agg_name}_step{step}.json"
            with open(pca_path, "w") as f:
                json.dump(pca_data, f, indent=2)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: Tensor, b: Tensor) -> float:
        """Cosine similarity between two flat tensors. Returns 0.0 if either is zero."""
        denom = a.norm() * b.norm()
        if denom.item() < 1e-12:
            return 0.0
        return (torch.dot(a.flatten(), b.flatten()) / denom).item()
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
cd /Users/qerywan/Developer/Imperial/SparseLoCo
python -m pytest tests/test_analysis.py -v
```

Expected output:
```
tests/test_analysis.py::TestCosine::test_identical_vectors PASSED
tests/test_analysis.py::TestCosine::test_orthogonal_vectors PASSED
tests/test_analysis.py::TestCosine::test_opposite_vectors PASSED
tests/test_analysis.py::TestCosine::test_zero_vector_returns_zero PASSED
tests/test_analysis.py::TestCosine::test_scale_invariant PASSED
tests/test_analysis.py::TestSavePCA::test_pca_cache_populated PASSED
tests/test_analysis.py::TestSavePCA::test_pca_projections_are_finite PASSED
tests/test_analysis.py::TestSavePCA::test_pca_step_field_matches PASSED
tests/test_analysis.py::TestWriteMetrics::test_writes_metrics_json PASSED
tests/test_analysis.py::TestWriteMetrics::test_writes_pca_json_files PASSED
tests/test_analysis.py::TestWriteMetrics::test_writes_multiple_pca_steps PASSED
tests/test_analysis.py::TestMetricComputation::test_perfect_aggregator_cosine_to_oracle_is_one PASSED
tests/test_analysis.py::TestMetricComputation::test_norms_before_length PASSED
tests/test_analysis.py::TestMetricComputation::test_norms_after_inflated_for_byzantine PASSED
tests/test_analysis.py::TestMetricComputation::test_honest_cosine_sim_uniform_grads_is_one PASSED
tests/test_analysis.py::TestMetricComputation::test_no_byzantine_workers_skips_byz_metrics PASSED
```

If a test fails, diagnose and fix before committing.

Also run the existing test suite to confirm no regressions:
```bash
python -m pytest tests/ -v --ignore=tests/test_analysis.py
```

- [ ] **Step 5: Commit**

```bash
git add simulation/analysis.py tests/test_analysis.py
git commit -m "feat: add InstrumentedSimulation for pseudo-gradient analysis (W2)"
```

---

### Task 2: `experiments/analyse_pseudograds.py` — driver + plotting

**Files:**
- Create: `experiments/analyse_pseudograds.py`

**Interfaces:**
- Consumes: `InstrumentedSimulation` from `simulation/analysis.py`; `build_model`, `param_count` from `simulation/model.py`; `Worker`, `SimConfig` from `simulation/workers.py`; all 5 aggregators from `simulation/aggregators.py`; `MagnitudeAttack` from `simulation/perturbations.py`; data loading from `experiments/run_baseline.py` (copied inline — do not import from `run_baseline.py`)
- Produces: `experiments/results/analysis/pseudograd_magnitude_f2_s10/metrics_{agg}.json` (×5), `pca_{agg}_step{N}.json` (×20), `plots/fig1_norms.png`, `plots/fig2_cohesion_oracle.png`, `plots/fig3_pca_snapshots.png`

- [ ] **Step 1: Write the failing smoke test**

Add to `tests/test_analysis.py`:

```python
# ---------------------------------------------------------------------------
# Driver smoke test
# ---------------------------------------------------------------------------

import subprocess
import sys

class TestDriverSmoke:
    def test_smoke_creates_output_files(self, tmp_path):
        """
        Run the driver in smoke mode (tiny model, H=2, 3 outer steps, 2 inner steps)
        and verify all expected output files exist.
        """
        result = subprocess.run(
            [
                sys.executable,
                "experiments/analyse_pseudograds.py",
                "--smoke",
                "--out-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd="/Users/qerywan/Developer/Imperial/SparseLoCo",
        )
        assert result.returncode == 0, (
            f"Driver exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        for agg_name in ["mean", "trimmed", "median", "rfa", "krum"]:
            assert (tmp_path / f"metrics_{agg_name}.json").exists(), \
                f"Missing metrics_{agg_name}.json"
        for fig in ["fig1_norms.png", "fig2_cohesion_oracle.png", "fig3_pca_snapshots.png"]:
            assert (tmp_path / "plots" / fig).exists(), f"Missing {fig}"
```

Run to confirm failure:
```bash
cd /Users/qerywan/Developer/Imperial/SparseLoCo
python -m pytest tests/test_analysis.py::TestDriverSmoke -v
```

Expected: `FileNotFoundError` or similar (script doesn't exist yet).

- [ ] **Step 2: Write `experiments/analyse_pseudograds.py`**

```python
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
from pathlib import Path

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
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: honest cluster cohesion for RFA (representative)
    rfa_metrics = all_metrics["rfa"]
    steps = [m["outer_step"] for m in rfa_metrics]
    ax_top.plot(
        steps, [m["honest_cosine_sim"] for m in rfa_metrics],
        color="steelblue", linewidth=2, label="honest worker cosine sim (RFA)"
    )
    ax_top.plot(
        steps, [m["byzantine_cosine_sim"] for m in rfa_metrics],
        color="firebrick", linestyle="--", linewidth=2,
        label="Byzantine cosine to oracle (RFA)"
    )
    ax_top.set_ylabel("cosine similarity")
    ax_top.set_title(
        "Cluster cohesion at H=500 — honest workers converge in direction\n"
        "(pre-perturbation, RFA run shown as representative)"
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
```

- [ ] **Step 3: Run the smoke test**

```bash
cd /Users/qerywan/Developer/Imperial/SparseLoCo
python experiments/analyse_pseudograds.py --smoke --out-dir /tmp/pseudograd_smoke_test
```

Expected: runs without error, prints step logs for all 5 aggregators, outputs:
```
/tmp/pseudograd_smoke_test/metrics_mean.json
/tmp/pseudograd_smoke_test/metrics_trimmed.json
/tmp/pseudograd_smoke_test/metrics_median.json
/tmp/pseudograd_smoke_test/metrics_rfa.json
/tmp/pseudograd_smoke_test/metrics_krum.json
/tmp/pseudograd_smoke_test/pca_mean_step1.json
/tmp/pseudograd_smoke_test/plots/fig1_norms.png
/tmp/pseudograd_smoke_test/plots/fig2_cohesion_oracle.png
/tmp/pseudograd_smoke_test/plots/fig3_pca_snapshots.png
```

- [ ] **Step 4: Run the pytest smoke test**

```bash
cd /Users/qerywan/Developer/Imperial/SparseLoCo
python -m pytest tests/test_analysis.py::TestDriverSmoke -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full test suite**

```bash
cd /Users/qerywan/Developer/Imperial/SparseLoCo
python -m pytest tests/ -v
```

Expected: all tests pass. If any of the pre-existing tests fail, investigate (they were passing before this task).

- [ ] **Step 6: Commit**

```bash
git add experiments/analyse_pseudograds.py tests/test_analysis.py
git commit -m "feat: add analyse_pseudograds.py driver and three thesis figures (W2)"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| `InstrumentedSimulation(Simulation)` subclass | Task 1 |
| `run_outer_step` fully overrides parent | Task 1 |
| Three measurement points per step | Task 1 (`_collect_metrics`) |
| `per_worker_norms_before` | Task 1 |
| `honest_cosine_sim` (honest pairs, pre-perturbation) | Task 1 |
| `byzantine_cosine_sim` (Byzantine vs oracle, pre-perturbation) | Task 1 |
| `oracle_honest_mean` = mean(grads[:n-f]) | Task 1 |
| `per_worker_norms_after` | Task 1 |
| `cosine_to_oracle` after aggregation | Task 1 |
| PCA at steps {1,10,25,50} in-memory, only (n,2) stored | Task 1 (`_save_pca`) |
| `analysis_history` → `metrics_{agg}.json` | Task 1 (`write_metrics`) |
| `pca_{agg}_step{N}.json` | Task 1 (`write_metrics`) |
| `analyse_pseudograds.py` driver, all 5 aggregators | Task 2 |
| MagnitudeAttack f=2 scale=10 hardcoded | Task 2 |
| NanoGPT hparams default, CLI for device + data-path | Task 2 |
| 50 outer steps (Tier-1 match) | Task 2 (`FULL_CFG`) |
| Output to `experiments/results/analysis/pseudograd_magnitude_f2_s10/` | Task 2 |
| `fig1_norms.png` — 1×5, log scale, honest blue / Byzantine red | Task 2 (`plot_norms`) |
| `fig2_cohesion_oracle.png` — 2 panels, RFA top, all aggs bottom | Task 2 (`plot_cohesion_oracle`) |
| `fig3_pca_snapshots.png` — 5×2 scatter, step 1 and step 25 | Task 2 (`plot_pca_snapshots`) |
| Production code untouched | Verified — no edits to workers.py, run_experiment.py, sweep.py |
| Disk < 10 MB | Enforced — no full tensors on disk, only (n,2) PCA projections |

**No placeholder scan:** No "TBD", "TODO", or incomplete sections found.

**Type consistency:** `_cosine` returns `float` in Task 1 and is consumed as `float` in Task 2 `plot_*` functions. `write_metrics(agg_name: str)` matches Task 2 calls `sim.write_metrics(agg_name)`. `pca_steps` is `set[int]` — passed from `SMOKE_CFG["pca_steps"]` which is `{1}` (set literal). ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-19-pseudograd-analysis.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints

Which approach?
