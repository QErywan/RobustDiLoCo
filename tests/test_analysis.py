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
