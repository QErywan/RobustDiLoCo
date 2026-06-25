"""
Tests for simulation/aggregators.py.

Strategy: use synthetic inputs with known closed-form outputs so each test
is a direct numerical assertion, not just a shape/type check.

Byzantine-tolerance structure (shared across robust aggregator tests):
    n=8 workers, f=2 Byzantine (last 2).
    Clean grads: 6 workers each sending `ones(D)`.
    Byzantine grads: 2 workers sending `BSCALE * ones(D)` (huge outlier).

    - MeanAggregator: output = (6*1 + 2*BSCALE)/8  (far from 1.0).
    - Robust aggregators: output close to 1.0 (outliers filtered or down-weighted).
"""

import warnings

import torch
import pytest
from simulation.aggregators import (
    MeanAggregator,
    TrimmedMeanAggregator,
    CoordMedianAggregator,
    GeometricMedianAggregator,
    MultiKrumAggregator,
)

N_WORKERS = 8
D = 512  # flat pseudo-gradient dimension


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_grads():
    """All workers send the same vector — aggregate must equal that vector."""
    g = torch.randn(D)
    return [g.clone() for _ in range(N_WORKERS)]


@pytest.fixture
def zero_grads():
    return [torch.zeros(D) for _ in range(N_WORKERS)]


@pytest.fixture
def known_mean_grads():
    """Grads are 1, 2, ..., n scaled vectors — mean is (n+1)/2 * ones."""
    return [torch.full((D,), float(i + 1)) for i in range(N_WORKERS)]


# ---------------------------------------------------------------------------
# MeanAggregator
# ---------------------------------------------------------------------------

class TestMeanAggregator:
    def setup_method(self):
        self.agg = MeanAggregator()

    def test_output_shape(self, uniform_grads):
        out = self.agg.aggregate(uniform_grads)
        assert out.shape == (D,)

    def test_output_is_tensor(self, uniform_grads):
        out = self.agg.aggregate(uniform_grads)
        assert isinstance(out, torch.Tensor)

    def test_uniform_input_returns_same_vector(self, uniform_grads):
        out = self.agg.aggregate(uniform_grads)
        assert torch.allclose(out, uniform_grads[0])

    def test_zero_input_returns_zero(self, zero_grads):
        out = self.agg.aggregate(zero_grads)
        assert torch.allclose(out, torch.zeros(D))

    def test_known_mean(self, known_mean_grads):
        # grads are [1,1,...], [2,2,...], ..., [8,8,...]
        # mean = (1+2+...+8)/8 = 4.5
        out = self.agg.aggregate(known_mean_grads)
        expected = torch.full((D,), 4.5)
        assert torch.allclose(out, expected)

    def test_single_worker(self):
        g = torch.randn(D)
        out = self.agg.aggregate([g])
        assert torch.allclose(out, g)

    def test_does_not_mutate_inputs(self, uniform_grads):
        original = [g.clone() for g in uniform_grads]
        self.agg.aggregate(uniform_grads)
        for orig, after in zip(original, uniform_grads):
            assert torch.allclose(orig, after)

    def test_output_finite(self):
        grads = [torch.randn(D) for _ in range(N_WORKERS)]
        out = self.agg.aggregate(grads)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BSCALE = 1000.0   # Byzantine outlier scale for Byzantine-tolerance tests
F = 2             # Byzantine workers (last F of N_WORKERS=8)


def make_byzantine_grads(clean_val: float = 1.0, byz_val: float = BSCALE) -> list:
    """6 clean workers + 2 Byzantine workers (last 2) sending a large vector."""
    clean = [torch.full((D,), clean_val) for _ in range(N_WORKERS - F)]
    byz   = [torch.full((D,), byz_val)   for _ in range(F)]
    return clean + byz


def check_shape_finite_no_mutation(agg, grads) -> torch.Tensor:
    """Run aggregate and assert shape, finiteness, and no in-place mutation."""
    original = [g.clone() for g in grads]
    out = agg.aggregate(grads)
    assert out.shape == (D,), f"Expected shape ({D},), got {out.shape}"
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    for orig, after in zip(original, grads):
        assert torch.allclose(orig, after), "Aggregator mutated input in place"
    return out


# ---------------------------------------------------------------------------
# TrimmedMeanAggregator
# ---------------------------------------------------------------------------

class TestTrimmedMeanAggregator:
    def test_output_shape_finite_no_mutation(self):
        agg = TrimmedMeanAggregator(f=F)
        check_shape_finite_no_mutation(agg, [torch.randn(D) for _ in range(N_WORKERS)])

    def test_uniform_input_equals_input(self, uniform_grads):
        agg = TrimmedMeanAggregator(f=F)
        out = agg.aggregate(uniform_grads)
        assert torch.allclose(out, uniform_grads[0])

    def test_known_output_f1(self):
        # Grads are [1,1,...], [2,2,...], ..., [8,8,...].
        # With f=1: sort → [1,2,3,4,5,6,7,8], trim 1 from each end → [2,3,4,5,6,7]
        # mean = 27/6 = 4.5
        grads = [torch.full((D,), float(i + 1)) for i in range(N_WORKERS)]
        agg = TrimmedMeanAggregator(f=1)
        out = agg.aggregate(grads)
        assert torch.allclose(out, torch.full((D,), 4.5))

    def test_known_output_f2(self):
        # f=2: trim 2 from each end → [3,4,5,6], mean = 4.5
        grads = [torch.full((D,), float(i + 1)) for i in range(N_WORKERS)]
        agg = TrimmedMeanAggregator(f=2)
        out = agg.aggregate(grads)
        assert torch.allclose(out, torch.full((D,), 4.5))

    def test_byzantine_tolerance(self):
        # Mean would be (6*1 + 2*1000)/8 = 250.75; trimmed mean should be ~1.0
        grads = make_byzantine_grads()
        agg = TrimmedMeanAggregator(f=F)
        out = agg.aggregate(grads)
        mean_agg = MeanAggregator().aggregate(grads)
        assert out.mean().item() < 2.0, f"Expected output near 1.0, got {out.mean().item():.3f}"
        assert out.mean().item() < mean_agg.mean().item() / 10, \
            "TrimmedMean should be far less influenced by outliers than Mean"

    def test_f_cap_warns_and_runs(self):
        # f=4 = n/2 is invalid; should warn and cap to 3
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agg = TrimmedMeanAggregator(f=4, n_workers=8)
            assert len(w) == 1
            assert "Capping" in str(w[0].message)
        assert agg.f == 3
        grads = [torch.randn(D) for _ in range(N_WORKERS)]
        out = agg.aggregate(grads)
        assert torch.isfinite(out).all()

    def test_f0_equals_mean(self):
        # f=0 → no trimming → result numerically equals MeanAggregator.
        # Use atol=1e-6 because sorting then averaging can differ from a direct
        # mean in the last ULP due to floating-point accumulation order.
        grads = [torch.randn(D) for _ in range(N_WORKERS)]
        mean_out = MeanAggregator().aggregate(grads)
        trim_out = TrimmedMeanAggregator(f=0).aggregate(grads)
        assert torch.allclose(mean_out, trim_out, atol=1e-6)


# ---------------------------------------------------------------------------
# CoordMedianAggregator
# ---------------------------------------------------------------------------

class TestCoordMedianAggregator:
    def test_output_shape_finite_no_mutation(self):
        agg = CoordMedianAggregator()
        check_shape_finite_no_mutation(agg, [torch.randn(D) for _ in range(N_WORKERS)])

    def test_uniform_input_equals_input(self, uniform_grads):
        agg = CoordMedianAggregator()
        out = agg.aggregate(uniform_grads)
        assert torch.allclose(out, uniform_grads[0])

    def test_known_output(self):
        # Grads [1,2,3,4,5,6,7,8] per coordinate.
        # torch.median of 8 values returns the lower middle (index 3 in sorted) = 4.0
        grads = [torch.full((D,), float(i + 1)) for i in range(N_WORKERS)]
        agg = CoordMedianAggregator()
        out = agg.aggregate(grads)
        assert torch.allclose(out, torch.full((D,), 4.0))

    def test_byzantine_tolerance(self):
        # Median of [1,1,1,1,1,1,1000,1000] = 1.0 (lower middle)
        grads = make_byzantine_grads()
        agg = CoordMedianAggregator()
        out = agg.aggregate(grads)
        assert torch.allclose(out, torch.ones(D)), \
            f"Expected CoordMedian output = 1.0, got {out.mean().item():.3f}"

    def test_single_outlier(self):
        # One very large outlier should not move coordinate-wise median significantly
        grads = [torch.ones(D) for _ in range(N_WORKERS - 1)] + [torch.full((D,), 1e6)]
        out = CoordMedianAggregator().aggregate(grads)
        assert out.mean().item() < 10.0


# ---------------------------------------------------------------------------
# GeometricMedianAggregator
# ---------------------------------------------------------------------------

class TestGeometricMedianAggregator:
    def test_output_shape_finite_no_mutation(self):
        agg = GeometricMedianAggregator()
        check_shape_finite_no_mutation(agg, [torch.randn(D) for _ in range(N_WORKERS)])

    def test_uniform_input_equals_input(self, uniform_grads):
        # Geometric median of identical vectors must be that vector
        agg = GeometricMedianAggregator()
        out = agg.aggregate(uniform_grads)
        assert torch.allclose(out, uniform_grads[0], atol=1e-4)

    def test_clean_grads_close_to_mean(self):
        # With i.i.d. clean grads, geometric median ≈ arithmetic mean
        torch.manual_seed(0)
        grads = [torch.randn(D) for _ in range(N_WORKERS)]
        gm = GeometricMedianAggregator().aggregate(grads)
        mean = MeanAggregator().aggregate(grads)
        # Should be in the same ballpark — not identical, but close in norm
        rel_diff = (gm - mean).norm() / mean.norm()
        assert rel_diff < 0.5, f"Clean-input GeoMedian too far from mean: {rel_diff:.3f}"

    def test_byzantine_robustness(self):
        # Two Byzantine workers sending 1000x outliers.
        # Geometric median should stay close to 1.0 (the clean cluster).
        grads = make_byzantine_grads()
        gm_out  = GeometricMedianAggregator().aggregate(grads)
        mean_out = MeanAggregator().aggregate(grads)
        assert gm_out.mean().item() < 5.0, \
            f"GeoMedian too influenced by outliers: {gm_out.mean().item():.3f}"
        assert gm_out.mean().item() < mean_out.mean().item() / 10, \
            "GeometricMedian should be far less influenced by outliers than Mean"

    def test_convergence_flag_via_eps(self):
        # Very loose eps — should still converge in 1 iteration, just less precise
        grads = [torch.ones(D) for _ in range(N_WORKERS)]
        agg = GeometricMedianAggregator(max_iter=1, eps=1.0)
        out = agg.aggregate(grads)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MultiKrumAggregator
# ---------------------------------------------------------------------------

class TestMultiKrumAggregator:
    def test_output_shape_finite_no_mutation(self):
        agg = MultiKrumAggregator(f=F)
        check_shape_finite_no_mutation(agg, [torch.randn(D) for _ in range(N_WORKERS)])

    def test_uniform_input_close_to_input(self, uniform_grads):
        # All workers identical → any selection gives the same vector
        agg = MultiKrumAggregator(f=F)
        out = agg.aggregate(uniform_grads)
        assert torch.allclose(out, uniform_grads[0])

    def test_byzantine_robustness(self):
        # 2 Byzantine workers sending 1000x vectors; honest workers send ones.
        # MultiKrum should select from the clean cluster.
        grads = make_byzantine_grads()
        agg = MultiKrumAggregator(f=F)
        out = agg.aggregate(grads)
        mean_out = MeanAggregator().aggregate(grads)
        assert out.mean().item() < 5.0, \
            f"MultiKrum output unexpectedly large: {out.mean().item():.3f}"
        assert out.mean().item() < mean_out.mean().item() / 10, \
            "MultiKrum should exclude Byzantine outliers from the final mean"

    def test_plain_krum_m1(self):
        # m=1 (plain Krum): selects a single worker, output = that worker's vector
        grads = make_byzantine_grads()
        agg = MultiKrumAggregator(f=F, m=1)
        out = agg.aggregate(grads)
        # The selected worker should be one of the clean ones (norm ≈ sqrt(D))
        # rather than a Byzantine one (norm ≈ 1000 * sqrt(D))
        clean_norm = torch.ones(D).norm().item()
        assert abs(out.norm().item() - clean_norm) < 1.0, \
            "Plain Krum should select a clean worker's vector"

    def test_f4_warns_but_runs(self):
        # f=4 violates 2f+2 < n but n-f-2=2 > 0, so it should warn and run
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agg = MultiKrumAggregator(f=4, n_workers=8)
            assert any("theoretical" in str(warning.message).lower() for warning in w), \
                "Expected a warning about theoretical guarantees"
        out = agg.aggregate([torch.randn(D) for _ in range(N_WORKERS)])
        assert torch.isfinite(out).all()

    def test_invalid_f_raises(self):
        # f=7 → n-f-2 = 8-7-2 = -1 ≤ 0 → ValueError
        with pytest.raises(ValueError, match="n-f-2"):
            MultiKrumAggregator(f=7, n_workers=8)
