"""
Tests for simulation/aggregators.py.

Strategy: use synthetic inputs with known closed-form outputs so each test
is a direct numerical assertion, not just a shape/type check.
"""

import torch
import pytest
from simulation.aggregators import MeanAggregator

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
