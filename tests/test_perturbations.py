"""
Tests for simulation/perturbations.py.

Each perturbation is tested with known inputs and asserted outputs.
n=8 workers, d=512 dimensional pseudo-gradients throughout.
"""

import torch
import pytest
from simulation.perturbations import (
    NoPerturbation,
    WorkerDropout,
    GaussianNoise,
    MagnitudeAttack,
)

N = 8
D = 512


def _unit_grads(n=N, d=D):
    """All workers send a unit vector — easy to reason about norms."""
    g = torch.ones(d) / (d ** 0.5)
    return [g.clone() for _ in range(n)]


def _randn_grads(n=N, d=D):
    return [torch.randn(d) for _ in range(n)]


# ---------------------------------------------------------------------------
# NoPerturbation
# ---------------------------------------------------------------------------

class TestNoPerturbation:
    def setup_method(self):
        self.p = NoPerturbation()

    def test_returns_same_values(self):
        grads = _unit_grads()
        out = self.p.apply(grads)
        for orig, result in zip(grads, out):
            assert torch.allclose(orig, result)

    def test_returns_new_list(self):
        grads = _unit_grads()
        out = self.p.apply(grads)
        assert out is not grads

    def test_length_preserved(self):
        grads = _randn_grads()
        assert len(self.p.apply(grads)) == N


# ---------------------------------------------------------------------------
# WorkerDropout
# ---------------------------------------------------------------------------

class TestWorkerDropout:
    def test_honest_workers_unchanged(self):
        p = WorkerDropout(n_workers=N, f=2)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2):
            assert torch.allclose(out[i], grads[i])

    def test_dropped_workers_are_zero(self):
        p = WorkerDropout(n_workers=N, f=2)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2, N):
            assert torch.allclose(out[i], torch.zeros(D))

    def test_length_preserved(self):
        p = WorkerDropout(n_workers=N, f=1)
        assert len(p.apply(_unit_grads())) == N

    def test_f_equal_n_raises(self):
        with pytest.raises(ValueError):
            WorkerDropout(n_workers=N, f=N)

    def test_does_not_mutate_input(self):
        p = WorkerDropout(n_workers=N, f=2)
        grads = _unit_grads()
        originals = [g.clone() for g in grads]
        p.apply(grads)
        for orig, after in zip(originals, grads):
            assert torch.allclose(orig, after)


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    def test_honest_workers_unchanged(self):
        p = GaussianNoise(n_workers=N, f=2, sigma_scale=1.0)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2):
            assert torch.allclose(out[i], grads[i])

    def test_noisy_workers_differ_from_input(self):
        p = GaussianNoise(n_workers=N, f=2, sigma_scale=10.0)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2, N):
            assert not torch.allclose(out[i], grads[i])

    def test_output_finite(self):
        p = GaussianNoise(n_workers=N, f=4, sigma_scale=1.0)
        out = p.apply(_randn_grads())
        for g in out:
            assert torch.isfinite(g).all()

    def test_length_preserved(self):
        p = GaussianNoise(n_workers=N, f=2, sigma_scale=0.1)
        assert len(p.apply(_unit_grads())) == N

    def test_zero_sigma_scale_returns_input(self):
        p = GaussianNoise(n_workers=N, f=2, sigma_scale=0.0)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2, N):
            assert torch.allclose(out[i], grads[i])

    def test_does_not_mutate_input(self):
        p = GaussianNoise(n_workers=N, f=2, sigma_scale=1.0)
        grads = _unit_grads()
        originals = [g.clone() for g in grads]
        p.apply(grads)
        for orig, after in zip(originals, grads):
            assert torch.allclose(orig, after)


# ---------------------------------------------------------------------------
# MagnitudeAttack
# ---------------------------------------------------------------------------

class TestMagnitudeAttack:
    def test_honest_workers_unchanged(self):
        p = MagnitudeAttack(n_workers=N, f=2, scale=100.0)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2):
            assert torch.allclose(out[i], grads[i])

    def test_byzantine_workers_scaled(self):
        p = MagnitudeAttack(n_workers=N, f=2, scale=100.0)
        grads = _unit_grads()
        out = p.apply(grads)
        for i in range(N - 2, N):
            assert torch.allclose(out[i], grads[i] * 100.0)

    def test_scale_1_is_identity_for_all(self):
        p = MagnitudeAttack(n_workers=N, f=4, scale=1.0)
        grads = _unit_grads()
        out = p.apply(grads)
        for orig, result in zip(grads, out):
            assert torch.allclose(orig, result)

    def test_norm_ratio_matches_scale(self):
        scale = 1000.0
        p = MagnitudeAttack(n_workers=N, f=1, scale=scale)
        grads = _randn_grads()
        out = p.apply(grads)
        ratio = out[-1].norm() / grads[-1].norm()
        assert torch.isclose(ratio, torch.tensor(scale), rtol=1e-4)

    def test_length_preserved(self):
        p = MagnitudeAttack(n_workers=N, f=2, scale=10.0)
        assert len(p.apply(_unit_grads())) == N

    def test_f_equal_n_raises(self):
        with pytest.raises(ValueError):
            MagnitudeAttack(n_workers=N, f=N, scale=10.0)

    def test_does_not_mutate_input(self):
        p = MagnitudeAttack(n_workers=N, f=2, scale=100.0)
        grads = _unit_grads()
        originals = [g.clone() for g in grads]
        p.apply(grads)
        for orig, after in zip(originals, grads):
            assert torch.allclose(orig, after)
