"""
Perturbation types for DiLoCo pseudo-gradient experiments.

Interface: apply(pseudo_grads: list[Tensor]) -> list[Tensor]
    - Input:  list of n flat tensors, one per worker
    - Output: list of n flat tensors (same length), possibly modified

Perturbations are applied BEFORE aggregation inside Simulation.run_outer_step.
They must not modify the input list in place — return a new list.

Byzantine fraction f is set at construction. For n=8 workers the thesis uses
f ∈ {1, 2, 4} (12.5%, 25%, 50%). Byzantine workers are always the last f
workers in the list (indices n-f to n-1) — this is a simulation convention,
not a real adversarial assumption.
"""

import torch
from torch import Tensor


class NoPerturbation:
    """Pass-through — used for clean baseline runs."""

    def apply(self, pseudo_grads: list[Tensor]) -> list[Tensor]:
        return list(pseudo_grads)


class WorkerDropout:
    """
    Simulate straggler / dropped workers by replacing f workers' pseudo-
    gradients with zeros. Corresponds to those workers failing to communicate.

    This is a natural perturbation (not adversarial) — the aggregator receives
    fewer informative updates. Severity levels in the thesis: f ∈ {1, 2, 4}.

    Note: designed for per-step SGD dropout analysis (Blanchard et al., 2017),
    but used here as a natural fault model. No published theoretical guarantee
    for the H=500 pseudo-gradient regime.
    """

    def __init__(self, n_workers: int, f: int):
        if f >= n_workers:
            raise ValueError(f"f={f} must be < n_workers={n_workers}")
        self.n_workers = n_workers
        self.f = f

    def apply(self, pseudo_grads: list[Tensor]) -> list[Tensor]:
        result = list(pseudo_grads)
        for i in range(self.n_workers - self.f, self.n_workers):
            result[i] = torch.zeros_like(pseudo_grads[i])
        return result


class GaussianNoise:
    """
    Add isotropic Gaussian noise to f workers' pseudo-gradients.

    Noise scale sigma is set relative to the mean pseudo-gradient norm across
    all workers so severity is comparable across model sizes and training stages.

    Natural perturbation — models gradient estimation noise, hardware faults,
    or stale updates. No Byzantine adversary assumed.
    """

    def __init__(self, n_workers: int, f: int, sigma_scale: float):
        """
        Args:
            sigma_scale: noise std = sigma_scale * mean_norm(pseudo_grads).
                         Thesis severity levels: sigma_scale ∈ {0.1, 0.5, 1.0}.
        """
        if f >= n_workers:
            raise ValueError(f"f={f} must be < n_workers={n_workers}")
        self.n_workers = n_workers
        self.f = f
        self.sigma_scale = sigma_scale

    def apply(self, pseudo_grads: list[Tensor]) -> list[Tensor]:
        mean_norm = sum(g.norm().item() for g in pseudo_grads) / len(pseudo_grads)
        sigma = self.sigma_scale * mean_norm

        result = list(pseudo_grads)
        for i in range(self.n_workers - self.f, self.n_workers):
            noise = torch.randn_like(pseudo_grads[i]) * sigma
            result[i] = pseudo_grads[i] + noise
        return result


class MagnitudeAttack:
    """
    Adversarial attack: f Byzantine workers scale their pseudo-gradient norm
    by a large factor, overwhelming the aggregated mean.

    Directly targets FedAvg's breakdown point (Blanchard et al., 2017, Lemma 1).
    The thesis severity levels are scale ∈ {10, 100, 1000}.

    Byzantine workers know the aggregation rule but not other workers' updates
    (non-adaptive threat model — see wiki/concepts/byzantine-robustness.md).
    """

    def __init__(self, n_workers: int, f: int, scale: float):
        if f >= n_workers:
            raise ValueError(f"f={f} must be < n_workers={n_workers}")
        self.n_workers = n_workers
        self.f = f
        self.scale = scale

    def apply(self, pseudo_grads: list[Tensor]) -> list[Tensor]:
        result = list(pseudo_grads)
        for i in range(self.n_workers - self.f, self.n_workers):
            result[i] = pseudo_grads[i] * self.scale
        return result
