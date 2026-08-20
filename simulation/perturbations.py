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
    Simulate straggler / dropped workers. Two models of a dropped worker, selected
    by `mode`:

      - "zero"    (default): replace the f dropped workers' pseudo-gradients with zero
                  vectors. The aggregator still receives n vectors, f of them at the
                  origin. This is the original model.
      - "exclude": remove the f dropped workers entirely; the aggregator receives only
                  the n-f surviving vectors. This is how FedAvg and real federated
                  systems handle a non-responding client.

    At f=4 the zeroing model puts half the inputs at the origin, where a geometric
    median (RFA) collapses; exclusion leaves it unaffected. Running both distinguishes
    a real effect from an artefact of the zeroing choice.

    Natural (non-adversarial) perturbation; the aggregator receives fewer informative
    updates. Thesis severity levels: f in {1, 2, 4}. Designed for per-step SGD dropout
    (Blanchard et al., 2017) and applied here at H=500 without a published guarantee.
    """

    def __init__(self, n_workers: int, f: int, mode: str = "zero"):
        if f >= n_workers:
            raise ValueError(f"f={f} must be < n_workers={n_workers}")
        if mode not in ("zero", "exclude"):
            raise ValueError(f"mode must be 'zero' or 'exclude', got {mode!r}")
        self.n_workers = n_workers
        self.f = f
        self.mode = mode

    def apply(self, pseudo_grads: list[Tensor]) -> list[Tensor]:
        if self.mode == "exclude":
            # Drop the last f workers outright — the aggregator sees n-f survivors.
            return list(pseudo_grads[: self.n_workers - self.f])
        # "zero": keep n vectors, put the last f at the origin.
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
