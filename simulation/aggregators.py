"""
Aggregation rules for DiLoCo pseudo-gradient aggregation.

Interface: aggregate(pseudo_grads: list[Tensor]) -> Tensor
    - Input: list of n flat tensors, one per worker, all same shape (d,)
    - Output: single flat tensor (d,) — the aggregated pseudo-gradient

All aggregators operate on flat pseudo-gradient vectors. The Simulation
orchestrator stacks/unstacks as needed; aggregators are architecture-agnostic.

Design note on setting appropriateness
---------------------------------------
Krum, Trimmed Mean, and Coordinate-wise Median were designed for per-step
distributed SGD (τ=1 local update). RFA (geometric median via Weiszfeld) is
the only aggregator here with theoretical guarantees for FL with multiple local
update steps (τ>1), making it the closest match to DiLoCo's H=500 inner AdamW
regime. Results for Krum/Trimmed Mean/CoordMedian in this setting should be
interpreted empirically, not as direct applications of their published
theoretical guarantees.

References:
    - Pillutla et al. (2022) "Robust Aggregation for Federated Learning"
      https://arxiv.org/abs/1912.13445  (RFA / geometric median)
    - Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine
      Tolerant Gradient Descent"  https://arxiv.org/abs/1703.02757  (Krum)
    - Yin et al. (2018) "Byzantine-Robust Distributed Learning: Towards
      Optimal Statistical Rates"  https://arxiv.org/abs/1803.01498
      (Trimmed Mean and Coordinate-wise Median)
"""

import warnings

import torch
from torch import Tensor


class MeanAggregator:
    """
    Plain weighted mean — DiLoCo's default aggregation rule (FedAvg).

    Breakdown point: 0. A single Byzantine worker can drive the aggregate to
    any arbitrary vector (Blanchard et al., 2017, Lemma 1).

    This is the baseline all other aggregators are compared against.
    """

    def aggregate(self, pseudo_grads: list[Tensor]) -> Tensor:
        stacked = torch.stack(pseudo_grads)   # (n, d)
        return stacked.mean(dim=0)


class TrimmedMeanAggregator:
    """
    Coordinate-wise trimmed mean (Yin et al., 2018, Algorithm 1).

    Per coordinate, discards the f largest and f smallest values across workers
    and averages the remaining n-2f values. Breakdown point f/(n-2f+1) under
    Theorem 1 of Yin et al. (2018), assuming i.i.d. gradients and per-step
    SGD (τ=1).

    Validity constraint: requires 2f < n (at least one un-trimmed value per
    coordinate). For n=8, valid thesis Byzantine fractions are f ∈ {1, 2, 3}.
    At f=4 (=n/2) no values would remain; f is automatically capped to
    min(f, n//2 - 1) with a warning.

    NOTE on setting: Designed for per-step SGD (τ=1). Applied empirically here
    to DiLoCo's H=500 pseudo-gradients; theoretical guarantees do not carry
    over to the multi-step regime. See module docstring.

    Reference: Yin et al. (2018), "Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates." https://arxiv.org/abs/1803.01498
    """

    def __init__(self, f: int, n_workers: int = 8):
        """
        Args:
            f: number of Byzantine workers to guard against (used as trim width).
            n_workers: total worker count n. Used to validate f and issue a
                       warning if f >= n//2.
        """
        if f < 0:
            raise ValueError(f"f must be >= 0, got {f}")
        max_valid_f = n_workers // 2 - 1
        if f > max_valid_f:
            warnings.warn(
                f"TrimmedMeanAggregator: f={f} >= n/2 (n={n_workers}) leaves "
                f"zero un-trimmed values. Capping to f={max_valid_f}. "
                f"Results at this f are degenerate; interpret with caution.",
                UserWarning,
                stacklevel=2,
            )
            f = max_valid_f
        self.f = f

    def aggregate(self, pseudo_grads: list[Tensor]) -> Tensor:
        stacked = torch.stack(pseudo_grads)            # (n, d)
        n = stacked.shape[0]
        sorted_grads, _ = stacked.sort(dim=0)          # sort along worker axis
        trimmed = sorted_grads[self.f : n - self.f]    # (n-2f, d)
        return trimmed.mean(dim=0)


class CoordMedianAggregator:
    """
    Coordinate-wise median (Yin et al., 2018, Algorithm 2).

    Computes the per-coordinate median across workers. Equivalent to Trimmed
    Mean with f = ⌊n/2⌋ - 1 and a single remaining value, but more stable
    because it does not require choosing f explicitly.

    Breakdown point: ⌊(n-1)/2⌋ / n under Theorem 2 of Yin et al. (2018),
    assuming i.i.d. gradients and τ=1. For n=8 this is ~43% Byzantine
    tolerance.

    NOTE on setting: Designed for per-step SGD (τ=1). Applied empirically here
    to DiLoCo's H=500 pseudo-gradients; theoretical guarantees do not carry
    over to the multi-step regime. See module docstring.

    Reference: Yin et al. (2018), "Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates." https://arxiv.org/abs/1803.01498
    """

    def aggregate(self, pseudo_grads: list[Tensor]) -> Tensor:
        stacked = torch.stack(pseudo_grads)   # (n, d)
        return stacked.median(dim=0).values


class GeometricMedianAggregator:
    """
    Geometric median via Weiszfeld's algorithm — also known as RFA
    (Robust Federated Aggregation, Pillutla et al., 2022).

    Computes argmin_{mu} sum_i ||mu - g_i||_2 via iterative reweighting:
        w_i^(t) = 1 / max(||mu^(t) - g_i||_2, eps)
        mu^(t+1) = (sum_i w_i^(t) g_i) / (sum_i w_i^(t))

    Initialised at the arithmetic mean; iterates until convergence (change <
    eps) or max_iter is reached.

    This is the ONLY aggregator here with theoretical guarantees for the
    multi-step FL regime (τ>1). Theorem 1 of Pillutla et al. (2022) shows
    that RFA converges for any number of local update steps H, making it the
    most principled choice for DiLoCo's H=500 regime. See module docstring.

    Implementation based on Pillutla et al. (2022) and the reference code at
    github.com/krishnap25/RFA.

    Reference: Pillutla et al. (2022), "Robust Aggregation for Federated
    Learning." https://arxiv.org/abs/1912.13445
    """

    def __init__(self, max_iter: int = 100, eps: float = 1e-5):
        """
        Args:
            max_iter: maximum Weiszfeld iterations.
            eps: convergence threshold (stopping criterion: ||mu_new - mu|| < eps)
                 and minimum denominator for inverse-distance weights.
        """
        self.max_iter = max_iter
        self.eps = eps

    def aggregate(self, pseudo_grads: list[Tensor]) -> Tensor:
        stacked = torch.stack(pseudo_grads)   # (n, d)

        # Initialise with arithmetic mean (warm start, speeds convergence)
        mu = stacked.mean(dim=0)              # (d,)

        for _ in range(self.max_iter):
            # Per-worker L2 distance to current estimate: (n,)
            dists = torch.norm(stacked - mu.unsqueeze(0), dim=1)

            # Inverse-distance weights, clamped to avoid division by zero
            weights = 1.0 / torch.clamp(dists, min=self.eps)   # (n,)
            weights = weights / weights.sum()                    # normalise to sum=1

            mu_new = (weights.unsqueeze(1) * stacked).sum(dim=0)   # (d,)

            if torch.norm(mu_new - mu).item() < self.eps:
                mu = mu_new
                break
            mu = mu_new

        return mu


class MultiKrumAggregator:
    """
    Multi-Krum aggregator (Blanchard et al., 2017, Algorithm 1).

    For each worker i, computes a score:
        s(i) = sum of squared L2 distances to the (n-f-2) nearest other workers.
    Selects the m workers with the smallest scores and returns their mean.
    Setting m=1 recovers plain Krum; m=n-f (the default) is Multi-Krum.

    The score function is specifically designed so that Byzantine workers (whose
    pseudo-gradients are far from the honest cluster) accumulate large scores and
    are excluded from the final mean.

    Validity constraint: Blanchard et al. require n >= 2f+3, equivalently
    n-f-2 >= f+1 > 0. For n=8 this limits f to {1, 2} for full theoretical
    guarantees. At f=3 (n-f-2=3 >= 1) the algorithm runs but the Blanchard
    guarantee is not satisfied; at f=4 (n-f-2=2 >= 1) likewise. A warning is
    issued when 2f+2 >= n. The algorithm still runs as long as n-f-2 > 0.

    NOTE on setting: Designed for per-step SGD (τ=1). Applied empirically here
    to DiLoCo's H=500 pseudo-gradients; theoretical guarantees do not carry
    over to the multi-step regime. See module docstring.

    Reference: Blanchard et al. (2017), "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent." https://arxiv.org/abs/1703.02757
    """

    def __init__(self, f: int, n_workers: int = 8, m: int | None = None):
        """
        Args:
            f: number of Byzantine workers assumed. Should satisfy n >= 2f+3
               for theoretical guarantees; must satisfy n-f-2 > 0 to run.
            n_workers: total number of workers n.
            m: number of workers to select for the final mean.
               Defaults to n-f (standard Multi-Krum). Use m=1 for plain Krum.
        """
        n_neighbors = n_workers - f - 2
        if n_neighbors <= 0:
            raise ValueError(
                f"MultiKrumAggregator: n-f-2={n_neighbors} <= 0 for n={n_workers}, "
                f"f={f}. Reduce f or increase n_workers."
            )
        if 2 * f + 2 >= n_workers:
            warnings.warn(
                f"MultiKrumAggregator: 2f+2={2*f+2} >= n={n_workers}. "
                f"Blanchard et al. (2017) require n >= 2f+3; theoretical "
                f"Byzantine-resilience guarantees do not hold at this f. "
                f"Results are empirical only.",
                UserWarning,
                stacklevel=2,
            )
        self.f = f
        self.n_workers = n_workers
        self.n_neighbors = n_neighbors
        self.m = m if m is not None else (n_workers - f)

    def aggregate(self, pseudo_grads: list[Tensor]) -> Tensor:
        n = len(pseudo_grads)
        stacked = torch.stack(pseudo_grads)   # (n, d)

        # Pairwise squared L2 distances (n, n) via expansion identity:
        #   ||g_i - g_j||^2 = ||g_i||^2 + ||g_j||^2 - 2 * g_i . g_j
        # Avoids the (n, n, d) intermediate that OOMs at d=30M+ parameters.
        norms_sq = (stacked ** 2).sum(dim=1)                            # (n,)
        sq_dists = (norms_sq.unsqueeze(1) + norms_sq.unsqueeze(0)
                    - 2.0 * (stacked @ stacked.T))                      # (n, n)
        sq_dists.clamp_(min=0.0)                                        # fp rounding guard

        # Zero out the diagonal (self-distance) by setting it to inf so it is
        # never selected as a nearest neighbour.
        sq_dists.fill_diagonal_(float("inf"))

        # Score(i) = sum of the (n-f-2) smallest pairwise squared distances
        sorted_dists, _ = sq_dists.sort(dim=1)                       # (n, n)
        scores = sorted_dists[:, : self.n_neighbors].sum(dim=1)      # (n,)

        # Select m workers with the smallest scores
        _, top_m_idx = scores.topk(self.m, largest=False)
        selected = stacked[top_m_idx]    # (m, d)
        return selected.mean(dim=0)
