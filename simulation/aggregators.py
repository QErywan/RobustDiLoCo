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

Reference: Pillutla et al. (2022) for RFA; Blanchard et al. (2017) for Krum;
Yin et al. (2018) for Trimmed Mean and Coordinate-wise Median.
"""

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
