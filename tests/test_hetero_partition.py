"""
Correctness tests for the heterogeneous (non-IID) Dirichlet partition.

The partition is the correctness-critical core of the hetero perturbation: if the split
isn't actually skewed at low alpha (or not uniform at high alpha), the whole condition is
inert. These run on CPU with synthetic cluster ids — no data / GPU needed.
"""

import numpy as np
import torch

from simulation.hetero_data import (
    dirichlet_partition,
    partition_blocks_to_loaders,
    HeterogeneousShadedDataset,
    tokens_to_blocks,
)


def _make_cluster_ids(n_blocks=4000, n_clusters=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=n_blocks)


def _worker_top_cluster_share(cluster_ids, assignment, n_workers):
    """For each worker, the fraction of its blocks belonging to its single most-common cluster."""
    shares = []
    for r in range(n_workers):
        blocks = cluster_ids[assignment == r]
        if len(blocks) == 0:
            shares.append(1.0)  # degenerate; treat as maximally concentrated
            continue
        _, counts = np.unique(blocks, return_counts=True)
        shares.append(counts.max() / len(blocks))
    return np.array(shares)


# ---------------------------------------------------------------------------
# 1. It's a true partition — every block assigned exactly once.
# ---------------------------------------------------------------------------

def test_partition_is_complete_and_disjoint():
    cluster_ids = _make_cluster_ids()
    n_workers = 8
    a = dirichlet_partition(cluster_ids, n_workers, alpha=0.5, seed=42)
    assert a.shape == (len(cluster_ids),)
    assert a.min() >= 0 and a.max() < n_workers
    # Every block assigned (no -1), and total count conserved.
    assert (a >= 0).all()
    assert sum((a == r).sum() for r in range(n_workers)) == len(cluster_ids)


# ---------------------------------------------------------------------------
# 2. Low alpha => highly skewed; high alpha => near-uniform.
# ---------------------------------------------------------------------------

def test_low_alpha_is_more_skewed_than_high_alpha():
    cluster_ids = _make_cluster_ids(n_blocks=8000, n_clusters=10)
    n_workers = 8
    a_lo = dirichlet_partition(cluster_ids, n_workers, alpha=0.1, seed=1)
    a_hi = dirichlet_partition(cluster_ids, n_workers, alpha=1.0, seed=1)

    skew_lo = _worker_top_cluster_share(cluster_ids, a_lo, n_workers).mean()
    skew_hi = _worker_top_cluster_share(cluster_ids, a_hi, n_workers).mean()

    # Under strong skew each worker is dominated by one cluster => high top-share.
    assert skew_lo > 0.5, f"alpha=0.1 not skewed enough (mean top-share {skew_lo:.2f})"
    # Under alpha=1.0 workers see a mix => much lower top-share than the 0.1 case.
    assert skew_hi < skew_lo - 0.15, f"alpha=1.0 ({skew_hi:.2f}) not clearly less skewed than 0.1 ({skew_lo:.2f})"


def test_high_alpha_workers_see_many_clusters():
    cluster_ids = _make_cluster_ids(n_blocks=8000, n_clusters=10)
    n_workers = 8
    a = dirichlet_partition(cluster_ids, n_workers, alpha=1.0, seed=3)
    # At alpha=1.0 every worker should see most clusters (near-IID).
    for r in range(n_workers):
        seen = np.unique(cluster_ids[a == r])
        assert len(seen) >= 7, f"worker {r} saw only {len(seen)}/10 clusters at alpha=1.0"


# ---------------------------------------------------------------------------
# 3. Determinism.
# ---------------------------------------------------------------------------

def test_determinism_same_seed_alpha():
    cluster_ids = _make_cluster_ids()
    a1 = dirichlet_partition(cluster_ids, 8, alpha=0.5, seed=42)
    a2 = dirichlet_partition(cluster_ids, 8, alpha=0.5, seed=42)
    assert np.array_equal(a1, a2)


def test_different_seed_differs():
    cluster_ids = _make_cluster_ids()
    a1 = dirichlet_partition(cluster_ids, 8, alpha=0.5, seed=42)
    a2 = dirichlet_partition(cluster_ids, 8, alpha=0.5, seed=7)
    assert not np.array_equal(a1, a2)


# ---------------------------------------------------------------------------
# 4. End-to-end loader building: min-per-worker guard + shapes.
# ---------------------------------------------------------------------------

def test_partition_blocks_to_loaders_shapes_and_guard():
    n_blocks, seq_len, n_workers, batch_size = 4000, 16, 8, 4
    tokens = torch.arange(n_blocks * seq_len, dtype=torch.long)
    blocks = tokens_to_blocks(tokens, seq_len)
    assert blocks.shape == (n_blocks, seq_len)

    cluster_ids = _make_cluster_ids(n_blocks=n_blocks, n_clusters=10)
    loaders = partition_blocks_to_loaders(
        blocks, cluster_ids, n_workers, batch_size, alpha=0.1, seed=42
    )
    assert len(loaders) == n_workers

    total = 0
    for ld in loaders:
        ds = ld.dataset
        assert isinstance(ds, HeterogeneousShadedDataset)
        assert len(ds) >= batch_size  # min-per-worker guard held
        item = ds[0]
        assert item.shape == (seq_len,)
        total += len(ds)
    # No blocks invented or lost by the guard (it only moves them).
    assert total == n_blocks


def test_determinism_of_loader_partition():
    n_blocks, seq_len = 4000, 16
    blocks = tokens_to_blocks(torch.arange(n_blocks * seq_len, dtype=torch.long), seq_len)
    cluster_ids = _make_cluster_ids(n_blocks=n_blocks)
    l1 = partition_blocks_to_loaders(blocks, cluster_ids, 8, 4, alpha=0.3, seed=5)
    l2 = partition_blocks_to_loaders(blocks, cluster_ids, 8, 4, alpha=0.3, seed=5)
    for a, b in zip(l1, l2):
        assert torch.equal(a.dataset.blocks, b.dataset.blocks)
