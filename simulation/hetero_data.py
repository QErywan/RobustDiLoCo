"""
Heterogeneous (non-IID) data partitioning for the DiLoCo simulation.

This is the data-level "hetero" perturbation: instead of corrupting pseudo-gradients
(magnitude/gaussian) or dropping workers (dropout), each worker is given a *non-IID*
slice of the corpus, so honest workers legitimately train on different distributions
and their pseudo-gradients diverge in direction. This is the natural / non-adversarial
end of the perturbation axis (f = 0 — no Byzantine worker; the skew *is* the perturbation).

Method (standard adaptation of Dirichlet non-IID to unlabeled LM data):
  1. Each `seq_len` token block is assigned a cluster id (a pseudo-label) by k-means in
     a pretrained model's feature space — precomputed offline by
     `experiments/cluster_shards.py` and cached as `cluster_ids.npy`.
  2. For each cluster c, worker proportions p_c ~ Dir(alpha) are drawn, and cluster-c
     blocks are split among workers by p_c. Low alpha -> each worker dominated by a few
     clusters (very heterogeneous); high alpha -> near-IID. This is the NIID-Bench
     distribution-based label-skew recipe (Li et al., ICDE 2022) with clusters as labels.

This module lives in the `simulation/` thesis layer. It reuses upstream
`tplr.data.ShadedDataset` (world_size=1) to load the token stream *without modifying it*.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# The pure partition function — the correctness-critical, unit-tested core.
# ---------------------------------------------------------------------------

def dirichlet_partition(
    cluster_ids: np.ndarray,
    n_workers: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """
    Assign each block to a worker via Dirichlet(alpha) skew over clusters.

    Args:
        cluster_ids: int array, one cluster id per block (length = n_blocks).
        n_workers:   number of workers to partition across.
        alpha:       Dirichlet concentration. Low (0.1) -> highly skewed / heterogeneous;
                     high (1.0+) -> near-uniform / IID.
        seed:        RNG seed — same (alpha, seed) gives an identical partition.

    Returns:
        int array of length n_blocks, each entry in [0, n_workers), giving the worker
        each block is assigned to. Every block is assigned exactly once (a true
        partition); a worker may receive zero blocks under extreme skew (callers that
        need a training guarantee should handle that — see partition_blocks_to_loaders).
    """
    cluster_ids = np.asarray(cluster_ids)
    n_blocks = len(cluster_ids)
    assignment = np.full(n_blocks, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    # Sort clusters for deterministic iteration order (unique() already sorts).
    for c in np.unique(cluster_ids):
        c_idx = np.where(cluster_ids == c)[0]
        rng.shuffle(c_idx)  # in-place, deterministic under rng
        p = rng.dirichlet(np.full(n_workers, alpha))
        # Cumulative boundaries guarantee a full, non-overlapping split summing to len(c_idx).
        bounds = (np.cumsum(p) * len(c_idx)).astype(np.int64)
        bounds[-1] = len(c_idx)  # last worker absorbs any rounding remainder
        start = 0
        for r in range(n_workers):
            end = int(bounds[r])
            if end > start:
                assignment[c_idx[start:end]] = r
            start = end

    # Sanity: every block assigned (no cluster empty of blocks would leave -1).
    if (assignment < 0).any():
        # Only possible if a cluster id had zero blocks — cannot happen from unique(),
        # but guard defensively so a bug surfaces loudly rather than silently mis-trains.
        raise RuntimeError("dirichlet_partition left blocks unassigned")
    return assignment


# ---------------------------------------------------------------------------
# Dataset — a thin view over one worker's pre-assigned blocks.
# ---------------------------------------------------------------------------

class HeterogeneousShadedDataset(Dataset):
    """
    Holds one worker's assigned token blocks as a (num_samples, seq_len) tensor.

    Same interface as tplr.data.ShadedDataset (`__getitem__` returns a 1-D seq_len
    LongTensor), so the DataLoader / Worker path is unchanged.
    """

    def __init__(self, blocks: torch.Tensor):
        super().__init__()
        if blocks.ndim != 2:
            raise ValueError(f"blocks must be 2-D (n, seq_len); got shape {tuple(blocks.shape)}")
        self.blocks = blocks
        self.num_samples = blocks.shape[0]
        self.pin_to_gpu = False  # parity with ShadedDataset attr read by get_dataloader

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.blocks[idx]


# ---------------------------------------------------------------------------
# Loader builder — partitions blocks and returns one DataLoader per worker.
# ---------------------------------------------------------------------------

def partition_blocks_to_loaders(
    blocks: torch.Tensor,
    cluster_ids: np.ndarray,
    n_workers: int,
    batch_size: int,
    alpha: float,
    seed: int,
) -> list[DataLoader]:
    """
    Dirichlet-partition `blocks` across workers by cluster and build a DataLoader each.

    Guarantees each worker receives at least `batch_size` blocks (so every worker can
    form at least one batch) — under extreme skew a worker may draw zero blocks from the
    Dirichlet split, so any starved worker is topped up from the most-loaded worker, with
    a warning. This is a training-robustness guard; the reported skew is essentially
    unchanged when n_blocks >> n_workers.
    """
    n_blocks = blocks.shape[0]
    cluster_ids = np.asarray(cluster_ids)[:n_blocks]
    if len(cluster_ids) != n_blocks:
        raise ValueError(
            f"cluster_ids length {len(cluster_ids)} < n_blocks {n_blocks} — "
            "clustering did not cover the whole token budget"
        )

    assignment = dirichlet_partition(cluster_ids, n_workers, alpha, seed)

    # Per-worker block indices.
    worker_idx = [np.where(assignment == r)[0] for r in range(n_workers)]

    # Guard: guarantee each worker >= batch_size blocks so it can form a batch.
    rng = np.random.default_rng(seed + 1)
    counts = [len(ix) for ix in worker_idx]
    for r in range(n_workers):
        while counts[r] < batch_size:
            donor = int(np.argmax(counts))
            if donor == r or counts[donor] <= batch_size:
                # No donor can spare blocks — corpus too small for this config.
                raise ValueError(
                    f"Worker {r} has {counts[r]} blocks (< batch_size={batch_size}) and no "
                    f"donor can spare any. n_blocks={n_blocks} too small for n_workers="
                    f"{n_workers}, alpha={alpha}."
                )
            take = rng.choice(worker_idx[donor], size=1)
            worker_idx[donor] = np.setdiff1d(worker_idx[donor], take, assume_unique=False)
            worker_idx[r] = np.concatenate([worker_idx[r], take])
            counts[donor] -= 1
            counts[r] += 1
            print(f"[hetero] worker {r} starved — moved 1 block from worker {donor} "
                  f"(guard: min {batch_size}/worker)")

    # Partition-sanity log — the "did heterogeneity actually happen" gate.
    # Prints each worker's block count and top-cluster share (fraction of its blocks in
    # its single most-common cluster). Low alpha -> shares near 1.0 (skewed); high alpha
    # -> shares near 1/n_clusters (near-IID).
    n_clusters = len(np.unique(cluster_ids))
    shares = []
    for r in range(n_workers):
        cl = cluster_ids[worker_idx[r]]
        if len(cl) == 0:
            shares.append(1.0); continue
        _, cnts = np.unique(cl, return_counts=True)
        shares.append(cnts.max() / len(cl))
    print(f"[hetero] alpha={alpha} seed={seed} | {n_clusters} clusters, "
          f"{n_blocks} blocks | per-worker top-cluster share "
          f"(1.0=fully skewed, {1/n_clusters:.2f}=uniform): "
          f"[{', '.join(f'{s:.2f}' for s in shares)}]  mean={np.mean(shares):.2f}")

    loaders = []
    for r in range(n_workers):
        ds = HeterogeneousShadedDataset(blocks[worker_idx[r]])
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders


def load_or_make_cluster_ids(
    n_blocks: int,
    cluster_ids_path: str | None,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """
    Load cached cluster ids, or (smoke/no-cache fallback) synthesise random ones.

    Real runs pass a path to `cluster_ids.npy` produced by cluster_shards.py. When no
    path exists (smoke tests, synthetic data), random cluster ids let the plumbing run
    end-to-end — the partition is still valid, just not topically meaningful. A warning
    makes clear this is not a real heterogeneity clustering.
    """
    if cluster_ids_path is not None:
        ids = np.load(cluster_ids_path)
        if len(ids) < n_blocks:
            raise ValueError(
                f"cached cluster_ids has {len(ids)} entries < n_blocks {n_blocks}"
            )
        return ids[:n_blocks]
    print("[hetero] WARNING: no cluster_ids cache — using RANDOM cluster ids "
          "(smoke/plumbing only, NOT a real feature-clustering).")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=n_blocks)


def tokens_to_blocks(tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Chop a 1-D token stream into a (n_blocks, seq_len) tensor (drops the remainder)."""
    n_blocks = tokens.shape[0] // seq_len
    if n_blocks == 0:
        raise ValueError(f"token stream length {tokens.shape[0]} < seq_len {seq_len}")
    return tokens[: n_blocks * seq_len].reshape(n_blocks, seq_len)
