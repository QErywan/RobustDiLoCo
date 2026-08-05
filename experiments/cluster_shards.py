"""
Offline clustering of tokenized shards → pseudo-labels for the hetero (non-IID) split.

C4 is unlabeled, so standard Dirichlet-over-labels has nothing to key on. The standard
adaptation (and DiLoCo's own non-IID method) is to cluster each `seq_len` token block in
a feature space to create pseudo-labels, then Dirichlet-partition over those clusters
(see simulation/hetero_data.py). This script produces the `cluster_ids.npy` cache that
the hetero loader reads.

Run ONCE per (shards, seq_len, K); it's cached and reused across all 15 hetero cells.

    python -m experiments.cluster_shards --data-path /vol/bitbucket/.../c4_gpt2 \
        --seq-len 1024 --n-clusters 10 --features pretrained --device cuda

Feature options:
    --features pretrained   forward a fixed pretrained GPT-2, mean-pool last hidden state
                            (principled; matches the interim plan / DiLoCo). Needs a GPU
                            for any real corpus size.
    --features histogram    per-block hashed token-frequency histogram (cheap, no model,
                            no GPU). De-risk fallback when the forward pass is too slow.

Self-contained torch k-means (k-means++ init + Lloyd iterations) — no sklearn dependency.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tplr.data import ShadedDataset  # noqa: E402
from experiments.run_baseline import _load_shard_meta  # noqa: E402
from simulation.hetero_data import tokens_to_blocks  # noqa: E402


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def histogram_features(blocks: torch.Tensor, n_buckets: int = 256) -> torch.Tensor:
    """Per-block normalized hashed token-frequency histogram. Cheap, no model/GPU."""
    n = blocks.shape[0]
    feats = torch.zeros(n, n_buckets, dtype=torch.float32)
    hashed = (blocks % n_buckets).long()
    feats.scatter_add_(1, hashed, torch.ones_like(hashed, dtype=torch.float32))
    feats /= feats.sum(dim=1, keepdim=True).clamp_min(1.0)
    return feats


def pretrained_features(blocks: torch.Tensor, model_name: str, device: str,
                        batch_size: int = 64) -> torch.Tensor:
    """Mean-pooled last-hidden-state from a fixed pretrained GPT-2. Needs GPU for scale."""
    from transformers import GPT2Model
    model = GPT2Model.from_pretrained(model_name).to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, blocks.shape[0], batch_size):
            batch = blocks[i:i + batch_size].to(device)
            out = model(input_ids=batch).last_hidden_state  # (b, seq, hidden)
            feats.append(out.mean(dim=1).cpu())  # mean-pool over tokens
            if (i // batch_size) % 50 == 0:
                print(f"  features {i + len(batch)}/{blocks.shape[0]}", flush=True)
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# Self-contained k-means (no sklearn)
# ---------------------------------------------------------------------------

def kmeans(feats: torch.Tensor, k: int, seed: int, n_iter: int = 50) -> np.ndarray:
    """Lloyd's algorithm with k-means++ init. Returns an int cluster id per row."""
    g = torch.Generator().manual_seed(seed)
    n = feats.shape[0]
    if n < k:
        raise ValueError(f"n_blocks {n} < n_clusters {k}")

    # k-means++ init
    centers = torch.empty(k, feats.shape[1])
    first = torch.randint(0, n, (1,), generator=g).item()
    centers[0] = feats[first]
    closest = torch.cdist(feats, centers[:1]).squeeze(1) ** 2
    for c in range(1, k):
        probs = closest / closest.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers[c] = feats[idx]
        d = torch.cdist(feats, centers[c:c + 1]).squeeze(1) ** 2
        closest = torch.minimum(closest, d)

    # Lloyd iterations
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(n_iter):
        new_assign = torch.cdist(feats, centers).argmin(dim=1)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                centers[c] = feats[mask].mean(dim=0)
            else:  # empty cluster — reseed to a random point
                centers[c] = feats[torch.randint(0, n, (1,), generator=g).item()]
    return assign.numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Cluster tokenized shards into pseudo-labels for hetero split.")
    p.add_argument("--data-path", required=True, help="Directory of pre-tokenized .npy shards")
    p.add_argument("--seq-len", type=int, required=True, help="Block length (must match the run's seq_len)")
    p.add_argument("--n-clusters", type=int, default=10)
    p.add_argument("--features", choices=["pretrained", "histogram"], default="pretrained")
    p.add_argument("--feature-model", type=str, default="gpt2", help="pretrained: HF model for features")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=64, help="pretrained: forward batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="Output .npy (default: <data-path>/cluster_ids.npy)")
    p.add_argument("--force", action="store_true", help="Recompute even if the cache exists")
    args = p.parse_args()

    out = Path(args.out) if args.out else Path(args.data_path) / "cluster_ids.npy"
    meta_out = out.with_name("cluster_meta.json")
    if out.exists() and not args.force:
        print(f"[cluster] {out} already exists — skipping (use --force to recompute).")
        return

    # Load the full token stream (world_size=1 reuses upstream ShadedDataset unmodified).
    meta = _load_shard_meta(args.data_path)
    print(f"[cluster] loading {meta['n_train_tokens']/1e6:.0f}M tokens from {args.data_path}")
    full = ShadedDataset(
        shards_path=args.data_path, token_budget=meta["n_train_tokens"],
        sequence_length=args.seq_len, rank=0, world_size=1,
        device=torch.device(args.device), shard_token_size=meta["tokens_per_shard"], split="train",
    )
    blocks = tokens_to_blocks(full.worker_tokens, args.seq_len)
    print(f"[cluster] {blocks.shape[0]} blocks of length {args.seq_len}")

    if args.features == "histogram":
        print("[cluster] extracting hashed token-histogram features")
        feats = histogram_features(blocks)
    else:
        print(f"[cluster] extracting pretrained features from {args.feature_model}")
        feats = pretrained_features(blocks, args.feature_model, args.device, args.batch_size)

    print(f"[cluster] k-means: K={args.n_clusters}")
    cluster_ids = kmeans(feats, args.n_clusters, args.seed)

    _, counts = np.unique(cluster_ids, return_counts=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, cluster_ids)
    with open(meta_out, "w") as fh:
        json.dump({
            "n_clusters": args.n_clusters, "seq_len": args.seq_len,
            "features": args.features, "feature_model": args.feature_model,
            "n_blocks": int(blocks.shape[0]), "seed": args.seed,
            "cluster_sizes": counts.tolist(),
        }, fh, indent=2)
    print(f"[cluster] wrote {out} ({blocks.shape[0]} ids) and {meta_out}")
    print(f"[cluster] cluster sizes: {counts.tolist()}")


if __name__ == "__main__":
    main()
