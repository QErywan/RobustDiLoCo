"""
Pre-tokenize C4 (English) into memory-mapped .npy shards.

Runs ONCE on Imperial, writes shards to /vol/bitbucket/qe25/data/c4_gpt2/ (or any
target dir).  Every subsequent sweep cell reads the shards via ShadedDataset
instead of streaming + tokenizing on the fly.

Why this matters
----------------
The current HFStreamingDataset.__iter__ calls tokenizer.encode() per C4 example
inside the training loop with num_workers=0.  On a 4080, the GPU sits idle waiting
for the CPU tokenizer: ~3 hours/cell for a 30M model.  After pre-tokenization,
data fetch is a memory-mapped int16 array slice: negligible cost.

Output layout (mirrors what ShadedDataset expects)
--------------------------------------------------
<out_dir>/
    train_000.npy, train_001.npy, …   — training token shards (int16, ~200MB each)
    validation_000.npy                — validation tokens for held-out eval
    meta.json                         — token counts, shard size, tokenizer

ShadedDataset reads these via np.load() and casts to int32 on load; int16 works
because the GPT-2 vocab (50257 tokens) fits in uint16 (max 65535).

Token budget
------------
Default: 500M train tokens + 50M validation.
At batch=4, seq=512 → 2048 tokens/step, a Tier-1 cell needs:
    8 workers × H=500 × 50 outer steps × 2048 = ~410M tokens
500M training tokens means each worker (of 8) gets 62.5M tokens — about 30,500
sequences of length 512, comfortably covering 25,000 inner steps without looping.

Disk usage
----------
500M tokens @ int16 (2 bytes) = 1 GB train
 50M tokens @ int16           = 0.1 GB val
Total: ~1.1 GB — tiny compared to checkpoint files.

Usage
-----
# On Imperial (run in a tmux, NOT in the sweep tmux):
python experiments/pretokenize.py \\
    --out-dir /vol/bitbucket/qe25/data/c4_gpt2 \\
    --n-train-tokens 500_000_000 \\
    --n-val-tokens 50_000_000

# Smaller test on Mac (CPU, faster to confirm the script works):
python experiments/pretokenize.py \\
    --out-dir /tmp/c4_gpt2_test \\
    --n-train-tokens 5_000_000 \\
    --n-val-tokens 500_000

# After running, verify the shards:
python experiments/pretokenize.py --verify --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKENIZER_NAME  = "gpt2"
DATASET_NAME    = "allenai/c4"
DATASET_CONFIG  = "en"
TOKENS_PER_SHARD = 100_000_000   # 100M tokens per file → ~200MB per shard at int16
DTYPE           = np.uint16      # GPT-2 vocab (50257) fits in uint16 (max 65535); ShadedDataset casts to int32 on load


# ---------------------------------------------------------------------------
# Core: stream-and-pack
# ---------------------------------------------------------------------------

def tokenize_split(
    split: str,
    n_tokens: int,
    out_dir: Path,
    shard_size: int = TOKENS_PER_SHARD,
    tokenizer_name: str = TOKENIZER_NAME,
    verbose: bool = True,
) -> int:
    """
    Stream the C4 `split`, tokenize with `tokenizer_name`, pack into a flat
    token buffer, and flush to .npy shards when the buffer reaches `shard_size`.

    Returns the total number of tokens written across all shards.
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        streaming=True,
        split=split,
        trust_remote_code=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    buffer      = []
    total_toks  = 0
    shard_idx   = 0
    t0          = time.perf_counter()
    n_examples  = 0

    for example in ds:
        tokens = tokenizer.encode(example["text"])
        buffer.extend(tokens)
        n_examples += 1

        # Flush one shard whenever buffer is big enough
        while len(buffer) >= shard_size:
            chunk = np.array(buffer[:shard_size], dtype=DTYPE)
            buffer = buffer[shard_size:]
            fname = out_dir / f"{split}_{shard_idx:03d}.npy"
            np.save(fname, chunk)
            total_toks += shard_size
            shard_idx  += 1
            elapsed = time.perf_counter() - t0
            if verbose:
                rate = total_toks / elapsed / 1e6
                print(f"  [{split}] shard {shard_idx:02d} written: "
                      f"{total_toks/1e6:.0f}M / {n_tokens/1e6:.0f}M tokens  "
                      f"({rate:.1f}M tok/s)", flush=True)

        if total_toks >= n_tokens:
            break

    # Write any remaining tokens as a partial shard (if non-empty)
    if buffer and total_toks < n_tokens:
        leftover = min(len(buffer), n_tokens - total_toks)
        chunk = np.array(buffer[:leftover], dtype=DTYPE)
        fname = out_dir / f"{split}_{shard_idx:03d}.npy"
        np.save(fname, chunk)
        total_toks += leftover
        shard_idx  += 1
        if verbose:
            print(f"  [{split}] partial shard {shard_idx:02d} written: "
                  f"{leftover/1e6:.1f}M tokens")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  [{split}] Done: {total_toks/1e6:.0f}M tokens in {shard_idx} shards "
              f"({n_examples} examples, {elapsed/60:.1f} min)")

    return total_toks


# ---------------------------------------------------------------------------
# Meta file
# ---------------------------------------------------------------------------

def write_meta(out_dir: Path, n_train: int, n_val: int, shard_size: int) -> None:
    """Write meta.json so the data loader knows the exact token counts."""
    meta = {
        "tokenizer":        TOKENIZER_NAME,
        "dataset":          f"{DATASET_NAME} ({DATASET_CONFIG})",
        "n_train_tokens":   n_train,
        "n_val_tokens":     n_val,
        "tokens_per_shard": shard_size,
        "dtype":            "uint16",
        "note": (
            "uint16 covers GPT-2 vocab (50257 < 65535). "
            "ShadedDataset loads with np.load().astype(np.int32), which correctly "
            "widens uint16 values to int32 without sign issues."
        ),
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[pretokenize] meta.json → {out_dir / 'meta.json'}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(out_dir: Path) -> None:
    """Load and spot-check each shard to confirm shapes and dtype."""
    import glob

    print(f"\n[pretokenize] Verifying shards in {out_dir}…")
    for pattern in ["train_*.npy", "validation_*.npy"]:
        files = sorted(glob.glob(str(out_dir / pattern)))
        if not files:
            print(f"  WARNING: no files matching {pattern}")
            continue
        total = 0
        for f in files:
            arr = np.load(f, allow_pickle=False)
            assert arr.ndim == 1, f"Expected 1-D array, got {arr.shape}"
            n = len(arr)
            total += n
            print(f"  {Path(f).name}: {n:>12,} tokens  dtype={arr.dtype}  "
                  f"min={arr.min()}  max={arr.max()}")
        print(f"  → {pattern.replace('*.npy','')}: {total:,} tokens total")

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"\n  meta.json: n_train={meta['n_train_tokens']:,}  "
              f"n_val={meta['n_val_tokens']:,}  shard_size={meta['tokens_per_shard']:,}")
        # Confirm actual vs declared
        train_files = sorted((out_dir).glob("train_*.npy"))
        actual_train = sum(len(np.load(f, allow_pickle=False)) for f in train_files)
        val_files    = sorted((out_dir).glob("validation_*.npy"))
        actual_val   = sum(len(np.load(f, allow_pickle=False)) for f in val_files)
        print(f"  Actual:    n_train={actual_train:,}  n_val={actual_val:,}")
        if actual_train != meta["n_train_tokens"] or actual_val != meta["n_val_tokens"]:
            print("  WARNING: mismatch between meta.json and actual shard contents!")
        else:
            print("  ✓ meta.json matches actual shard token counts.")
    else:
        print(f"\n  WARNING: no meta.json found in {out_dir}")

    print("[pretokenize] Verification complete.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-tokenize C4 into .npy shards for the DiLoCo simulation sweep."
    )
    p.add_argument("--out-dir", type=str,
                   default="/vol/bitbucket/qe25/data/c4_gpt2",
                   help="Output directory for shard files and meta.json.")
    p.add_argument("--n-train-tokens", type=lambda x: int(x.replace("_", "")),
                   default=500_000_000,
                   help="Total training tokens to write (default: 500M).")
    p.add_argument("--n-val-tokens", type=lambda x: int(x.replace("_", "")),
                   default=50_000_000,
                   help="Total validation tokens to write (default: 50M).")
    p.add_argument("--shard-size", type=lambda x: int(x.replace("_", "")),
                   default=TOKENS_PER_SHARD,
                   help="Tokens per shard file (default: 100M).")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training split (useful to only regenerate val).")
    p.add_argument("--skip-val", action="store_true",
                   help="Skip validation split.")
    p.add_argument("--verify", action="store_true",
                   help="After writing (or standalone), verify the shards and exit.")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify existing shards, do not write anything.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.verify_only:
        verify(out_dir)
        return

    print(f"\n{'='*60}")
    print(f"  DiLoCo C4 Pre-tokenizer")
    print(f"  out_dir    : {out_dir}")
    print(f"  train toks : {args.n_train_tokens:,}")
    print(f"  val toks   : {args.n_val_tokens:,}")
    print(f"  shard size : {args.shard_size:,} tokens ({args.shard_size*2/1e9:.2f}GB per shard @ uint16)")
    print(f"  tokenizer  : {TOKENIZER_NAME}")
    print(f"  dataset    : {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"{'='*60}\n")

    n_train_written = 0
    n_val_written   = 0

    if not args.skip_train:
        print(f"[pretokenize] Writing {args.n_train_tokens/1e6:.0f}M training tokens…")
        n_train_written = tokenize_split(
            split="train",
            n_tokens=args.n_train_tokens,
            out_dir=out_dir,
            shard_size=args.shard_size,
        )

    if not args.skip_val:
        print(f"\n[pretokenize] Writing {args.n_val_tokens/1e6:.0f}M validation tokens…")
        n_val_written = tokenize_split(
            split="validation",
            n_tokens=args.n_val_tokens,
            out_dir=out_dir,
            shard_size=args.shard_size,
        )

    write_meta(out_dir, n_train=n_train_written, n_val=n_val_written,
               shard_size=args.shard_size)

    if args.verify:
        verify(out_dir)

    print(f"\n[pretokenize] All done. Shards in {out_dir}")
    print(f"  Run with: --dataset c4 --data-path {out_dir}")


if __name__ == "__main__":
    main()
