"""
Streaming dataset for simulation experiments.

Streams C4 (or any HuggingFace text dataset) on the fly and tokenizes into
fixed-length sequences. Each worker gets its own shard so data is non-overlapping.
No local download required — HuggingFace handles streaming from the hub.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset


class HFStreamingDataset(IterableDataset):
    """
    Streams a HuggingFace text dataset and packs tokens into fixed-length sequences.

    Each worker rank receives a distinct shard so workers see non-overlapping data.
    Tokenization happens on the fly — no pre-processing or disk storage needed.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        seq_len: int,
        rank: int,
        world_size: int,
        dataset_config: str = "en",
        split: str = "train",
    ):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        ds = load_dataset(
            dataset_name,
            dataset_config,
            streaming=True,
            split=split,
            trust_remote_code=True,
        )
        # Each worker sees a non-overlapping shard
        self.ds = ds.shard(num_shards=world_size, index=rank)

    def __iter__(self):
        buffer = []
        for example in self.ds:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                yield torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                buffer = buffer[self.seq_len :]


def make_hf_loaders(
    dataset_name: str,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    n_workers: int,
) -> list[DataLoader]:
    """One streaming DataLoader per worker, each seeing a distinct shard."""
    loaders = []
    for rank in range(n_workers):
        ds = HFStreamingDataset(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
            rank=rank,
            world_size=n_workers,
        )
        # num_workers=0 required for IterableDataset with manual sharding
        loaders.append(DataLoader(ds, batch_size=batch_size, num_workers=0))
    return loaders
