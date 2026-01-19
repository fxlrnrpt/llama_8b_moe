"""Data loading utilities for training."""

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer


class StreamingDataset(IterableDataset):
    """
    Wrapper for HuggingFace streaming datasets that distributes data across GPUs.

    Each GPU only processes samples where (sample_index % world_size == rank).
    """

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for i, item in enumerate(self.dataset):
            if i % self.world_size == self.rank:
                yield item


def create_collate_fn(tokenizer: PreTrainedTokenizer, max_seq_len: int):
    """
    Create a collate function for language modeling.

    Tokenizes text and creates input/label pairs shifted by 1 position.
    """

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len + 1,  # +1 for shifting to create labels
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        return {
            "input_ids": input_ids[:, :-1],  # All tokens except last
            "labels": input_ids[:, 1:],  # All tokens except first (shifted)
        }

    return collate_fn


def create_dataloader(
    dataset_name: str,
    dataset_subset: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a distributed DataLoader for streaming datasets.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb-edu")
        dataset_subset: Dataset subset/config (e.g., "sample-10BT")
        tokenizer: Tokenizer for encoding text
        max_seq_len: Maximum sequence length for training
        batch_size: Batch size per GPU
        world_size: Total number of GPUs
        rank: Current GPU rank
        num_workers: Number of dataloader workers

    Returns:
        DataLoader configured for distributed streaming training.
    """
    # Load streaming dataset (no download required)
    dataset = load_dataset(
        dataset_name,
        dataset_subset,
        split="train",
        streaming=True,
    )

    # Wrap with distributed sampler logic
    streaming_ds = StreamingDataset(dataset, rank, world_size)

    # Create collate function
    collate_fn = create_collate_fn(tokenizer, max_seq_len)

    dataloader = DataLoader(
        streaming_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
