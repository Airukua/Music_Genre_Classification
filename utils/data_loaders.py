import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from glob import glob

def Loaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits a dataset into training, validation, and test sets, and returns corresponding DataLoaders.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset to split.
        batch_size (int): Number of samples per batch for each DataLoader.
        train_ratio (float, optional): Proportion of the dataset to use for training. Default is 0.7.
        val_ratio (float, optional): Proportion of the dataset to use for validation. Default is 0.2.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train, validation, and test sets.

    Raises:
        ValueError: If the sum of `train_ratio` and `val_ratio` exceeds 1.0.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must sum to less than 1.")
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, lengths=[train_size, val_size, test_size], generator=generator
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
