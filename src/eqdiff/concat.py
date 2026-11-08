from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, WeightedRandomSampler


def weights(variables: Sequence[object], alpha: float, beta: float) -> List[float]:
    n = len(variables)
    if n == 0:
        return []
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not 0 <= beta <= 1:
        raise ValueError(f"beta must be between 0 and 1, got {beta}")
    if n == 1:
        return [1.0]
    if n == 2:
        return [alpha, 1 - alpha]

    m = n - 2
    b = beta ** (1.0 / m)
    return [alpha * b] + [alpha * (1 - b) / m] * m + [1 - alpha]


class Concatenator:
    """Build concatenated datasets and weighted training samplers."""

    def __init__(
        self,
        dataset_cls: Callable[..., Dataset],
        root: str,
        dataset_kwargs: dict,
    ):
        self.dataset_cls = dataset_cls
        self.root = Path(root)
        self.dataset_kwargs = dataset_kwargs

    def dataset_roots(self) -> List[str]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        roots = sorted(path for path in self.root.iterdir() if path.is_dir())
        if not roots:
            raise FileNotFoundError(
                f"No dataset subdirectories found under {self.root}"
            )
        return [str(path) for path in roots]

    def datasets(self, split: str) -> List[Dataset]:
        return [
            self.dataset_cls(split=split, root=root, **self.dataset_kwargs)
            for root in self.dataset_roots()
        ]

    @staticmethod
    def concatenate(datasets: Iterable[Dataset]) -> ConcatDataset:
        datasets = list(datasets)
        if not datasets:
            raise ValueError("At least one dataset is required for concatenation")
        return ConcatDataset(datasets)

    @staticmethod
    def sample_weights(
        dataset: Dataset,
        alpha: float,
        beta: float,
    ) -> torch.DoubleTensor:
        if isinstance(dataset, Subset):
            base_weights = Concatenator.sample_weights(dataset.dataset, alpha, beta)
            return base_weights[torch.as_tensor(dataset.indices, dtype=torch.long)]

        if not isinstance(dataset, ConcatDataset):
            return torch.ones(len(dataset), dtype=torch.double)

        dataset_weights = weights(dataset.datasets, alpha, beta)
        if any(len(child_dataset) == 0 for child_dataset in dataset.datasets):
            raise ValueError("Cannot sample from an empty dataset")
        sample_weights = [
            dataset_weight / len(child_dataset)
            for child_dataset, dataset_weight in zip(
                dataset.datasets, dataset_weights
            )
            for _ in range(len(child_dataset))
        ]
        return torch.tensor(sample_weights, dtype=torch.double)

    @classmethod
    def sampler(
        cls,
        dataset: Dataset,
        alpha: float,
        beta: float,
    ) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            cls.sample_weights(dataset, alpha, beta),
            num_samples=len(dataset),
            replacement=True,
        )
