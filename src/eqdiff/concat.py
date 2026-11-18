import copy
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, WeightedRandomSampler

_DATASET_DIR = re.compile(r"dataset_(\d+)")


def weights(
    variables: Sequence[object],
    alpha: float,
    beta: float,
    min_first: float = 0.0,
) -> List[float]:
    """Sampling share of each dataset, ordered oldest (dataset_0) -> newest.

    - newest dataset: 1 - alpha
    - each middle dataset: w = alpha / (m + beta), m = number of middle datasets
    - original dataset: beta * w (beta < 1 keeps it below the middles)
    - min_first: lower bound for the original dataset's share (replay floor
      against catastrophic forgetting). The missing share is taken from the
      newest dataset; the middle datasets keep their share.
    """
    n = len(variables)
    if n == 0:
        return []
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not 0 <= beta <= 1:
        raise ValueError(f"beta must be between 0 and 1, got {beta}")
    if not 0 <= min_first < 1:
        raise ValueError(f"min_first must be in [0, 1), got {min_first}")
    if n == 1:
        return [1.0]
    if n == 2:
        shares = [alpha, 1 - alpha]
    else:
        m = n - 2
        w = alpha / (m + beta)
        shares = [alpha - m * w] + [w] * m + [1 - alpha]

    if shares[0] < min_first:
        shares[-1] -= min_first - shares[0]
        shares[0] = min_first
        if shares[-1] <= 0:
            raise ValueError(
                f"min_first={min_first} leaves no share for the newest dataset "
                f"(alpha={alpha}, beta={beta}, {n} datasets)"
            )
    return shares


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
        """Return `dataset_<n>` subdirectories ordered by n (oldest first).

        Only directories named `dataset_<int>` are used, so stray folders such as
        `raw/` or `processed/` in the root are ignored. Sorting is numeric so that
        `dataset_10` comes after `dataset_9` (the last one gets weight 1 - alpha).
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        roots = []
        for path in self.root.iterdir():
            match = _DATASET_DIR.fullmatch(path.name)
            if path.is_dir() and match:
                roots.append((int(match.group(1)), path))
        if not roots:
            raise FileNotFoundError(
                f"No dataset_<n> subdirectories found under {self.root}"
            )
        return [str(path) for _, path in sorted(roots)]

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
    def merge_statistics(
        datasets: Sequence[Dataset], dataset_weights: Optional[Sequence[float]] = None
    ):
        """Merge `.statistics` of several datasets.

        num_nodes histograms are summed exactly (so every molecule size seen in
        any dataset stays in the support). Normalised marginals stored as tensors
        (atom/bond/charge types, ...) are averaged with `dataset_weights`
        (e.g. the sampler weights, so the prior matches what the model is trained
        on) or, if None, weighted by dataset size. Dict-valued stats
        (valencies, bond_lengths) are taken from the first dataset.
        """
        stats = [d.statistics for d in datasets]
        if len(stats) == 1:
            return stats[0]
        if dataset_weights is None:
            dataset_weights = [len(d) for d in datasets]
        w = torch.tensor(list(dataset_weights), dtype=torch.double)
        w = w / w.sum()

        merged = copy.copy(stats[0])
        num_nodes = Counter()
        for s in stats:
            num_nodes.update(dict(s.num_nodes))
        merged.num_nodes = dict(num_nodes)

        for name, value in vars(stats[0]).items():
            if not torch.is_tensor(value):
                continue
            values = [getattr(s, name) for s in stats]
            if any(v is None or v.shape != value.shape for v in values):
                continue  # incompatible shapes (e.g. histograms) -> keep first
            avg = sum(wi * v.double() for wi, v in zip(w, values))
            setattr(merged, name, avg.to(value.dtype))
        return merged

    @staticmethod
    def sample_weights(
        dataset: Dataset,
        alpha: float,
        beta: float,
        min_first: float = 0.0,
    ) -> torch.DoubleTensor:
        if isinstance(dataset, Subset):
            base_weights = Concatenator.sample_weights(
                dataset.dataset, alpha, beta, min_first
            )
            return base_weights[torch.as_tensor(dataset.indices, dtype=torch.long)]

        if not isinstance(dataset, ConcatDataset):
            return torch.ones(len(dataset), dtype=torch.double)

        dataset_weights = weights(dataset.datasets, alpha, beta, min_first)
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
        num_samples: Optional[int] = None,
        min_first: float = 0.0,
    ) -> WeightedRandomSampler:
        """Weighted sampler with replacement.

        `num_samples` is the number of draws per epoch (default: len(dataset)).
        Lower it to limit how often the small, high-weight newest dataset is
        repeated per epoch.
        """
        return WeightedRandomSampler(
            cls.sample_weights(dataset, alpha, beta, min_first),
            num_samples=num_samples or len(dataset),
            replacement=True,
        )
