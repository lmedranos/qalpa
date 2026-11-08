from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AcquiScorer(ABC):
    @abstractmethod
    def score(self) -> np.ndarray:
        """Return one novelty score per query point."""

    @abstractmethod
    def add(self, point: np.ndarray, weight: float = 1.0) -> None:
        """Add a selected point to the scorer's reference set."""
