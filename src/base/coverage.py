from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CoverageEstimator(ABC):
    @abstractmethod
    def estimate(self, requested: np.ndarray, candidates: np.ndarray,
                 center: np.ndarray, scale: np.ndarray,
                 minimum_effective_count: float) -> float:
        """Return the fraction of candidates covered by requested points."""
