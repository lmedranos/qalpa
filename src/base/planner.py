from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.components.trajectory import Target


@dataclass(frozen=True)
class PlanningRequest:
    trajectory: Any
    state: Any
    config: Mapping[str, Any]
    iteration: int
    total_samples: int
    rng: np.random.Generator


class Planner(ABC):
    @abstractmethod
    def plan(self, request: PlanningRequest) -> list[Target]:
        """Create targets for one active-learning iteration."""
