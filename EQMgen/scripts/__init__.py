"""
EQMgen.scripts - Main workflow entry points

Provides:
- forward_main: Forward pass (generation step)
- backward_main: Backward pass (training/evaluation step)
- init_experiment: Initialize new experiment with scaffolding
"""

from .forward import forward_main
from .backward import backward_main
from .init_experiment import init_experiment

__all__ = [
    "forward_main",
    "backward_main",
    "init_experiment",
]
