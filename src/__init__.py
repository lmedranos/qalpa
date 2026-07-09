"""
QALPA - Quantum-Aware Learning for Property-Space Augmentation

This package provides tools for:
- Data management (src.utils)
- Pipeline processing (src.pipeline_blocks)
- Workflow execution (src.scripts)
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Make key modules easily importable
from . import utils
from . import pipeline_blocks
from . import scripts

__all__ = ["utils", "pipeline_blocks", "scripts"]
