"""
EQMgen - Active Learning Framework for Molecular Generation and Property Prediction

This package provides tools for:
- Data management (EQMgen.utils)
- Pipeline processing (EQMgen.pipeline_blocks)
- Workflow execution (EQMgen.scripts)
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Make key modules easily importable
from . import utils
from . import pipeline_blocks
from . import scripts

__all__ = ["utils", "pipeline_blocks", "scripts"]
