"""
Quick reference for init_experiment() function

Initialize new experiments with automatic scaffolding.
"""

# ============================================================================
# FROM CLI
# ============================================================================

# Basic usage
python EQMgen/scripts/init_experiment.py my_experiment

# With explicit name argument
python EQMgen/scripts/init_experiment.py --name my_experiment

# Quiet mode (no verbose output)
python EQMgen/scripts/init_experiment.py my_experiment --quiet

# With custom base path
python EQMgen/scripts/init_experiment.py my_experiment --base-path /path/to/project


# ============================================================================
# FROM PYTHON
# ============================================================================

# In a script or notebook
from EQMgen.scripts import init_experiment
from pathlib import Path

# Simple usage
init_experiment("my_experiment")

# With custom base path
init_experiment("my_experiment", base_path=Path("/path/to/project"))

# Suppress verbose output
init_experiment("my_experiment", verbose=False)

# Get the experiment directory path
exp_dir = init_experiment("my_experiment")
print(f"Experiment created at: {exp_dir}")


# ============================================================================
# WHAT IT CREATES
# ============================================================================

# After calling init_experiment("my_experiment"), you get:
#
# Experiments/my_experiment/
# ├── configs/
# │   └── experiment_config.yaml          (auto-populated template)
# ├── data/                               (empty)
# ├── jobs/                               
# │   └── forward_activediff.sh           (auto-populated template)
# │   └── backwark_activediff.sh          (auto-populated template)
# ├── results/                            (empty)
# └── logs/                               (empty)


# ============================================================================
# NEXT STEPS
# ============================================================================

# 1. Edit the experiment config
#    Experiments/my_experiment/configs/experiment_config.yaml
#    - Set base_dataset_path
#    - Customize data.props and data.scores
#    - Adjust training/geometry parameters

# 2. Run Forward and Backword jobs
