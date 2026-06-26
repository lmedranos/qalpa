# `init_experiment()` - Experiment Initialization Function

## Overview

`init_experiment()` is a helper function that scaffolds a new experiment with proper directory structure and configuration template.

Instead of manually creating directories and copying files, you can create a new experiment in one command.

---

## Quick Start

### CLI
```bash
python EQMgen/scripts/init_experiment.py my_experiment
```

### Python
```python
from EQMgen.scripts import init_experiment
init_experiment("my_experiment")
```

---

## What It Creates

```
Experiments/my_experiment/
├── configs/
│   └── experiment_config.yaml          # Auto-populated template
├── data/                               # Data directory
├── results/                            # Results directory
└── logs/                               # Logs directory
```

The `experiment_config.yaml` is automatically populated with:
- Correct experiment name
- Correct data paths
- Full defaults hierarchy (root → framework → experiment)
- All sections ready for customization

---

## Usage Examples

### Basic CLI
```bash
python EQMgen/scripts/init_experiment.py my_experiment
```

### CLI with Options
```bash
python EQMgen/scripts/init_experiment.py --name my_experiment
python EQMgen/scripts/init_experiment.py my_experiment --quiet
python EQMgen/scripts/init_experiment.py my_experiment --base-path /path/to/project
```

### Python Function
```python
from EQMgen.scripts import init_experiment
from pathlib import Path

# Create new experiment
exp_dir = init_experiment("my_experiment")

# With options
exp_dir = init_experiment(
    "my_experiment",
    base_path=Path("/custom/project/path"),
    verbose=True
)
```

---

## Function Signature

```python
def init_experiment(
    exp_name: str,
    base_path: Path = None,
    verbose: bool = True
) -> Path:
    """
    Initialize a new experiment with directory structure and configuration template.
    
    Args:
        exp_name: Name of the experiment
        base_path: Base path to project root (auto-detected if None)
        verbose: Print status messages (default: True)
    
    Returns:
        Path to the created experiment directory
    
    Raises:
        FileExistsError: If experiment already exists
        FileNotFoundError: If template config not found
    """
```

---

## After Initialization

1. **Edit configuration**
   ```bash
   vim Experiments/my_experiment/configs/experiment_config.yaml
   ```
   
   Update:
   - `base_dataset_path` - Point to your dataset
   - `data.props` - Properties to track
   - `data.scores` - Scores to track
   - Any training/geometry/properties overrides

2. **Run experiment**
   ```bash
   python EQMgen/scripts/run_experiment.py \
       --config-path Experiments/my_experiment/configs \
       --config-name experiment_config
   ```

3. **With parameter overrides (optional)**
   ```bash
   python EQMgen/scripts/run_experiment.py \
       --config-path Experiments/my_experiment/configs \
       --config-name experiment_config \
       training.num_iterations=5 \
       geometry.optimizer=EquiDTB \
       system.seed=123
   ```

---

## Workflow Example

```bash
# 1. Create experiment
python EQMgen/scripts/init_experiment.py protein_fold

# 2. Edit configuration
vim Experiments/protein_fold/configs/experiment_config.yaml
# Change base_dataset_path, props, scores, etc.

# 3. Run experiment
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/protein_fold/configs \
    --config-name experiment_config

# 4. Run with different parameters
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/protein_fold/configs \
    --config-name experiment_config \
    training.num_iterations=20 \
    geometry.optimizer=MACE
```

---

## Error Handling

### Experiment Already Exists
```python
FileExistsError: Experiment 'my_experiment' already exists at Experiments/my_experiment
```

**Solution**: Use a different experiment name

### Template Config Not Found
```python
FileNotFoundError: Template config not found at Experiments/allostery/configs/experiment_config.yaml
```

**Solution**: Ensure the allostery experiment exists with the template config

---

## See Also

- [HIERARCHICAL_CONFIG_GUIDE.md](HIERARCHICAL_CONFIG_GUIDE.md) - Configuration system details
- [RUN_EXPERIMENT.sh](RUN_EXPERIMENT.sh) - Command examples
- [THREE_TIER_CONFIG_SUMMARY.md](THREE_TIER_CONFIG_SUMMARY.md) - Quick config summary
- [EQMgen/scripts/init_experiment.py](EQMgen/scripts/init_experiment.py) - Source code
- [INIT_EXPERIMENT_REFERENCE.py](INIT_EXPERIMENT_REFERENCE.py) - Quick reference in Python format
