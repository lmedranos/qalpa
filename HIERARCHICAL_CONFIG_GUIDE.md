# Three-Tier Hierarchical Configuration System

## Overview

The configuration is now organized in three tiers, allowing for maximum flexibility:

```
┌─────────────────────────────────────────────────────────┐
│ Root Configs (/configs/)                                │
│ • system.yaml      - Global settings (seed, device)     │
│ • training.yaml    - Default training params            │
│ • paths.yaml       - Root directory paths               │
│ └─→ SHARED ACROSS ALL EXPERIMENTS                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ EQMgen Configs (EQMgen/configs/)                        │
│ • eqmgen_paths.yaml      - EQMgen code paths            │
│ • eqmgen_pipeline.yaml   - Pipeline stages              │
│ • eqmgen_model.yaml      - Diffusion model config       │
│ • eqmgen_geometry.yaml   - Geometry optimizer           │
│ • eqmgen_properties.yaml - Property calculator          │
│ └─→ FRAMEWORK-SPECIFIC SETTINGS                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Experiment Configs (Experiments/{exp_name}/configs/)    │
│ • exp_data.yaml        - Dataset paths, properties      │
│ • exp_training.yaml    - Iteration overrides            │
│ • exp_geometry.yaml    - Optimizer overrides            │
│ • exp_properties.yaml  - Calculator overrides           │
│ • experiment_config.yaml - Ties everything together     │
│ └─→ EXPERIMENT-SPECIFIC SETTINGS                        │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
activediffusion/
├── configs/                             # Root-level (GLOBAL)
│   ├── system.yaml                      # Seed, device, determinism
│   ├── training.yaml                    # Default batch size, iterations
│   ├── paths.yaml                       # Root directory paths
│   └── config.yaml                      # Root defaults file
│
├── EQMgen/
│   └── configs/                         # Framework-level (IMPLEMENTATION)
│       ├── eqmgen_paths.yaml            # Code directory paths
│       ├── eqmgen_pipeline.yaml         # Pipeline blocks
│       ├── eqmgen_model.yaml            # Model parameters
│       ├── eqmgen_geometry.yaml         # Geometry optimizer
│       ├── eqmgen_properties.yaml       # Properties calculator
│       └── eqmgen_config.yaml           # EQMgen defaults
│
└── Experiments/
    └── allostery/                       # Experiment-level (SPECIFIC)
        └── configs/
            └── experiment_config.yaml   # All-in-one experiment config
```

---

## Configuration Resolution Order

When you run an experiment, Hydra resolves configs in this priority:

1. **Experiment-specific** (highest priority) - `Experiments/{exp_name}/configs/`
2. **EQMgen framework** (medium priority) - `EQMgen/configs/`
3. **Root global** (lowest priority) - `configs/`

**Later configs override earlier ones**, so you can override any setting at any level.

---

## Usage Examples

### Running Experiment from Experiment Directory

```bash
cd Experiments/allostery

# Run with full config hierarchy (best practice)
python ../../EQMgen/scripts/backward.py \
    --config-path configs \
    --config-name experiment_config
```

Or with Hydra:

```python
from hydra import main as hydra_main

@hydra_main(
    version_base=None,
    config_path="Experiments/allostery/configs",
    config_name="experiment_config"
)
def run(cfg):
    # cfg now has full hierarchy: root → EQMgen → experiment
    pass
```

### Overriding at Runtime

```bash
# Override experiment-level config
python backward.py geometry.optimizer=EquiDTB

# Override EQMgen-level config
python backward.py EQdiffParam.num_samples=500

# Override root-level config
python backward.py training.num_iterations=20
```

### Creating New Experiments

**Quick way using `init_experiment()`:**

```bash
# From CLI
python EQMgen/scripts/init_experiment.py my_experiment

# Or with full options
python EQMgen/scripts/init_experiment.py --name my_experiment
```

**Or from Python:**

```python
from EQMgen.scripts import init_experiment

# Create new experiment with scaffolding
init_experiment("my_experiment")
```

This creates:
- `Experiments/my_experiment/configs/experiment_config.yaml` (auto-populated template)
- `Experiments/my_experiment/data/` (data directory)
- `Experiments/my_experiment/results/` (results directory)
- `Experiments/my_experiment/logs/` (logs directory)

Then edit the config file and run:

```bash
python EQMgen/scripts/run_experiment.py \
    --config-path Experiments/my_experiment/configs \
    --config-name experiment_config
```

---

## What Goes Where?

### ✅ Root Configs (`/configs/`)
- System settings that don't change often
- Default training parameters
- Base paths

**Edit when:** You need framework-wide defaults

### ✅ EQMgen Configs (`EQMgen/configs/`)
- Pipeline structure
- Model architecture
- Geometry optimizer settings
- Property calculator settings
- Code directory paths

**Edit when:** You update EQMgen implementation

### ✅ Experiment Configs (`Experiments/{exp_name}/configs/`)
- Dataset paths
- Experiment-specific parameters
- Overrides for this experiment
- Property selections

**Edit when:** You run a new experiment

---

## Example Configurations

### Root: Global Defaults

**`/configs/system.yaml`**
```yaml
system:
  seed: 42
  device: cuda
  deterministic: true
```

**`/configs/training.yaml`**
```yaml
training:
  batch_size: 32
  max_parallel_tasks: 1
  num_iterations: 10  # Default, can override
```

### EQMgen: Framework Settings

**`EQMgen/configs/eqmgen_model.yaml`**
```yaml
model:
  type: EQdiff
  device: ${system.device}  # Reference root config!

EQdiffParam:
  num_samples: 200
  n_nodes: 20
```

### Experiment: All-In-One Configuration

**`Experiments/allostery/configs/experiment_config.yaml`**
```yaml
defaults:
  - /configs/system
  - /configs/training
  - /configs/paths
  - /EQMgen/configs/eqmgen_config

# Experiment metadata
experiment_name: allostery
experiment:
  name: allostery
  description: "Allostery-focused molecular generation experiment"
  version: 1.0

# Data configuration
base_dataset_path: /path/to/data.hdf5
data:
  exp_data_path: ${paths.experiments}/allostery/data
  props: ["HLgap", "eMBD"]
  scores: ["SAScore"]

# Training overrides
training:
  num_iterations: 10
  batch_size: 32
  max_parallel_tasks: 1

# Geometry overrides
geometry:
  optimizer: DFTB+
  calculate_forces: true
  filter_scheme: [score, accuracy]

# Properties overrides
properties:
  calculator: DFTB+
  filteration_threshold: 0.1
```

---

## Benefits

| Feature | Benefit |
|---------|---------|
| **Three-tier hierarchy** | Separation of concerns, reduces duplication |
| **Global defaults** | Consistent settings across experiments |
| **Framework settings** | Implementation details separate from experiments |
| **Experiment overrides** | Easy to customize per experiment |
| **Hydra composition** | Clean CLI overrides (no hard-coded paths) |
| **Reusability** | Same framework configs for different experiments |

---

## Migration Guide

### Before
- Single monolithic `experiment_config.yaml`
- Hard to reuse settings across experiments
- Difficult to track which configs are global vs experiment-specific

### After
- Three-level hierarchy
- Reuse global and framework configs
- Clear which settings are experiment-specific
- Easy to create new experiments from templates

---

## Template: New Experiment Structure

```bash
# Create experiment directory
mkdir -p Experiments/my_experiment/configs

# Create single all-in-one experiment config file
cat > Experiments/my_experiment/configs/experiment_config.yaml << 'EOF'
defaults:
  - /configs/system
  - /configs/training
  - /configs/paths
  - /EQMgen/configs/eqmgen_config

# Experiment metadata
experiment_name: my_experiment
experiment:
  name: my_experiment
  description: "My custom experiment"
  version: 1.0

# Data configuration (experiment-specific)
base_dataset_path: /path/to/dataset.hdf5
data:
  exp_data_path: ${paths.experiments}/my_experiment/data
  props: ["property1", "property2"]
  scores: ["score1"]

# Training configuration (overrides for this experiment)
training:
  num_iterations: 10
  batch_size: 32

# Geometry configuration (overrides for this experiment)
geometry:
  optimizer: DFTB+
  calculate_forces: true

# Properties configuration (overrides for this experiment)
properties:
  calculator: DFTB+
  filteration_threshold: 0.1
EOF
```

That's it! One file per experiment with all settings clearly organized in sections.


---

## Questions?

- **How to override a setting?** Use Hydra CLI: `python script.py setting.key=value`
- **How to add new framework config?** Add to `EQMgen/configs/` with `eqmgen_` prefix
- **How to add new experiment?** Copy `Experiments/allostery/configs/` and customize
- **How to reference another config value?** Use `${path.to.value}` syntax (OmegaConf)
