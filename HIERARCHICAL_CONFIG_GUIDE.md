# Configuration Hierarchy Guide

## Overview

EQMgen uses Hydra/OmegaConf composition with three levels:

1. Root shared config
2. Framework config pack
3. Experiment-specific config

The experiment config created by init_experiment is the entrypoint and imports the lower levels through defaults.

## Current Sources

### Root level

- global_config.yaml

### Framework level

- EQMgen/configs/eqmgen_paths.yaml
- EQMgen/configs/eqdiff_model.yaml
- Additional assets under EQMgen/configs (for example nctr1.in, SKfiles)

### Experiment level

- Experiments/<exp_name>/configs/experiment_config.yaml

## How Composition Works

In experiment_config.yaml, defaults include:

- /global_config
- /EQMgen/configs/eqmgen_paths
- /EQMgen/configs/eqdiff_model

init_experiment injects hydra.searchpath with the absolute repo root so these absolute defaults resolve correctly.

## Runtime Resolution Order

Hydra merges from base to specific:

1. global_config.yaml
2. EQMgen/configs/*.yaml imported by defaults
3. experiment_config.yaml local values/overrides
4. CLI overrides

Later values override earlier ones.

## Running with Experiment Config

### Forward step

```bash
python -m EQMgen.scripts.forward.py \
  --config-path Experiments/<exp_name>/configs \
  --config-name experiment_config \
  loop.current_iter=0
```

### Backward step

```bash
python -m EQMgen.scripts.backward.py \
  --config-path Experiments/<exp_name>/configs \
  --config-name experiment_config \
  loop.current_iter=0
```

The generated Slurm job scripts in Experiments/<exp_name>/jobs call these same entrypoints.

## What to Edit Where

### Edit global_config.yaml when:

- You need shared defaults across experiments.
- You change common paths/system settings.

### Edit EQMgen/configs/*.yaml when:

- You change framework behavior (pipeline block paths, EQdiff script paths, model defaults).

### Edit Experiments/<exp_name>/configs/experiment_config.yaml when:

- You set dataset, trajectory, query props/scores, loop settings, and experiment-specific overrides.

## Common Override Examples

```bash
python -m EQMgen.scripts.forward.py \
  --config-path Experiments/<exp_name>/configs \
  --config-name experiment_config \
  loop.current_iter=3 \
  EQdiffParam.num_epochs=100 \
  geometry.optimizer=EquiDTB
```

## Initialization Reminder

Use init_experiment to scaffold a new experiment and path wiring:

```bash
python EQMgen/scripts/init_experiment.py --name my_exp --dataset-path /path/to/base_dataset.hdf5
```

This creates a versioned experiment directory and prepared job scripts under Experiments/.
