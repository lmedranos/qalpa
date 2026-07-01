# init_experiment: Experiment Initialization

## Overview

init_experiment scaffolds a runnable experiment by creating directories, copying templates, and injecting absolute paths into config and job scripts.

It is implemented in EQMgen/scripts/init_experiment.py.

## Quick Start

### CLI

```bash
python EQMgen/scripts/init_experiment.py --name my_experiment --dataset-path /path/to/base_dataset.hdf5
```

### Python

```python
from EQMgen.scripts.init_experiment import init_experiment

init_experiment(
    exp_name="my_experiment",
    dataset_path="/path/to/base_dataset.hdf5",
)
```

## Function Signature

```python
def init_experiment(
    exp_name: str,
    dataset_path: str,
    base_path: str = None,
    verbose: bool = True,
):
    ...
```

## What It Creates

Given name my_experiment, the function creates a versioned experiment directory:

```text
Experiments/my_experiment_0/
├── configs/
│   └── experiment_config.yaml
├── jobs/
│   ├── eqmgen_forward.sh
│   └── eqmgen_backward.sh
├── logs/
└── results/

Data/my_experiment_0/
```

If the name already exists, it increments the suffix automatically (for example my_experiment_1, my_experiment_2, ...).

## Template Injection

During initialization, the script replaces placeholders:

- @@REPO_ROOT@@ in experiment_config.yaml and both job scripts
- @@EXP_CONFIG_DIR@@ in both job scripts
- template_experiment with the resolved versioned experiment name
- template_dataset with the provided dataset path

## CLI Usage

### Required

- Experiment name (positional or --name)
- --dataset-path

### Optional

- --base-path
- --quiet

Examples:

```bash
python EQMgen/scripts/init_experiment.py my_exp --dataset-path /data/base.hdf5
python EQMgen/scripts/init_experiment.py --name my_exp --dataset-path /data/base.hdf5 --quiet
python EQMgen/scripts/init_experiment.py --name my_exp --dataset-path /data/base.hdf5 --base-path /path/to/repo
```

## After Initialization

1. Edit config:

```bash
vim Experiments/my_exp_0/configs/experiment_config.yaml
```

2. Submit jobs from the created jobs directory:

```bash
cd Experiments/my_exp_0/jobs
sbatch eqmgen_forward.sh
sbatch eqmgen_backward.sh
```

The job scripts run:

- python -m EQMgen.scripts.forward_p
- python -m EQMgen.scripts.backward_p

using the generated experiment config directory.

## Error Handling

- FileNotFoundError if template config or job scripts are missing
- FileNotFoundError if dataset path does not exist

## Notes

- Initialization does not run the pipeline itself; it only scaffolds and wires paths.
- Iteration tracking in jobs is handled through MY_ITER.state in the jobs directory.
