# `init_experiment`

`init_experiment` scaffolds a runnable experiment: it creates the folders, copies the templates, and writes absolute paths into the configuration and the job scripts. It does not run the pipeline.

Source: {repo}`src/scripts/init_experiment.py`

## Command line

```bash
python src/scripts/init_experiment.py --name my_exp --dataset-path /path/to/initial_dataset.hdf5
```

| Argument | Required | Meaning |
|---|---|---|
| `name` or `--name` | Yes | Experiment name, as a positional argument or with `--name` |
| `--dataset-path` | Yes | Initial dataset (HDF5); must exist |
| `--base-path` | No | Repository root; detected automatically if not given |
| `--quiet` | No | Print nothing |

Examples:

```bash
python src/scripts/init_experiment.py my_exp --dataset-path /data/base.hdf5
python src/scripts/init_experiment.py --name my_exp --dataset-path /data/base.hdf5 --quiet
python src/scripts/init_experiment.py --name my_exp --dataset-path /data/base.hdf5 --base-path /path/to/repo
```

The Navigator's **Create experiment** button runs the same script ([Step 1: Experiment](../navigator/using-the-app.md#step-1-experiment)).

## Python

```python
from src.scripts.init_experiment import init_experiment

init_experiment(
    exp_name="my_exp",
    dataset_path="/path/to/initial_dataset.hdf5",
    base_path=None,      # repository root; detected if None
    verbose=True,
)
```

## What it creates

```text
Experiments/my_exp_0/
├── configs/
│   └── experiment_config.yaml     # copy of Experiments/templates/configs/experiment_config.yaml
├── jobs/
│   ├── qalpa_forward.sh           # copy of Experiments/templates/jobs/qalpa_forward.sh
│   └── qalpa_backward.sh          # copy of Experiments/templates/jobs/qalpa_backward.sh
├── logs/
└── results/

Data/my_exp_0/
```

Names are versioned: if `my_exp_0` exists, the next call creates `my_exp_1`, then `my_exp_2`, and so on.

## Placeholders it fills in

| Placeholder | Replaced with | In |
|---|---|---|
| `template_experiment` | The versioned experiment name, e.g. `my_exp_0` | `experiment_config.yaml` |
| `template_dataset` | The `--dataset-path` value | `experiment_config.yaml` |
| `@@REPO_ROOT@@` | Absolute repository path | `experiment_config.yaml` and both job scripts |
| `@@EXP_CONFIG_DIR@@` | Absolute path of `Experiments/<exp>/configs/` | Both job scripts |
| `@@JOB_DIR@@` | Absolute path of `Experiments/<exp>/jobs/` (the jobs' working directory) | Both job scripts |

## After initialization

1. Copy the trajectory CSV into `Experiments/my_exp_0/configs/` ([Designing a trajectory](../user-guide/trajectory.md)).
2. Review `Experiments/my_exp_0/configs/experiment_config.yaml` ([Quick start, step 3](../getting-started/quickstart.md#3-review-the-configuration)).
3. Submit the jobs from `Experiments/my_exp_0/jobs/` ([Submitting jobs](../user-guide/running-on-slurm.md#submitting-jobs)).

The iteration counter is `loop.current_iter` in `experiment_config.yaml`. It starts at 0, and every backward job advances it.

## Errors

| Error | Cause |
|---|---|
| `FileNotFoundError: Template config not found …` | `Experiments/templates/configs/experiment_config.yaml` is missing |
| `FileNotFoundError: Template forward/backward job script not found …` | A job template in `Experiments/templates/jobs/` is missing |
| `FileNotFoundError` for the dataset | `--dataset-path` does not exist |
| `Error: experiment name is required` | Neither `name` nor `--name` was given |
