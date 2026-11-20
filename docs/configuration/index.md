# Configuration overview

QALPA composes its configuration with [Hydra](https://hydra.cc/) and OmegaConf. Settings live at three levels, from shared to specific, and an experiment's `experiment_config.yaml` is the entry point that pulls in the lower levels.

| Level | Files | Holds | Scope |
|---|---|---|---|
| **Root** | {repo}`global_config.yaml` | Shared paths, seed, FHI-aims executable | All experiments |
| **Framework** | {repo}`src/configs/src_paths.yaml`, {repo}`src/configs/eqdiff_model.yaml`, {repo}`src/configs/al.yaml`, plus assets such as `nctr1.in` and the Slater–Koster files | Code paths; diffusion-model architecture and optimizer; active-learning defaults | All experiments |
| **Experiment** | `Experiments/<experiment>/configs/experiment_config.yaml` | Everything specific to one experiment: dataset, trajectory, loop, model, QM and planner settings | One experiment |

(composition-order)=
## How the configuration is assembled

When a job starts, Hydra composes the configuration from these sources. Later sources override earlier ones.

| Order | Source | Merged | Scope |
|:-:|---|---|---|
| 1 | {repo}`global_config.yaml` | At the top level | All experiments |
| 2 | {repo}`src/configs/src_paths.yaml` | At the top level | All experiments |
| 3 | {repo}`src/configs/eqdiff_model.yaml` | At the top level; training also reads the file directly | All experiments |
| 4 | `Experiments/<experiment>/configs/experiment_config.yaml` | Primary configuration | One experiment |
| 5 | `key=value` arguments to `forward.py` or `backward.py` | Command-line overrides | One job |
| – | {repo}`src/configs/al.yaml` | Under `al:`, by `src/utils/al_config.py` | Every `al` key the experiment does not set |

Levels 1 to 3 are imported through the `defaults` list at the top of the experiment configuration:

```yaml
defaults:
  - /global_config
  - /src/configs/src_paths@_global_
  - /src/configs/eqdiff_model@_global_

hydra:
  searchpath:
    - file://@@REPO_ROOT@@     # replaced by the absolute repository path
```

The defaults are absolute (they start with `/`), so Hydra resolves them from the repository root. `init_experiment.py` writes that root into `hydra.searchpath`.

## Life of an experiment's configuration

Over the life of an experiment, the configuration goes through four stages:

1. **Initialization.** `init_experiment.py` copies the template and fills in `hydra.searchpath`, `experiment_name`, `experiment.name` and `base_dataset_path`. In the job scripts it fills in the repository, configuration and job paths. See [`init_experiment`](../reference/init-experiment.md).
2. **Composition and validation.** Each job composes the configuration, merges `al.yaml` under `al:` and checks the result ([Validation](validation.md)).
3. **Training.** The forward job runs the training script with `src/configs/eqdiff_model.yaml`, followed by flags derived from `EQdiffParam` ([Diffusion model](diffusion-model.md)).
4. **Write-back.** At the end of every backward job, the composed configuration, including any command-line overrides and the advanced `loop.current_iter`, is written back to `experiment_config.yaml`.

(write-back)=
```{important}
Because of the write-back, `experiment_config.yaml` becomes self-contained after the first backward job. The `defaults` list, `hydra.searchpath` and all comments are removed, and the values of levels 1 to 3 are copied in. From then on:

- Edits to `global_config.yaml` and `src_paths.yaml` no longer reach the experiment. Change those values in its `experiment_config.yaml`.
- Diffusion-model settings still belong in `src/configs/eqdiff_model.yaml`. Training reads that file, so the copied model keys in `experiment_config.yaml` have no effect.
- Command-line overrides given to a backward job stay in effect for all later iterations.
- `al.yaml` is still merged at run time and keeps supplying every `al` key the experiment does not set.

The commented template remains available in {repo}`Experiments/templates/configs/experiment_config.yaml`.
```

(what-to-edit-where)=
## What to edit where

| To change | Edit, before the first backward job | Edit, after it |
|---|---|---|
| Shared paths, seed, FHI-aims executable | `global_config.yaml` | The experiment's `experiment_config.yaml` |
| Code and script paths | `src/configs/src_paths.yaml` | The experiment's `experiment_config.yaml` |
| Diffusion-model architecture and optimizer | `src/configs/eqdiff_model.yaml` | `src/configs/eqdiff_model.yaml` |
| Active-learning defaults for all experiments | `src/configs/al.yaml` | `src/configs/al.yaml` |
| Anything experiment-specific | `experiment_config.yaml` | `experiment_config.yaml` |

## Overriding settings for one job

The job scripts run the pipeline as Python modules from the repository root. To run a step by hand, or to override keys for a single run, pass `key=value` arguments after the config name:

```bash
cd <repo>
export PYTHONPATH="$PWD:$PYTHONPATH"

python -m src.scripts.forward \
  --config-path "$PWD/Experiments/my_exp_0/configs" \
  --config-name experiment_config \
  EQdiffParam.num_epochs=100 \
  geometry.optimizer=EquiDTB
```

`backward.py` takes the same arguments. Pass `--config-path` as an absolute path: Hydra resolves a relative one against `src/scripts/`, not against your working directory.

```{warning}
Overrides given to a backward job are written back into `experiment_config.yaml` and stay in effect for every later iteration. To restart from a block without touching the config, use `QALPA_START_AT` instead ([Resuming](../user-guide/resuming.md)).
```

(edit-labels)=
## Reading the reference tables

The reference pages in this section list every key with its default and an **Edit** label:

| Label | Meaning |
|---|---|
| **Mandatory** | Set or verify for every experiment or installation |
| Optional | The shipped value works; change it to tune behaviour |
| Automatic | Written by `init_experiment.py` or by the jobs; edit only to repair or restart a run |
| Keep | The pipeline depends on this value; change it only if you move the code |
| Overridden | Replaced at run time by the key named in the explanation; editing it has no effect |
| Informational | Not read by `forward.py` or `backward.py` |

**Default** is the value shipped in the source file of each section. For the experiment configuration, it is the value in the template that `init_experiment.py` copies.

| Page | Covers |
|---|---|
| [Shared settings](shared.md) | `global_config.yaml` and `src/configs/src_paths.yaml` |
| [Diffusion model](diffusion-model.md) | `src/configs/eqdiff_model.yaml` |
| [Experiment settings](experiment.md) | `experiment_config.yaml`: data, loop and pipeline, model, MLflow, QM, screening, HDF5 |
| [Active learning](active-learning.md) | The `al` section and `src/configs/al.yaml` |
| [Validation](validation.md) | Checks run at job start |
| [Recipes](recipes.md) | Snippets for common changes |
