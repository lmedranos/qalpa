# QALPA Configuration Reference

Every setting QALPA reads, where it lives, what it does and when to change it. For a guided first run, start with the [README](README.md).

**Edit labels**

| Label | Meaning |
|---|---|
| **Mandatory** | Set or verify for every experiment or installation |
| Optional | The shipped value works; change it to tune behaviour |
| Automatic | Written by `init_experiment.py` or by the jobs; edit only to repair or restart a run |
| Keep | The pipeline depends on this value; change it only if you move the code |
| Overridden | Replaced at run time by the key named in the explanation; editing it has no effect |
| Informational | Not read by `forward.py` or `backward.py` |

**Default** is the value shipped in the source file of each section. For the experiment configuration, it is the value in the template that `init_experiment.py` copies.

## Contents

1. [How the configuration is assembled](#1-how-the-configuration-is-assembled)
2. [Root configuration](#2-root-configuration)
3. [Diffusion model](#3-diffusion-model)
4. [Code paths](#4-code-paths)
5. [Experiment configuration](#5-experiment-configuration): [metadata and data](#51-metadata-and-data) · [loop and pipeline](#52-loop-and-pipeline) · [generative model](#53-generative-model) · [MLflow](#54-mlflow) · [geometry optimization](#55-geometry-optimization) · [trajectory and screening](#56-trajectory-and-screening) · [property calculation](#57-property-calculation) · [HDF5 output](#58-hdf5-output)
6. [Active learning](#6-active-learning): [switch and strategy](#61-switch-and-strategy) · [anchor mode](#62-anchor-mode) · [corridor and frontier modes](#63-corridor-and-frontier-modes) · [KDE and adaptive scoring](#64-kde-and-adaptive-scoring) · [hit rule and labelling](#65-hit-rule-and-labelling) · [deduplication](#66-deduplication)
7. [Keys with built-in defaults](#7-keys-with-built-in-defaults)
8. [Validation](#8-validation)
9. [Recipes](#9-recipes)

## 1. How the configuration is assembled

When a job starts, Hydra composes the configuration from these sources. Later sources override earlier ones.

| Order | Source | Merged | Scope |
|:-:|---|---|---|
| 1 | [`global_config.yaml`](global_config.yaml) | At the top level | All experiments |
| 2 | [`src/configs/src_paths.yaml`](src/configs/src_paths.yaml) | At the top level | All experiments |
| 3 | [`src/configs/eqdiff_model.yaml`](src/configs/eqdiff_model.yaml) | At the top level; training also reads the file directly | All experiments |
| 4 | `Experiments/<experiment>/configs/experiment_config.yaml` | Primary configuration | One experiment |
| 5 | `key=value` arguments to `forward.py` or `backward.py` | Command-line overrides | One job |
| – | [`src/configs/al.yaml`](src/configs/al.yaml) | Under `al:`, by `src/utils/al_config.py` | Every `al` key the experiment does not set |

Over the life of an experiment, the configuration goes through four stages:

1. **Initialization.** `init_experiment.py` copies the template and fills in `hydra.searchpath`, `experiment_name`, `experiment.name` and `base_dataset_path`. In the job scripts it fills in the repository, configuration and job paths.
2. **Composition and validation.** Each job composes the configuration, merges `al.yaml` under `al:` and checks the result ([Section 8](#8-validation)).
3. **Training.** The forward job runs the training script with `src/configs/eqdiff_model.yaml`, followed by flags derived from `EQdiffParam` ([Section 3](#3-diffusion-model)).
4. **Write-back.** At the end of every backward job, the composed configuration, including any command-line overrides and the advanced `loop.current_iter`, is written back to `experiment_config.yaml`.

> [!IMPORTANT]
> Because of the write-back, `experiment_config.yaml` becomes self-contained after the first backward job. The `defaults` list, `hydra.searchpath` and all comments are removed, and the values of levels 1 to 3 are copied in. From then on:
> - Edits to `global_config.yaml` and `src_paths.yaml` no longer reach the experiment. Change those values in its `experiment_config.yaml`.
> - Diffusion-model settings still belong in `src/configs/eqdiff_model.yaml`. Training reads that file, so the copied model keys in `experiment_config.yaml` have no effect.
> - Command-line overrides given to a backward job stay in effect for all later iterations.
> - `al.yaml` is still merged at run time and keeps supplying every `al` key the experiment does not set.
>
> The commented template remains available in `Experiments/templates/configs/experiment_config.yaml`.

| To change | Edit, before the first backward job | Edit, after it |
|---|---|---|
| Shared paths, seed, FHI-aims executable | `global_config.yaml` | The experiment's `experiment_config.yaml` |
| Code and script paths | `src/configs/src_paths.yaml` | The experiment's `experiment_config.yaml` |
| Diffusion-model architecture and optimizer | `src/configs/eqdiff_model.yaml` | `src/configs/eqdiff_model.yaml` |
| Active-learning defaults for all experiments | `src/configs/al.yaml` | `src/configs/al.yaml` |
| Anything experiment-specific | `experiment_config.yaml` | `experiment_config.yaml` |

## 2. Root configuration

Source: [`global_config.yaml`](global_config.yaml)

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `root` | `${repo_root:}` | Repository root, resolved from the location of `forward.py` and `backward.py`; all paths below derive from it | Keep |
| `Data_path` | `${root}/Data` | Root of all experiment data; point it to another filesystem to keep data elsewhere | Optional |
| `Experiments_path` | `${root}/Experiments` | Root of the experiment folders; `init_experiment.py` always creates them in `<repo>/Experiments` | Keep |
| `Code_path` | `${root}/src` | Source directory | Keep |
| `paths.data` | `${Data_path}` | Base of every data path in the experiment configuration; change `Data_path` instead | Keep |
| `paths.experiments` | `${Experiments_path}` | Base of `exp_config_path` and `query.trajectory_csv` | Keep |
| `paths.code` | `${Code_path}` | Available for interpolation | Informational |
| `paths.config` | `${root}/configs` | Available for interpolation; the framework configs live in `src/configs/` | Informational |
| `system.seed` | `42` | Seed of target planning (`plan`) and coverage estimation (`evaluate`). Training uses `seed` in `eqdiff_model.yaml`, and the data split uses `EQdiffParam.split_seed` | Optional |
| `system.fih_aims_exc` | `/software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x` | FHI-aims executable used by `prop_calc` | **Mandatory** with FHI-aims |

## 3. Diffusion model

Source: [`src/configs/eqdiff_model.yaml`](src/configs/eqdiff_model.yaml)

The forward job trains with `run_train.py --conf src/configs/eqdiff_model.yaml <flags>`. The file is applied first and the flags second, so keys marked *Overridden* take their value from `EQdiffParam`. The flags also set the save directory, dataset root, GPU count, number of epochs, test interval, checkpoint, property conditioning and replay options, so adding those keys here has no effect. Every key must be a known argument of the training script ([`hparams.py`](src/eqdiff/e3mol_aqm/experiments/hparams.py)); an unknown key stops training with `Unknown argument in config file`. Changes apply to every experiment trained afterwards.

**Model**

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `sdim` | `256` | Scalar feature dimension of the equivariant (EQGAT) layers | Optional |
| `vdim` | `256` | Vector feature dimension | Optional |
| `edim` | `128` | Edge feature dimension | Optional |
| `num_layers` | `12` | Number of message-passing layers | Optional |
| `use_cross_product` | `false` | Cross-product interactions in the vector updates | Optional |
| `edge_mp` | `false` | Edge message passing | Optional |
| `fully_connected` | `true` | Fully connected molecular graphs | Optional |
| `local_global_model` | `false` | Combined local and global message passing | Optional |
| `vector_aggr` | `mean` | Aggregation of vector features | Optional |
| `use_pos_norm` | `true` | Normalization of coordinates inside the network | Optional |
| `bond_prediction` | `false` | Auxiliary bond prediction | Optional |
| `bond_model_guidance` | `false` | Guide sampling with a separate bond model (needs a bond-model checkpoint) | Optional |
| `additional_feats` | `false` | Extra atom features (aromaticity, ring membership, hybridization); `true` selects a different model class | Keep |

**Diffusion**

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `continuous` | `false` | Continuous-time diffusion; `false` is the discrete diffusion model QALPA uses | Keep |
| `noise_scheduler` | `adaptive` | Noise schedule | Optional |
| `timesteps` | `500` | Number of diffusion steps | Optional |
| `eps_min` | `1.e-3` | Minimum epsilon of the schedule | Optional |
| `beta_min`, `beta_max` | `1.e-4`, `2.e-2` | Beta range of the schedule | Optional |
| `loss_weighting` | `snr_t` | Loss weighting over noise levels | Optional |

**Training and optimizer**

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `seed` | `42` | Training seed | Optional |
| `batch_size` | `200` | Training batch size; replaced by `EQdiffParam.batch_size_tr` | Overridden |
| `inference_batch_size` | `70` | Batch size of validation and test loading and of the sampling evaluations during training | Optional |
| `lr` | `2.e-4` | Initial learning rate | Optional |
| `lr_scheduler` | `reduce_on_plateau` | `reduce_on_plateau`, `cosine_annealing` or `cyclic` | Optional |
| `lr_factor`, `lr_patience`, `lr_cooldown` | `0.75`, `50`, `0` | Reduce-on-plateau factor, patience and cooldown | Optional |
| `lr_min`, `lr_step_size` | `5.e-5`, `5.e-5` | Lower learning rate and step size of the cyclic and cosine schedulers | Optional |
| `weight_decay` | `1.e-12` | Optimizer weight decay | Optional |
| `ema_decay` | `0.999` | Decay of the exponential moving average of the weights | Optional |
| `grad_clip_val` | `10` | Gradient clipping value | Optional |
| `accum_batch` | `1` | Gradient-accumulation steps | Optional |
| `max_num_neighbors` | `128` | Maximum number of graph neighbours | Optional |
| `gamma` | `0.99` | Not used by the discrete diffusion trainer | Informational |
| `max_time` | `06:00:00:00` | Not passed to the trainer; the training length is set by `EQdiffParam.num_epochs` | Informational |

**Data and runtime**

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `dataset` | `aqm` | Dataset loader; replaced by `EQdiffParam.dataset` | Overridden |
| `concat_datasets`, `alpha`, `beta` | `false`, `0.2`, `0.2` | Replay settings; replaced by the `EQdiffParam` values when `EQdiffParam.concat_datasets` is true | Overridden |
| `remove_hs` | `false` | Train without hydrogen atoms. Keep `false`: the QM steps need explicit hydrogens | Keep |
| `use_adaptive_loader` | `true` | Size-adaptive batching; not used by the `aqm_qm7x` loader | Informational |
| `id` | `0` | Run index; checkpoints go to `model_iter_<n>/run<id>/`, and the pipeline reads `run0/` | Keep |
| `num_workers` | `8` | Data-loader workers | Optional |
| `precision` | `32` | Numerical precision of training | Optional |

## 4. Code paths

Source: [`src/configs/src_paths.yaml`](src/configs/src_paths.yaml)

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `Code_path` | `${root}/src` | Source directory | Keep |
| `code_paths.EQdiff` | `${Code_path}/eqdiff/e3mol_aqm` | Diffusion-model code | Keep |
| `code_paths.pipeline_blocks` | `${Code_path}/pipeline_blocks` | Pipeline-block scripts | Keep |
| `code_paths.configs` | `${Code_path}/configs` | Framework configs, the EquiDTB model, the Slater–Koster files and the FHI-aims control file `nctr1.in` | Keep |
| `code_paths.actdiff_utils`, `code_paths.scripts` | `${Code_path}/utils`, `${Code_path}/scripts` | Available for interpolation | Informational |
| `EQdiff.split` | `${code_paths.EQdiff}/experiments/data/aqm/split_data.py` | Dataset split (`split` block) | Keep |
| `EQdiff.train` | `${code_paths.EQdiff}/experiments/run_train.py` | Training (`train` block) | Keep |
| `EQdiff.generate` | `${code_paths.EQdiff}/experiments/run_evaluation_qm.py` | Generation (`generate` block) | Keep |
| `pipeline_blocks.GeoRelax` | `${code_paths.pipeline_blocks}/optimize_geometry.py` | Geometry optimization (`relax_geo` block) | Keep |
| `pipeline_blocks.PropCalc` | `${code_paths.pipeline_blocks}/prop_calc.py` | Property calculation with EquiDTB or DFTB+ (`prop_calc` block) | Keep |
| `pipeline_blocks.SAScore`, `.PrePropCalc`, `.PostPropCalc`, `.MolFilter`, `.MolHDF5` | `${code_paths.pipeline_blocks}/<module>.py` | These blocks are imported as Python modules | Informational |
| `utils.*` | `${code_paths.actdiff_utils}/<module>.py` | Utility modules, imported directly | Informational |
| `scripts.forward`, `scripts.backward` | `${code_paths.scripts}/forward.py`, `.../backward.py` | The job scripts run `python -m src.scripts.forward` and `python -m src.scripts.backward` | Informational |
| `FHI-aims.ctrl` | `${code_paths.configs}/nctr1.in` | Property calculation with FHI-aims uses `<code_paths.configs>/nctr1.in` directly; replace that file to change the control settings | Informational |

## 5. Experiment configuration

Source: [`Experiments/templates/configs/experiment_config.yaml`](Experiments/templates/configs/experiment_config.yaml), copied to `Experiments/<experiment>/configs/experiment_config.yaml` by `init_experiment.py`.

### 5.1 Metadata and data

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `defaults` | `/global_config`, `/src/configs/src_paths`, `/src/configs/eqdiff_model` | Imports levels 1 to 3; removed by the write-back | Keep |
| `hydra.searchpath` | `file://@@REPO_ROOT@@` | Lets Hydra find the imported files; removed by the write-back | Automatic |
| `experiment_name` | `template_experiment` | Versioned name (`<name>_<v>`) of the experiment folders under `Experiments/` and `Data/`, and of the MLflow experiment | Automatic |
| `experiment.name` | `template_experiment` | Same as `experiment_name` | Informational |
| `experiment.description` | `molecular generation experiment` | Free text | Informational |
| `exp_config_path` | `${paths.experiments}/${experiment_name}/configs/experiment_config.yaml` | Target of the write-back; `al_state.json`, `density_reference.npy` and `al_metrics.csv` are stored two levels up | Keep |
| `base_dataset_path` | `template_dataset` | Initial dataset (HDF5), taken from `--dataset-path`. The first job copies it to `Data/<experiment>/hdf5/<stem>-input.hdf5`, which all later jobs use | Automatic |
| `data.exp_data_path` | `${paths.data}/${experiment_name}` | Data folder of the experiment | Optional |

### 5.2 Loop and pipeline

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `loop.num_iterations` | `8` | Last iteration index; iterations run from 0 to this value, and a job with a higher `current_iter` stops | **Mandatory** |
| `loop.current_iter` | `0` | Iteration the next job runs; every backward job increments it | Automatic |
| `pipeline.blocks` | `[split, train, plan, generate, dedup, score_filter, relax_geo, prop_calc, prop_filter, hdf5, compress]` | Blocks to run; each job runs the listed blocks it owns, in order | Optional |
| `pipeline.start_at` | `split` | First block to run in both jobs; earlier blocks are skipped. It is not reset automatically | Optional |
| `max_parallel_tasks` | `20` | Maximum number of relaxations or QM calculations running at once in the backward job | Optional |
| `processing_timeout_hours` | `2` | Time limit per relaxation or QM calculation; timed-out molecules are moved to a `killed/` folder | Optional |

A block runs only if it is listed in `pipeline.blocks` and does not come before `pipeline.start_at`.

| Block | Job | Runs if | Notes |
|---|---|---|---|
| `split` | Forward | Listed | Prepares `dataset_<n>`; skipped if it is already prepared |
| `train` | Forward | Listed | Starts from the previous iteration's checkpoint (`best_mol_stab.ckpt`, else `run0/last.ckpt`) |
| `plan` | Forward | `al.enabled` | Always placed right before `generate`; an existing `iter_<n>/targets.json` is reused |
| `generate` | Forward | Listed | One generation run per planned target |
| `dedup` | Forward | `al.enabled` and `al.dedup.enabled` | Inserted after `generate` automatically, and removed when deduplication is disabled |
| `score_filter` | Backward | `score` in `query.filter_scheme` | QED and SA screening |
| `relax_geo` | Backward | `geometry.optimizer` is supported | Geometry optimization |
| `prop_calc` | Backward | The trajectory has property columns | QM properties |
| `prop_filter` | Backward | `accuracy` in `query.filter_scheme` | With active learning, each molecule is compared with its own target |
| `hdf5` | Backward | Listed | Appends the labelled molecules to the HDF5 files |
| `evaluate` | Backward | `al.enabled` and listed | Runs after `hdf5`; appends hit rate and coverage to `Experiments/<experiment>/al_metrics.csv` |
| `compress` | – | – | Listed in the template but not implemented; it has no effect |

### 5.3 Generative model

All keys below sit under `EQdiffParam`.

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `num_samples` | `200` | Molecules generated per iteration. With active learning and `al.anchor_extra_fraction`, the anchor target gets this many; otherwise it is the total over all targets | **Mandatory** |
| `properties_list` | `['HLgap', 'eMBD']` | Conditioning properties, in context order. Each must exist in the initial dataset and as a trajectory column; with active learning, only `HLgap`, `eMBD`, `DIP`, `eH` and `eL` are supported. Set it before iteration 0 | **Mandatory** |
| `context_mapping` | `True` | Train the model conditioned on `properties_list`; property-guided generation needs it | Keep |
| `num_context_features` | `${len:${EQdiffParam.properties_list}}` | Derived; the forward job passes the length of `properties_list` itself | Informational |
| `dataset` | `aqm_qm7x` | Dataset loader of the training script, for AQM/QM7-X HDF5 data | Optional |
| `batch_size` | `20` | Generation batch size | Optional |
| `batch_size_tr` | `64` | Training batch size | Optional |
| `num_epochs` | `50` | Training epochs per iteration | Optional |
| `test_interval` | `25` | Epochs between sampling evaluations (1000 molecules); `best_mol_stab.ckpt` is saved only at these evaluations | Optional |
| `load_ckpt_from_pretrained` | `null` | Checkpoint to start iteration 0 from; later iterations always start from the previous one | Optional |
| `concat_datasets` | `true` | Keep one dataset per iteration and train on all of them with weighted sampling; needs `hdf5.hdf5_iter_flag: True`. With `false`, a single dataset is split again from the cumulative HDF5 every iteration | Optional |
| `alpha` | `0.6` | Sampling share of all datasets except the newest, which gets `1 − alpha`. Must lie in [0, 1] | Optional |
| `beta` | `0.9` | Weight of the initial dataset relative to one intermediate dataset; below 1 keeps it under them. Must lie in [0, 1] | Optional |
| `min_original_share` | `0.1` | Lower bound for the initial dataset's share, taken from the newest dataset; `0` disables it | Optional |
| `samples_per_epoch` | `1000` | Draws per epoch once more than one dataset exists; `null` uses the size of the training set | Optional |
| `split_seed` | `42` | Seed of the per-molecule train/validation/test split | Optional |
| `dataset_root` | `${paths.data}/${experiment_name}/diffusion_wdir/data/` | Prepared training data (`dataset_<n>/`) | Optional |
| `model_root` | `${paths.data}/${experiment_name}/diffusion_wdir/model_registry/` | Checkpoints (`model_iter_<n>/`); also read by the MLflow sync | Optional |
| `test_save_dir` | `${paths.data}/${experiment_name}/diffusion_wdir/test/` | Not passed to training | Informational |
| `save_dir` | `${paths.experiments}/${experiment_name}/logs/` | Not passed to training; training writes into `model_root` | Informational |

With three or more datasets and *m* intermediate ones, the sampling shares are: newest `1 − alpha`, each intermediate `alpha / (m + beta)`, initial `beta × alpha / (m + beta)`. With exactly two datasets, the initial dataset gets `alpha`. `min_original_share` then raises the initial dataset's share, if needed, at the expense of the newest.

### 5.4 MLflow

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `mlflow.enabled` | `false` | Mirror the training curves of all iterations into MLflow after every training run | Optional |
| `mlflow.python` | Site-specific path | Interpreter with `mlflow`, `tensorboard` and `matplotlib` that runs `src/scripts/sync_mlflow.py` | **Mandatory** if enabled |
| `mlflow.tracking_uri` | `file://${paths.data}/mlruns` | MLflow store shared by all experiments; each experiment appears under its `experiment_name` | Optional |

### 5.5 Geometry optimization

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `geometry.optimizer` | `EquiDTB` | `EquiDTB` (machine-learned), `DFTB+` or `FHI-aims`. With any other value, relaxation is skipped | **Mandatory** |
| `geometry.calculate_forces` | `True` | Also compute forces for the relaxed and unrelaxed geometries | Optional |
| `geometry.model` | `${code_paths.configs}/MACE_model_swa.model` | EquiDTB model file | **Mandatory** with EquiDTB |

### 5.6 Trajectory and screening

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `query.trajectory_csv` | `${paths.experiments}/${experiment_name}/configs/trajectory_*.csv` | Trajectory file; with a pattern, the first match in alphabetical order is used | **Mandatory** (copy the CSV) |
| `query.props` | `${EQdiffParam.properties_list}` | Conditioning properties read from the trajectory and passed to generation in this order; keep it identical to `properties_list` | Keep |
| `query.scores` | `["SAScore", "QED"]` | `trajectory` mode only: score columns read from the CSV. Each must be a column, and the Navigator names the QED column `QEDScore` | Optional |
| `query.score_reference` | `initial_mean` | `initial_mean`: thresholds from the mean QED and SA score of the initial dataset, computed once and cached next to the input HDF5. `trajectory`: per-waypoint thresholds from the CSV | Optional |
| `query.score_relax` | `{QED: 0.2, SAScore: 0.2}` | `initial_mean` only: a molecule passes if QED > (1 − r) × mean QED and SA < (1 + r) × mean SA. Each r lies in (−1, 1); r > 0 loosens and r < 0 tightens | Optional |
| `query.filter_scheme` | `[score, accuracy]` | `score` enables `score_filter`; `accuracy` enables `prop_filter` | Optional |
| `query.filter_threshold` | `0.1` | Relative tolerance of the property-accuracy filter; also the default relative tolerance of `al.tolerances` | Optional |

### 5.7 Property calculation

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `properties.calculator` | `EquiDTB` | `EquiDTB`, `DFTB+` or `FHI-aims`. FHI-aims runs `system.fih_aims_exc` with `src/configs/nctr1.in`; DFTB+ needs the patched ASE calculator and `DFTB_PREFIX`, which the backward job sets | **Mandatory** |
| `properties.model` | `${code_paths.configs}/MACE_model_swa.model` | EquiDTB model file | **Mandatory** with EquiDTB |

### 5.8 HDF5 output

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `hdf5.appending_flag` | `True` | Append labelled molecules to the cumulative `<stem>-output.hdf5`, which starts as a copy of the initial dataset. Required for active learning; with `False`, every job recreates the file | **Mandatory** with active learning |
| `hdf5.hdf5_iter_flag` | `True` | Also write each iteration's molecules to `iter_<n>/hdf5/output.hdf5`, the next iteration's `dataset_<n+1>`; required by `concat_datasets` | Optional |
| `hdf5.extra_properties_flag` | `False` | Also write additional QM properties, such as forces, charges and volumes, to separate `extra_properties` HDF5 files | Optional |

## 6. Active learning

Defaults: [`src/configs/al.yaml`](src/configs/al.yaml). The `al` section of the experiment configuration is merged over this file when a job starts, key by key inside nested sections such as `density`, so the experiment only needs the keys it changes. **Template** is the value in the experiment template; – means the template does not set the key, so the `al.yaml` value applies.

The template runs the KDE anchor planner: `query_strategy: kde_map`, `kde_region: anchor`, `anchor_score: relative_pct`. The [README](README.md#kde-target-planning) explains how it works.

### 6.1 Switch and strategy

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `enabled` | `false` | `true` | Turn active learning on: adds the `plan` and `dedup` blocks and labels every molecule against its own target | **Mandatory** decision |
| `query_strategy` | `kde_map` | `kde_map` | `kde_map` (KDE planning), `frontier` (advance along the trajectory by hit rate) or `trajectory` (waypoint *n* only) | Optional |
| `kde_region` | `corridor` | `anchor` | `kde_map` only: `anchor` samples a box around the current waypoint; `corridor` samples a padded band along the trajectory | Optional |
| `generation_budget_hours` | `12` | `6` | No further targets are started once generation has used this much wall time | Optional |

### 6.2 Anchor mode

These keys apply with `kde_region: anchor`.

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `include_anchor` | `true` | `true` | Waypoint *n* itself is target 0 | Optional |
| `anchor_box` | `0.1` | `0.1` | Candidates are sampled within ± this fraction of each anchor value; for values near 0, the trajectory span on that axis is used | Optional |
| `anchor_n_candidates` | `2000` | `200` | Candidates sampled in the box; `null` uses `density.n_candidates` | Optional |
| `anchor_n_targets` | `3` | `2` | Additional targets per iteration; the anchor comes on top | Optional |
| `anchor_score` | `kde` | `relative_pct` | `relative_pct`: distance to the anchor plus density difference, with farthest-point selection. `kde`: the adaptive score of [Section 6.4](#64-kde-and-adaptive-scoring), with repulsion | Optional |
| `anchor_min_datnum` | `1` | `1` | `relative_pct` only: minimum atom-count difference to the anchor. `anchor_box × atNUM` must allow it, or no candidate qualifies | Optional |
| `anchor_extra_fraction` | `null` | `0.2` | Molecules per additional target = round(fraction × `num_samples`), on top of `num_samples` at the anchor. `null` divides `num_samples` among all targets, by score with `kde` and equally with `relative_pct` | Optional |

### 6.3 Corridor and frontier modes

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `targets_per_iter` | `4` | – | Corridor: targets planned per iteration | Optional |
| `corridor_pad` | `0.1` | – | Corridor: padding around the trajectory polyline, as a fraction of its span on each axis | Optional |
| `corridor_lookahead` | `null` | – | Corridor: `null` uses the whole trajectory; `k` uses the waypoints up to `current_iter + k` | Optional |
| `advance_hit_rate` | `0.3` | – | Frontier: hit rate a waypoint must reach before the next one is used | Optional |

### 6.4 KDE and adaptive scoring

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `density.bandwidth` | `scott` | `scott` | KDE bandwidth in standardized units: `scott` (Scott's rule) or a number. Along `atNUM`, it is never narrower than one atom | Optional |
| `density.density_percentiles` | `null` | `[5.0, 15.0]` | Keep only candidates whose density under the KDE of the initial dataset lies between these percentiles of the initial dataset's own leave-one-out densities. Computed once and cached in `density_reference.npy`; `null` applies no band | Optional |
| `density.n_candidates` | `2000` | – | Corridor: candidates sampled per iteration; anchor mode uses it when `anchor_n_candidates` is `null` | Optional |
| `density.w_sparse` | `0.5` | – | Adaptive score: weight of data sparsity, the negative log density of all labelled molecules | Optional |
| `density.w_err` | `0.5` | – | Adaptive score: weight of the generator's past error near the candidate | Optional |
| `density.w_explore` | `0.5` | – | Adaptive score: weight of 1 / √(n_eff + 1), which favours rarely requested regions | Optional |
| `density.min_reach` | `0.05` | `0.0` | Drop candidates whose estimated hit rate is below this value once more than `min_neff` past requests lie nearby; `0` disables it | Optional |
| `density.logp_floor` | `-20.0` | – | Lower clip of the log density | Optional |
| `min_neff` | `5` | `5` | Kernel-weighted number of nearby past requests needed before a region can be judged unreachable; also the coverage threshold of `evaluate` | Optional |

### 6.5 Hit rule and labelling

A molecule is a *hit* if, for every conditioning property, abs(achieved − target) ≤ max(relative × abs(target), absolute). The atom count is not part of the rule.

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `tolerances.default.relative` | `null` | `null` | Relative tolerance; `null` uses `query.filter_threshold` | Optional |
| `tolerances.default.absolute` | `0.0` | `0.0` | Absolute tolerance | Optional |
| `tolerances.<property>` | – | – | Per-property override: a number (relative tolerance) or `{relative: …, absolute: …}`. Tolerances must be non-negative | Optional |
| `keep_off_target` | `true` | `true` | Without the `prop_filter` block, keep misses in the HDF5 files with `al_hit = 0`; `false` drops them. With `prop_filter`, misses are always dropped | Optional |

### 6.6 Deduplication

Deduplication runs at the end of the forward job, right after `generate`.

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `dedup.enabled` | `false` | `true` | Remove molecules with identical connectivity | Optional |
| `dedup.scope` | `all` | `all` | `all`: one copy across all targets, and the lowest target id wins. `target`: one copy within each target | Optional |
| `dedup.against_initial` | `true` | `true` | Also remove molecules whose connectivity occurs in the initial dataset; those connectivities are cached next to the input HDF5 | Optional |
| `dedup.cov_factor` | `1.3` | – | Two atoms are bonded if their distance is below `cov_factor` × the sum of their covalent radii | Optional |

## 7. Keys with built-in defaults

These keys are not in any shipped file. Add them to `experiment_config.yaml` to change the built-in value.

| Parameter | Built-in default | Explanation | Edit |
|---|---|---|---|
| `query.fix_num_nodes` | `true` | Use the trajectory's `atNUM` column as the atom count of generated molecules and as a planning axis. Keep `true` with active learning when the CSV has an `atNUM` column | Optional |
| `EQdiffParam.n_nodes` | Not set | Atom count for generation when the trajectory provides none: an integer, or a list with one value per iteration. If not set, atom counts are sampled from the training data | Optional |

## 8. Validation

These checks run when a job starts, or when the block that needs the setting runs. A failed check stops the job with an error that names the setting.

| Setting | Requirement |
|---|---|
| `loop.current_iter` | At most `loop.num_iterations` |
| `query.trajectory_csv` | A matching file exists and contains every `query.props` column (and, in `trajectory` mode, every `query.scores` column) |
| `query.props` | With active learning: not empty, and only `HLgap`, `eMBD`, `DIP`, `eH`, `eL` |
| `query.score_reference`, `query.score_relax` | `initial_mean` or `trajectory`; relax keys `QED` and `SAScore` with values in (−1, 1) |
| `EQdiffParam.context_mapping` | Needs a non-empty `properties_list` |
| `EQdiffParam.concat_datasets` | Needs `hdf5.hdf5_iter_flag: True`; from iteration 1, the previous backward job must have written `iter_<n−1>/hdf5/output.hdf5` |
| `EQdiffParam.alpha`, `beta`, `min_original_share` | `alpha` and `beta` in [0, 1]; `min_original_share` in [0, 1) and small enough to leave a share for the newest dataset |
| `hdf5.appending_flag` | `True` with active learning |
| `al.query_strategy`, `al.kde_region`, `al.anchor_score`, `al.dedup.scope` | One of the listed options |
| `al.kde_region: corridor` | At least two waypoints |
| `al.anchor_box` | Greater than 0 |
| `al.anchor_n_targets`, `al.targets_per_iter` | At least 0 with the anchor, otherwise at least 1; at most the number of candidates |
| Number of targets | At most `EQdiffParam.num_samples` (including the anchor), unless `anchor_extra_fraction` is set |
| `al.anchor_extra_fraction` | Greater than 0; needs `include_anchor: true`; round(fraction × `num_samples`) ≥ 1 |
| `al.anchor_score: relative_pct` | An `atNUM` column in the trajectory with `query.fix_num_nodes: true`; `anchor_min_datnum` ≥ 0 |
| `al.density.density_percentiles` | `[low, high]` with 0 ≤ low < high ≤ 100 |
| `al.tolerances` | Non-negative values |

## 9. Recipes

All snippets go into `Experiments/<experiment>/configs/experiment_config.yaml`.

**Compute properties with DFT (FHI-aims)**

```yaml
properties:
  calculator: FHI-aims
system:
  fih_aims_exc: /path/to/aims.x   # overrides global_config.yaml for this experiment
```

Also load the FHI-aims modules in the backward job script. The control settings come from `src/configs/nctr1.in`.

**Condition on other properties**

```yaml
EQdiffParam:
  properties_list: ['HLgap', 'DIP']
```

Set this before iteration 0. The initial dataset must contain these properties, and the trajectory CSV needs them as columns, plus `atNUM`; `query.props` follows automatically.

**Plan more and wider KDE targets**

```yaml
al:
  anchor_box: 0.2
  anchor_n_candidates: 500
  anchor_n_targets: 4
  anchor_extra_fraction: 0.25
```

**Tighten the hit rule for one property**

```yaml
al:
  tolerances:
    HLgap: 0.05                            # relative tolerance only
    eMBD: {relative: 0.1, absolute: 0.05}
```

**Keep off-target molecules**

```yaml
pipeline:
  blocks: [split, train, plan, generate, dedup, score_filter, relax_geo, prop_calc, hdf5, evaluate]
al:
  keep_off_target: true
```

Misses are then stored with `al_hit = 0`. Their QM labels are still correct, so they also join the next training set. `evaluate` logs the hit rate per iteration.

**Explore the whole trajectory corridor**

```yaml
al:
  kde_region: corridor
  targets_per_iter: 4
```

This needs at least two waypoints and uses the adaptive score of [Section 6.4](#64-kde-and-adaptive-scoring). Until the molecules of a first iteration have been labelled, it requests the plain waypoint.

**Run without active learning**

```yaml
al:
  enabled: false
```

The trajectory then needs one row per iteration, from 0 to `loop.num_iterations`.

**Start from a pretrained model**

```yaml
EQdiffParam:
  load_ckpt_from_pretrained: /path/to/model.ckpt
```

**Keep experiment data on another filesystem**

```yaml
Data_path: /scratch/<project>/qalpa-data   # top level; all data paths derive from it
```

Set it before the first job of the experiment.
