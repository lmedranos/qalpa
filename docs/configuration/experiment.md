# Experiment settings

Source: {repo}`Experiments/templates/configs/experiment_config.yaml`, copied to `Experiments/<experiment>/configs/experiment_config.yaml` by `init_experiment.py`.

The planner settings under `al:` have their own page: [Active learning](active-learning.md). Edit labels are explained in [Reading the reference tables](index.md#reading-the-reference-tables).

:::{dropdown} Full template: `experiment_config.yaml`
```{literalinclude} ../../Experiments/templates/configs/experiment_config.yaml
:language: yaml
```
:::

## Metadata and data

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

## Loop and pipeline

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `loop.num_iterations` | `8` | Last iteration index; iterations run from 0 to this value, and a job with a higher `current_iter` stops | **Mandatory** |
| `loop.current_iter` | `0` | Iteration the next job runs; every backward job increments it | Automatic |
| `pipeline.blocks` | `[split, train, plan, generate, dedup, score_filter, relax_geo, prop_calc, prop_filter, hdf5, compress]` | Blocks to run; each job runs the listed blocks it owns, in order | Optional |
| `pipeline.start_at` | `split` | First block to run in both jobs; earlier blocks are skipped. It is not reset automatically. For a single job, the environment variable `QALPA_START_AT` (the Navigator's **Start at**) takes priority and leaves this value unchanged | Optional |
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

## Generative model

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

**Sampling shares.** With three or more datasets and *m* intermediate ones:

| Dataset | Share |
|---|---|
| Newest | `1 − alpha` |
| Each intermediate | `alpha / (m + beta)` |
| Initial | `beta × alpha / (m + beta)` |

With exactly two datasets, the initial dataset gets `alpha`. `min_original_share` then raises the initial dataset's share, if needed, at the expense of the newest.

## MLflow

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `mlflow.enabled` | `false` | Mirror the training curves of all iterations into MLflow after every training run | Optional |
| `mlflow.python` | Site-specific path | Interpreter with `mlflow`, `tensorboard` and `matplotlib` that runs `src/scripts/sync_mlflow.py` | **Mandatory** if enabled |
| `mlflow.tracking_uri` | `file://${paths.data}/mlruns` | MLflow store shared by all experiments; each experiment appears under its `experiment_name` | Optional |

See [Monitoring training with MLflow](../user-guide/mlflow.md).

## Geometry optimization

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `geometry.optimizer` | `EquiDTB` | `EquiDTB` (machine-learned), `DFTB+` or `FHI-aims`. With any other value, relaxation is skipped | **Mandatory** |
| `geometry.calculate_forces` | `True` | Also compute forces for the relaxed and unrelaxed geometries | Optional |
| `geometry.model` | `${code_paths.configs}/MACE_model_swa.model` | EquiDTB model file | **Mandatory** with EquiDTB |

## Trajectory and screening

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `query.trajectory_csv` | `${paths.experiments}/${experiment_name}/configs/trajectory_*.csv` | Trajectory file; with a pattern, the first match in alphabetical order is used | **Mandatory** (copy the CSV) |
| `query.props` | `${EQdiffParam.properties_list}` | Conditioning properties read from the trajectory and passed to generation in this order; keep it identical to `properties_list` | Keep |
| `query.scores` | `["SAScore"]` | Scores that `score_filter` checks, in both modes. `initial_mean`: `SAScore`, `QED` or both; `["SAScore"]` skips the QED test. `trajectory`: each score is also a CSV column holding its per-waypoint threshold, and only `SAScore` works: no QED score is computed in this mode, so a `QED` entry stops the backward job | Optional |
| `query.score_reference` | `initial_mean` | `initial_mean`: thresholds from the mean QED and SA score of the initial dataset, computed once and cached next to the input HDF5. `trajectory`: per-waypoint thresholds from the CSV | Optional |
| `query.score_relax` | `{QED: 0.3, SAScore: 0.3}` | `initial_mean` only: a molecule passes if QED > (1 − r) × mean QED and SA < (1 + r) × mean SA; only the tests for the scores in `query.scores` are applied. Each r lies in (−1, 1); r > 0 loosens and r < 0 tightens | Optional |
| `query.filter_scheme` | `[score, accuracy]` | `score` enables `score_filter`; `accuracy` enables `prop_filter` | Optional |
| `query.filter_threshold` | `0.1` | Relative tolerance of the property-accuracy filter; also the default relative tolerance of `al.tolerances` | Optional |

The trajectory file format is described in [Designing a trajectory](../user-guide/trajectory.md#the-trajectory-file).

## Property calculation

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `properties.calculator` | `EquiDTB` | `EquiDTB`, `DFTB+` or `FHI-aims`. FHI-aims runs `system.fih_aims_exc` with `src/configs/nctr1.in`; DFTB+ needs the patched ASE calculator and `DFTB_PREFIX`, which the backward job sets | **Mandatory** |
| `properties.model` | `${code_paths.configs}/MACE_model_swa.model` | EquiDTB model file | **Mandatory** with EquiDTB |

## HDF5 output

| Parameter | Default | Explanation | Edit |
|---|---|---|---|
| `hdf5.appending_flag` | `True` | Append labelled molecules to the cumulative `<stem>-output.hdf5`, which starts as a copy of the initial dataset. Required for active learning; with `False`, every job recreates the file | **Mandatory** with active learning |
| `hdf5.hdf5_iter_flag` | `True` | Also write each iteration's molecules to `iter_<n>/hdf5/output.hdf5`, the next iteration's `dataset_<n+1>`; required by `concat_datasets` | Optional |
| `hdf5.extra_properties_flag` | `False` | Also write additional QM properties, such as forces, charges and volumes, to separate `extra_properties` HDF5 files | Optional |

## Keys with built-in defaults

These keys are not in any shipped file. Add them to `experiment_config.yaml` to change the built-in value.

| Parameter | Built-in default | Explanation | Edit |
|---|---|---|---|
| `query.fix_num_nodes` | `true` | Use the trajectory's `atNUM` column as the atom count of generated molecules and as a planning axis. Keep `true` with active learning when the CSV has an `atNUM` column | Optional |
| `EQdiffParam.n_nodes` | Not set | Atom count for generation when the trajectory provides none: an integer, or a list with one value per iteration. If not set, atom counts are sampled from the training data | Optional |
