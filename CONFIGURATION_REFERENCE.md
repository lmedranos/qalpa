# QALPA Configuration Reference

QALPA uses Hydra/OmegaConf composition. The experiment configuration is the
entry point; its `defaults` imports the root shared configuration and the two
framework configuration packs. Values defined later override earlier values,
and command-line overrides have the highest priority.

**Edit labels**

- **Mandatory**: set or verify for each experiment or deployment.
- **Optional**: the checked-in default is normally suitable, but it may be changed.

## 1. Root shared configuration

Source: [`global_config.yaml`](global_config.yaml)

| Object | Parameter | Explanation | Default | Edit |
|---|---|---|---|---|
| root | `root` | Repository root used to build CWD-independent paths. | `${repo_root:}` | Optional |
| root | `Data_path` | Shared data directory. | `${root}/Data` | Optional |
| root | `Experiments_path` | Shared experiment directory. | `${root}/Experiments` | Optional |
| root | `Code_path` | Source-code directory. | `${root}/src` | Optional |
| paths | `paths.config` | Root configuration directory. | `${root}/configs` | Optional |
| paths | `paths.data` | Base data path used by experiments. | `${Data_path}` | Optional |
| paths | `paths.experiments` | Base experiment path used by experiment configs. | `${Experiments_path}` | Optional |
| paths | `paths.code` | Base source-code path. | `${Code_path}` | Optional |
| system | `system.seed` | Default random seed for reproducible operations. | `42` | Optional |
| system | `system.fih_aims_exc` | Executable used for FHI-aims calculations. | `/software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x` | **Mandatory** when using FHI-aims |

## 2. Framework configuration pack A: EQdiff model

Source: [`src/configs/eqdiff_model.yaml`](src/configs/eqdiff_model.yaml)

| Object | Parameter | Explanation | Default | Edit |
|---|---|---|---|---|
| model | `bond_model_guidance` | Enables bond-model guidance. | `false` | Optional |
| model | `use_pos_norm` | Normalizes positional inputs. | `true` | Optional |
| model | `bond_prediction` | Enables bond prediction. | `false` | Optional |
| model | `seed` | EQdiff model seed. | `42` | Optional |
| training | `batch_size` | Training batch size. | `200` | Optional |
| training | `inference_batch_size` | Batch size during inference/generation. | `70` | Optional |
| training | `ema_decay` | Exponential moving-average decay. | `0.999` | Optional |
| training | `grad_clip_val` | Gradient clipping limit. | `10` | Optional |
| training | `accum_batch` | Number of gradient-accumulation batches. | `1` | Optional |
| training | `gamma` | Model/training gamma parameter. | `0.99` | Optional |
| data | `use_adaptive_loader` | Enables the adaptive data loader. | `true` | Optional |
| data | `remove_hs` | Removes hydrogen atoms from model inputs. | `false` | Optional |
| architecture | `sdim` | Scalar feature dimension. | `256` | Optional |
| architecture | `vdim` | Vector feature dimension. | `256` | Optional |
| architecture | `edim` | Edge feature dimension. | `128` | Optional |
| architecture | `num_layers` | Number of EQGAT/message-passing layers. | `12` | Optional |
| architecture | `use_cross_product` | Enables cross-product interactions. | `false` | Optional |
| architecture | `edge_mp` | Enables edge message passing. | `false` | Optional |
| architecture | `fully_connected` | Uses fully connected molecular graphs. | `true` | Optional |
| architecture | `local_global_model` | Enables local/global model mode. | `false` | Optional |
| architecture | `vector_aggr` | Vector aggregation method. | `mean` | Optional |
| data | `dataset` | EQdiff dataset identifier. | `aqm` | **Mandatory** if different from the experiment dataset |
| diffusion | `additional_feats` | Includes additional diffusion features. | `false` | Optional |
| diffusion | `continuous` | Uses continuous diffusion mode. | `false` | Optional |
| diffusion | `noise_scheduler` | Diffusion noise schedule. | `adaptive` | Optional |
| diffusion | `eps_min` | Minimum diffusion epsilon. | `1.e-3` | Optional |
| diffusion | `beta_min` | Minimum beta value. | `1.e-4` | Optional |
| diffusion | `beta_max` | Maximum beta value. | `2.e-2` | Optional |
| diffusion | `timesteps` | Number of diffusion timesteps. | `500` | Optional |
| diffusion | `max_time` | Maximum training/runtime duration. | `06:00:00:00` | Optional |
| diffusion | `loss_weighting` | Diffusion loss-weighting scheme. | `snr_t` | Optional |
| runtime | `id` | Configuration/model identifier. | `0` | Optional |
| optimizer | `lr_scheduler` | Learning-rate scheduler. | `reduce_on_plateau` | Optional |
| optimizer | `lr` | Initial learning rate. | `2.e-4` | Optional |
| optimizer | `lr_min` | Minimum learning rate for cyclic scheduling. | `5.e-5` | Optional |
| optimizer | `lr_step_size` | Cyclic scheduler step size. | `5.e-5` | Optional |
| optimizer | `lr_factor` | Reduction factor for the scheduler. | `0.75` | Optional |
| optimizer | `lr_patience` | Scheduler patience in validation checks. | `50` | Optional |
| optimizer | `lr_cooldown` | Scheduler cooldown period. | `0` | Optional |
| optimizer | `max_num_neighbors` | Maximum graph neighbors. | `128` | Optional |
| optimizer | `weight_decay` | Optimizer weight decay. | `1.e-12` | Optional |
| runtime | `num_workers` | Data-loader worker count. | `8` | Optional |
| runtime | `precision` | Training numerical precision. | `32` | Optional |

## 3. Framework configuration pack B: source paths

Source: [`src/configs/src_paths.yaml`](src/configs/src_paths.yaml)

| Object | Parameter | Explanation | Default | Edit |
|---|---|---|---|---|
| paths | `Code_path` | Source-code root. | `${root}/src` | Optional |
| paths | `code_paths.EQdiff` | EQdiff framework directory. | `${Code_path}/eqdiff/e3mol_aqm` | **Mandatory** if the framework moves |
| paths | `code_paths.pipeline_blocks` | Pipeline-block directory. | `${Code_path}/pipeline_blocks` | **Mandatory** if the framework moves |
| paths | `code_paths.actdiff_utils` | Utility-module directory. | `${Code_path}/utils` | **Mandatory** if the framework moves |
| paths | `code_paths.scripts` | Workflow-script directory. | `${Code_path}/scripts` | **Mandatory** if the framework moves |
| paths | `code_paths.configs` | Source configuration directory. | `${Code_path}/configs` | **Mandatory** if the framework moves |
| scripts | `EQdiff.split` | Dataset split script. | `${code_paths.EQdiff}/experiments/data/aqm/split_data.py` | Optional |
| scripts | `EQdiff.train` | EQdiff training script. | `${code_paths.EQdiff}/experiments/run_train.py` | Optional |
| scripts | `EQdiff.generate` | EQdiff generation/evaluation script. | `${code_paths.EQdiff}/experiments/run_evaluation_qm.py` | Optional |
| pipeline | `pipeline_blocks.SAScore` | SAScore calculation block. | `${code_paths.pipeline_blocks}/calculate_sascore.py` | Optional |
| pipeline | `pipeline_blocks.GeoRelax` | Geometry-relaxation block. | `${code_paths.pipeline_blocks}/optimize_geometry.py` | Optional |
| pipeline | `pipeline_blocks.PropCalc` | Property-calculation block. | `${code_paths.pipeline_blocks}/prop_calc.py` | Optional |
| pipeline | `pipeline_blocks.PrePropCalc` | Property-calculation preprocessing block. | `${code_paths.pipeline_blocks}/propcalc_preprocess.py` | Optional |
| pipeline | `pipeline_blocks.PostPropCalc` | Property-calculation postprocessing block. | `${code_paths.pipeline_blocks}/propcalc_postprocess.py` | Optional |
| pipeline | `pipeline_blocks.MolFilter` | Molecule-filtering block. | `${code_paths.pipeline_blocks}/filter_molecules.py` | Optional |
| pipeline | `pipeline_blocks.MolHDF5` | HDF5 conversion block. | `${code_paths.pipeline_blocks}/to_hdf5.py` | Optional |
| utilities | `utils.DataManager` | Data-management utility. | `${code_paths.actdiff_utils}/data_manager.py` | Optional |
| utilities | `utils.Trajectory` | Trajectory utility. | `${code_paths.actdiff_utils}/trajectory.py` | Optional |
| utilities | `utils.Parallelize` | Parallel-execution utility. | `${code_paths.actdiff_utils}/parallelize.py` | Optional |
| utilities | `utils.MolCalculator` | Molecular-calculation utility. | `${code_paths.actdiff_utils}/mol_calculate.py` | Optional |
| workflow | `scripts.forward` | Forward workflow entry point. | `${code_paths.scripts}/forward.py` | Optional |
| workflow | `scripts.backward` | Backward workflow entry point. | `${code_paths.scripts}/backward.py` | Optional |
| FHI-aims | `FHI-aims.ctrl` | FHI-aims control/input file. | `${code_paths.configs}/nctr1.in` | **Mandatory** when using FHI-aims |

## 4. Experiment-specific configuration

Source: [`Experiments/templates/configs/experiment_config.yaml`](Experiments/templates/configs/experiment_config.yaml)

| Object | Parameter | Explanation | Default | Edit |
|---|---|---|---|---|
| composition | `defaults` | Imports the root and framework configuration packs. | `global_config`, `src_paths`, `eqdiff_model` | Optional |
| Hydra | `hydra.searchpath` | Repository search path used to resolve absolute defaults. | `file://@@REPO_ROOT@@` before initialization | **Mandatory** for generated experiment configs |
| experiment | `experiment_name` | Experiment directory/name. | `template_experiment` | **Mandatory** |
| experiment | `experiment.name` | Human-readable experiment name. | `template_experiment` | **Mandatory** |
| experiment | `experiment.description` | Human-readable experiment description. | `molecular generation experiment` | Optional |
| experiment | `exp_config_path` | Path to the active experiment config. | `${paths.experiments}/${experiment_name}/configs/experiment_config.yaml` | Optional |
| data | `base_dataset_path` | Input HDF5 dataset used to initialize the experiment. | `template_dataset` | **Mandatory** |
| data | `data.exp_data_path` | Experiment output/data directory. | `${paths.data}/${experiment_name}` | Optional |
| loop | `loop.num_iterations` | Total number of workflow iterations. | `10` | **Mandatory** |
| loop | `loop.current_iter` | Iteration to execute/resume. | `0` | Optional |
| loop | `loop.stop` | Stops both workflow scripts cleanly when true. | `false` | Optional |
| pipeline | `pipeline.start_at` | First block to execute when resuming. | `split` | Optional |
| pipeline | `pipeline.blocks` | Ordered legacy pipeline blocks. AL replaces this list when enabled. | `[split, train, generate, score_filter, relax_geo, prop_calc, prop_filter, hdf5, compress]` | Optional |
| runtime | `max_parallel_tasks` | Maximum parallel relaxation/property tasks. | `20` | Optional |
| runtime | `processing_timeout_hours` | Timeout for processing tasks. | `2` | Optional |
| EQdiffParam | `EQdiffParam.num_samples` | Molecules generated per target in normal mode. AL generates this times `al.oversample`. | `200` | **Mandatory** |
| EQdiffParam | `EQdiffParam.batch_size` | EQdiff training/generation batch size. | `40` | Optional |
| EQdiffParam | `EQdiffParam.num_epochs` | Number of training epochs. | `50` | Optional |
| EQdiffParam | `EQdiffParam.test_interval` | Model test/evaluation interval. | `40` | Optional |
| EQdiffParam | `EQdiffParam.context_mapping` | Enables property-conditioned generation. | `true` | **Mandatory** for property-guided generation |
| EQdiffParam | `EQdiffParam.properties_list` | Properties supplied to/tracked by the diffusion model. | `[HLgap, eMBD]` | **Mandatory** |
| EQdiffParam | `EQdiffParam.num_context_features` | Number of conditioning features; derived from `properties_list`. | `${len:${EQdiffParam.properties_list}}` | Optional |
| EQdiffParam | `EQdiffParam.dataset_root` | Processed diffusion dataset directory. | `${paths.data}/${experiment_name}/diffusion_wdir/data/` | Optional |
| EQdiffParam | `EQdiffParam.model_root` | Model checkpoint registry. | `${paths.data}/${experiment_name}/diffusion_wdir/model_registry/` | Optional |
| EQdiffParam | `EQdiffParam.test_save_dir` | Model test-output directory. | `${paths.data}/${experiment_name}/diffusion_wdir/test/` | Optional |
| EQdiffParam | `EQdiffParam.load_ckpt_from_pretrained` | Optional initial checkpoint. | `null` | Optional |
| EQdiffParam | `EQdiffParam.save_dir` | Training log directory. | `${paths.experiments}/${experiment_name}/logs/` | Optional |
| EQdiffParam | `EQdiffParam.dataset` | Dataset loader name. | `aqm` | **Mandatory** |
| geometry | `geometry.optimizer` | Geometry optimizer: `FHI-aims`, `EquiDTB`, or `DFTB+`. | `DFTB+` | **Mandatory** |
| geometry | `geometry.calculate_forces` | Requests force calculations during relaxation. | `true` | Optional |
| geometry | `geometry.model` | Model path used by the selected optimizer when applicable. | `${code_paths.configs}/MACE_model_swa.model` | **Mandatory** for model-based optimizers |
| query | `query.trajectory_csv` | Trajectory CSV defining target property space. | `${paths.experiments}/${experiment_name}/configs/trajectory_*.csv` | **Mandatory** |
| query | `query.props` | Query properties; normally follows `EQdiffParam.properties_list`. | `${EQdiffParam.properties_list}` | **Mandatory** |
| query | `query.scores` | Molecular scores used by score filtering. | `[SAScore]` | Optional |
| query | `query.filter_scheme` | Enabled filters, such as `score` and `accuracy`. | `[score, accuracy]` | Optional |
| query | `query.filter_threshold` | Relative tolerance for property-hit testing. | `0.2` | Optional |
| properties | `properties.calculator` | QM/property calculator: `FHI-aims`, `EquiDTB`, or `DFTB+`. | `DFTB+` | **Mandatory** |
| properties | `properties.model` | Model path for the selected property calculator. | `${code_paths.configs}/MACE_model_swa.model` | **Mandatory** when required by calculator |
| al | `al.enabled` | Enables adaptive active learning; false preserves the legacy loop. | `true` | **Mandatory** decision |
| al | `al.query_strategy` | Target strategy: `trajectory`, `kde_map`, or `frontier`. | `kde_map` | Optional |
| al | `al.targets_per_iter` | Number of adaptive targets per iteration. | `4` | Optional |
| al | `al.oversample` | Generation multiplier before acquisition. | `5` | Optional |
| al | `al.budget` | Maximum candidates sent to relaxation/QM. | `200` | **Mandatory** when AL is enabled |
| al | `al.corridor_pad` | Fractional padding around the trajectory corridor. | `0.1` | Optional |
| al | `al.corridor_lookahead` | Limits KDE corridor to early trajectory rows; null means all rows. | `null` | Optional |
| al | `al.advance_hit_rate` | Frontier hit rate required to advance. | `0.3` | Optional |
| al | `al.keep_off_target` | Keeps valid QM molecules even when they miss their target. | `true` | Optional |
| al | `al.generation_budget_hours` | Wall-clock budget for target generation. | `12` | Optional |
| al | `al.min_neff` | Minimum effective sample count for coverage/reachability decisions. | `5` | Optional |
| al.tolerances | `al.tolerances.default.relative` | Relative hit tolerance fallback for AL properties; falls back to `query.filter_threshold` when omitted. | `0.2` | Optional |
| al.tolerances | `al.tolerances.default.absolute` | Absolute hit tolerance fallback added to the relative rule through `max(relative * abs(target), absolute)`. | `0.0` | Optional |
| al.tolerances | `al.tolerances.<property>.relative` | Property-specific relative hit tolerance. | `query.filter_threshold` | Optional |
| al.tolerances | `al.tolerances.<property>.absolute` | Property-specific absolute hit tolerance. | `0.0` | Optional |
| al.density | `al.density.bandwidth` | KDE bandwidth rule or numeric bandwidth. | `scott` | Optional |
| al.density | `al.density.n_candidates` | Number of sampled candidate targets. | `2000` | Optional |
| al.density | `al.density.w_sparse` | Weight for sparse labelled-data regions. | `0.5` | Optional |
| al.density | `al.density.w_err` | Weight for past target-achievement error. | `0.5` | Optional |
| al.density | `al.density.w_explore` | Weight for under-explored regions. | `0.5` | Optional |
| al.density | `al.density.min_reach` | Minimum estimated reachability before excluding a well-sampled target. | `0.05` | Optional |
| al.density | `al.density.logp_floor` | Lower bound for KDE log-density. | `-20.0` | Optional |
| al.acquisition | `al.acquisition.mode` | Acquisition mode: `score` or `random`. | `score` | Optional |
| al.acquisition | `al.acquisition.w_hit` | Hit-probability score weight. | `1.0` | Optional |
| al.acquisition | `al.acquisition.w_unc` | Surrogate-uncertainty score weight. | `0.5` | Optional |
| al.acquisition | `al.acquisition.w_nov` | Chemical-novelty score weight. | `0.5` | Optional |
| al.acquisition | `al.acquisition.w_div` | Diversity score weight. | `0.5` | Optional |
| al.acquisition | `al.acquisition.w_cost` | Computational-cost penalty weight. | `0.1` | Optional |
| al.acquisition | `al.acquisition.novelty_ref_max` | Maximum existing fingerprints used as novelty references. | `20000` | Optional |
| al.surrogate | `al.surrogate.backend` | Surrogate backend; currently `fingerprint_rf`. | `fingerprint_rf` | Optional |
| al.surrogate | `al.surrogate.n_members` | Number of ensemble members. | `5` | Optional |
| al.surrogate | `al.surrogate.n_trees` | Trees per random-forest member. | `30` | Optional |
| al.surrogate | `al.surrogate.min_train` | Minimum labels before using the surrogate instead of cold start. | `50` | Optional |
| al.surrogate | `al.surrogate.max_train` | Maximum randomly sampled training labels. | `20000` | Optional |
| al.surrogate | `al.surrogate.props` | Query-property subset predicted by the surrogate. | `${EQdiffParam.properties_list}` | Optional |
| al.validation | `al.validation.size` | Maximum fixed validation-set size. | `200` | Optional |
| al.early_stop | `al.early_stop.enabled` | Enables metric-based early stopping. | `false` | Optional |
| al.early_stop | `al.early_stop.patience` | Iterations considered for early stopping. | `3` | Optional |
| al.early_stop | `al.early_stop.min_delta` | Minimum improvement required. | `0.01` | Optional |
| hdf5 | `hdf5.appending_flag` | Appends labels to the cumulative HDF5. Required for AL. | `true` | **Mandatory** when AL is enabled |
| hdf5 | `hdf5.hdf5_iter_flag` | Writes an iteration-specific HDF5 file. | `false` | Optional |
| hdf5 | `hdf5.extra_properties_flag` | Writes additional calculated properties. | `false` | Optional |

## Practical editing rules

1. Change shared machine paths in `global_config.yaml` or `src_paths.yaml`,
   not in every experiment.
2. Change model architecture/training defaults in `eqdiff_model.yaml`.
3. Change datasets, trajectories, calculators, iteration counts, and active
   learning behavior in the experiment-specific config.
4. Keep `query.props`, `EQdiffParam.properties_list`, and
   `al.surrogate.props` compatible. The surrogate properties must be a subset
   of `query.props`.
5. Active learning requires `hdf5.appending_flag: true`, and `kde_map` requires
   at least two trajectory rows.
