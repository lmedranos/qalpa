# Shared settings

These two files apply to every experiment. After an experiment's first backward job, their values are copied into its `experiment_config.yaml`, and later edits here no longer reach it ({ref}`write-back <write-back>`). Edit labels are explained in [Reading the reference tables](index.md#reading-the-reference-tables).

## Root configuration

Source: {repo}`global_config.yaml`

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

:::{dropdown} Full file: `global_config.yaml`
```{literalinclude} ../../global_config.yaml
:language: yaml
```
:::

## Code paths

Source: {repo}`src/configs/src_paths.yaml`

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
