# Repository layout

```text
qalpa/
├── Navigator/              # Phase I: trajectory designer, browser app and server, example trajectory
├── Experiments/templates/  # experiment_config.yaml and the SLURM job templates
├── src/
│   ├── scripts/            # init_experiment, forward, backward, sync_mlflow
│   ├── pipeline_blocks/    # plan_targets, dedup_molecules, score_screen, optimize_geometry,
│   │                       # prop_calc, label_targets, to_hdf5, evaluate_targets, …
│   ├── utils/              # kde_plan, al_config, al_state, trajectory, data management, parallelization
│   ├── configs/            # al.yaml, eqdiff_model.yaml, src_paths.yaml, EquiDTB model, Slater–Koster files
│   └── eqdiff/             # equivariant diffusion model and weighted dataset concatenation
├── analysis/               # Phase III notebooks
├── docs/                   # this documentation (Sphinx, Read the Docs)
├── Data/                   # experiment data, written at run time
├── global_config.yaml      # shared configuration
└── environment_f.yml, environment_b.yml, environment-al.yml
```

## Entry points

| Script | Run by | Purpose |
|---|---|---|
| {repo}`src/scripts/init_experiment.py` | You, or the Navigator | Create an experiment ([reference](init-experiment.md)) |
| {repo}`src/scripts/forward.py` | `qalpa_forward.sh` as `python -m src.scripts.forward` | Forward job: `split` → `train` → `plan` → `generate` → `dedup` |
| {repo}`src/scripts/backward.py` | `qalpa_backward.sh` as `python -m src.scripts.backward` | Backward job: `score_filter` → `relax_geo` → `prop_calc` → `prop_filter` → `hdf5` (→ `evaluate`) |
| {repo}`src/scripts/sync_mlflow.py` | The forward job (with `mlflow.enabled`), or you | Mirror TensorBoard logs into MLflow ([MLflow](../user-guide/mlflow.md)) |
| {repo}`Navigator/navigator_server.py` | You | Navigator server ([setup](../navigator/index.md)) |

## Other markdown files

The repository root also holds the original long-form guides. This documentation supersedes them, and where they disagree with the code, this documentation follows the code:

| File | Covered here by |
|---|---|
| {repo}`README.md` | [Home](../index.md), [Getting started](../getting-started/installation.md) and the [User guide](../user-guide/how-it-works.md) |
| {repo}`CONFIGURATION_REFERENCE.md` | The [Configuration](../configuration/index.md) section |
| {repo}`HIERARCHICAL_CONFIG_GUIDE.md` | [Configuration overview](../configuration/index.md) |
| {repo}`INIT_EXPERIMENT_GUIDE.md` | [`init_experiment`](init-experiment.md) |
| {repo}`Navigator/README.md` | The [Navigator](../navigator/index.md) section |
