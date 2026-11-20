# Quick start

This page takes you from an initial dataset to the first iteration. Each step links to the page with the details.

You need an initial dataset in AQM/QM7-X HDF5 format: molecule groups with `atNUM`, `atXYZ` and the properties you want to condition on (for example `HLgap` and `eMBD`).

## 1. Design a trajectory

Draw waypoints through property space on top of the initial dataset and save them as `trajectory_<x>_<y>_<colour>.csv`. There are three ways to do it: the [Navigator](../navigator/index.md) browser app, the Jupyter notebook {repo}`Navigator/design_trajectory.ipynb`, or the stand-alone {repo}`Navigator/navigator.html`. See [Designing a trajectory](../user-guide/trajectory.md).

To try QALPA first, use the example {repo}`Navigator/trajectory_HLgap_eMBD_atNUM.csv`:

```{literalinclude} ../../Navigator/trajectory_HLgap_eMBD_atNUM.csv
:language: text
```

## 2. Initialize an experiment

```bash
python src/scripts/init_experiment.py --name my_exp --dataset-path /path/to/initial_dataset.hdf5
cp Navigator/trajectory_HLgap_eMBD_atNUM.csv Experiments/my_exp_0/configs/
```

```text
Experiments/my_exp_0/
├── configs/experiment_config.yaml   # experiment settings, paths already filled in
├── jobs/qalpa_forward.sh            # GPU job: split → train → plan → generate → dedup
├── jobs/qalpa_backward.sh           # CPU job: score_filter → relax_geo → prop_calc → prop_filter → hdf5
├── logs/
└── results/
```

Experiment names are versioned: running the command again creates `my_exp_1`. Keep exactly one `trajectory_*.csv` in `configs/`; if several match, the first in alphabetical order is used. All options are listed in the [`init_experiment` reference](../reference/init-experiment.md).

## 3. Review the configuration

The template in `Experiments/my_exp_0/configs/experiment_config.yaml` runs KDE-planned active learning as it is. Check at least:

| Setting | Why it matters |
|---|---|
| `EQdiffParam.properties_list` | Conditioning properties; must match the trajectory columns |
| `loop.num_iterations` | Iterations run from 0 up to this value |
| `geometry.optimizer`, `properties.calculator` | `EquiDTB` (template default), `DFTB+` or `FHI-aims` (also set `system.fih_aims_exc` in `global_config.yaml`) |
| `EQdiffParam.num_samples` | Molecules generated at the waypoint per iteration |
| `al.*` | How targets are planned, see [Active learning](../configuration/active-learning.md) |
| `mlflow.*` | Optional training dashboards, see [MLflow](../user-guide/mlflow.md) |

The [Configuration overview](../configuration/index.md) explains where each setting lives and which file to edit.

## 4. Adapt the job scripts

Set the partition, time limit, account, conda path and modules for your cluster, as described in [Running on SLURM](../user-guide/running-on-slurm.md). Edit the templates in `Experiments/templates/jobs/` once, and every new experiment inherits them.

## 5. Submit

One iteration is a forward job followed by a backward job. The backward job must start only after the forward job of the same iteration has finished successfully.

If both partitions are on one cluster, chain the two jobs with a dependency:

```bash
cd Experiments/my_exp_0/jobs
fwd=$(sbatch --parsable qalpa_forward.sh)
sbatch --dependency=afterok:$fwd qalpa_backward.sh
```

If the GPU and CPU partitions are on different clusters, submit each job by hand, or let the [Navigator](../navigator/running-jobs.md) chain them for you. [Submitting jobs](../user-guide/running-on-slurm.md#submitting-jobs) covers both cases and how to queue a whole run.

## 6. Monitor

```bash
squeue -u $USER
tail -f Experiments/my_exp_0/logs/fcheck-log.out    # forward job
tail -f Experiments/my_exp_0/logs/bcheck-log.out    # backward job
```

The results of iteration *n* appear in `Data/my_exp_0/iter_<n>/`. Start with `targets.json` (what was requested) and `prop_filter_report.json` (what was hit). See [Outputs and analysis](../user-guide/outputs.md).
