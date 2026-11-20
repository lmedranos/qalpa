# Running on SLURM

`init_experiment.py` copies both job scripts from {repo}`Experiments/templates/jobs/` and fills in the repository and experiment paths. Edit the templates once, and every new experiment inherits your cluster settings.

| | Forward: `qalpa_forward.sh` | Backward: `qalpa_backward.sh` |
|---|---|---|
| Blocks | `split` → `train` → `plan` → `generate` → `dedup` | `score_filter` → `relax_geo` → `prop_calc` → `prop_filter` → `hdf5` |
| Template resources | 1 node, 1 GPU, 6 CPUs, 8000 MB per CPU | 10 nodes × 32 tasks, 6000 MB per CPU |
| Template partition and time | `capella`, 20 h | `barnard`, 5 h |
| Environment | `conda_envs/env-qalpa-forward` | `conda_envs/env-qalpa-backward` and the FHI-aims modules |
| Log | `logs/fcheck-log.out`, `logs/fcheck-log.err` | `logs/bcheck-log.out`, `logs/bcheck-log.err` |

## Adapt the job scripts

Adapt these lines in both scripts before the first run:

```bash
#SBATCH --partition=<partition>          # GPU partition (forward) or CPU partition (backward)
#SBATCH --time=HH:MM:SS
#SBATCH --mail-user=<you@institution>    # or delete the two mail lines
#SBATCH -A <account>                     # add if your cluster requires an account

source <conda-install>/etc/profile.d/conda.sh   # replaces the site-specific Anaconda path
```

In the backward script, replace the module line with your cluster's equivalents (the template loads the compiler, MPI and FHI-aims modules). If you use FHI-aims, also set `system.fih_aims_exc` in `global_config.yaml` to your `aims.x`:

```bash
module load <release> <compiler> <mpi> <fhi-aims-module>
```

The backward job runs up to `max_parallel_tasks` relaxations or QM calculations at a time (template: 20), each limited to `processing_timeout_hours` (template: 2).

Both scripts run the pipeline as a Python module, with the experiment's configuration folder:

```bash
python -m src.scripts.forward  --config-path "$EXP_CONFIG_DIR" --config-name experiment_config
python -m src.scripts.backward --config-path "$EXP_CONFIG_DIR" --config-name experiment_config
```

(submitting-jobs)=
## Submitting jobs

One iteration is a forward job followed by a backward job. The backward job must start only after the forward job of the same iteration has finished successfully: it works on the generated molecules, and it is the backward job that advances `loop.current_iter` for the next forward job. Both jobs share state only through the experiment folders, so the GPU and CPU partitions may belong to different clusters as long as both mount the same file system.

### Forward and backward on different clusters (manual)

SLURM dependencies do not work across separate clusters, so submit each job yourself and wait for it to finish:

```bash
cd Experiments/my_exp_0/jobs

# on the GPU cluster
sbatch qalpa_forward.sh
# wait until it has finished: squeue -u $USER is empty and logs/fcheck-log.out ends with "Forward Job completed"

# on the CPU cluster
sbatch qalpa_backward.sh
# wait until logs/bcheck-log.out ends with "Backward Job completed"; loop.current_iter has advanced

# repeat both steps for the next iteration
```

```{warning}
Never run the forward job of the next iteration while the backward job is still running. The forward job would read the old `loop.current_iter` and run the same iteration again.
```

```{tip}
The [Navigator](../navigator/running-jobs.md) can do this for you: it submits each job on its own cluster, over SSH where needed, and chains forward and backward jobs automatically.
```

### Both partitions on one cluster (chained)

The forward and backward jobs of one iteration can be linked with a dependency:

```bash
cd Experiments/my_exp_0/jobs
fwd=$(sbatch --parsable qalpa_forward.sh)
sbatch --dependency=afterok:$fwd qalpa_backward.sh
```

To queue the whole run at once, chain the pairs. Each job reads `loop.current_iter` when it starts, and each backward job advances it:

```bash
prev=""
for n in $(seq 0 8); do                      # iterations 0 … loop.num_iterations
  fwd=$(sbatch --parsable ${prev:+--dependency=afterok:$prev} qalpa_forward.sh)
  prev=$(sbatch --parsable --dependency=afterok:$fwd qalpa_backward.sh)
  echo "iteration $n: forward $fwd, backward $prev"
done
```

If a job fails, the jobs queued after it cannot start (`DependencyNeverSatisfied`). Cancel them with `scancel`, fix the cause, and [resume](resuming.md).

## Monitoring

```bash
squeue -u $USER
tail -f Experiments/my_exp_0/logs/fcheck-log.out    # forward job
tail -f Experiments/my_exp_0/logs/bcheck-log.out    # backward job
```

The results of iteration *n* appear in `Data/my_exp_0/iter_<n>/`. Start with `targets.json` (what was requested) and `prop_filter_report.json` (what was hit).
