# Running jobs

Step 6 of the app submits the forward and backward jobs, follows their state, chains them, and shows logs, training curves and results.

```{tip}
Before chaining a full run, submit one forward job with a short time limit, such as `00:20:00`, to check the settings.
```

## Job settings

Press **Job settings…**. There is one column per job.

When you start the server, the fields are filled with defaults for **your** account:

| Field | Forward job | Backward job |
|---|---|---|
| Submit from | over SSH, unless the server runs on capella | this server's machine if the server runs on barnard, otherwise over SSH |
| Login node | `login1.capella.hpc.tu-dresden.de` | `login1.barnard.hpc.tu-dresden.de` |
| User | your user name | your user name |
| Port | 22 | 22 |
| Private key file | `~/.ssh/id_navigator` if it exists | same |
| Partition | `capella` | `barnard` |
| Time limit | `20:00:00` | `05:00:00` |
| Nodes × tasks per node | 1 × 1 | 10 × 32 |
| CPUs per task | 6 | 1 |
| GPUs per node | 1 | (none) |
| Memory per CPU (MB) | 8000 | 6000 |

The resources match the `#SBATCH` lines of the default job scripts.

**What to check or fill in:**

- **Submit from:** use *this server's machine* for the job whose cluster the server runs on. Use *another login node over SSH* for the other job.
- **Private key file:** `~/.ssh/id_navigator`. This is a path on the machine where the server runs.
- **Key passphrase:** the passphrase you chose when you {ref}`created the key <create-ssh-key>`. See [Enter the key passphrase](index.md#enter-the-key-passphrase) for when it is needed.
  - The server keeps it in memory only and never writes it to disk.
  - Tick **remember in this browser** to have the page hand it to the server again after a restart. Only tick it on your own computer.
  - **forget the passphrase** removes it.
- **Resources:** each filled field is passed to `sbatch` and overrides the job script. An empty field keeps the script's value. Use **Account** if your project needs `--account`, and **Extra sbatch options** for anything else, such as `--qos=short`.

Press **Save**. The settings are stored per user in `~/.qalpa_navigator.json`; delete that file to go back to the defaults.

## Submit and follow the jobs

- **Submit forward** and **Submit backward** run `sbatch` on the job scripts in `Experiments/<exp>/jobs/`. A notice shows the SLURM job ID.
- The **Jobs** tab (**Status**) shows:
  - the current iteration (`loop.current_iter` out of `N_iter`);
  - the state of the latest forward and backward job: pending, running, completed or failed;
  - every job submitted from the app, with the cluster it ran on (`capella` or `barnard`).

  Job states are read from SLURM every 10 minutes; **Refresh** reads them at once. The page and the chain share these checks, so each pending or running job costs about one `sacct` over SSH every 10 minutes. If a state cannot be checked (a missing passphrase, or a failed login), the header shows a ⚠ warning, the states may be out of date, and the chain waits. The job list is kept in `Experiments/<exp>/navigator_jobs.json`.
- **Cancel** appears next to each pending or running job in the table. It runs `scancel` on the cluster the job was submitted to, using the same Job settings, and also stops the chain.
- **Delete** appears next to each job that has ended (completed, failed, cancelled…). It removes only that row from the list; the job's output files stay. Running jobs cannot be deleted: cancel them first. Deleting a failed last job lets you restart the chain from the job before it.
- **Chain** submits the next job automatically: backward after forward completes, forward after backward completes. It stops when a job fails or is cancelled, or after the last iteration. The server checks every 10 minutes, so the chain keeps going with the browser closed, as long as the server runs. The next job is submitted up to 10 minutes after the previous one ends.

## SSH connections

The **Jobs** tab also lists this server's SSH connections. Looking at this list makes no SSH request.

- **Login nodes:** for each job, its login node, whether the key passphrase is given, and the login state:
  - *connected*: logged in; the connection is reused for every request.
  - *not connected*: the server logs in at the next status check, submission or cancel.
  - *waiting for passphrase*: no login until every passphrase is given.
  - *paused*: the last login failed, with the reason. **Retry** logs in once more.
- **Open connections:** every SSH connection the server holds: the job connections, the project-files connection opened with **⚙ Settings**, and *ghost* connections that nothing uses any more. **Close** closes one. **Close all** closes every connection except the project-files one; job connections open again when next needed. *Dropped* connections are already closed; **Remove** clears them from the list.
- **Recent SSH requests:** the last 20 logins and commands, with how long each took and its result.
- **Other Navigator servers of yours on this machine**, if any, for example one left running by an earlier session. Each one checks job states and logs in on its own. **Stop** ends it.

## Restart from a block (Start at)

**Start at** restarts one job from a pipeline block, for example after a failed job, without changing the config.

- Forward blocks: `split`, `train`, `plan`, `generate`, `dedup`. Backward blocks: `score_filter`, `relax_geo`, `prop_calc`, `prop_filter`, `hdf5`, `evaluate`.
- The block applies to the **next submission only**: yours, or the first job the chain submits. Then the list goes back to **From config**, and every later job, chained ones included, uses `pipeline.start_at` from the config again.
- Only the job that runs the chosen block can be submitted: the other job's **Submit** button is disabled.
- The block reaches the job as the environment variable `QALPA_START_AT`, which takes priority over `pipeline.start_at`. The config file is not changed. The job fails at once if the block is not in the experiment's `pipeline.blocks`. In the **Jobs** tab, the job shows a **from \<block\>** tag.
- With a block chosen, turning **Chain** on resumes after a failed job: the chain first submits the job that runs that block, then continues normally. Without a block, the chain does not start after a failed job, and the message under the switch says so.

Typical recovery: clean the failed stage's outputs in the [Clean](#clean) tab, choose the failed block in **Start at**, and turn **Chain** on (or press **Submit** for that job). [Resuming and re-planning](../user-guide/resuming.md) explains the same restarts from the command line.

## Logs

The **Logs** tab shows the job log files of the experiment: **Forward .out**, **Forward .err**, **Backward .out** and **Backward .err**, as named by `#SBATCH -o` / `-e` in the job scripts, plus any other file in `Experiments/<exp>/logs/`.

- With **follow** ticked, the page reloads the file every 10 seconds and scrolls to the end.
- Large files show their last 200 KB.
- Each run overwrites these files, so they always belong to the latest forward and backward job.

## Training

The **Training** tab plots the training curves of the diffusion model, with one line per iteration. Older iterations are lighter and the newest is darker.

- Pick the metric from the list, for example `val/loss`, `train/loss` or `test_dataset_0/loss`. **log scale** helps when the loss drops by orders of magnitude.
- Hover over the chart to read the value of every iteration at a step.
- The table below the chart lists the last, minimum and maximum value of each iteration.
- The tab reloads every 60 seconds, so you can watch a forward job train.

The curves come from the TensorBoard files that training writes to `Data/<exp>/diffusion_wdir/model_registry/model_iter_<q>/run0/lightning_logs/version_<k>/`. If a run has several versions, the newest is used. The Navigator reads these files itself, so it needs neither TensorBoard nor MLflow.

For the full MLflow view (parameters, sampling evaluations, test-loss matrix), set `mlflow.enabled: true` in the config and open the MLflow UI as described in [Monitoring training with MLflow](../user-guide/mlflow.md).

## Evaluation

The **Evaluation** tab shows what each iteration generated and how good it is. It reads `Data/<exp>/iter_<n>/` and `Experiments/<exp>/al_metrics.csv`, and reloads every 60 seconds.

- **KPIs** for the chosen iteration: validity, novelty, uniqueness and molecule stability (from `generated/t<id>/evaluation.csv`, weighted by molecules sampled per target), plus hit rate and corridor coverage from the backward job. Each one shows the change from the previous iteration.
- **Across iterations:** any sampling metric over iterations, as the mean (line) and per target (dots). Click an iteration to select it.
- **Targets:** each KDE target from `targets.json` with its properties, region, density, sparsity and score, its sampling metrics, the deduplication result (kept / duplicate / already in the initial dataset) and its hit rate.
- **Pipeline funnel:** molecules left after each stage (generated → deduplicated → relaxed → prop_calc → filters → hdf5).
- **All sampling metrics:** every column of `evaluation.csv`, per target.
- **KDE map:** the reference dataset as hexagons coloured by its KDE density percentile (`density_reference.npy`), hexagons inside the density band outlined, the candidates the planner scored, the chosen targets (◆), and every computed molecule at its computed properties (● hit, ✕ miss) with a line to its own target (from `prop_filter_properties.csv`, written by the backward job). Pick any two planning properties as axes, toggle the layers, and hover for details. The first load reads the reference HDF5 (about a minute for 50k molecules) and is cached afterwards; **Load density** reloads.
- **Candidate scoring:** every row of `anchor_candidates.csv`: KDE density (log) against the score or any other column. Selected candidates are orange, eligible ones blue, and those inside the density band (shaded) outlined.
- **Molecules:** browse any stage folder of the iteration (generated, deduplicated, dedup_removed, query, rejected, …), filter by target, id or SMILES, and see each molecule's dedup status, hit/miss and rejection reason. Click a row to view the 3D structure; drag to rotate.

On the **Map**, tick **KDE targets** to draw the targets of every iteration on the property map (◆, newer iterations darker, size = molecules requested, dark ◆ = anchor). Both axes must be target properties.

## Clean

The **Clean** tab, set apart at the right end of the tab bar, clears the outputs of a failed iteration so you can run it again.

- Choose the iteration (`Data/<exp>/iter_<n>/`). Its folders and files are grouped by pipeline stage, in pipeline order: forward (plan, generate, dedup), then backward (score filter, relax, prop calc, prop filter, HDF5), then the `query/` working folder. Anything else is listed under **Other**.
- Tick single items, or tick a stage's header to select all of that stage's items. Each row shows its file count and size.
- **Clean…** shows what will be cleaned and asks you to confirm. Selected folders are **emptied but kept**. Selected top-level files, such as `targets.json`, are deleted.
- Cleaning is refused while a job of the experiment is pending or running. Cancel it first.
- Cleaning does not change `current_iter` in the experiment config.
