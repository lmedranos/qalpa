# QALPA Navigator

The Navigator is a browser app for running a QALPA experiment from start to finish. You start a small server on an HPC login node and open the app in your browser. From there you:

1. **Experiment:** create a new experiment or open an existing one.
2. **Map:** plot the molecules of the initial dataset on two properties.
3. **Waypoints:** click the map to draw the design trajectory.
4. **Export:** save the trajectory into the experiment.
5. **Configuration:** edit `experiment_config.yaml`.
6. **Run:** submit the forward and backward jobs, follow their status, and chain them automatically.

The forward job runs on **capella** (GPU) and the backward job on **barnard** (CPU). The server submits each job on its own cluster: directly when the server runs on that cluster, otherwise over SSH to that cluster's login node.

---

## 1. Install the packages

The server needs Python 3.9 or newer and four packages. Install them for the Python you will start the server with:

```bash
module load Anaconda3            # or activate any QALPA conda environment
pip install --user numpy h5py pyyaml paramiko
python -c "import numpy, h5py, yaml, paramiko; print('ok')"
```

`paramiko` is only needed to submit jobs over SSH, or to browse another machine's files.

## 2. Create an SSH key for the Navigator (once)

To submit a job on the other cluster, the server logs in to that cluster's login node with an SSH key. Create a key just for the Navigator and allow it in your account:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_navigator        # choose a passphrase you will remember
cat ~/.ssh/id_navigator.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Your home folder is shared by all ZIH clusters, so this one entry works on both capella and barnard.

Test the key from the login node where you will run the server. Answer `yes` the first time it asks to trust the host:

```bash
ssh -i ~/.ssh/id_navigator login1.capella.hpc.tu-dresden.de hostname
ssh -i ~/.ssh/id_navigator login1.barnard.hpc.tu-dresden.de hostname
```

Each command should ask for the key's passphrase and print a host name. If it asks for your account password instead, the key is not in `authorized_keys` yet.

> **Note:** the passphrase protects the key file. It is not your HPC account password.

> **Note:** after many failed logins in a short time, a login node may refuse connections for a while ("Connection reset by peer"). Wait about 30 minutes before trying again.

## 3. Start the server

On a login node, from the repository folder:

```bash
python Navigator/navigator_server.py                    # project folder = this repository
python Navigator/navigator_server.py --port 8800        # pick another port if 8765 is taken
```

It prints:

```
  QALPA Navigator — project folder /data/horse/ws/.../QAPLA2

  Open:   http://localhost:8765/?token=3fa1c0...
  Token:  3fa1c0...   (paste it if the page asks)
```

Keep this terminal open: the server and the job chain stop with it. To keep it running after you log out, start it inside `tmux` or `screen`.

### Enter the key passphrase

The server needs the passphrase of `~/.ssh/id_navigator` to submit jobs on the other cluster **and to check their state**. Without it, job states stay at their last known value and the chain does not continue.

The passphrase is kept in the server's memory only, never on disk. **You enter it again each time the server starts**, but only once per start. There are three ways, and you need only one:

1. **In the terminal, at start-up (recommended).** Right after starting, the server asks:

   ```
     Passphrase for ~/.ssh/id_navigator (forward jobs on login1.capella.hpc.tu-dresden.de, empty to skip):
     ✓ logged in to login1.capella.hpc.tu-dresden.de
   ```

   It tests the passphrase by logging in, and lets you retry up to 3 times. If both jobs use the same key, it asks once. Press Enter on an empty prompt to skip.
2. **Remembered in the browser.** If you ticked **remember in this browser** in [Job settings](#job-settings), the page hands the passphrase to the server when you open it. You type nothing.
3. **In the page.** If the server still has no passphrase when you open the page, the page says so and opens **Job settings** for you to enter it.

If the server was started without a terminal (e.g. with `nohup`), there is no prompt: use 2 or 3.

> **Never type it at all?** Create a second key without a passphrase, only for the Navigator, and set it as **Private key file** in Job settings:
>
> ```bash
> ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_navigator_nopass
> cat ~/.ssh/id_navigator_nopass.pub >> ~/.ssh/authorized_keys
> chmod 600 ~/.ssh/id_navigator_nopass
> ```
>
> Anyone who obtains that file can log in as you, so keep it readable only by you.

## 4. Open the app and sign in

- **In VS Code** (Remote SSH): click the **Open** link in the terminal. VS Code forwards the port to your computer.
- **Without VS Code:** forward the port from your computer first, then open the link in your browser:

  ```bash
  ssh -L 8765:localhost:8765 <user>@<login node the server runs on>
  ```

The link already contains the token. If the page shows **Sign in**, copy the token printed by the server and paste it there. The browser remembers it for this server.

The token stays the same across restarts (it is kept in `~/.qalpa_navigator_token`, readable only by you). Delete that file to get a new one, e.g. if the token was shared by mistake.

The top bar shows **Connected** and the project folder. **⚙ Settings** lets you switch to another project folder, or to a different machine over SSH.

## 5. Use the app

### Step 1: Experiment

- **New:** enter a name, pick the initial dataset (`.hdf5`) with **Browse…** and press **Create experiment**. This runs `init_experiment.py` and creates `Experiments/<name>_<n>/`. The log is shown when it finishes.
- **Existing:** choose an experiment from the list and press **Use this experiment**. **Refresh** reloads the list.

The experiment's dataset is selected for the map automatically.

### Step 2: Map

1. Choose the properties for the **x axis**, **y axis** and **colour**.
   - An axis can be **none**: it is left out of the trajectory.
   - An axis can be **constant**: pick the property name and type its value, and every waypoint gets that value.
2. Set **Molecules to read**. The default is 1000; 0 reads all of them.
3. Press **Show map**.

The progress bar shows how many molecules have been read, out of the total in the file. To read more, raise the number and press **Show map** again.

### Step 3: Waypoints

Click anywhere on the map to add waypoints, from the start to the end of the trajectory. The colour value of each waypoint is taken from the nearest molecules.

- **Undo** removes the last waypoint.
- **Clear** removes all of them.

### Step 4: Export

- **Save to experiment** writes `trajectory_<x>_<y>_<colour>.csv` into `Experiments/<exp>/configs/`. It also sets `EQdiffParam.properties_list` in the experiment's config. Older `trajectory_*.csv` files there are renamed to `.bak`, because the pipeline uses the first trajectory file it finds.
- **Save as…**, **Download CSV** and **Download PNG** save copies elsewhere.

### Step 5: Configuration

**Edit config** opens `Experiments/<exp>/configs/experiment_config.yaml` in the **Config** tab.

- **Save** (or Ctrl+S) checks the YAML syntax before writing. The previous version is kept as `experiment_config.yaml.bak`.
- **Reload** discards your unsaved edits.

Avoid editing the config while a backward job is running: that job rewrites the file when it finishes, and the page shows a warning meanwhile.

### Step 6: Run

#### Job settings

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
- **Key passphrase:** the passphrase you chose in [section 2](#2-create-an-ssh-key-for-the-navigator-once). See [Enter the key passphrase](#enter-the-key-passphrase) for when it is needed.
  - The server keeps it in memory only and never writes it to disk.
  - Tick **remember in this browser** to have the page hand it to the server again after a restart. Only tick it on your own computer.
  - **forget the passphrase** removes it.
- **Resources:** each filled field is passed to `sbatch` and overrides the job script. An empty field keeps the script's value. Use **Account** if your project needs `--account`, and **Extra sbatch options** for anything else, such as `--qos=short`.

Press **Save**. The settings are stored per user in `~/.qalpa_navigator.json`; delete that file to go back to the defaults.

#### Submit and follow the jobs

- **Submit forward** and **Submit backward** run `sbatch` on the job scripts in `Experiments/<exp>/jobs/`. A notice shows the SLURM job ID.
- The **Jobs** tab (**Status**) shows:
  - the current iteration (`loop.current_iter` out of `N_iter`);
  - the state of the latest forward and backward job: pending, running, completed or failed;
  - every job submitted from the app, with the cluster it ran on (`capella` or `barnard`).

  It refreshes every 20 seconds. If a state cannot be checked (usually: the server has no key passphrase), the header shows a ⚠ warning, the states may be out of date, and the chain waits. The job list is kept in `Experiments/<exp>/navigator_jobs.json`.
- **Cancel** appears next to each pending or running job in the table. It runs `scancel` on the cluster the job was submitted to, using the same Job settings, and also stops the chain.
- **Delete** appears next to each job that has ended (completed, failed, cancelled…). It removes only that row from the list; the job's output files stay. Running jobs cannot be deleted: cancel them first. Deleting a failed last job lets you restart the chain from the job before it.
- **Chain** submits the next job automatically: backward after forward completes, forward after backward completes. It stops when a job fails or is cancelled, or after the last iteration. The server checks every 30 seconds, so the chain keeps going with the browser closed, as long as the server runs.

#### Restart from a block (Start at)

**Start at** restarts one job from a pipeline block, for example after a failed job, without changing the config.

- Forward blocks: `split`, `train`, `plan`, `generate`, `dedup`. Backward blocks: `score_filter`, `relax_geo`, `prop_calc`, `prop_filter`, `hdf5`, `evaluate`.
- The block applies to the **next submission only**: yours, or the first job the chain submits. Then the list goes back to **From config**, and every later job, chained ones included, uses `pipeline.start_at` from the config again.
- Only the job that runs the chosen block can be submitted: the other job's **Submit** button is disabled.
- The block reaches the job as the environment variable `QALPA_START_AT`, which takes priority over `pipeline.start_at`. The config file is not changed. The job fails at once if the block is not in the experiment's `pipeline.blocks`. In the **Jobs** tab, the job shows a **from \<block\>** tag.
- With a block chosen, turning **Chain** on resumes after a failed job: the chain first submits the job that runs that block, then continues normally. Without a block, the chain does not start after a failed job, and the message under the switch says so.

Typical recovery: clean the failed stage's outputs in the **Clean** tab, choose the failed block in **Start at**, and turn **Chain** on (or press **Submit** for that job).

#### Logs

The **Logs** tab shows the job log files of the experiment: **Forward .out**, **Forward .err**, **Backward .out** and **Backward .err**, as named by `#SBATCH -o` / `-e` in the job scripts, plus any other file in `Experiments/<exp>/logs/`.

- With **follow** ticked, the page reloads the file every 10 seconds and scrolls to the end.
- Large files show their last 200 KB.
- Each run overwrites these files, so they always belong to the latest forward and backward job.

#### Training

The **Training** tab plots the training curves of the diffusion model, with one line per iteration. Older iterations are lighter and the newest is darker.

- Pick the metric from the list, for example `val/loss`, `train/loss` or `test_dataset_0/loss`. **log scale** helps when the loss drops by orders of magnitude.
- Hover over the chart to read the value of every iteration at a step.
- The table below the chart lists the last, minimum and maximum value of each iteration.
- The tab reloads every 60 seconds, so you can watch a forward job train.

The curves come from the TensorBoard files that training writes to `Data/<exp>/diffusion_wdir/model_registry/model_iter_<q>/run0/lightning_logs/version_<k>/`. If a run has several versions, the newest is used. The Navigator reads these files itself, so it needs neither TensorBoard nor MLflow.

For the full MLflow view (parameters, sampling evaluations, test-loss matrix), set `mlflow.enabled: true` in the config and open the MLflow UI as described in the main README.

#### Evaluation

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

#### Clean

The **Clean** tab, set apart at the right end of the tab bar, clears the outputs of a failed iteration so you can run it again.

- Choose the iteration (`Data/<exp>/iter_<n>/`). Its folders and files are grouped by pipeline stage, in pipeline order: forward (plan, generate, dedup), then backward (score filter, relax, prop calc, prop filter, HDF5), then the `query/` working folder. Anything else is listed under **Other**.
- Tick single items, or tick a stage's header to select all of that stage's items. Each row shows its file count and size.
- **Clean…** shows what will be cleaned and asks you to confirm. Selected folders are **emptied but kept**. Selected top-level files, such as `targets.json`, are deleted.
- Cleaning is refused while a job of the experiment is pending or running. Cancel it first.
- Cleaning does not change `current_iter` in the experiment config.

> **Tip:** before chaining a full run, submit one forward job with a short time limit, such as `00:20:00`, to check the settings.

---

## Without the server

Double-click [`navigator.html`](navigator.html) to use the map, waypoints and CSV download on files from your own computer. Steps 1, 5 and 6 need the server.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `SSH needs the paramiko package` | Install `paramiko` for the Python that runs the server, then restart the server. |
| A finished job still shows *running*, and the chain does not continue | The server has no key passphrase, so it cannot ask the cluster. Enter it in **Job settings** (see [Enter the key passphrase](#enter-the-key-passphrase)); the state updates within 30 s. |
| **Delete** does nothing / "not found" | The server is older than the page. Restart the server. |
| `checkints do not match`, or `private key file is encrypted` | Wrong or missing **Key passphrase**. Use the key's passphrase, not your account password. Check it with `ssh-keygen -y -f ~/.ssh/id_navigator`. |
| `Authentication failed` | The public key is not in `~/.ssh/authorized_keys`, or the **User** is wrong. |
| `Connection reset by peer` / `Error reading SSH protocol banner` | The login node is refusing connections for a while, usually after many failed logins. Wait about 30 minutes. |
| `invalid partition specified` | The job was submitted on the wrong cluster. Check **Submit from** and **Login node**. |
| Page asks for a token | Paste the token the server printed at start-up. |
