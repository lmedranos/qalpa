# Navigator: setup

The Navigator is a browser app for running a QALPA experiment from start to finish. You start a small server on an HPC login node and open the app in your browser. From there you:

1. **Experiment:** create a new experiment or open an existing one.
2. **Map:** plot the molecules of the initial dataset on two properties.
3. **Waypoints:** click the map to draw the design trajectory.
4. **Export:** save the trajectory into the experiment.
5. **Configuration:** edit `experiment_config.yaml`.
6. **Run:** submit the forward and backward jobs, follow their status, and chain them automatically.

Steps 1 to 5 are described in [Using the app](using-the-app.md), step 6 in [Running jobs](running-jobs.md).

The forward job runs on **capella** (GPU) and the backward job on **barnard** (CPU), the two clusters of the template job scripts. The server submits each job on its own cluster: directly when the server runs on that cluster, otherwise over SSH to that cluster's login node.

```{note}
The host names and defaults on these pages are those of the ZIH clusters at TU Dresden. On another site, change the login nodes and partitions in [Job settings](running-jobs.md#job-settings).
```

## 1. Install the packages

The server needs Python 3.9 or newer and four packages. Install them for the Python you will start the server with:

```bash
module load Anaconda3            # or activate any QALPA conda environment
pip install --user numpy h5py pyyaml paramiko
python -c "import numpy, h5py, yaml, paramiko; print('ok')"
```

`paramiko` is only needed to submit jobs over SSH, or to browse another machine's files.

(create-ssh-key)=
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

```{note}
The passphrase protects the key file. It is not your HPC account password.
```

```{note}
After many failed logins in a short time, a login node may refuse connections for a while ("Connection reset by peer"). Wait about 30 minutes before trying again.
```

## 3. Start the server

On a login node, from the repository folder:

```bash
python Navigator/navigator_server.py                    # project folder = this repository
python Navigator/navigator_server.py --port 8800        # pick another port if 8765 is taken
```

| Option | Default | Meaning |
|---|---|---|
| `--root` | The repository that contains the script | Project folder; the server reads and writes only inside it |
| `--port` | `8765` | Port on `localhost` |

It prints:

```text
  QALPA Navigator — project folder /data/horse/ws/.../QAPLA2

  Open:   http://localhost:8765/?token=3fa1c0...
  Token:  3fa1c0...   (paste it if the page asks)
```

Keep this terminal open: the server and the job chain stop with it. To keep it running after you log out, start it inside `tmux` or `screen`.

(enter-the-key-passphrase)=
### Enter the key passphrase

The server needs the passphrase of `~/.ssh/id_navigator` to submit jobs on the other cluster **and to check their state**. Without it, job states stay at their last known value and the chain does not continue.

The passphrase is kept in the server's memory only, never on disk. **You enter it again each time the server starts**, but only once per start. There are three ways, and you need only one:

1. **In the terminal, at start-up (recommended).** Right after starting, the server asks:

   ```text
     Passphrase for ~/.ssh/id_navigator (forward jobs on login1.capella.hpc.tu-dresden.de, empty to skip):
     ✓ logged in to login1.capella.hpc.tu-dresden.de
   ```

   It checks each passphrase by opening the key on this machine, so a wrong passphrase never reaches the login node. You get up to 3 tries. If both jobs use the same key, it asks once. Press Enter on an empty prompt to skip. Once every passphrase is given, the server logs in to each login node once.
2. **Remembered in the browser.** If you ticked **remember in this browser** in [Job settings](running-jobs.md#job-settings), the page hands the passphrase to the server when you open it. You type nothing.
3. **In the page.** If the server still has no passphrase when you open the page, the page says so and opens **Job settings** for you to enter it.

If the server was started without a terminal (e.g. with `nohup`), there is no prompt: use 2 or 3.

**No SSH login is made until every key passphrase is given.** A passphrase entered in the page is checked the same way: if it is wrong, **Job settings** shows the error and nothing is saved. If a login still fails (for example the key is not in `authorized_keys`), the server does not try that login node again on its own. The page shows a notice and a **⚠ SSH login failed** button in the top bar until you press **Retry** under [SSH connections](running-jobs.md#ssh-connections), or save new login details in Job settings.

:::{dropdown} Never type the passphrase at all?
Create a second key without a passphrase, only for the Navigator, and set it as **Private key file** in Job settings:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_navigator_nopass
cat ~/.ssh/id_navigator_nopass.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_navigator_nopass
```

```{warning}
Anyone who obtains that file can log in as you, so keep it readable only by you.
```
:::

## 4. Open the app and sign in

- **In VS Code** (Remote SSH): click the **Open** link in the terminal. VS Code forwards the port to your computer.
- **Without VS Code:** forward the port from your computer first, then open the link in your browser:

  ```bash
  ssh -L 8765:localhost:8765 <user>@<login node the server runs on>
  ```

The link already contains the token. If the page shows **Sign in**, copy the token printed by the server and paste it there. The browser remembers it for this server.

The token stays the same across restarts (it is kept in `~/.qalpa_navigator_token`, readable only by you). Delete that file to get a new one, e.g. if the token was shared by mistake.

The top bar shows **Connected** and the project folder. **⚙ Settings** lets you switch to another project folder, or to a different machine over SSH. **Docs ↗** opens this documentation in a new tab.

## Other ways to run it

**Server on your own computer.** Install the packages (`pip install numpy h5py pyyaml paramiko`), start the same script there, and enter your HPC host, user and SSH key under **⚙ Settings**. The server then reads the project folder on the cluster over SSH.

**Without a server.** Double-click {repo}`Navigator/navigator.html` to use the map, waypoints and CSV download on files from your own computer. Steps 1, 5 and 6 need the server.

## Files the Navigator keeps

| File | Contents |
|---|---|
| `~/.qalpa_navigator_token` | Sign-in token, readable only by you; delete it to get a new one |
| `~/.qalpa_navigator.json` | Your Job settings; delete it to go back to the defaults |
| `Experiments/<exp>/navigator_jobs.json` | Jobs submitted from the app for this experiment |
| `Experiments/<exp>/configs/experiment_config.yaml.bak` | Previous version of the config, written when you save in the app |
