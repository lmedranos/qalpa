#!/usr/bin/env python3
"""QALPA Navigator server: the Navigator app in your browser, working on project files.

Two ways to run it:

  On the HPC (files are local to the server):
      python Navigator/navigator_server.py
      -> open the printed link; VS Code forwards the port (or: ssh -L 8765:localhost:8765 <login-node>)

  On your own computer (files reached over SSH):
      pip install numpy h5py paramiko
      python navigator_server.py
      -> open the printed link, click the gear icon, and enter host, user and SSH key of the HPC.

The server listens on localhost only and every request needs the token printed at start-up (the browser
remembers it after the first visit). It reads and writes only inside the chosen project folder.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import posixpath
import secrets
import stat
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE / "navigator.html"
DATA_EXT = (".h5", ".hdf5", ".csv")
COOKIE = "qalpa_navigator_token"


# ------------------------------------------------------------------------------------------- file backends
class LocalFS:
    kind = "local"

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.label = str(self.root)

    def _abs(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError("outside the project folder")
        return p

    def listdir(self, rel):
        d = self._abs(rel)
        if not d.is_dir():
            raise NotADirectoryError(rel)
        return [(e.name, e.is_dir(), None if e.is_dir() else e.stat().st_size) for e in d.iterdir()]

    def isdir(self, rel):
        return self._abs(rel).is_dir()

    def exists(self, rel):
        return self._abs(rel).exists()

    def open(self, rel):
        return open(self._abs(rel), "rb")

    def write_text(self, rel, text):
        self._abs(rel).write_text(text)

    def read_text(self, rel):
        return self._abs(rel).read_text()

    def rename(self, a, b):
        self._abs(a).replace(self._abs(b))

    def abspath(self, rel):
        return str(self._abs(rel))

    def remove_file(self, rel):
        """Delete one file (never a folder)."""
        p = self._abs(rel)
        if p.is_dir():
            raise IsADirectoryError(rel)
        p.unlink()

    def clear_dir(self, rel):
        """Delete everything inside a folder (files and subfolders); the folder itself is kept."""
        import shutil
        p = self._abs(rel)
        if p == self.root or not p.is_dir():
            raise PermissionError(f"not a folder that can be emptied: {rel}")
        for c in p.iterdir():
            shutil.rmtree(c) if c.is_dir() and not c.is_symlink() else c.unlink()

    def du(self, rel):
        """(bytes, files) under a folder."""
        b = n = 0
        for f in self._abs(rel).rglob("*"):
            if f.is_file() and not f.is_symlink():
                b += f.stat().st_size; n += 1
        return b, n

    def alive(self):
        """True while the project folder can be reached (read by /api/info for the page's connection dot)."""
        return self.root.is_dir()

    def run(self, args):
        """Run a command in the project folder; returns (exit code, combined output)."""
        import subprocess
        p = subprocess.run([sys.executable if args[0] == "python" else args[0], *args[1:]], cwd=self.root,
                           capture_output=True, text=True, timeout=300)
        return p.returncode, p.stdout + p.stderr

    def close(self):
        pass


class SSHFS:
    kind = "ssh"

    def __init__(self, host, user, port=22, key=None, password=None, root="~", python="python3"):
        self.python = python or "python3"
        try:
            import paramiko
        except ImportError as e:
            raise RuntimeError("SSH needs the paramiko package: pip install paramiko") from e
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(host, port=int(port or 22), username=user or None,
                            key_filename=str(Path(key).expanduser()) if key else None,
                            password=password or None, look_for_keys=not (key or password), timeout=20)
        self.client.get_transport().set_keepalive(30)    # a dropped link closes the transport, so alive() turns False
        self.sftp = self.client.open_sftp()
        root = root or "~"
        if root.startswith("~"):
            root = self.sftp.normalize(".") + root[1:]
        self.root = posixpath.normpath(self.sftp.normalize(root))
        self.label = f"{user + '@' if user else ''}{host}:{self.root}"

    def _abs(self, rel: str) -> str:
        p = posixpath.normpath(posixpath.join(self.root, rel or "."))
        if p != self.root and not p.startswith(self.root.rstrip("/") + "/"):
            raise PermissionError("outside the project folder")
        return p

    def listdir(self, rel):
        out = []
        for a in self.sftp.listdir_attr(self._abs(rel)):
            d = stat.S_ISDIR(a.st_mode)
            out.append((a.filename, d, None if d else a.st_size))
        return out

    def isdir(self, rel):
        try:
            return stat.S_ISDIR(self.sftp.stat(self._abs(rel)).st_mode)
        except FileNotFoundError:
            return False

    def exists(self, rel):
        try:
            self.sftp.stat(self._abs(rel))
            return True
        except FileNotFoundError:
            return False

    def open(self, rel):
        f = self.sftp.open(self._abs(rel), "rb")
        f.prefetch()                     # faster sequential reads
        return f

    def write_text(self, rel, text):
        with self.sftp.open(self._abs(rel), "w") as f:
            f.write(text)

    def read_text(self, rel):
        with self.sftp.open(self._abs(rel), "r") as f:
            return f.read().decode()

    def rename(self, a, b):
        self.sftp.posix_rename(self._abs(a), self._abs(b))

    def abspath(self, rel):
        return self._abs(rel)

    def alive(self):
        """True while the SSH session to the host is open."""
        t = self.client.get_transport()
        return bool(t and t.is_active())

    def remove_file(self, rel):
        """Delete one file (never a folder) on the remote host."""
        if self.isdir(rel):
            raise IsADirectoryError(rel)
        self.sftp.remove(self._abs(rel))

    def clear_dir(self, rel):
        """Delete everything inside a remote folder (files and subfolders); the folder itself is kept."""
        import shlex
        p = self._abs(rel)
        if p == self.root or not self.isdir(rel):
            raise PermissionError(f"not a folder that can be emptied: {rel}")
        _, out, err = self.client.exec_command(f"find {shlex.quote(p)} -mindepth 1 -delete", timeout=600)
        if out.channel.recv_exit_status():
            raise RuntimeError(err.read().decode().strip() or "find -delete failed")

    def du(self, rel):
        import shlex
        p = shlex.quote(self._abs(rel))
        _, out, _ = self.client.exec_command(f"du -sb {p} | cut -f1; find {p} -type f | wc -l", timeout=120)
        try:
            b, n = out.read().decode().split()[:2]
            return int(b), int(n)
        except ValueError:
            return None, None

    def run(self, args):
        """Run a command in the project folder on the remote host; returns (exit code, combined output)."""
        import shlex
        cmd = [self.python if args[0] == "python" else args[0], *args[1:]]
        _, out, err = self.client.exec_command(f"cd {shlex.quote(self.root)} && {shlex.join(cmd)}", timeout=300)
        code = out.channel.recv_exit_status()
        return code, out.read().decode() + err.read().decode()

    def close(self):
        self.client.close()


# -------------------------------------------------------------------------------------------- dataset readers
PROGRESS = {"done": 0, "total": 0, "active": False}    # read by /api/progress while a dataset loads


def property_names(fs, rel: str) -> list:
    """Property columns of a dataset without loading the molecules (first molecule of an HDF5, CSV header)."""
    with fs.open(rel) as f:
        if rel.lower().endswith(".csv"):
            return [h.strip() for h in io.TextIOWrapper(f, encoding="utf-8").readline().split(",") if h.strip()]
        import h5py
        import numpy as np
        with h5py.File(f, "r") as h:
            for top in h:
                if not isinstance(h[top], h5py.Group):
                    continue
                for name in h[top]:
                    mol = h[top][name]
                    if isinstance(mol, h5py.Group):
                        return sorted(k for k, d in mol.items() if isinstance(d, h5py.Dataset) and
                                      (k == "atNUM" or (d.size == 1 and np.issubdtype(d.dtype, np.number))))
    return []


# ------------------------------------------------------------------------------------------------ experiments
EXP_DIR = "Experiments"
NON_CONDITIONING = {"atNUM", "SAScore", "QED", "QEDScore"}


def _config_value(text: str, key: str):
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip().strip("'\"") or None
    return None


def list_experiments(fs) -> list:
    out = []
    if not fs.isdir(EXP_DIR):
        return out
    for name, is_dir, _ in sorted(fs.listdir(EXP_DIR)):
        cfg = f"{EXP_DIR}/{name}/configs/experiment_config.yaml"
        if not is_dir or name == "templates" or not fs.exists(cfg):
            continue
        text = fs.read_text(cfg)
        trajs = sorted(n for n, d, _ in fs.listdir(f"{EXP_DIR}/{name}/configs")
                       if not d and n.startswith("trajectory_") and n.endswith(".csv"))
        out.append({"name": name, "dataset": _config_value(text, "base_dataset_path"),
                    "iteration": _config_value(text, "current_iter"), "trajectories": trajs})
    return out


def init_experiment(fs, name: str, dataset_rel: str) -> dict:
    """Run src/scripts/init_experiment.py of the project; returns the versioned experiment name and its output."""
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError("use letters, digits, - and _ in the experiment name")
    if not fs.exists("src/scripts/init_experiment.py"):
        raise ValueError("the project folder has no src/scripts/init_experiment.py; point Settings at the QALPA repository")
    if not fs.exists(dataset_rel) or not dataset_rel.lower().endswith((".h5", ".hdf5")):
        raise ValueError("choose an .h5 or .hdf5 dataset")
    code, out = fs.run(["python", "src/scripts/init_experiment.py", "--name", name,
                        "--dataset-path", fs.abspath(dataset_rel)])
    created = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Creating experiment:")), None)
    if code != 0 or not created:
        raise RuntimeError(out.strip().splitlines()[-1] if out.strip() else f"init_experiment failed (exit {code})")
    return {"name": created, "log": out}


def save_trajectory(fs, experiment: str, filename: str, content: str, conditioning: list) -> dict:
    """Write the trajectory into the experiment's configs/, keep it the only trajectory_*.csv there,
    and set EQdiffParam.properties_list to its conditioning properties."""
    folder = f"{EXP_DIR}/{posixpath.basename(experiment)}/configs"
    cfg = f"{folder}/experiment_config.yaml"
    if not fs.exists(cfg):
        raise ValueError(f"{experiment} has no configs/experiment_config.yaml")
    filename = posixpath.basename(filename)
    if not (filename.startswith("trajectory_") and filename.endswith(".csv")):
        raise ValueError("the file name must look like trajectory_*.csv")
    moved = []
    for n, d, _ in fs.listdir(folder):
        if not d and n.startswith("trajectory_") and n.endswith(".csv") and n != filename:
            fs.rename(f"{folder}/{n}", f"{folder}/{n}.bak")
            moved.append(n)
    fs.write_text(f"{folder}/{filename}", content)
    props = [p for p in conditioning if p not in NON_CONDITIONING]
    updated = False
    if props:
        lines = fs.read_text(cfg).splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.lstrip().startswith("properties_list:"):
                indent = line[:len(line) - len(line.lstrip())]
                comment = "  #" + line.split("#", 1)[1] if "#" in line else "\n"
                lines[i] = f"{indent}properties_list: {props!r}{comment}"
                updated = True
                break
        if updated:
            fs.write_text(cfg, "".join(lines))
    return {"saved": f"{folder}/{filename}", "backed_up": moved, "properties_list": props if updated else None}


def read_hdf5(fileobj, max_mol: int) -> dict:
    """Scalar properties of every molecule (atNUM -> atom count), as {property: [values]}."""
    import h5py
    import numpy as np

    cols, n = {}, 0
    with h5py.File(fileobj, "r") as f:
        total = sum(len(f[t]) for t in f if isinstance(f[t], h5py.Group))
        PROGRESS.update(done=0, total=min(total, max_mol), file_total=total)
        for top in f:
            grp = f[top]
            if not isinstance(grp, h5py.Group):
                continue
            for name in grp:
                mol = grp[name]
                if not isinstance(mol, h5py.Group):
                    continue
                for key, ds in mol.items():
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    if key == "atNUM":
                        v = float(ds.shape[0])
                    elif ds.size == 1 and np.issubdtype(ds.dtype, np.number):
                        v = float(np.asarray(ds).reshape(-1)[0])
                    else:
                        continue
                    cols.setdefault(key, [float("nan")] * n).append(v)
                n += 1
                PROGRESS["done"] = n
                for vals in cols.values():
                    if len(vals) < n:
                        vals.append(float("nan"))
                if n >= max_mol:
                    return cols
    return cols


def read_csv(fileobj, max_mol: int) -> dict:
    cols: dict = {}
    rows = 0
    for i, row in enumerate(csv.DictReader(io.TextIOWrapper(fileobj, encoding="utf-8"))):
        rows += 1                                    # keep counting past the limit for the file total
        if i >= max_mol:
            continue
        for k, v in row.items():
            try:
                cols.setdefault(k.strip(), []).append(float(v))
            except (TypeError, ValueError):
                cols.setdefault(k.strip(), []).append(float("nan"))
    PROGRESS["file_total"] = rows
    return {k: v for k, v in cols.items() if any(x == x for x in v)}


# ------------------------------------------------------------------------------------------- experiment config
def config_path(experiment: str) -> str:
    return f"{EXP_DIR}/{posixpath.basename(experiment)}/configs/experiment_config.yaml"


def save_config(fs, experiment: str, text: str) -> dict:
    """Check the YAML, keep the previous version as .bak, write the new one."""
    try:
        import yaml
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("the config must be a YAML mapping (key: value lines)")
    except ImportError:
        pass
    except Exception as e:                           # yaml.YAMLError carries line and column
        raise ValueError(f"invalid YAML: {e}") from e
    path = config_path(experiment)
    if not fs.exists(path):
        raise ValueError(f"{experiment} has no configs/experiment_config.yaml")
    fs.write_text(path + ".bak", fs.read_text(path))
    fs.write_text(path, text)
    return {"saved": path}


# ------------------------------------------------------------------------------------------------------- SLURM jobs
JOBS = ("forward", "backward")
SETTINGS_FILE = Path.home() / ".qalpa_navigator.json"
JOB_FIELDS = ["via", "host", "user", "port", "key", "partition", "account", "time", "nodes", "ntasks_per_node",
              "cpus_per_task", "gpus", "mem_per_cpu", "extra"]
FINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
         "DEADLINE"}
# Pipeline blocks in order, and the job that runs each (src/scripts/forward.py, backward.py).
# "Start at" sends one of them to a single job as QALPA_START_AT (read by src/utils/al_config.resolve_start);
# it takes priority over pipeline.start_at for that job only, and the config file is not changed.
STAGE_JOB = {"split": "forward", "train": "forward", "plan": "forward", "generate": "forward", "dedup": "forward",
             "score_filter": "backward", "relax_geo": "backward", "prop_calc": "backward", "prop_filter": "backward",
             "hdf5": "backward", "evaluate": "backward"}
START_ONCE_ENV = "QALPA_START_AT"


def _check_start(job: str, start_at) -> str:
    start_at = (start_at or "").strip()
    if start_at and STAGE_JOB.get(start_at) != job:
        owner = STAGE_JOB.get(start_at)
        raise ValueError(f"'{start_at}' is not a pipeline block" if not owner
                         else f"'{start_at}' runs in the {owner} job; submit the {owner} job to start there")
    return start_at


def _export_var(opts: list, name: str, value: str) -> list:
    """sbatch options plus one exported variable; the rest of the environment is passed on as by default."""
    out, done, i = [], False, 0
    while i < len(opts):
        o = opts[i]
        if o == "--export" and i + 1 < len(opts):        # "--export VALUE" form in the extra options
            o, i = f"--export={opts[i + 1]}", i + 1
        if o.startswith("--export="):
            v = o.split("=", 1)[1]
            o = f"--export={name}={value}" if v.upper() == "NONE" else f"--export={v},{name}={value}"
            done = True
        out.append(o)
        i += 1
    return out if done else out + [f"--export=ALL,{name}={value}"]


CLUSTERS = {"forward": ("capella", "login1.capella.hpc.tu-dresden.de"),
            "backward": ("barnard", "login1.barnard.hpc.tu-dresden.de")}
SCRIPT_RESOURCES = {  # same values as the #SBATCH lines of the default job scripts
    "forward": {"time": "20:00:00", "nodes": 1, "ntasks_per_node": 1, "cpus_per_task": 6, "gpus": 1, "mem_per_cpu": 8000},
    "backward": {"time": "05:00:00", "nodes": 10, "ntasks_per_node": 32, "cpus_per_task": 1, "gpus": "",
                 "mem_per_cpu": 6000},
}


TOKEN_FILE = Path.home() / ".qalpa_navigator_token"


def load_token() -> str:
    """Same token on every start (stored readable only by you), so a browser that signed in once stays signed in."""
    try:
        tok = TOKEN_FILE.read_text().strip()
        if len(tok) >= 16:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(16)
    TOKEN_FILE.touch(mode=0o600)
    TOKEN_FILE.chmod(0o600)
    TOKEN_FILE.write_text(tok)
    return tok


def default_job_settings() -> dict:
    """Defaults for whoever starts the server: their user name and ~/.ssh/id_navigator (if it exists);
    a job whose cluster is the one this server runs on is submitted here, the other one over SSH."""
    import getpass
    import socket
    here = socket.getfqdn().lower()
    key = Path.home() / ".ssh" / "id_navigator"
    out = {}
    for j, (cluster, host) in CLUSTERS.items():
        out[j] = {f: "" for f in JOB_FIELDS} | SCRIPT_RESOURCES[j] | {
            "via": "local" if f".{cluster}." in here else "ssh", "host": host, "port": 22,
            "user": getpass.getuser(), "key": "~/.ssh/id_navigator" if key.exists() else "", "partition": cluster}
    return out


def load_job_settings() -> dict:
    # Saved settings (~/.qalpa_navigator.json, one per user) win over the defaults, field by field.
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        saved = {}
    defaults = default_job_settings()
    return {j: defaults[j] | {k: v for k, v in saved.get(j, {}).items() if k in JOB_FIELDS} for j in JOBS}


JOB_SETTINGS = load_job_settings()


def public_job_settings() -> dict:
    # The key passphrase lives only in this server's memory: never written to disk or sent to the page.
    return {j: {k: v for k, v in s.items() if not k.startswith("_")} | {"passphrase_set": bool(s.get("_passphrase")),
                                                                        "passphrase_needed": passphrase_needed(s)}
            for j, s in JOB_SETTINGS.items()}


def key_encrypted(s: dict) -> bool:
    """True when the job's SSH private key file is protected by a passphrase."""
    if s.get("via") != "ssh" or not s.get("key"):
        return False
    path = Path(s["key"]).expanduser()
    try:
        import paramiko
    except ImportError:
        return False
    if not path.exists():
        return False
    try:
        for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
            try:
                cls.from_private_key_file(str(path))
                return False
            except paramiko.PasswordRequiredException:
                raise
            except Exception:
                continue
        return False
    except paramiko.PasswordRequiredException:
        return True
    except Exception:
        return False


def passphrase_needed(s: dict) -> bool:
    return not s.get("_passphrase") and key_encrypted(s)


def ask_passphrases_at_startup():
    """Ask in the terminal for the passphrase of every encrypted SSH key, and check it by logging in,
    so that job states and chains work from the start. Skipped when there is no terminal (the page asks then)."""
    import getpass
    if not sys.stdin.isatty():
        return
    for job, s in JOB_SETTINGS.items():
        if not passphrase_needed(s):
            continue
        other = next((o for o in JOB_SETTINGS.values() if o is not s and o.get("_passphrase")
                      and o.get("key") == s.get("key")), None)
        if other:                                     # same key file as the other job: reuse it
            s["_passphrase"] = other["_passphrase"]
            continue
        for _ in range(3):
            pw = getpass.getpass(f"  Passphrase for {s['key']} ({job} jobs on {s['host']}, empty to skip): ")
            if not pw:
                break
            s["_passphrase"] = pw
            try:
                code, _ = Runner(s).run(["true"], "/")
                print(f"  ✓ logged in to {s['host']}")
                break
            except Exception as err:
                s.pop("_passphrase", None)
                print(f"  ✗ {type(err).__name__}: {err}")


def local_cluster_name() -> str:
    import socket
    here = socket.getfqdn().lower()
    return next((c for c, _ in CLUSTERS.values() if f".{c}." in here), here.split(".")[0])


def cluster_name(s: dict) -> str:
    """Short cluster name (capella, barnard, ...) for where a job runs, never 'local'."""
    if s.get("via") != "ssh":
        return local_cluster_name()
    host = s.get("host", "")
    return next((c for c, _ in CLUSTERS.values() if f".{c}." in f".{host}."), host.split(".")[1] if host.count(".") > 1 else host)


def fix_local_via():
    """The settings file in $HOME is shared by all clusters, so a 'local' saved while the server ran on barnard
    would, on capella, submit and query that job on capella. Go over SSH to the job's configured host instead."""
    here = local_cluster_name()
    for job, s in JOB_SETTINGS.items():
        want = cluster_name({"via": "ssh", "host": s.get("host", "")}) if s.get("host") else here
        if s.get("via") != "ssh" and want != here:
            s["via"] = "ssh"
            print(f"  note: {job} jobs run on {want}, not on this machine ({here}): they go over SSH to {s['host']}")


def settings_for(entry: dict) -> dict:
    """Job settings to query or cancel a recorded job with; refuses when they point at another cluster than the
    one the job was submitted to (its id means nothing there, and sacct would just return nothing)."""
    s = JOB_SETTINGS[entry["job"]]
    ran, now = entry.get("cluster"), cluster_name(s)
    if ran and ran != now:
        raise RuntimeError(f"{entry['job']} job {entry['id']} ran on {ran}, but the {entry['job']} job settings now "
                           f"reach {now}: in Job settings, submit {entry['job']} jobs over SSH to {ran}")
    return s


class Runner:
    """Runs SLURM commands where a job is submitted: here, or on that cluster's login node over SSH."""
    _clients: dict = {}

    def __init__(self, s: dict):
        self.s = s

    def run(self, args: list, cwd: str) -> tuple:
        import shlex
        import subprocess
        if self.s.get("via") != "ssh":
            p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120)
            return p.returncode, p.stdout + p.stderr
        key = (self.s["host"], self.s.get("user"), int(self.s.get("port") or 22), self.s.get("key"))
        client = Runner._clients.get(key)
        if client is None or not client.get_transport() or not client.get_transport().is_active():
            try:
                import paramiko
            except ImportError as e:
                raise RuntimeError("SSH needs the paramiko package: pip install paramiko") from e
            auth = dict(key_filename=str(Path(key[3]).expanduser()) if key[3] else None, timeout=20,
                        passphrase=self.s.get("_passphrase") or None)
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(key[0], port=key[2], username=key[1] or None, **auth)
            Runner._clients[key] = client
        _, out, err = client.exec_command(f"cd {shlex.quote(cwd)} && {shlex.join(args)}", timeout=120)
        code = out.channel.recv_exit_status()
        return code, out.read().decode() + err.read().decode()


def sbatch_options(s: dict) -> list:
    opts = []
    for field, flag in (("partition", "--partition"), ("account", "--account"), ("time", "--time"),
                        ("nodes", "--nodes"), ("ntasks_per_node", "--ntasks-per-node"),
                        ("cpus_per_task", "--cpus-per-task"), ("mem_per_cpu", "--mem-per-cpu")):
        if str(s.get(field, "")).strip():
            opts.append(f"{flag}={str(s[field]).strip()}")
    if str(s.get("gpus", "")).strip():
        opts.append(f"--gres=gpu:{str(s['gpus']).strip()}")
    import shlex
    opts += shlex.split(s.get("extra", "") or "")
    return opts


def _script_logs(fs, experiment: str) -> dict:
    """Map each log path named in the job scripts (#SBATCH -o/-e) to (job, 'out'|'err')."""
    import re
    jobs_dir = f"{EXP_DIR}/{posixpath.basename(experiment)}/jobs"
    found = {}
    for job in JOBS:
        try:
            text = fs.read_text(f"{jobs_dir}/qalpa_{job}.sh")
        except OSError:
            continue
        for m in re.finditer(r"^#SBATCH\s+(-o|--output|-e|--error)[=\s]+(\S+)", text, re.M):
            kind = "out" if m.group(1) in ("-o", "--output") else "err"
            found[posixpath.normpath(posixpath.join(jobs_dir, m.group(2)))] = (job, kind)
    return found


# ---------------------------------------------------------------- training curves (TensorBoard event files)
# Training logs to TensorBoard (PyTorch Lightning), and src/scripts/sync_mlflow.py mirrors the same files into
# MLflow. The Navigator reads the event files directly, so it needs neither tensorboard nor mlflow installed.

def _varint(b: bytes, i: int):
    shift = val = 0
    while True:
        c = b[i]
        i += 1
        val |= (c & 0x7F) << shift
        if c < 0x80:
            return val, i
        shift += 7


def _fields(b: bytes):
    """Yield (field number, wire type, value) of a protobuf message."""
    import struct
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 1:
            v = struct.unpack_from("<d", b, i)[0]
            i += 8
        elif w == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif w == 5:
            v = struct.unpack_from("<f", b, i)[0]
            i += 4
        else:
            return
        yield f, w, v


def _scalar_of(value_msg: bytes):
    """(tag, value) of a Summary.Value: simple_value, or a one-element float/double tensor."""
    import struct
    tag, val = None, None
    for f, w, v in _fields(value_msg):
        if f == 1 and w == 2:
            tag = v.decode("utf-8", "replace")
        elif f == 2 and w == 5:
            val = v
        elif f == 8 and w == 2:                                   # TensorProto
            for tf, tw, tv in _fields(v):
                if tf == 5 and tw == 5:
                    val = tv
                elif tf == 5 and tw == 2 and len(tv) >= 4:
                    val = struct.unpack_from("<f", tv)[0]
                elif tf == 6 and tw == 1:
                    val = tv
                elif tf == 6 and tw == 2 and len(tv) >= 8:
                    val = struct.unpack_from("<d", tv)[0]
                elif tf == 4 and tw == 2 and len(tv) in (4, 8):     # tensor_content
                    val = struct.unpack_from("<f" if len(tv) == 4 else "<d", tv)[0]
    return tag, val


def read_tfevents(fh, scalars: dict):
    """Add {tag: {step: value}} from one TensorBoard event file (TFRecord of Event protos)."""
    import struct
    data = fh.read()
    i, n = 0, len(data)
    while i + 12 <= n:
        ln = struct.unpack_from("<Q", data, i)[0]
        rec = data[i + 12:i + 12 + ln]
        i += 12 + ln + 4                                           # length, crc, data, crc
        if len(rec) < ln:
            break                                                  # file still being written
        step = 0
        summary = None
        for f, w, v in _fields(rec):
            if f == 2 and w == 0:
                step = v
            elif f == 5 and w == 2:
                summary = v
        if summary is None:
            continue
        for f, w, v in _fields(summary):
            if f == 1 and w == 2:
                tag, val = _scalar_of(v)
                if tag is not None and val is not None and val == val:   # skip NaN
                    scalars.setdefault(tag, {})[step] = float(val)


TB_CACHE: dict = {}          # event file -> (size, scalars)


def training_curves(fs, experiment: str, max_points: int = 1500) -> dict:
    """Scalars of every AL iteration: Data/<exp>/diffusion_wdir/model_registry/model_iter_NN/run*/lightning_logs/
    version_*/events.out.tfevents.* (newest version with events per iteration)."""
    import re
    exp = posixpath.basename(experiment)
    root = f"Data/{exp}/diffusion_wdir/model_registry"
    out = {"root": root, "iterations": []}
    if not fs.isdir(root):
        return out
    for name, is_dir, _ in sorted(fs.listdir(root)):
        m = re.fullmatch(r"model_iter_(\d+)", name)
        if not (is_dir and m):
            continue
        best = None                                                 # (version, dir, files)
        for run, rd, _ in fs.listdir(f"{root}/{name}"):
            ll = f"{root}/{name}/{run}/lightning_logs"
            if not (rd and run.startswith("run") and fs.isdir(ll)):
                continue
            for ver, vd, _ in fs.listdir(ll):
                vm = re.fullmatch(r"version_(\d+)", ver)
                if not (vd and vm):
                    continue
                files = [(f, s) for f, d, s in fs.listdir(f"{ll}/{ver}") if not d and f.startswith("events.out.tfevents")]
                if files and (best is None or int(vm.group(1)) > best[0]):
                    best = (int(vm.group(1)), f"{ll}/{ver}", files)
        if best is None:
            continue
        scalars: dict = {}
        for f, size in sorted(best[2]):
            path = f"{best[1]}/{f}"
            cached = TB_CACHE.get(path)
            if cached is None or cached[0] != size:
                part: dict = {}
                with fs.open(path) as fh:
                    read_tfevents(fh, part)
                TB_CACHE[path] = cached = (size, part)
            for tag, pts in cached[1].items():
                scalars.setdefault(tag, {}).update(pts)
        series = {}
        for tag, pts in scalars.items():
            steps = sorted(pts)
            if len(steps) > max_points:                             # thin out, keep the last point
                k = len(steps) / max_points
                steps = [steps[int(j * k)] for j in range(max_points)] + [steps[-1]]
            series[tag] = [[s, pts[s]] for s in steps]
        out["iterations"].append({"iteration": int(m.group(1)), "dir": best[1], "scalars": series})
    return out


# ------------------------------------------------------------- generation, KDE targets and their evaluation
# Each iteration folder Data/<exp>/iter_<n>/ holds what the forward and backward jobs produced:
#   targets.json                 targets planned by the KDE (properties, density, sparsity, score, n_samples)
#   generated/t<id>/evaluation.csv   sampling metrics per target (validity, novelty, uniqueness, stability, ...)
#   dedup_report.csv             kept / duplicate / in_initial per generated molecule
#   prop_filter_report.json      hit rate against each molecule's own target (backward job)
#   <stage>/                     molecules left after each pipeline stage
# Experiments/<exp>/al_metrics.csv holds hit rate and corridor coverage per iteration.
# Pipeline order (forward: generate, dedup; backward: score filter, relax, prop calc, per-target filter, HDF5).
# Side folders hold what a stage removed or produced besides its output; query/ is the working folder each
# stage overwrites, so it is shown last.
STAGES = ["generated", "deduplicated", "score_filtered", "relaxed", "prop_calc", "accuracy_filtered", "hdf5"]
SIDE_STAGES = {"dedup_removed": "deduplicated", "force_calc": "relaxed", "rejected": "accuracy_filtered"}


def _read_csv_text(text: str) -> list:
    return list(csv.DictReader(io.StringIO(text)))


def _num(v):
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return v if v not in ("", None) else None


def _count_molecules(fs, folder: str) -> int:
    """.xyz files in a stage folder and its direct subfolders (generated/t<id>/); prop_calc holds one .json each."""
    ext = ".json" if posixpath.basename(folder) == "prop_calc" else ".xyz"
    n = 0
    for name, is_dir, _ in fs.listdir(folder):
        if is_dir:
            n += sum(1 for f, d, _ in fs.listdir(f"{folder}/{name}") if not d and f.endswith(ext))
        elif name.endswith(ext):
            n += 1
    return n


def generation_report(fs, experiment: str) -> dict:
    import re
    exp = posixpath.basename(experiment)
    root = f"Data/{exp}"
    out = {"root": root, "iterations": [], "al_metrics": []}
    am = f"{EXP_DIR}/{exp}/al_metrics.csv"
    if fs.exists(am):
        out["al_metrics"] = [{k: _num(v) for k, v in r.items()} for r in _read_csv_text(fs.read_text(am))]
    if not fs.isdir(root):
        return out
    iters = sorted((int(m.group(1)), n) for n, d, _ in fs.listdir(root)
                   if d and (m := re.fullmatch(r"iter_(\d+)", n)))
    for it, name in iters:
        d = f"{root}/{name}"
        entries = {n: is_dir for n, is_dir, _ in fs.listdir(d)}
        rec = {"iteration": it, "dir": d, "targets": [], "strategy": None, "prop_names": [], "evaluation": {},
               "dedup": {}, "funnel": [], "prop_filter": None}
        if "targets.json" in entries:
            try:
                tj = json.loads(fs.read_text(f"{d}/targets.json"))
                rec.update(strategy=tj.get("strategy"), prop_names=tj.get("prop_names", []), targets=tj.get("targets", []))
            except ValueError:
                pass
        if entries.get("generated"):
            for t, td, _ in sorted(fs.listdir(f"{d}/generated")):
                if td and fs.exists(f"{d}/generated/{t}/evaluation.csv"):
                    rows = _read_csv_text(fs.read_text(f"{d}/generated/{t}/evaluation.csv"))
                    if rows:
                        rec["evaluation"][t] = {k.strip(): _num(v) for k, v in rows[-1].items() if k.strip()}
        if "dedup_report.csv" in entries:
            for r in _read_csv_text(fs.read_text(f"{d}/dedup_report.csv")):
                c = rec["dedup"].setdefault(str(r.get("target_id", "")), {})
                c[r.get("status", "")] = c.get(r.get("status", ""), 0) + 1
        if "prop_filter_report.json" in entries:
            try:
                rec["prop_filter"] = json.loads(fs.read_text(f"{d}/prop_filter_report.json"))
            except ValueError:
                pass
        order = []
        for s in STAGES:
            order.append((s, None))
            order += [(x, s) for x, parent in SIDE_STAGES.items() if parent == s]
        known = {x for x, _ in order} | {"query"}
        order += [(n, None) for n in sorted(entries) if entries[n] and n not in known] + [("query", None)]
        for s, parent in order:
            if entries.get(s):
                n = _count_molecules(fs, f"{d}/{s}")
                if s == "score_filtered":
                    # the passed molecules stay in query/; score_filtered/ keeps only rejected/ and the summary
                    try:
                        c = json.loads(fs.read_text(f"{d}/score_filtered/score_screen_summary.json"))["counts"]
                        n = int(c["passed"])
                    except Exception:
                        rej = f"{d}/score_filtered/rejected"
                        dedup_n = next((f["n"] for f in rec["funnel"] if f["stage"] == "deduplicated"), None)
                        if fs.isdir(rej) and dedup_n is not None:
                            n = max(0, dedup_n - _count_molecules(fs, rej))
                rec["funnel"].append({"stage": s, "n": n, "side_of": parent})
        out["iterations"].append(rec)
    return out


KDE_CACHE: dict = {}         # (hdf5 path, size, props) -> property rows of the reference dataset


def _reference_rows(fs, rel: str, prop_names: list) -> list:
    """Rows of the reference HDF5 in file order with every property finite: the same rows (and order) as
    load_state(...).labelled_props(), so they line up with density_reference.npy."""
    import h5py
    import numpy as np
    size = next((s for n, _, s in fs.listdir(posixpath.dirname(rel) or ".") if n == posixpath.basename(rel)), None)
    key = (rel, size, tuple(prop_names))
    if key in KDE_CACHE:
        return KDE_CACHE[key]
    rows = []
    with fs.open(rel) as fh, h5py.File(fh, "r") as f:
        for top in f:
            if not isinstance(f[top], h5py.Group):
                continue
            for name in f[top]:
                mol = f[top][name]
                if not isinstance(mol, h5py.Group):
                    continue
                row = []
                for p in prop_names:
                    if p not in mol:
                        row.append(float("nan"))
                    elif p == "atNUM":
                        row.append(float(mol[p].shape[0]))
                    else:
                        row.append(float(np.asarray(mol[p]).reshape(-1)[0]))
                if all(v == v and abs(v) != float("inf") for v in row):
                    rows.append(row)
    KDE_CACHE.clear()
    KDE_CACHE[key] = rows
    return rows


def kde_report(fs, experiment: str, max_points: int = 15000) -> dict:
    """What the KDE planner saw and did: the reference dataset with its density, and per iteration the
    candidates it scored, the targets it chose and where the computed molecules landed."""
    import re
    exp = posixpath.basename(experiment)
    out = {"reference": None, "reference_msg": "", "iterations": []}
    root_abs = fs.abspath("").rstrip("/") + "/"
    # reference density (density_reference.npy + al_state.json written by the planner)
    st_path = f"{EXP_DIR}/{exp}/al_state.json"
    try:
        st = json.loads(fs.read_text(st_path)).get("density_reference") if fs.exists(st_path) else None
        if not st:
            out["reference_msg"] = "no density_reference in al_state.json yet (the planner writes it in anchor / density-band mode)"
        else:
            src, names = st["key"]["source"], st["key"]["prop_names"]
            rel = src[len(root_abs):] if src.startswith(root_abs) else None
            if rel is None or not fs.exists(rel):
                out["reference_msg"] = f"reference dataset {src} is not inside the project folder"
            else:
                import numpy as np
                with fs.open(f"{EXP_DIR}/{exp}/{st.get('file', 'density_reference.npy')}") as fh:
                    dens = np.load(io.BytesIO(fh.read()))
                rows = _reference_rows(fs, rel, names)
                if len(rows) != len(dens):
                    out["reference_msg"] = f"{len(dens)} densities but {len(rows)} molecules in {rel}: they do not line up"
                else:
                    srt = np.sort(dens)
                    pct = 100.0 * np.searchsorted(srt, dens, side="right") / len(srt)
                    idx = np.arange(len(rows)) if len(rows) <= max_points else \
                        np.random.default_rng(0).choice(len(rows), max_points, replace=False)
                    arr = np.asarray(rows)[idx]
                    out["reference"] = {"source": rel, "n": len(rows), "prop_names": names,
                                        "cols": {p: arr[:, j].round(6).tolist() for j, p in enumerate(names)},
                                        "density": dens[idx].tolist(), "pct": pct[idx].round(2).tolist()}
    except Exception as e:
        out["reference_msg"] = f"{type(e).__name__}: {e}"
    root = f"Data/{exp}"
    if fs.isdir(root):
        for n, d, _ in sorted(fs.listdir(root)):
            m = re.fullmatch(r"iter_(\d+)", n)
            if not (d and m):
                continue
            rec = {"iteration": int(m.group(1)), "candidates": [], "landed": []}
            for fname, key in (("anchor_candidates.csv", "candidates"), ("prop_filter_properties.csv", "landed")):
                if fs.exists(f"{root}/{n}/{fname}"):
                    rec[key] = [{k: (v if k in ("molecule", "status", "reason") else _num(v)) for k, v in r.items()}
                                for r in _read_csv_text(fs.read_text(f"{root}/{n}/{fname}"))]
            out["iterations"].append(rec)
    return out


def list_molecules(fs, experiment: str, iteration: int, stage: str, limit: int = 3000) -> dict:
    """Molecules of one stage folder with their target, dedup status, SMILES and (if computed) hit/miss."""
    exp = posixpath.basename(experiment)
    d = f"Data/{exp}/iter_{int(iteration)}"
    stage = posixpath.basename(stage)
    out = {"dir": d, "stages": [], "molecules": [], "truncated": False}
    if not fs.isdir(d):
        return out
    out["stages"] = [n for n, isd, _ in sorted(fs.listdir(d)) if isd]
    dedup, landed, reasons, meta = {}, {}, {}, {}
    if fs.exists(f"{d}/dedup_report.csv"):
        dedup = {r["name"]: r for r in _read_csv_text(fs.read_text(f"{d}/dedup_report.csv"))}
    if fs.exists(f"{d}/prop_filter_properties.csv"):
        landed = {r["molecule"]: r for r in _read_csv_text(fs.read_text(f"{d}/prop_filter_properties.csv"))}
    if fs.exists(f"{d}/rejected/reasons.json"):
        try:
            reasons = json.loads(fs.read_text(f"{d}/rejected/reasons.json"))
        except ValueError:
            pass
    if fs.exists(f"{d}/candidates_meta.json"):
        try:
            meta = json.loads(fs.read_text(f"{d}/candidates_meta.json"))
        except ValueError:
            pass
    if stage not in out["stages"]:
        return out
    files = []
    for n, isd, _ in fs.listdir(f"{d}/{stage}"):
        if isd:
            files += [(f"{n}_{f[:-4]}", f"{stage}/{n}/{f}") for f, fd, _ in fs.listdir(f"{d}/{stage}/{n}")
                      if not fd and f.endswith(".xyz")]
        elif n.endswith(".xyz"):
            files.append((n[:-4], f"{stage}/{n}"))
    import re
    files.sort(key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x[0])])
    out["truncated"] = len(files) > limit
    for mid, path in files[:limit]:
        base = re.sub(r"^(opt-)+", "", mid)                   # relaxed molecules are renamed opt-<id>
        dd, ld = dedup.get(base, {}), landed.get(mid) or landed.get(base) or {}
        tid = meta.get(base, {}).get("target_id", dd.get("target_id", ld.get("target_id")))
        out["molecules"].append({"id": mid, "path": path, "target_id": _num(tid), "dedup": dd.get("status"),
                                 "smiles": dd.get("connectivity"), "status": ld.get("status"),
                                 "hit": _num(ld.get("hit")), "reason": reasons.get(mid + ".xyz") or ld.get("reason") or None})
    return out


def read_xyz(fs, experiment: str, iteration: int, rel: str) -> dict:
    d = f"Data/{posixpath.basename(experiment)}/iter_{int(iteration)}"
    p = posixpath.normpath(f"{d}/{rel}")
    if not p.startswith(d + "/") or not p.endswith(".xyz"):
        raise PermissionError("not an .xyz file of this iteration")
    return {"path": p, "text": fs.read_text(p)}


def list_logs(fs, experiment: str) -> list:
    """Log files of an experiment: the files named in the job scripts plus everything in logs/."""
    exp_dir = f"{EXP_DIR}/{posixpath.basename(experiment)}"
    named = _script_logs(fs, experiment)
    paths = {p for p in named if p.startswith(exp_dir + "/")}
    for d in (f"{exp_dir}/logs",):
        if fs.isdir(d):
            paths |= {f"{d}/{n}" for n, is_dir, _ in fs.listdir(d) if not is_dir}
    out = []
    for p in sorted(paths):
        job, kind = named.get(p, ("", p.rsplit(".", 1)[-1] if "." in p else ""))
        size = None
        if fs.exists(p):
            size = next((s for n, _, s in fs.listdir(posixpath.dirname(p)) if n == posixpath.basename(p)), None)
        out.append({"path": p[len(exp_dir) + 1:], "job": job, "kind": kind, "size": size})
    order = {"forward": 0, "backward": 1, "": 2}
    return sorted(out, key=lambda e: (order.get(e["job"], 2), e["kind"] != "out", e["path"]))


def read_log(fs, experiment: str, rel: str, tail: int = 200_000) -> dict:
    exp_dir = f"{EXP_DIR}/{posixpath.basename(experiment)}"
    p = posixpath.normpath(f"{exp_dir}/{rel}")
    if not p.startswith(exp_dir + "/"):
        raise PermissionError("outside the experiment folder")
    if not fs.exists(p):
        return {"path": rel, "size": 0, "text": "", "missing": True}
    with fs.open(p) as f:
        f.seek(0, 2)
        size = f.tell()
        start = max(0, size - tail)
        f.seek(start)
        data = f.read()
    text = data.decode("utf-8", "replace")
    if start:
        text = text.split("\n", 1)[-1]          # drop the partial first line
    return {"path": rel, "size": size, "text": text, "truncated": start > 0}


# ---- clean: empty the output folders of a failed iteration so it can be re-run
# Everything an iteration writes into Data/<exp>/iter_<n>/, in pipeline order: (name, job, block).
# Folders are only emptied, never removed; query/ is the working folder each stage overwrites, so it comes last.
ITER_LAYOUT = [
    ("anchor_candidates.csv", "forward", "plan"),
    ("targets.json", "forward", "plan"),
    ("generated", "forward", "generate"),
    ("candidates_meta.json", "forward", "generate"),
    ("deduplicated", "forward", "dedup"),
    ("dedup_removed", "forward", "dedup"),
    ("dedup_report.csv", "forward", "dedup"),
    ("score_filtered", "backward", "score filter"),
    ("relaxed", "backward", "relax"),
    ("force_calc", "backward", "relax"),
    ("prop_calc", "backward", "prop calc"),
    ("accuracy_filtered", "backward", "prop filter"),
    ("rejected", "backward", "prop filter"),
    ("prop_filter_properties.csv", "backward", "prop filter"),
    ("prop_filter_report.json", "backward", "prop filter"),
    ("hdf5", "backward", "HDF5"),
    ("query", "", "working folder"),
]


def _iter_dirs(fs, experiment: str) -> list:
    d = f"Data/{posixpath.basename(experiment)}"
    if not fs.isdir(d):
        return []
    its = [int(n[5:]) for n, isd, _ in fs.listdir(d) if isd and n.startswith("iter_") and n[5:].isdigit()]
    return sorted(its)


def clean_listing(fs, experiment: str, iteration=None) -> dict:
    """Iterations of an experiment and, for the chosen one, its folders/files with sizes."""
    its = _iter_dirs(fs, experiment)
    out = {"iterations": its, "iteration": None, "entries": []}
    if not its:
        return out
    it = int(iteration) if iteration not in (None, "") else its[-1]
    if it not in its:
        return out
    out["iteration"] = it
    d = f"Data/{posixpath.basename(experiment)}/iter_{it}"
    rank = {name: (i, job, block) for i, (name, job, block) in enumerate(ITER_LAYOUT)}
    # pipeline order; anything not in ITER_LAYOUT goes after it, alphabetically
    for n, isd, size in sorted(fs.listdir(d), key=lambda e: (rank.get(e[0], (len(rank),))[0], e[0])):
        files = None
        if isd:
            size, files = fs.du(f"{d}/{n}")
        _, job, block = rank.get(n, (None, "", ""))
        out["entries"].append({"name": n, "dir": isd, "size": size, "files": files, "job": job, "block": block})
    return out


def clean_iteration(fs, experiment: str, iteration, names: list) -> dict:
    """Empty the chosen folders of one iteration (the folders stay) and delete the chosen top-level files."""
    exp = posixpath.basename(experiment)
    it = int(iteration)
    d = f"Data/{exp}/iter_{it}"
    if not fs.isdir(d):
        raise ValueError(f"{d} does not exist")
    with JOB_LOCK:
        try:
            _experiment_status(fs, exp)          # refresh job states from Slurm before deciding
        except Exception:
            pass
        active = [e for e in read_history(fs, exp) if e.get("state") not in FINAL]
        if active:
            raise ValueError("a job of this experiment is still pending/running ("
                             + ", ".join(f"{e['job']} {e['id']}" for e in active) + "); cancel it first")
        present = {n: isd for n, isd, _ in fs.listdir(d)}
        emptied, removed = [], []
        for n in names:
            n = posixpath.basename(str(n))
            if n not in present:
                continue
            if present[n]:
                fs.clear_dir(f"{d}/{n}")
                emptied.append(f"{d}/{n}")
            else:
                fs.remove_file(f"{d}/{n}")
                removed.append(f"{d}/{n}")
        return {"emptied": emptied, "removed": removed}


def _history_path(experiment):
    return f"{EXP_DIR}/{posixpath.basename(experiment)}/navigator_jobs.json"


def read_history(fs, experiment) -> list:
    try:
        return json.loads(fs.read_text(_history_path(experiment)))
    except Exception:
        return []


def write_history(fs, experiment, hist):
    fs.write_text(_history_path(experiment), json.dumps(hist, indent=1))


import threading
JOB_LOCK = threading.RLock()   # serialises history.json updates and chain submissions across server threads


def submit_job(fs, experiment: str, job: str, start_at=None) -> dict:
    with JOB_LOCK:
        return _submit_job(fs, experiment, job, start_at)


def _submit_job(fs, experiment: str, job: str, start_at=None) -> dict:
    if job not in JOBS:
        raise ValueError("job must be forward or backward")
    start_at = _check_start(job, start_at)
    jobs_dir = f"{EXP_DIR}/{posixpath.basename(experiment)}/jobs"
    script = f"qalpa_{job}.sh"
    if not fs.exists(f"{jobs_dir}/{script}"):
        raise ValueError(f"{jobs_dir}/{script} not found")
    s = JOB_SETTINGS[job]
    opts = sbatch_options(s)
    if start_at:
        opts = _export_var(opts, START_ONCE_ENV, start_at)
    code, out = Runner(s).run(["sbatch", "--parsable", *opts, script], fs.abspath(jobs_dir))
    jobid = out.strip().splitlines()[-1].split(";")[0] if out.strip() else ""
    if code != 0 or not jobid.isdigit():
        raise RuntimeError(out.strip() or f"sbatch failed (exit {code})")
    cfg = fs.read_text(config_path(experiment))
    import time
    entry = {"id": jobid, "job": job, "iteration": _config_value(cfg, "current_iter"),
             "cluster": cluster_name(s), "state": "PENDING",
             "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    if start_at:
        entry["start_at"] = start_at
    hist = read_history(fs, experiment)
    hist.append(entry)
    write_history(fs, experiment, hist)
    return entry


def job_state(job_entry: dict) -> str:
    s = settings_for(job_entry)
    try:
        code, out = Runner(s).run(["sacct", "-j", job_entry["id"], "-n", "-X", "-P", "-o", "State"], "/")
        state = out.strip().splitlines()[0].split()[0] if code == 0 and out.strip() else ""
        if not state:
            code2, out2 = Runner(s).run(["squeue", "-h", "-j", job_entry["id"], "-o", "%T"], "/")
            state = out2.strip() if code2 == 0 and out2.strip() else ""
        if not state:
            if code != 0:      # sacct itself failed: report it rather than pass the last known state off as current
                raise RuntimeError(f"sacct failed: {out.strip()[:200] or f'exit {code}'}")
            state = job_entry.get("state", "UNKNOWN")    # e.g. just submitted, not in the accounting yet
        return state.rstrip("+")
    except Exception as err:   # e.g. SSH key passphrase not entered since the server started
        raise RuntimeError(f"cannot query {job_entry['job']} job {job_entry['id']}: {type(err).__name__}: {err}") from err


def experiment_status(fs, experiment: str) -> dict:
    with JOB_LOCK:
        return _experiment_status(fs, experiment)


def _experiment_status(fs, experiment: str) -> dict:
    hist = read_history(fs, experiment)
    changed = False
    errors = []
    for e in hist:
        c = e.get("cluster", "")              # older entries: "local" or the login node's full name
        if c == "local" or "." in c:
            # "local" meant the cluster that job type runs on (forward: capella, backward: barnard)
            e["cluster"] = CLUSTERS[e["job"]][0] if c == "local" else cluster_name({"via": "ssh", "host": c})
            changed = True
        if e.get("state") not in FINAL:
            try:
                new = job_state(e)
            except RuntimeError as err:
                errors.append(str(err))
                continue
            if new != e.get("state"):
                e["state"] = new
                changed = True
    if changed:
        write_history(fs, experiment, hist)
    cfg = fs.read_text(config_path(experiment))
    return {"current_iter": _config_value(cfg, "current_iter"), "num_iterations": _config_value(cfg, "num_iterations"),
            "start_at": _config_value(cfg, "start_at"),
            "jobs": hist[-30:][::-1], "chain": CHAINS.get(experiment), "errors": errors}


# ---- chain: submit the other job once the previous one has completed
def cancel_job(fs, experiment: str, jobid: str) -> dict:
    """scancel a job submitted from the Navigator, on the cluster it was submitted to. Stops the chain."""
    entry = next((e for e in read_history(fs, experiment) if str(e.get("id")) == str(jobid)), None)
    if entry is None:
        raise ValueError(f"job {jobid} was not submitted from the Navigator")
    s = settings_for(entry)             # never scancel the same number on another cluster
    jobs_dir = f"{EXP_DIR}/{posixpath.basename(experiment)}/jobs"
    code, out = Runner(s).run(["scancel", str(entry["id"])], fs.abspath(jobs_dir))
    if code != 0:
        state = job_state(entry)
        if state in FINAL:
            raise RuntimeError(f"job {entry['id']} has already ended ({state.lower()})")
        raise RuntimeError(out.strip() or f"scancel failed (exit {code})")
    exp = posixpath.basename(experiment)
    if CHAINS.get(exp, {}).get("active"):
        CHAINS[exp].update(active=False, message=f"stopped: you cancelled {entry['job']} job {entry['id']}")
    return {"id": entry["id"], "job": entry["job"]}


def delete_job_record(fs, experiment: str, jobid: str) -> dict:
    """Remove an ended job from the Navigator's history (the job itself and its output files are untouched)."""
    with JOB_LOCK:
        hist = read_history(fs, experiment)
        entry = next((e for e in hist if str(e.get("id")) == str(jobid)), None)
        if entry is None:
            raise ValueError(f"job {jobid} is not in the history")
        if entry.get("state") not in FINAL:
            raise ValueError(f"job {jobid} is still {str(entry.get('state', 'unknown')).lower()}: cancel it first")
        write_history(fs, experiment, [e for e in hist if e is not entry])
    return {"id": entry["id"], "job": entry["job"]}


CHAINS: dict = {}            # experiment -> {"active", "message"}


def chain_step(fs, experiment: str):
    with JOB_LOCK:          # loop thread and /api/chain may call this at once: never submit twice
        _chain_step(fs, experiment)


def _chain_step(fs, experiment: str):
    ch = CHAINS.get(experiment)
    if not ch or not ch["active"]:
        return
    st = experiment_status(fs, experiment)
    last = st["jobs"][0] if st["jobs"] else None
    if last and last["state"] not in FINAL and st["errors"]:
        ch["message"] = f"cannot check job state ({st['errors'][0]}) - enter the SSH key passphrase in Job settings"
        return
    start = ch.get("start_at")      # one-off "Start at": used for the chain's first submission only
    if last and last["state"] not in FINAL:
        ch["message"] = (f"waiting for {last['job']} job {last['id']} ({last['state'].lower()})"
                         + (f"; then {STAGE_JOB[start]} job from {start}" if start else ""))
        return
    if last and last["state"] != "COMPLETED" and not start:   # with a start block the chain resumes a failed run
        ch.update(active=False, message=f"stopped: {last['job']} job {last['id']} ended {last['state'].lower()}. "
                                        "To resume, choose a block in Start at, then turn the chain on again")
        return
    nxt = STAGE_JOB[start] if start else "forward" if not last or last["job"] == "backward" else "backward"
    if nxt == "forward" and last:
        try:
            if int(st["current_iter"]) > int(st["num_iterations"]):
                ch.update(active=False, message="finished: all iterations done")
                return
        except (TypeError, ValueError):
            pass
    try:
        e = submit_job(fs, experiment, nxt, start)
        ch.pop("start_at", None)
        ch["message"] = f"submitted {nxt} job {e['id']}" + (f" (start at {start})" if start else "")
    except Exception as err:
        ch.update(active=False, message=f"stopped: could not submit {nxt}: {err}")


def chain_loop(interval=30):
    import threading
    import time

    def loop():
        while True:
            for exp in list(CHAINS):
                try:
                    chain_step(Handler.fs, exp)
                except Exception as err:
                    CHAINS[exp]["message"] = f"error: {err}"
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()


# ------------------------------------------------------------------------------------------------- web server
class Handler(BaseHTTPRequestHandler):
    fs = None                 # active backend
    default_root: Path
    token: str

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype, cookie=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Strict")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False, default=str).encode(), "application/json")

    def _error(self, msg, code=400):
        self._json({"error": msg}, code)

    def _token_given(self, q):
        cookies = dict(c.strip().split("=", 1) for c in self.headers.get("Cookie", "").split(";") if "=" in c)
        return q.get("token", [""])[0] or self.headers.get("X-Token", "") or cookies.get(COOKIE, "")

    def _authorized(self, q):
        return secrets.compare_digest(self._token_given(q), self.token)

    def _info(self):
        fs = Handler.fs
        try:
            alive = fs.alive()
        except Exception:
            alive = False
        return {"mode": fs.kind, "root": fs.label, "name": posixpath.basename(str(fs.root).rstrip("/")) or "/",
                "alive": alive}

    def do_GET(self):
        url, q = urlparse(self.path), parse_qs(urlparse(self.path).query)
        if url.path in ("/", "/navigator.html"):
            # the page itself is public: without a valid token it shows the sign-in screen
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8", cookie=self._authorized(q))
        if not self._authorized(q):
            return self._error("invalid token", 403)
        try:
            if url.path == "/api/info":
                return self._json(self._info())
            if url.path == "/api/progress":
                return self._json(PROGRESS)
            if url.path == "/api/config":
                exp = q.get("experiment", [""])[0]
                return self._json({"experiment": exp, "text": Handler.fs.read_text(config_path(exp))})
            if url.path == "/api/training":
                return self._json(training_curves(Handler.fs, q.get("experiment", [""])[0]))
            if url.path == "/api/generation":
                return self._json(generation_report(Handler.fs, q.get("experiment", [""])[0]))
            if url.path == "/api/kde":
                return self._json(kde_report(Handler.fs, q.get("experiment", [""])[0]))
            if url.path == "/api/molecules":
                return self._json(list_molecules(Handler.fs, q.get("experiment", [""])[0], q.get("iteration", ["0"])[0],
                                                 q.get("stage", [""])[0]))
            if url.path == "/api/xyz":
                return self._json(read_xyz(Handler.fs, q.get("experiment", [""])[0], q.get("iteration", ["0"])[0],
                                           q.get("path", [""])[0]))
            if url.path == "/api/clean":
                return self._json(clean_listing(Handler.fs, q.get("experiment", [""])[0], q.get("iteration", [""])[0]))
            if url.path == "/api/logs":
                return self._json({"files": list_logs(Handler.fs, q.get("experiment", [""])[0])})
            if url.path == "/api/log":
                return self._json(read_log(Handler.fs, q.get("experiment", [""])[0], q.get("path", [""])[0],
                                           int(q.get("tail", ["200000"])[0] or 200000)))
            if url.path == "/api/status":
                return self._json(experiment_status(Handler.fs, q.get("experiment", [""])[0]))
            if url.path == "/api/job_settings":
                return self._json(public_job_settings())
            if url.path == "/api/props":
                rel = q.get("path", [""])[0].strip("/")
                return self._json({"path": rel, "props": property_names(Handler.fs, rel)})
            if url.path == "/api/experiments":
                exps = list_experiments(Handler.fs)
                root = str(Handler.fs.root).rstrip("/") + "/"
                for e in exps:      # dataset path relative to the project folder, when it lies inside it
                    d = e["dataset"] or ""
                    e["dataset_rel"] = d[len(root):] if d.startswith(root) else None
                return self._json({"experiments": exps})
            if url.path == "/api/list":
                rel = q.get("path", [""])[0].strip("/")
                if not Handler.fs.isdir(rel):
                    return self._error("not a directory")
                items = [{"name": n, "dir": d, "size": s} for n, d, s in Handler.fs.listdir(rel)
                         if not n.startswith(".") and (d or n.lower().endswith(DATA_EXT))]
                items.sort(key=lambda it: (not it["dir"], it["name"].lower()))
                return self._json({"path": rel, "items": items})
            if url.path == "/api/load":
                rel = q.get("path", [""])[0].strip("/")
                max_mol = int(q.get("max", ["1000"])[0]) or sys.maxsize       # 0 = all molecules
                PROGRESS.update(done=0, total=0, file_total=None, active=True)
                try:
                    with Handler.fs.open(rel) as f:
                        if rel.lower().endswith(".csv"):
                            cols = read_csv(f, max_mol)
                        elif rel.lower().endswith((".h5", ".hdf5")):
                            cols = read_hdf5(f, max_mol)
                        else:
                            return self._error("choose an .h5, .hdf5 or .csv file")
                finally:
                    PROGRESS["active"] = False
                clean = {k: [None if v != v else v for v in vals] for k, vals in cols.items()}
                return self._json({"name": posixpath.basename(rel), "columns": clean,
                                   "file_total": PROGRESS.get("file_total")})
        except PermissionError as e:
            return self._error(str(e), 403)
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}", 500)
        self._error("not found", 404)

    def do_POST(self):
        url, q = urlparse(self.path), parse_qs(urlparse(self.path).query)
        if not self._authorized(q):
            return self._error("invalid token", 403)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if url.path == "/api/connect":
                if body.get("mode") == "ssh":
                    new = SSHFS(body.get("host", "").strip(), body.get("user", "").strip(), body.get("port") or 22,
                                body.get("key", "").strip() or None, body.get("password") or None,
                                body.get("root", "").strip() or "~", body.get("python", "").strip() or "python3")
                else:
                    new = LocalFS(Path(body.get("root") or Handler.default_root).expanduser())
                    if not new.root.is_dir():
                        return self._error(f"{new.root} is not a folder")
                old, Handler.fs = Handler.fs, new
                old.close()
                return self._json(self._info())
            if url.path == "/api/init":
                res = init_experiment(Handler.fs, body.get("name", "").strip(), body.get("dataset", "").strip("/"))
                return self._json(res)
            if url.path == "/api/config":
                return self._json(save_config(Handler.fs, body.get("experiment", ""), body.get("text", "")))
            if url.path == "/api/job_settings":
                for j in JOBS:
                    JOB_SETTINGS[j].update({k: v for k, v in body.get(j, {}).items() if k in JOB_FIELDS})
                    pw = body.get(j, {}).get("passphrase")
                    if pw == "\x00clear":
                        JOB_SETTINGS[j].pop("_passphrase", None)
                    elif pw:  # empty keeps the passphrase already held
                        JOB_SETTINGS[j]["_passphrase"] = pw
                Runner._clients.clear()
                saved = {j: {k: v for k, v in s.items() if k in JOB_FIELDS} for j, s in JOB_SETTINGS.items()}
                SETTINGS_FILE.write_text(json.dumps(saved, indent=1))
                return self._json(public_job_settings())
            if url.path == "/api/submit":
                return self._json(submit_job(Handler.fs, body.get("experiment", ""), body.get("job", ""),
                                             body.get("start_at")))
            if url.path == "/api/cancel":
                return self._json(cancel_job(Handler.fs, body.get("experiment", ""), body.get("id", "")))
            if url.path == "/api/clean":
                if not body.get("confirm"):
                    return self._error("deletion not confirmed")
                return self._json(clean_iteration(Handler.fs, body.get("experiment", ""), body.get("iteration"),
                                                  body.get("names", [])))
            if url.path == "/api/delete_job":
                return self._json(delete_job_record(Handler.fs, body.get("experiment", ""), body.get("id", "")))
            if url.path == "/api/chain":
                exp = posixpath.basename(body.get("experiment", ""))
                if body.get("on"):
                    start = (body.get("start_at") or "").strip()
                    if start and start not in STAGE_JOB:
                        return self._error(f"'{start}' is not a pipeline block")
                    CHAINS[exp] = {"active": True, "message": "starting…", **({"start_at": start} if start else {})}
                    chain_step(Handler.fs, exp)
                elif exp in CHAINS:
                    CHAINS[exp].update(active=False, message="stopped by you")
                    CHAINS[exp].pop("start_at", None)
                return self._json(CHAINS.get(exp) or {"active": False, "message": ""})
            if url.path == "/api/save_trajectory":
                return self._json(save_trajectory(Handler.fs, body.get("experiment", ""), body.get("name", ""),
                                                  body.get("content", ""), body.get("columns", [])))
            if url.path == "/api/save":
                folder = body.get("folder", "").strip("/")
                name = posixpath.basename(body.get("name", ""))
                if not name.lower().endswith(".csv") or not Handler.fs.isdir(folder):
                    return self._error("save a .csv file into an existing folder")
                rel = posixpath.join(folder, name) if folder else name
                if Handler.fs.exists(rel) and not body.get("overwrite"):
                    return self._json({"exists": True, "path": rel}, 409)
                Handler.fs.write_text(rel, body.get("content", ""))
                return self._json({"saved": rel})
        except PermissionError as e:
            return self._error(str(e), 403)
        except ValueError as e:
            return self._error(str(e), 400)
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}", 500)
        self._error("not found", 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE.parent, help="project folder (default: this repository)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.default_root = args.root.resolve()
    Handler.fs = LocalFS(Handler.default_root)
    Handler.token = load_token()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)       # port busy: take a free one
    port = srv.server_address[1]
    fix_local_via()
    ask_passphrases_at_startup()
    chain_loop()
    print(f"\n  QALPA Navigator — project folder {Handler.default_root}\n")
    print(f"  Open:   http://localhost:{port}/?token={Handler.token}")
    print(f"  Token:  {Handler.token}   (paste it if the page asks)\n")
    print(f"  Remote machine? Forward the port first: ssh -L {port}:localhost:{port} <login-node>")
    print("  Stop with Ctrl+C.\n")
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        Handler.fs.close()
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
