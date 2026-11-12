"""Import diffusion training/evaluation curves of all AL iterations into MLflow.

Training logs to TensorBoard (unchanged). This script reads the TensorBoard
event files of every iteration and mirrors them into an MLflow tracking store:

  experiment  = QALPA experiment name (e.g. exp_allostric_0)
  run iter_NN = one run per iteration: all scalars (train/val/test losses,
                per-dataset val_dataset_<k>/... and test_dataset_<k>/...,
                sampling metrics val/validity, val/mol_stable, ...), the
                training hparams and the sampling shares (sampling.json)
  run overview = one point per iteration (step = iteration) with the final
                value of every epoch-level metric ("final/...") and the
                sampling share of each dataset ("share/dataset_<k>"), so
                forgetting/overfitting can be followed across iterations,
                e.g. final/val_dataset_0/loss_epoch.

It is incremental and can be re-run at any time, also while training runs.
Needs only `mlflow` and `tensorboard` (no torch).

Usage:
  python src/scripts/sync_mlflow.py \
      --model-root Data/<exp>/diffusion_wdir/model_registry \
      --tracking-uri file:///abs/path/Data/mlruns --experiment <exp>
  mlflow ui --backend-store-uri file:///abs/path/Data/mlruns --port 5000
"""
import argparse
import json
import re
from pathlib import Path

import yaml
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

ITER_DIR = re.compile(r"model_iter_(\d+)")
VERSION_DIR = re.compile(r"version_(\d+)")
BATCH = 900  # MLflow allows up to 1000 metrics per log_batch call
MAX_PARAM_LEN = 500


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-root", required=True, help="EQdiffParam.model_root")
    parser.add_argument("--tracking-uri", required=True, help="e.g. file:///abs/path/mlruns")
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    return parser.parse_args()


def iteration_dirs(model_root: Path):
    found = []
    for path in model_root.iterdir() if model_root.is_dir() else []:
        match = ITER_DIR.fullmatch(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def latest_log_dir(iter_dir: Path):
    """Newest lightning_logs/version_<k> that contains event files."""
    versions = []
    for path in (iter_dir / "run0" / "lightning_logs").glob("version_*"):
        match = VERSION_DIR.fullmatch(path.name)
        if match and any(path.glob("events.out.tfevents.*")):
            versions.append((int(match.group(1)), path))
    return max(versions)[1] if versions else None


def read_scalars(log_dir: Path):
    """{tag: [(wall_time, step, value), ...]} from all event files in log_dir."""
    scalars = {}
    for event_file in sorted(log_dir.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            scalars.setdefault(tag, []).extend(
                (e.wall_time, e.step, e.value) for e in acc.Scalars(tag)
            )
    return scalars


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that turns unknown tags (python objects) into plain strings."""


_TolerantLoader.add_multi_constructor(
    "", lambda loader, suffix, node: f"<{suffix}>"
)


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, val in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), val, out)
    else:
        out[prefix] = str(value)[:MAX_PARAM_LEN]
    return out


def run_params(iter_dir: Path, log_dir: Path):
    params = {}
    hparams_file = log_dir / "hparams.yaml"
    if hparams_file.is_file():
        hparams = yaml.load(hparams_file.read_text(), Loader=_TolerantLoader) or {}
        hparams = hparams.get("hparams", hparams) if isinstance(hparams, dict) else {}
        flatten("", hparams, params)
    sampling = load_sampling(iter_dir)
    if sampling:
        for ds in sampling.get("datasets", []):
            for key in ("share", "n_train", "n_val", "n_test", "draws_per_sample_per_epoch"):
                params[f"sampling.{ds['name']}.{key}"] = str(ds[key])
        params["sampling.samples_per_epoch"] = str(sampling.get("samples_per_epoch"))
    return params


def load_sampling(iter_dir: Path):
    path = iter_dir / "sampling.json"
    return json.loads(path.read_text()) if path.is_file() else None


def find_run(client, experiment_id, **tags):
    query = " and ".join(f"tags.`{k}` = '{v}'" for k, v in tags.items())
    runs = client.search_runs([experiment_id], filter_string=query, max_results=1)
    return runs[0] if runs else None


def log_metrics(client, run_id, metrics):
    for i in range(0, len(metrics), BATCH):
        client.log_batch(run_id, metrics=metrics[i : i + BATCH])


def sync_iteration(client, experiment_id, n, iter_dir, log_dir):
    """Create/update run iter_NN; returns the TensorBoard scalars."""
    version = log_dir.name
    run = find_run(client, experiment_id, **{"qalpa.iteration": n})
    if run is not None and run.data.tags.get("qalpa.tb_version") != version:
        # training of this iteration was restarted -> replace the run
        client.delete_run(run.info.run_id)
        run = None
    if run is None:
        run = client.create_run(
            experiment_id,
            run_name=f"iter_{n:02d}",
            tags={
                "qalpa.iteration": str(n),
                "qalpa.tb_version": version,
                "qalpa.log_dir": str(log_dir),
            },
        )
        params = [Param(k, v) for k, v in run_params(iter_dir, log_dir).items()]
        params.append(Param("iteration", str(n)))
        for i in range(0, len(params), 100):
            client.log_batch(run.info.run_id, params=params[i : i + 100])
        for name in ("sampling.json",):
            if (iter_dir / name).is_file():
                client.log_artifact(run.info.run_id, str(iter_dir / name))
        if (log_dir / "hparams.yaml").is_file():
            client.log_artifact(run.info.run_id, str(log_dir / "hparams.yaml"))

    scalars = read_scalars(log_dir)
    new = []
    for key, events in scalars.items():
        logged = client.get_metric_history(run.info.run_id, key)
        last_step = max((m.step for m in logged), default=-1)
        n_at_last = sum(m.step == last_step for m in logged)
        events = sorted(events, key=lambda e: (e[1], e[0]))
        # skip what is already logged (same step may hold several values)
        seen_last = 0
        for wall, step, value in events:
            if step < last_step:
                continue
            if step == last_step and seen_last < n_at_last:
                seen_last += 1
                continue
            new.append(Metric(key, float(value), int(wall * 1000), int(step)))
    log_metrics(client, run.info.run_id, new)
    client.set_terminated(run.info.run_id, "FINISHED")
    print(f"  iter_{n:02d}: {version}, {len(scalars)} metrics, {len(new)} new points")
    return scalars


def is_epoch_metric(key: str) -> bool:
    if key.endswith("_step") or key in ("epoch",) or key.startswith("lr"):
        return False
    return True


def sync_overview(client, experiment_id, per_iteration):
    """Rebuild run 'overview': final value of each metric, step = iteration."""
    old = find_run(client, experiment_id, **{"qalpa.overview": "true"})
    if old is not None:
        client.delete_run(old.info.run_id)
    run = client.create_run(
        experiment_id, run_name="overview", tags={"qalpa.overview": "true"}
    )
    metrics = []
    for n, (iter_dir, scalars) in sorted(per_iteration.items()):
        finals = {}
        for key, events in scalars.items():
            if is_epoch_metric(key) and events:
                wall, step, value = max(events, key=lambda e: (e[1], e[0]))
                finals[key] = (wall, value)
        sampling = load_sampling(iter_dir) or {}
        datasets = sampling.get("datasets", [])
        if len(datasets) == 1:
            # a single dataset has no per-dataset loaders: val/... == val_<name>/...
            name = datasets[0]["name"]
            for key in list(finals):
                stage, _, rest = key.partition("/")
                if stage in ("val", "test") and rest:
                    finals.setdefault(f"{stage}_{name}/{rest}", finals[key])
        for key, (wall, value) in finals.items():
            metrics.append(Metric(f"final/{key}", float(value), int(wall * 1000), n))
        for ds in datasets:
            metrics.append(Metric(f"share/{ds['name']}", float(ds["share"]), 0, n))
            metrics.append(
                Metric(
                    f"draws_per_sample_per_epoch/{ds['name']}",
                    float(ds["draws_per_sample_per_epoch"]),
                    0,
                    n,
                )
            )
    log_metrics(client, run.info.run_id, metrics)
    client.set_terminated(run.info.run_id, "FINISHED")
    print(f"  overview: {len(per_iteration)} iterations, {len(metrics)} points")


def main():
    args = get_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(args.experiment)
    )
    print(f"MLflow sync: {args.model_root} -> {args.tracking_uri} [{args.experiment}]")

    per_iteration = {}
    for n, iter_dir in iteration_dirs(Path(args.model_root)):
        log_dir = latest_log_dir(iter_dir)
        if log_dir is None:
            print(f"  iter_{n:02d}: no TensorBoard logs yet, skipped")
            continue
        per_iteration[n] = (
            iter_dir,
            sync_iteration(client, experiment_id, n, iter_dir, log_dir),
        )
    if per_iteration:
        sync_overview(client, experiment_id, per_iteration)


if __name__ == "__main__":
    main()
