"""Import diffusion training/evaluation curves of all AL iterations into MLflow.

Training logs to TensorBoard (unchanged). This script reads the TensorBoard
event files (and the sampling evaluation.csv) of every iteration and mirrors
them into an MLflow tracking store:

  experiment  = QALPA experiment name (e.g. exp_allostric_0)
  run iter_NN = one run per iteration: all scalars (train/val/test losses,
                per-dataset val_dataset_<k>/... and test_dataset_<k>/...,
                sampling metrics val/validity, val/mol_stable, ...), all
                sampling-evaluation metrics as gen/<name> (step = epoch), the
                training hparams and the sampling shares (sampling.json).
                Artifacts:
                  test/test_metrics.csv|png  test losses on all test data
                      ("all") and on every dataset alone
                  generation/evaluation.csv|png  sampling evaluations
  run overview = one point per iteration (step = iteration) with the final
                value of every epoch-level metric ("final/..."), the sampling
                share of each dataset ("share/dataset_<k>") and the forgetting
                of each dataset ("forgetting/dataset_<k>": test loss minus its
                best earlier test loss, > 0 = forgetting).
                Artifacts:
                  test/test_loss_matrix.csv|png  test loss of every model
                      (row = iteration) on every test set (column)
                  test/test_metrics_all_iterations.csv  all test losses

It is incremental and can be re-run at any time, also while training runs.
Needs only `mlflow`, `tensorboard` and `matplotlib` (no torch).

Usage:
  python src/scripts/sync_mlflow.py \
      --model-root Data/<exp>/diffusion_wdir/model_registry \
      --tracking-uri file:///abs/path/Data/mlruns --experiment <exp>
  mlflow ui --backend-store-uri file:///abs/path/Data/mlruns --port 5000
"""
import argparse
import io
import json
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from mlflow.entities import Metric, Param  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

ITER_DIR = re.compile(r"model_iter_(\d+)")
VERSION_DIR = re.compile(r"version_(\d+)")
TEST_KEY = re.compile(r"test_(dataset_\d+)/")
BATCH = 900  # MLflow allows up to 1000 metrics per log_batch call
MAX_PARAM_LEN = 500
LOSSES = ("loss", "coords_loss", "atoms_loss", "charges_loss", "bonds_loss")
GEN_SKIP = {"step", "epoch", "run_time", "ngraphs"}  # evaluation.csv bookkeeping


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


# --------------------------------------------------------------------------
# test losses per test set
# --------------------------------------------------------------------------
def dataset_index(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1


def set_order(name: str):
    return (name != "all", dataset_index(name))


def set_color(name: str):
    return "#4d4d4d" if name == "all" else plt.cm.tab10(dataset_index(name) % 10)


def test_table(scalars, sampling):
    """Test losses of the full test split ("all") and of every dataset alone.

    One row per test set with n_test, train_share and the epoch means of the
    diffusion losses (run_train.py tests best_mol_stab.ckpt after training).
    None while the iteration has no test results.
    """
    info = {d["name"]: d for d in (sampling or {}).get("datasets", [])}
    names = set(info) | {m.group(1) for k in scalars if (m := TEST_KEY.match(k))}
    names = sorted(names, key=dataset_index)
    rows = []
    for test_set in ["all"] + names:
        if test_set == "all" or f"test_{test_set}/loss_epoch" not in scalars:
            if test_set != "all" and len(names) != 1:
                continue
            # one dataset has no extra dataloader: its test split is the full one
            prefix = "test"
        else:
            prefix = f"test_{test_set}"
        values = {
            loss: max(scalars[key], key=lambda e: e[0])[2]  # newest test run
            for loss in LOSSES
            if (key := f"{prefix}/{loss}_epoch") in scalars
        }
        if not values:
            continue
        if test_set == "all":
            n_test = sum(d["n_test"] for d in info.values()) if info else None
            share = 1.0 if info else None
        else:
            n_test = info.get(test_set, {}).get("n_test")
            share = info.get(test_set, {}).get("share")
        rows.append({"test_set": test_set, "n_test": n_test, "train_share": share, **values})
    return pd.DataFrame(rows) if rows else None


def plot_test_table(df, title, path):
    cols = [c for c in LOSSES if c in df]
    fig, axes = plt.subplots(
        1, len(cols), figsize=(2.6 * len(cols) + 0.8, 3.8), squeeze=False
    )
    x = np.arange(len(df))
    for ax, col in zip(axes[0], cols):
        values = df[col].to_numpy(float)
        ax.bar(x, values, color=[set_color(s) for s in df.test_set])
        ax.set_xticks(x, df.test_set, rotation=45, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        for xi, value in zip(x, values):
            if np.isfinite(value):
                ax.annotate(f"{value:.3g}", (xi, value), ha="center", va="bottom", fontsize=7)
        ax.margins(y=0.15)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loss_matrix(loss, path):
    """Heatmap + curves of the test loss: model (row) x test set (column)."""
    fig, (ax_map, ax_line) = plt.subplots(
        1, 2, figsize=(13, max(3.8, 0.45 * len(loss) + 2.2))
    )
    values = loss.to_numpy(float)
    image = ax_map.imshow(np.ma.masked_invalid(values), cmap="viridis", aspect="auto")
    ax_map.set_xticks(range(len(loss.columns)), loss.columns, rotation=45, ha="right")
    ax_map.set_yticks(range(len(loss.index)), [f"iter_{n:02d}" for n in loss.index])
    ax_map.set_xlabel("test set")
    ax_map.set_ylabel("model")
    ax_map.set_title("test loss (lower is better)", fontsize=10)
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            light = image.norm(value) < 0.5
            ax_map.text(j, i, f"{value:.3g}", ha="center", va="center",
                        fontsize=8, color="w" if light else "k")
        else:
            ax_map.text(j, i, "-", ha="center", va="center", fontsize=8, color="0.5")
    fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.04)

    for test_set in loss.columns:
        series = loss[test_set].dropna()
        ax_line.plot(series.index, series.values, marker="o", label=test_set,
                     color=set_color(test_set), lw=2.2 if test_set == "all" else 1.5)
    ax_line.set_xlabel("AL iteration (model)")
    ax_line.set_ylabel("test loss")
    ax_line.set_xticks(list(loss.index))
    ax_line.set_title("test loss per test set across iterations", fontsize=10)
    ax_line.legend(fontsize=8)
    ax_line.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# sampling evaluations (evaluation.csv, every test_interval epochs)
# --------------------------------------------------------------------------
def generation_table(iter_dir: Path):
    path = iter_dir / "run0" / "evaluation.csv"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header = lines[0]  # the file is appended to, header repeated per evaluation
    body = [line for line in lines[1:] if line != header]
    df = pd.read_csv(io.StringIO("\n".join([header] + body)))
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "epoch" in df and len(df) > 1:
        # keep only the latest training (a restart appends again from epoch 0)
        resets = np.flatnonzero(np.diff(df["epoch"].to_numpy()) <= 0)
        if len(resets):
            df = df.iloc[resets[-1] + 1 :]
    return df.reset_index(drop=True)


def generation_columns(df):
    return [
        c for c in df.columns
        if c not in GEN_SKIP and pd.api.types.is_numeric_dtype(df[c])
    ]


def generation_scalars(df, wall):
    """gen/<metric>: [(wall_time, epoch, value), ...]."""
    if df is None or "epoch" not in df:
        return {}
    out = {}
    for col in generation_columns(df):
        events = [
            (wall + 1e-3 * i, int(epoch), float(value))
            for i, (epoch, value) in enumerate(zip(df["epoch"], df[col]))
            if np.isfinite(value)
        ]
        if events:
            out[f"gen/{col}"] = events
    return out


def plot_generation(df, title, path):
    cols = generation_columns(df)
    ncol = 4
    nrow = -(-len(cols) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.3 * nrow), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.plot(df["epoch"], df[col], marker="o", color="#1f77b4")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("epoch", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def log_files(client, run_id, files, artifact_path):
    for path in files:
        client.log_artifact(run_id, str(path), artifact_path=artifact_path)


def log_iteration_artifacts(client, run_id, n, test_df, gen_df):
    """(Re)write the per-iteration tables/plots; the file store overwrites them."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if test_df is not None:
            test_df.to_csv(tmp / "test_metrics.csv", index=False)
            plot_test_table(
                test_df,
                f"iter_{n:02d}: test losses on all test data and on each dataset alone",
                tmp / "test_metrics.png",
            )
            log_files(client, run_id, [tmp / "test_metrics.csv", tmp / "test_metrics.png"], "test")
        if gen_df is not None and len(gen_df):
            gen_df.to_csv(tmp / "evaluation.csv", index=False)
            files = [tmp / "evaluation.csv"]
            if "epoch" in gen_df:
                plot_generation(
                    gen_df,
                    f"iter_{n:02d}: sampling evaluation (every test_interval epochs)",
                    tmp / "generation_metrics.png",
                )
                files.append(tmp / "generation_metrics.png")
            log_files(client, run_id, files, "generation")


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
def sync_iteration(client, experiment_id, n, iter_dir, log_dir):
    """Create/update run iter_NN; returns (scalars, test table)."""
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
    gen_df = generation_table(iter_dir)
    if gen_df is not None:
        wall = (iter_dir / "run0" / "evaluation.csv").stat().st_mtime
        scalars.update(generation_scalars(gen_df, wall))
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

    test_df = test_table(scalars, load_sampling(iter_dir))
    log_iteration_artifacts(client, run.info.run_id, n, test_df, gen_df)
    client.set_terminated(run.info.run_id, "FINISHED")
    tested = ", ".join(test_df.test_set) if test_df is not None else "none yet"
    print(
        f"  iter_{n:02d}: {version}, {len(scalars)} metrics, {len(new)} new points, "
        f"test sets: {tested}"
    )
    return scalars, test_df


def is_epoch_metric(key: str) -> bool:
    if key.endswith("_step") or key in ("epoch",) or key.startswith("lr"):
        return False
    return True


def overview_test_artifacts(client, run_id, tables):
    """Loss matrix (model x test set) artifacts; returns forgetting metrics."""
    if not tables:
        return []
    long = pd.concat(
        [df.assign(iteration=n) for n, df in sorted(tables.items())], ignore_index=True
    )
    long = long[["iteration"] + [c for c in long.columns if c != "iteration"]]
    sets = sorted(long.test_set.unique(), key=set_order)
    loss = long.pivot(index="iteration", columns="test_set", values="loss").reindex(columns=sets)

    # forgetting/<dataset>: test loss minus the best (lowest) earlier test loss
    metrics = []
    for test_set in sets:
        if test_set == "all":
            continue
        best = None
        for n, value in loss[test_set].items():
            if not np.isfinite(value):
                continue
            gap = 0.0 if best is None else float(value - best)
            metrics.append(Metric(f"forgetting/{test_set}", gap, 0, int(n)))
            best = value if best is None else min(best, value)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        long.to_csv(tmp / "test_metrics_all_iterations.csv", index=False)
        loss.to_csv(tmp / "test_loss_matrix.csv")
        plot_loss_matrix(loss, tmp / "test_loss_matrix.png")
        log_files(
            client,
            run_id,
            [tmp / "test_loss_matrix.png", tmp / "test_loss_matrix.csv",
             tmp / "test_metrics_all_iterations.csv"],
            "test",
        )
    return metrics


def sync_overview(client, experiment_id, per_iteration):
    """Rebuild run 'overview': final value of each metric, step = iteration."""
    old = find_run(client, experiment_id, **{"qalpa.overview": "true"})
    if old is not None:
        client.delete_run(old.info.run_id)
    run = client.create_run(
        experiment_id, run_name="overview", tags={"qalpa.overview": "true"}
    )
    metrics = []
    tables = {}
    for n, (iter_dir, scalars, test_df) in sorted(per_iteration.items()):
        if test_df is not None:
            tables[n] = test_df
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
    metrics += overview_test_artifacts(client, run.info.run_id, tables)
    log_metrics(client, run.info.run_id, metrics)
    client.set_terminated(run.info.run_id, "FINISHED")
    print(
        f"  overview: {len(per_iteration)} iterations, {len(metrics)} points, "
        f"test results of {len(tables)} iterations"
    )


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
            *sync_iteration(client, experiment_id, n, iter_dir, log_dir),
        )
    if per_iteration:
        sync_overview(client, experiment_id, per_iteration)


if __name__ == "__main__":
    main()
