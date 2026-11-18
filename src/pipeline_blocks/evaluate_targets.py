"""Evaluate block: per-iteration hit rate and KDE coverage of the trajectory corridor (ported from qalpa2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.al_state import load_state
from src.utils.kde_plan import estimate_coverage, robust_center_scale
from src.pipeline_blocks.plan_targets import build_region
from src.pipeline_blocks.label_targets import read_prop_filter_report

METRIC_COLUMNS = ["iteration", "n_labelled", "hit_rate", "hit_rate_by_target", "coverage"]


def compute_metrics(state, iteration: int, corridor, min_neff: float, seed: int, n_coverage: int = 1000) -> dict:
    out = {c: np.nan for c in METRIC_COLUMNS}
    out["iteration"] = int(iteration)
    new = (state.iteration == iteration) & np.isfinite(state.props).all(axis=1)
    out["n_labelled"] = int(new.sum())

    rec = new & (state.hit >= 0)
    if rec.any():
        out["hit_rate"] = float(np.mean(state.hit[rec] == 1))
        by_t = {int(t): float(np.mean(state.hit[rec & (state.target_id == t)] == 1))
                for t in np.unique(state.target_id[rec])}
        out["hit_rate_by_target"] = ";".join(f"{t}:{v:.3f}" for t, v in sorted(by_t.items()))

    # coverage: share of the corridor with at least min_neff kernel-weighted past requests nearby
    records, labelled = state.record_mask(), state.labelled_mask()
    if corridor is not None and records.any() and labelled.any():
        center, scale = robust_center_scale(state.props[labelled])
        candidates = corridor.sample(n_coverage, np.random.default_rng([int(seed), int(iteration), 2]))
        out["coverage"] = estimate_coverage(state.requested[records], candidates, center, scale, min_neff)
    return out


def run_evaluate(*, traj, al: dict, hdf5_path, iteration: int, seed: int, metrics_file, iter_dir=None,
                 log=print) -> dict:
    """Append this iteration's metrics to metrics_file and return them.

    If prop_filter ran (<iter_dir>/prop_filter_report.json), the HDF5 file only holds the hits, so the hit rates
    are taken from that report (all evaluated molecules, misses included).
    """
    state = load_state(hdf5_path, traj.plan_names)
    corridor = None
    if len(traj) >= 2 or al.get("kde_region", "corridor") == "anchor":
        corridor = build_region(traj, al, iteration)   # the corridor, or the anchor box in anchor mode
    row = compute_metrics(state, iteration, corridor, al["min_neff"], seed)
    report = read_prop_filter_report(iter_dir) if iter_dir is not None else None
    if report is not None and int(report.get("iteration", -1)) == int(iteration) and report.get("evaluated"):
        row["hit_rate"] = float(report["hit_rate"])
        row["hit_rate_by_target"] = ";".join(f"{t}:{v['hit_rate']:.3f}" for t, v in report["per_target"].items())
        log("  Hit rates from prop_filter_report.json (before the misses were removed)")
    path = Path(metrics_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=METRIC_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
    log(f"  Hit rate {row['hit_rate']}, coverage {row['coverage']} (appended to {path})")
    return row
