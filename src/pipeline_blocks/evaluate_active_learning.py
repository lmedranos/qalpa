from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.pipeline_blocks.acquire_candidates import ACQUISITION_FILE
from src.components.dataset_labels import NUM_ATOMS_PROP
from src.components.kde_plan import robust_center_scale
from src.components.candidates import canonical_id
from src.pipeline_blocks.plan_targets import Corridor
from src.components.al_state import load_state
from src.base import CoverageEstimator
from src.components.kde_plan import KDECoverageEstimator

METRIC_COLUMNS = ["iteration", "n_labelled", "n_selected", "hit_rate", "hit_rate_by_target", "surrogate_nmae",
                  "surrogate_rmse", "calib_1sigma", "calib_2sigma", "coverage", "mean_acq_score"]


def compute_metrics(state, iteration: int, acq: Optional[pd.DataFrame], corridor, min_neff: float,
                    seed: int, n_coverage: int = 1000,
                    coverage_estimator: CoverageEstimator | None = None) -> dict:
    """Compute metrics for molecules labelled in one iteration."""
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

    cols = [j for j, p in enumerate(state.prop_names) if p != NUM_ATOMS_PROP]
    labelled = state.labelled_mask()
    if acq is not None and len(acq):
        selected = acq[acq["selected"].astype(bool)]
        out["n_selected"] = int(len(selected))
        scores = selected["score"].to_numpy(dtype=float)
        out["mean_acq_score"] = float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan

        achieved = {canonical_id(str(k).split("/")[-1]): state.props[i]
                    for i, k in enumerate(state.keys) if new[i]}
        names = [state.prop_names[j] for j in cols]
        pred, truth, sigma = [], [], []
        for _, row in selected.iterrows():
            cid = canonical_id(row["name"])
            mu_row = [row.get(f"mu_{p}", np.nan) for p in names]
            sigma_row = [row.get(f"sigma_{p}", np.nan) for p in names]
            if cid in achieved and np.isfinite(mu_row).all() and np.isfinite(sigma_row).all():
                pred.append(mu_row)
                sigma.append(sigma_row)
                truth.append(achieved[cid][cols])
        if pred and labelled.any():
            pred, truth, sigma = np.array(pred), np.array(truth), np.array(sigma)
            _, scale = robust_center_scale(state.props[labelled][:, cols])
            error = pred - truth
            out["surrogate_nmae"] = float(np.mean(np.abs(error) / scale))
            out["surrogate_rmse"] = float(np.sqrt(np.mean((error / scale) ** 2)))
            out["calib_1sigma"] = float(np.mean(np.abs(error) <= sigma))
            out["calib_2sigma"] = float(np.mean(np.abs(error) <= 2 * sigma))

    records = state.record_mask()
    if corridor is not None and records.any() and labelled.any():
        center, scale = robust_center_scale(state.props[labelled])
        candidates = corridor.sample(n_coverage, np.random.default_rng([int(seed), int(iteration), 2]))
        estimator = coverage_estimator or KDECoverageEstimator()
        out["coverage"] = estimator.estimate(
            state.requested[records], candidates, center, scale, min_neff
        )
    return out


def append_metrics(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{c: row.get(c, np.nan) for c in METRIC_COLUMNS}]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def early_stop(path, patience: int, min_delta: float) -> tuple:
    """Return whether recent metrics satisfy the configured early-stop rule."""
    path = Path(path)
    if not path.exists():
        return False, ""
    df = pd.read_csv(path).drop_duplicates("iteration", keep="last")
    if len(df) < patience + 1:
        return False, ""
    previous, recent = df.iloc[:-patience], df.iloc[-patience:]
    previous_hit, recent_hit = previous["hit_rate"].dropna(), recent["hit_rate"].dropna()
    previous_error, recent_error = previous["surrogate_nmae"].dropna(), recent["surrogate_nmae"].dropna()
    hit_gain = (recent_hit.max() - previous_hit.max()) if len(previous_hit) and len(recent_hit) else 0.0
    error_gain = ((previous_error.min() - recent_error.min()) / previous_error.min()
                  if len(previous_error) and len(recent_error) and previous_error.min() > 0 else 0.0)
    if hit_gain < min_delta and error_gain < min_delta:
        return True, (f"hit rate gain {hit_gain:.3f} and surrogate error gain {error_gain:.3f} "
                      f"are below {min_delta} over {patience} iterations")
    return False, ""


def run_evaluate(*, traj, al: dict, hdf5_path, iter_dir, iteration: int, seed: int, metrics_file, log=print) -> tuple:
    """Append this iteration's metrics; returns (row, stop, reason) where stop reflects the early-stop rule."""
    state = load_state(hdf5_path, traj.prop_names)
    acq_path = Path(iter_dir) / ACQUISITION_FILE
    acq = pd.read_csv(acq_path) if acq_path.exists() else None
    corridor = None
    if len(traj) >= 2:
        corridor = Corridor.from_trajectory(traj, al["corridor_pad"], al["corridor_lookahead"], iteration)
    estimator = KDECoverageEstimator()
    row = compute_metrics(state, iteration, acq, corridor, al["min_neff"], seed,
                          coverage_estimator=estimator)
    append_metrics(metrics_file, row)
    log(f"  Hit rate {row['hit_rate']}, surrogate nMAE {row['surrogate_nmae']}, coverage {row['coverage']}")

    stop, reason = False, ""
    es = al["early_stop"]
    if es["enabled"]:
        stop, reason = early_stop(metrics_file, int(es["patience"]), float(es["min_delta"]))
    return row, stop, reason
