from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.active_learning.constants import NUM_ATOMS_PROP
from src.active_learning.density import kernel_regress, robust_center_scale
from src.active_learning.ids import canonical_id
from src.active_learning.state import ALState

METRIC_COLUMNS = ["iteration", "n_labelled", "n_selected", "hit_rate", "hit_rate_by_target", "surrogate_nmae",
                  "surrogate_rmse", "calib_1sigma", "calib_2sigma", "coverage", "mean_acq_score"]


def compute_metrics(state: ALState, iteration: int, acq: Optional[pd.DataFrame], corridor, min_neff: float,
                    seed: int, n_coverage: int = 1000) -> dict:
    """Metrics for the molecules labelled in `iteration`; NaN where a quantity is unavailable."""
    out = {c: np.nan for c in METRIC_COLUMNS}
    out["iteration"] = int(iteration)
    new = (state.iteration == iteration) & np.isfinite(state.props).all(axis=1)
    out["n_labelled"] = int(new.sum())

    rec = new & (state.hit >= 0)
    if rec.any():
        out["hit_rate"] = float(np.mean(state.hit[rec] == 1))
        by_t = {int(t): float(np.mean(state.hit[rec & (state.target_id == t)] == 1)) for t in np.unique(state.target_id[rec])}
        out["hit_rate_by_target"] = ";".join(f"{t}:{v:.3f}" for t, v in sorted(by_t.items()))

    cols = [j for j, p in enumerate(state.prop_names) if p != NUM_ATOMS_PROP]
    lab = state.labelled_mask()
    if acq is not None and len(acq):
        sel = acq[acq["selected"].astype(bool)]
        out["n_selected"] = int(len(sel))
        scores = sel["score"].to_numpy(dtype=float)
        out["mean_acq_score"] = float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan

        achieved = {canonical_id(str(k).split("/")[-1]): state.props[i] for i, k in enumerate(state.keys) if new[i]}
        names = [state.prop_names[j] for j in cols]
        pred, truth, sig = [], [], []
        for _, r in sel.iterrows():
            cid = canonical_id(r["name"])
            mu_row = [r.get(f"mu_{p}", np.nan) for p in names]
            sg_row = [r.get(f"sigma_{p}", np.nan) for p in names]
            if cid in achieved and np.isfinite(mu_row).all() and np.isfinite(sg_row).all():
                pred.append(mu_row)
                sig.append(sg_row)
                truth.append(achieved[cid][cols])
        if pred and lab.any():
            pred, truth, sig = np.array(pred), np.array(truth), np.array(sig)
            _, scale = robust_center_scale(state.props[lab][:, cols])
            err = pred - truth
            out["surrogate_nmae"] = float(np.mean(np.abs(err) / scale))
            out["surrogate_rmse"] = float(np.sqrt(np.mean((err / scale) ** 2)))
            out["calib_1sigma"] = float(np.mean(np.abs(err) <= sig))
            out["calib_2sigma"] = float(np.mean(np.abs(err) <= 2 * sig))

    allrec = state.record_mask()
    if corridor is not None and allrec.any() and lab.any():
        center, scale = robust_center_scale(state.props[lab])
        cand = corridor.sample(n_coverage, np.random.default_rng([int(seed), int(iteration), 2]))
        _, n_eff = kernel_regress(state.requested[allrec], np.zeros(int(allrec.sum())), cand, center, scale)
        out["coverage"] = float(np.mean(n_eff >= min_neff))
    return out


def append_metrics(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in METRIC_COLUMNS}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def early_stop(path, patience: int, min_delta: float) -> tuple:
    """(stop, reason): no gain in best hit rate nor in surrogate error over the last `patience` iterations."""
    path = Path(path)
    if not path.exists():
        return False, ""
    df = pd.read_csv(path).drop_duplicates("iteration", keep="last")
    if len(df) < patience + 1:
        return False, ""
    prev, last = df.iloc[:-patience], df.iloc[-patience:]
    hp, hl = prev["hit_rate"].dropna(), last["hit_rate"].dropna()
    ep, el = prev["surrogate_nmae"].dropna(), last["surrogate_nmae"].dropna()
    hit_gain = (hl.max() - hp.max()) if len(hp) and len(hl) else 0.0
    err_gain = ((ep.min() - el.min()) / ep.min()) if len(ep) and len(el) and ep.min() > 0 else 0.0
    if hit_gain < min_delta and err_gain < min_delta:
        return True, f"hit rate gain {hit_gain:.3f} and surrogate error gain {err_gain:.3f} are below {min_delta} over {patience} iterations"
    return False, ""
