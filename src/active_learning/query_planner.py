from __future__ import annotations

from typing import Optional

import numpy as np

from src.active_learning.constants import NUM_ATOMS_PROP
from src.active_learning.density import PropertyDensity, kernel_regress, repulsion
from src.active_learning.ids import Target
from src.active_learning.state import ALState, read_al_state_file, write_al_state_file


def is_count_prop(name: str) -> bool:
    return name == NUM_ATOMS_PROP


def _snap_counts(points: np.ndarray, prop_names) -> np.ndarray:
    out = np.array(points, dtype=float, copy=True)
    for j, p in enumerate(prop_names):
        if is_count_prop(p):
            out[:, j] = np.maximum(np.round(out[:, j]), 1.0)
    return out


class Corridor:
    """Padded polyline through trajectory rows; targets are sampled inside it."""

    def __init__(self, points: np.ndarray, prop_names, pad: float):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if len(points) < 2:
            raise ValueError("The corridor needs at least 2 trajectory rows; add rows to the trajectory CSV.")
        self.points = points
        self.prop_names = list(prop_names)
        span = points.max(axis=0) - points.min(axis=0)
        ref = np.maximum(np.abs(points).mean(axis=0), 1.0)
        self.pad = pad * np.where(span > 1e-12, span, ref)

    @classmethod
    def from_trajectory(cls, traj, pad: float, lookahead: Optional[int], iteration: int) -> "Corridor":
        names = list(traj.prop_names)
        pts = np.column_stack([np.asarray(traj.properties[p], dtype=float) for p in names])
        if lookahead is not None:
            end = min(iteration + int(lookahead), len(pts) - 1)
            pts = pts[: max(end + 1, 2)]
        return cls(pts, names, pad)

    def bounds(self) -> tuple:
        return self.points.min(axis=0) - self.pad, self.points.max(axis=0) + self.pad

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        seg = rng.integers(0, len(self.points) - 1, size=n)
        t = rng.random(n)[:, None]
        base = self.points[seg] + t * (self.points[seg + 1] - self.points[seg])
        noise = (rng.random(base.shape) * 2.0 - 1.0) * self.pad
        return _snap_counts(base + noise, self.prop_names)


def trajectory_box(traj, pad: float) -> tuple:
    """Padded bounding box (lo, hi) of all trajectory rows; works for a single row."""
    pts = np.column_stack([np.asarray(traj.properties[p], dtype=float) for p in traj.prop_names])
    span = pts.max(axis=0) - pts.min(axis=0)
    margin = pad * np.where(span > 1e-12, span, np.maximum(np.abs(pts).mean(axis=0), 1.0))
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def allocate(total: int, weights, minimum: int = 1) -> np.ndarray:
    """Integer shares summing to `total`, each at least `minimum`, proportional to `weights`."""
    w = np.asarray(weights, dtype=float)
    k = len(w)
    if total < k * minimum:
        raise ValueError(f"Cannot give {k} targets at least {minimum} samples from a pool of {total}")
    rest = total - k * minimum
    p = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
    raw = p * rest
    base = np.floor(raw).astype(int)
    order = np.argsort(-(raw - base))
    base[order[: rest - base.sum()]] += 1
    return base + minimum


def _row_target(traj, row: int, tid: int, n_samples: int, meta: dict) -> Target:
    props = {p: float(traj.properties[p][row]) for p in traj.prop_names}
    return Target(tid, props, n_samples, dict(meta, row=int(row)))


def plan_trajectory(traj, iteration: int, total: int) -> list:
    row = min(iteration, len(traj) - 1)
    if row != iteration:
        print(f"  Trajectory has {len(traj)} rows; iteration {iteration} reuses the last row.")
    return [_row_target(traj, row, 0, total, {"strategy": "trajectory"})]


def _last_hit_rate_at_row(state: ALState, traj, row: int) -> Optional[float]:
    rec = state.record_mask()
    if not rec.any():
        return None
    target = np.array([float(traj.properties[p][row]) for p in traj.prop_names])
    same = rec & np.all(np.isclose(state.requested, target, rtol=1e-6, atol=1e-9), axis=1)
    if not same.any():
        return None
    last = state.iteration[same].max()
    sel = same & (state.iteration == last)
    return float(np.mean(state.hit[sel] == 1))


def plan_frontier(traj, state: ALState, al: dict, iteration: int, total: int, state_file) -> list:
    saved = read_al_state_file(state_file)
    front = saved.get("frontier", {})
    if front.get("iteration") == iteration:
        row = int(front["row"])
    else:
        row = int(front.get("row", 0))
        rate = _last_hit_rate_at_row(state, traj, row)
        if rate is not None and rate >= al["advance_hit_rate"]:
            row = min(row + 1, len(traj) - 1)
        saved["frontier"] = {"row": row, "iteration": int(iteration)}
        write_al_state_file(state_file, saved)
    return [_row_target(traj, row, 0, total, {"strategy": "frontier"})]


def score_targets(cand: np.ndarray, state: ALState, prop_names, d: dict, min_neff: float) -> tuple:
    """(score, excluded mask, scale, parts) for candidate targets from data density and generator response."""
    min_bw = np.array([1.0 if is_count_prop(p) else 0.0 for p in prop_names])
    labelled = state.labelled_props()
    if len(labelled):
        dens = PropertyDensity(labelled, bandwidth=d["bandwidth"], min_bandwidth=min_bw, logp_floor=d["logp_floor"])
        sparsity = -dens.logp(cand)
        center, scale = dens.center, dens.scale
    else:
        sparsity = np.zeros(len(cand))
        center = cand.mean(axis=0)
        scale = np.where(cand.std(axis=0) > 1e-12, cand.std(axis=0), 1.0)

    rec = state.record_mask()
    cols = [j for j, p in enumerate(prop_names) if not is_count_prop(p)]
    if rec.any():
        req, ach = state.requested[rec], state.props[rec]
        err_samples = np.mean(np.abs(ach[:, cols] - req[:, cols]) / scale[cols], axis=1)
        err, n_eff = kernel_regress(req, err_samples, cand, center, scale, prior=float(err_samples.mean()))
        reach, _ = kernel_regress(req, state.hit[rec].astype(float), cand, center, scale, prior=1.0)
    else:
        err, n_eff, reach = np.zeros(len(cand)), np.zeros(len(cand)), np.ones(len(cand))

    score = (d["w_sparse"] * _norm01(sparsity) + d["w_err"] * _norm01(err)
             + d["w_explore"] / np.sqrt(n_eff + 1.0))
    excluded = (reach < d["min_reach"]) & (n_eff > min_neff)
    if excluded.all():
        print("  All candidate targets look unreachable; keeping them all.")
        excluded[:] = False
    return score, excluded, scale, {"sparsity": sparsity, "err": err, "n_eff": n_eff, "reach": reach}


def plan_kde_map(traj, state: ALState, al: dict, iteration: int, total: int, rng: np.random.Generator) -> list:
    corridor = Corridor.from_trajectory(traj, al["corridor_pad"], al["corridor_lookahead"], iteration)
    d = al["density"]
    cand = corridor.sample(int(d["n_candidates"]), rng)
    score, excluded, scale, parts = score_targets(cand, state, corridor.prop_names, d, al["min_neff"])

    k = min(int(al["targets_per_iter"]), len(cand), total)
    mask = np.ones(len(cand))
    chosen = []
    for _ in range(k):
        adj = np.where(excluded, -np.inf, score * mask)
        adj[chosen] = -np.inf
        i = int(np.argmax(adj))
        chosen.append(i)
        mask *= repulsion(cand[i], cand, scale)

    shares = allocate(total, np.exp(score[chosen] - score[chosen].max()))
    return [
        Target(tid, {p: float(cand[i, j]) for j, p in enumerate(corridor.prop_names)}, int(n),
               {"strategy": "kde_map", "score": float(score[i]), "n_eff": float(parts["n_eff"][i]),
                "reach": float(parts["reach"][i])})
        for tid, (i, n) in enumerate(zip(chosen, shares))
    ]


def plan_targets(traj, state: ALState, al: dict, iteration: int, total: int, seed: int, state_file=None) -> list:
    """Targets for one iteration. Deterministic given seed, iteration and state."""
    strategy = al["query_strategy"]
    if strategy == "trajectory":
        return plan_trajectory(traj, iteration, total)
    if strategy == "frontier":
        return plan_frontier(traj, state, al, iteration, total, state_file)
    if not state.record_mask().any():
        return plan_trajectory(traj, iteration, total)
    rng = np.random.default_rng([int(seed), int(iteration)])
    return plan_kde_map(traj, state, al, iteration, total, rng)
