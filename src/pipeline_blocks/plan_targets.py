"""Plan block: choose this iteration's generation targets in property space (ported from qalpa2).

Strategies (experiment_config.yaml -> al.query_strategy):
  - trajectory: the trajectory row of the current iteration (the original behaviour, one target).
  - frontier:   stay on a trajectory row until its hit rate reaches al.advance_hit_rate, then move on.
  - kde_map:    sample candidate targets inside a padded corridor around the trajectory and pick the
                al.targets_per_iter best by data sparsity (Gaussian KDE of the labelled molecules),
                the generator's past error, and how little that region has been requested so far.
                With al.kde_region: anchor, candidates are instead sampled in a box of +/- al.anchor_box
                (fraction of each value) around the current iteration's trajectory row, and the row itself is
                target 0 (al.include_anchor). al.density.density_percentiles = [lo, hi] keeps only candidates
                whose density under a fixed KDE of the initial dataset lies between those percentiles of the
                initial dataset's own (leave-one-out) density; the band is computed once and cached.

The planning axes are the trajectory's conditioning properties plus atNUM if the trajectory has an
atNUM column (Trajectory.plan_names).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.al_config import NUM_ATOMS_PROP, kde_counts
from src.utils.al_state import ALState, load_state, read_al_state_file, write_al_state_file

AL_GROUP_PREFIX = "G-active-loop-"   # HDF5 groups written by run_to_hdf5; everything else is the initial dataset
from src.utils.kde_plan import PropertyDensity, kernel_regress, repulsion
from src.utils.trajectory import Target, read_targets, write_targets


def is_count_prop(name: str) -> bool:
    return name == NUM_ATOMS_PROP


def _snap_counts(points: np.ndarray, prop_names) -> np.ndarray:
    out = np.array(points, dtype=float, copy=True)
    for j, p in enumerate(prop_names):
        if is_count_prop(p):
            out[:, j] = np.maximum(np.round(out[:, j]), 1.0)
    return out


def _traj_points(traj) -> np.ndarray:
    axes = traj.plan_properties
    return np.column_stack([np.asarray(axes[p], dtype=float) for p in traj.plan_names])


class Corridor:
    """Padded polyline through trajectory rows; candidate targets are sampled inside it."""

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
        pts = _traj_points(traj)
        if lookahead is not None:
            end = min(iteration + int(lookahead), len(pts) - 1)
            pts = pts[: max(end + 1, 2)]
        return cls(pts, traj.plan_names, pad)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        seg = rng.integers(0, len(self.points) - 1, size=n)
        t = rng.random(n)[:, None]
        base = self.points[seg] + t * (self.points[seg + 1] - self.points[seg])
        noise = (rng.random(base.shape) * 2.0 - 1.0) * self.pad
        return _snap_counts(base + noise, self.prop_names)


class AnchorBox:
    """Box of +/- `box` (fraction of each value) around one trajectory row; candidate targets are sampled inside it."""

    def __init__(self, anchor: np.ndarray, prop_names, box: float, ref_span: np.ndarray, row: int):
        self.anchor = _snap_counts(np.atleast_2d(np.asarray(anchor, dtype=float)), prop_names)[0]
        self.prop_names = list(prop_names)
        self.row = int(row)
        # half-width = box * |value|; a value at ~0 falls back to box * the trajectory span on that axis (or 1)
        ref = np.where(ref_span > 1e-12, ref_span, 1.0)
        # denominators for relative (%) differences from the anchor: |value|, or the fallback for values at ~0
        self.denom = np.where(np.abs(self.anchor) > 1e-12, np.abs(self.anchor), ref)
        self.half = float(box) * self.denom

    def relative_pct(self, points: np.ndarray) -> np.ndarray:
        """Signed difference from the anchor in % of the anchor value, per axis."""
        return 100.0 * (np.atleast_2d(points) - self.anchor) / self.denom

    @classmethod
    def from_trajectory(cls, traj, box: float, iteration: int) -> "AnchorBox":
        pts = _traj_points(traj)
        row = min(int(iteration), len(pts) - 1)
        return cls(pts[row], traj.plan_names, box, pts.max(axis=0) - pts.min(axis=0), row)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        noise = (rng.random((n, len(self.anchor))) * 2.0 - 1.0) * self.half
        return _snap_counts(self.anchor + noise, self.prop_names)


def build_region(traj, al: dict, iteration: int):
    """Sampling region for kde_map candidates (al.kde_region: corridor | anchor)."""
    if al.get("kde_region", "corridor") == "anchor":
        return AnchorBox.from_trajectory(traj, al["anchor_box"], iteration)
    return Corridor.from_trajectory(traj, al["corridor_pad"], al["corridor_lookahead"], iteration)


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


def samples_per_extra(total: int, fraction: float) -> int:
    """Molecules requested at each additional target: round(fraction * num_samples), at least 1."""
    return max(int(round(float(fraction) * total)), 1)


def _row_target(traj, row: int, tid: int, n_samples: int, meta: dict) -> Target:
    axes = traj.plan_properties
    props = {p: float(axes[p][row]) for p in traj.plan_names}
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
    target = _traj_points(traj)[row]
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


def fit_density(state: ALState, prop_names, d: dict) -> Optional[PropertyDensity]:
    """Gaussian KDE of the labelled molecules' properties, or None without labelled data."""
    labelled = state.labelled_props()
    if not len(labelled):
        return None
    min_bw = np.array([1.0 if is_count_prop(p) else 0.0 for p in prop_names])
    return PropertyDensity(labelled, bandwidth=d["bandwidth"], min_bandwidth=min_bw, logp_floor=d["logp_floor"])


def score_targets(cand: np.ndarray, state: ALState, prop_names, d: dict, min_neff: float,
                  dens: Optional[PropertyDensity] = None) -> tuple:
    """(score, excluded mask, scale, parts) for candidate targets from data density and generator response."""
    if dens is None:
        dens = fit_density(state, prop_names, d)
    if dens is not None:
        # sparsity: -log p(y) of the labelled molecules' properties; high where data is scarce
        sparsity = -dens.logp(cand)
        center, scale = dens.center, dens.scale
    else:
        sparsity = np.zeros(len(cand))
        center = cand.mean(axis=0)
        scale = np.where(cand.std(axis=0) > 1e-12, cand.std(axis=0), 1.0)

    rec = state.record_mask()
    cols = [j for j, p in enumerate(prop_names) if not is_count_prop(p)]
    if rec.any():
        # generator response: past error and hit rate, smoothed over requested targets
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


def band_eligible(density: np.ndarray, band: tuple, excluded: np.ndarray, needed: int) -> tuple:
    """(eligible mask, in-band mask): candidates inside the density band, topped up with the closest ones if too few."""
    lo, hi = band
    in_band = (density >= lo) & (density <= hi)
    eligible = in_band & ~excluded
    short = needed - int(eligible.sum())
    if short > 0:
        # top up with a pool of the candidates closest to the band in log density (the pool is larger than
        # the shortfall so the greedy repulsion can still spread the picks out)
        pool = max(10 * short, 50)
        print(f"  Only {int(eligible.sum())} candidate targets lie in the density band [{lo:.3e}, {hi:.3e}]; "
              f"choosing from the {pool} closest to it (widen anchor_box or the band, or raise n_candidates).")
        logd = np.log(density + 1e-300)
        dist = np.maximum(np.log(lo + 1e-300) - logd, 0.0) + np.maximum(logd - np.log(hi + 1e-300), 0.0)
        dist[eligible | excluded] = np.inf
        extra = np.argsort(dist)[:pool]
        eligible[extra[np.isfinite(dist[extra])]] = True
    return eligible, in_band


def initial_props(state: ALState, reference_hdf5=None, prop_names=None) -> np.ndarray:
    """Labelled properties of the initial dataset: the input HDF5 if given, else the non-AL groups of `state`."""
    if reference_hdf5 is not None and Path(reference_hdf5).exists():
        return load_state(reference_hdf5, prop_names or state.prop_names).labelled_props()
    base = np.array([not str(k).startswith(AL_GROUP_PREFIX) for k in state.keys], dtype=bool)
    return state.props[base & state.labelled_mask()]


class DensityReference:
    """Fixed KDE of the initial dataset, its molecules' own (leave-one-out) densities, and the optional density band.

    All of it depends only on the initial dataset, so it is the same for every iteration; the reference densities
    are computed once and cached next to the AL state file (density_reference.npy).
    """

    def __init__(self, dens: PropertyDensity, ref_sorted: np.ndarray, band: Optional[tuple]):
        self.dens, self.ref_sorted, self.band, self.n = dens, ref_sorted, band, len(ref_sorted)

    def percentile(self, density: np.ndarray) -> np.ndarray:
        """Percentile rank (0-100) of each density among the initial dataset's own densities."""
        return 100.0 * np.searchsorted(self.ref_sorted, np.asarray(density, dtype=float), side="right") / self.n

    @classmethod
    def build(cls, props: np.ndarray, prop_names, d: dict, state_file=None, source: str = "", log=print):
        if not len(props):
            return None
        min_bw = np.array([1.0 if is_count_prop(p) else 0.0 for p in prop_names])
        dens = PropertyDensity(props, bandwidth=d["bandwidth"], min_bandwidth=min_bw, logp_floor=d["logp_floor"])
        key = {"source": str(source), "n": int(len(dens.Z)), "prop_names": list(prop_names),
               "bandwidth": d["bandwidth"]}
        saved = read_al_state_file(state_file) if state_file else {}
        npy = Path(state_file).with_name("density_reference.npy") if state_file else None
        ref = None
        if npy is not None and saved.get("density_reference", {}).get("key") == key and npy.exists():
            ref = np.load(npy)
            if len(ref) != key["n"]:
                ref = None
        if ref is None:
            log(f"  Computing the reference density of the initial dataset ({key['n']} molecules) ...")
            ref = dens.reference_density()
            if npy is not None:
                np.save(npy, ref)
                saved["density_reference"] = {"key": key, "file": npy.name}
                write_al_state_file(state_file, saved)
        else:
            log(f"  Reference density of the initial dataset ({key['n']} molecules) loaded from {npy.name}")
        ref_sorted = np.sort(ref)
        band = None
        pct = d.get("density_percentiles")
        if pct is not None:
            lo, hi = np.percentile(ref_sorted, [float(pct[0]), float(pct[1])])
            band = (float(lo), float(hi))
            log(f"  Density band: percentiles {[float(p) for p in pct]} of the initial dataset -> [{lo:.3e}, {hi:.3e}]")
        return cls(dens, ref_sorted, band)


def relative_scores(region: "AnchorBox", cand: np.ndarray, density: np.ndarray,
                    reference: Optional[DensityReference]) -> dict:
    """anchor_score = relative_pct: per candidate, the sum over the conditioning properties of
    100 * |y - anchor| / |anchor|, plus the relative error (%) of its KDE density with respect to the anchor's,
    100 * |p(y) - p(anchor)| / p(anchor), both under the fixed KDE of the initial dataset."""
    names = region.prop_names
    rel = region.relative_pct(cand)                                   # signed %, all axes (incl. atNUM)
    cond = [j for j, p in enumerate(names) if not is_count_prop(p)]
    err_sum = np.abs(rel[:, cond]).sum(axis=1)
    if reference is not None:
        anchor_dens = float(reference.dens.density(region.anchor[None, :])[0])
        dens_pct = 100.0 * np.abs(density - anchor_dens) / max(anchor_dens, 1e-300)
    else:
        anchor_dens, dens_pct = float("nan"), np.zeros(len(cand))
    count_axes = [j for j, p in enumerate(names) if is_count_prop(p)]
    d_atnum = cand[:, count_axes[0]] - region.anchor[count_axes[0]] if count_axes else np.zeros(len(cand))
    return {"rel_pct": rel, "err_sum": err_sum, "dens_pct": dens_pct, "score": err_sum + dens_pct, "d_atnum": d_atnum,
            "anchor_density": anchor_dens}


def farthest_point_pick(vectors: np.ndarray, score: np.ndarray, eligible: np.ndarray, n_pick: int,
                        fixed: list) -> tuple:
    """Pick n_pick eligible rows: the best score first, then each time the row farthest (Euclidean, in `vectors`)
    from everything already chosen, including the `fixed` rows (the anchor). Returns (picked, min distance at pick)."""
    picked, dist_at = [], []
    mind = np.full(len(vectors), np.inf)
    for i in fixed:
        mind = np.minimum(mind, np.linalg.norm(vectors - vectors[i], axis=1))
    avail = eligible.copy()
    avail[list(fixed)] = False
    for _ in range(n_pick):
        if not avail.any():
            break
        if not picked:
            i = int(np.argmax(np.where(avail, score, -np.inf)))
        else:
            # farthest from the chosen set; equal distances go to the higher score
            key = np.where(avail, mind, -np.inf)
            best = np.flatnonzero(key >= key.max() - 1e-9)
            i = int(best[np.argmax(score[best])])
        picked.append(i)
        dist_at.append(float(mind[i]) if np.isfinite(mind[i]) else float("nan"))
        avail[i] = False
        mind = np.minimum(mind, np.linalg.norm(vectors - vectors[i], axis=1))
    return picked, dist_at


def write_candidate_report(path, region, cand, rs: dict, density, in_band, eligible, picked: list) -> None:
    """CSV with the score components of every candidate (selection order in `selected`, 0 = not selected)."""
    import csv
    order = {i: r + 1 for r, i in enumerate(picked)}
    names = region.prop_names
    with Path(path).open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["candidate", *names, "d_atnum", *[f"relpct_{p}" for p in names], "rel_err_pct_sum",
                    "density", "density_rel_err_pct", "score", "in_band", "eligible", "selected"])
        for i in range(len(cand)):
            w.writerow([i, *[f"{v:.6g}" for v in cand[i]], f"{rs['d_atnum'][i]:g}",
                        *[f"{v:.4f}" for v in rs["rel_pct"][i]], f"{rs['err_sum'][i]:.4f}", f"{density[i]:.6e}",
                        f"{rs['dens_pct'][i]:.4f}", f"{rs['score'][i]:.4f}", int(in_band[i]), int(eligible[i]),
                        order.get(i, 0)])


def plan_kde_map(traj, state: ALState, al: dict, iteration: int, total: int, rng: np.random.Generator,
                 reference: Optional[DensityReference] = None, report_dir=None) -> list:
    region = build_region(traj, al, iteration)
    anchored = isinstance(region, AnchorBox)
    d = al["density"]
    n_cand, n_extra, include_anchor = kde_counts(al)
    cand = region.sample(n_cand, rng)
    if include_anchor:
        cand = np.vstack([region.anchor[None, :], cand])   # row 0 = the trajectory row itself

    dens = fit_density(state, region.prop_names, d)
    score, excluded, scale, parts = score_targets(cand, state, region.prop_names, d, al["min_neff"], dens)
    # with anchor_extra_fraction each target gets its own samples on top of num_samples, so no budget cap applies
    fixed = include_anchor and al.get("anchor_extra_fraction") is not None
    n_extra = min(n_extra, n_cand) if fixed else min(n_extra, n_cand, total - int(include_anchor))
    k = n_extra + int(include_anchor)

    relative = anchored and al.get("anchor_score", "kde") == "relative_pct"
    if relative:
        # only candidates whose atom count differs from the anchor's by at least anchor_min_datnum
        j_at = region.prop_names.index(NUM_ATOMS_PROP)
        d_at = np.abs(cand[:, j_at] - region.anchor[j_at])
        min_d = float(al.get("anchor_min_datnum", 1))
        excluded = excluded | (d_at < min_d)
        if region.half[j_at] + 0.5 <= min_d:
            print(f"  anchor_box allows atNUM {region.anchor[j_at]:g} +/- {region.half[j_at]:.2f}, so no candidate can "
                  f"differ by >= {min_d:g} atoms; widen anchor_box.")

    # density band: candidates must be as sparse, under the fixed KDE of the initial dataset, as the given
    # percentile range of the initial dataset's own molecules
    density = reference.dens.density(cand) if reference is not None else np.full(len(cand), np.nan)
    band = None
    eligible = ~excluded
    in_band = np.ones(len(cand), dtype=bool)
    if reference is not None and reference.band is not None:
        band = reference.band
        eligible, in_band = band_eligible(density, band, excluded, n_extra)
    elif d.get("density_percentiles") is not None:
        print("  No initial-dataset molecules with properties; the density band is not applied.")

    chosen = [0] if include_anchor else []
    rs, dist_at = None, {}
    if relative:
        # score = sum of relative % errors to the anchor + relative % error of the KDE density to the anchor's; then spread the picks out in
        # the space of relative % differences from the anchor (farthest-point selection)
        if reference is None:
            print("  No initial-dataset reference; the density term is 0.")
        rs = relative_scores(region, cand, density, reference)
        picked, dists = farthest_point_pick(rs["rel_pct"], rs["score"], eligible, n_extra, fixed=list(chosen))
        dist_at = dict(zip(picked, dists))
        chosen += picked
        if len(picked) < n_extra:
            print(f"  Only {len(picked)} eligible candidate targets; planning {len(picked)} instead of {n_extra}.")
        score = rs["score"]
        if report_dir is not None:
            write_candidate_report(Path(report_dir) / "anchor_candidates.csv", region, cand, rs, density,
                                   in_band, eligible, picked)

    # greedy picks; each pick suppresses nearby candidates so the targets spread out
    mask = np.ones(len(cand))
    if include_anchor and not relative:
        mask *= repulsion(cand[0], cand, scale)
    while not relative and len(chosen) < k:
        adj = np.where(eligible, score * mask, -np.inf)
        adj[chosen] = -np.inf
        if not np.isfinite(adj).any():
            print(f"  Only {len(chosen)} eligible candidate targets; planning {len(chosen)} instead of {k}.")
            break
        i = int(np.argmax(adj))
        chosen.append(i)
        mask *= repulsion(cand[i], cand, scale)

    off = int(include_anchor)   # candidate rows from `off` on were sampled in the region
    n_band = int(in_band[off:].sum()) if band is not None else n_cand
    print(f"  Candidates: {n_cand} sampled in the {'anchor box' if anchored else 'corridor'}, {n_band} in the density "
          f"band, {int(eligible[off:].sum())} eligible; selected {len(chosen) - off} additional targets"
          f"{' + the anchor row' if include_anchor else ''}.")
    weights = np.ones(len(chosen)) if relative else np.exp(score[chosen] - score[chosen].max())
    if fixed:
        # the anchor keeps num_samples; each additional target gets anchor_extra_fraction * num_samples on top
        per = samples_per_extra(total, al["anchor_extra_fraction"])
        shares = np.array([total] + [per] * (len(chosen) - 1))
        print(f"  Samples: anchor {total}, {len(chosen) - 1} additional targets x {per} "
              f"({float(al['anchor_extra_fraction']):g} x {total}); {int(shares.sum())} in total")
    else:
        shares = allocate(total, weights)
    if relative:
        print(f"  Additional targets (relative_pct score = sum of |rel. error| % + |rel. density error| %; "
              f"anchor density {rs['anchor_density']:.3e}):")
        for r, i in enumerate(chosen[int(include_anchor):]):
            print(f"    {r + 1}: {dict(zip(region.prop_names, np.round(cand[i], 4).tolist()))}  dAtNUM={rs['d_atnum'][i]:+g}  "
                  f"err%={rs['err_sum'][i]:.2f}  dens_err%={rs['dens_pct'][i]:.2f}  score={rs['score'][i]:.2f}  "
                  f"min dist%={dist_at.get(i, float('nan')):.2f}")
    targets = []
    for tid, (i, n) in enumerate(zip(chosen, shares)):
        meta = {"strategy": "kde_map", "region": "anchor" if anchored else "corridor",
                "score": float(score[i]), "sparsity": float(parts["sparsity"][i]), "density": float(density[i]),
                "err": float(parts["err"][i]), "n_eff": float(parts["n_eff"][i]), "reach": float(parts["reach"][i])}
        if anchored:
            meta.update(anchor_row=region.row, is_anchor=bool(include_anchor and i == 0))
        if band is not None:
            meta.update(in_band=bool(in_band[i]), band=[band[0], band[1]])
        if rs is not None:
            meta.update(score_mode="relative_pct", rel_err_pct_sum=float(rs["err_sum"][i]),
                        density_rel_err_pct=float(rs["dens_pct"][i]), d_atnum=float(rs["d_atnum"][i]),
                        min_dist_pct=dist_at.get(i))
        targets.append(Target(tid, {p: float(cand[i, j]) for j, p in enumerate(region.prop_names)}, int(n), meta))
    return targets


def plan_targets(traj, state: ALState, al: dict, iteration: int, total: int, seed: int, state_file=None,
                 reference_hdf5=None, report_dir=None) -> list:
    """Targets for one iteration. Deterministic given seed, iteration and state."""
    strategy = al["query_strategy"]
    if strategy == "trajectory":
        return plan_trajectory(traj, iteration, total)
    if strategy == "frontier":
        return plan_frontier(traj, state, al, iteration, total, state_file)
    if strategy != "kde_map":
        raise ValueError(f"Unknown planner: {strategy!r}")
    anchored = al.get("kde_region", "corridor") == "anchor"
    # corridor mode needs generator feedback; anchor mode only needs labelled data for the density band
    ready = state.labelled_mask().any() if anchored else state.record_mask().any()
    if not ready:
        print("  No labelled AL molecules yet; kde_map uses the trajectory row this iteration.")
        return plan_trajectory(traj, iteration, total)
    rng = np.random.default_rng([int(seed), int(iteration)])
    reference = None
    relative = anchored and al.get("anchor_score", "kde") == "relative_pct"
    if al["density"].get("density_percentiles") is not None or relative:
        src = reference_hdf5 if reference_hdf5 is not None and Path(reference_hdf5).exists() else "output HDF5 (non-AL groups)"
        reference = DensityReference.build(initial_props(state, reference_hdf5, traj.plan_names), traj.plan_names,
                                           al["density"], state_file, source=src)
    return plan_kde_map(traj, state, al, iteration, total, rng, reference, report_dir)


def run_plan(*, traj, al: dict, hdf5_path, iter_dir, iteration: int, total_samples: int, seed: int,
             state_file, reference_hdf5=None, log=print) -> list:
    """Write <iter_dir>/targets.json (an existing plan is reused) and return the targets.

    reference_hdf5: the initial dataset (DataManager.input_hdf5); it fixes the KDE density band for all iterations.
    """
    existing = read_targets(iter_dir)
    if existing is not None:
        log(f"  Reusing the plan in {Path(iter_dir) / 'targets.json'} (delete it to re-plan)")
        return existing[1]

    state = load_state(hdf5_path, traj.plan_names)
    log(f"  Loaded {int(state.labelled_mask().sum())} labelled molecules "
        f"({int(state.record_mask().sum())} with AL target records) from {hdf5_path}")
    targets = plan_targets(traj, state, al, iteration, total_samples, seed, state_file, reference_hdf5, iter_dir)
    write_targets(iter_dir, iteration, al["query_strategy"], traj.plan_names, targets)
    for t in targets:
        log(f"  Target {t.id}: {t.props} x {t.n_samples}")
    return targets
