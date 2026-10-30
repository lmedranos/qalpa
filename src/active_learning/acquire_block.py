from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.active_learning.acquisition import select_batch
from src.active_learning.constants import NUM_ATOMS_PROP
from src.active_learning.density import PropertyDensity
from src.active_learning.features import MolFeaturizer, hdf5_atoms
from src.active_learning.ids import canonical_id, read_candidates_meta, read_targets
from src.active_learning.labels import read_xyz
from src.active_learning.query_planner import allocate
from src.active_learning.state import load_state
from src.active_learning.surrogate import build_surrogate

ACQUISITION_FILE = "acquisition.csv"


def _shares(budget: int, targets: list, available: dict) -> dict:
    """Budget share per target id, proportional to the planned samples, limited to targets with candidates."""
    live = [t for t in targets if available.get(t.id, 0) > 0]
    if not live:
        return {}
    minimum = 1 if budget >= len(live) else 0
    return dict(zip([t.id for t in live], allocate(budget, [t.n_samples for t in live], minimum)))


def run_acquire(*, al: dict, prop_names: list, query_dir, iter_dir, hdf5_path, cache_dir, threshold: float,
                seed: int, iteration: int, featurizer=None, log=print) -> dict:
    """Keep the best `al.budget` candidates in query_dir (others are deleted) and write acquisition.csv."""
    query_dir, iter_dir = Path(query_dir), Path(iter_dir)
    plan = read_targets(iter_dir)
    if plan is None:
        raise FileNotFoundError(f"No targets.json in {iter_dir}; run the plan and generate blocks first.")
    _, targets = plan
    by_id = {t.id: t for t in targets}
    meta = read_candidates_meta(iter_dir)
    acq = al["acquisition"]
    rng = np.random.default_rng([int(seed), int(iteration), 1])

    files = sorted(f for f in query_dir.iterdir() if f.is_file() and f.suffix == ".xyz")
    cids = [canonical_id(f.name) for f in files]
    tids = np.array([meta.get(c, {}).get("target_id", -1) for c in cids], dtype=int)
    n = len(files)
    status = np.where(np.isin(tids, list(by_id)), "ok", "no_target").astype(object)

    cols = [j for j, p in enumerate(prop_names) if p != NUM_ATOMS_PROP]
    sp_names = [prop_names[j] for j in cols]
    mu = np.full((n, len(cols)), np.nan)
    sigma = np.full((n, len(cols)), np.nan)
    diag = {k: np.full(n, np.nan) for k in ("p_hit", "unc", "nov", "cost", "div", "score")}
    selected = np.zeros(n, dtype=bool)
    summary = {"n_candidates": n, "cold_start": False, "n_train": 0}

    available = {t.id: int(np.sum((tids == t.id))) for t in targets}
    shares = _shares(int(al["budget"]), targets, available)

    if acq["mode"] == "random":
        for tid, share in shares.items():
            idx = np.flatnonzero((tids == tid) & (status == "ok"))
            pick = rng.choice(idx, size=min(share, len(idx)), replace=False)
            selected[pick] = True
    else:
        featurizer = featurizer or MolFeaturizer(Path(cache_dir) / "fingerprints.npz")
        state = load_state(hdf5_path, prop_names)
        lab = state.labelled_mask()
        ref_fps, ref_smiles, rows = [], set(), []
        if lab.any():
            import h5py

            with h5py.File(hdf5_path, "r") as h5:
                for key, y in zip(state.keys[lab], state.props[lab]):
                    smi, fp = featurizer.features(str(key), lambda k=str(key): hdf5_atoms(h5, k))
                    if fp is None:
                        continue
                    ref_smiles.add(smi)
                    ref_fps.append(fp)
                    rows.append(y[cols])
        n_bits = featurizer.n_bits
        ref_fps = np.array(ref_fps, dtype=bool).reshape(-1, n_bits) if len(ref_fps) else np.zeros((0, n_bits), dtype=bool)
        ref_y = np.array(rows, dtype=float).reshape(-1, len(cols))
        featurizer.save()

        fps = np.zeros((n, ref_fps.shape[1]), dtype=bool)
        n_heavy = np.zeros(n)
        seen = set(ref_smiles)
        for i, f in enumerate(files):
            if status[i] != "ok":
                continue
            symbols, pos = read_xyz(f)
            n_heavy[i] = sum(s != "H" for s in symbols)
            smi, fp = featurizer.features(f"cand/{cids[i]}", lambda s=symbols, p=pos: (s, p), use_cache=False)
            if fp is None:
                status[i] = "invalid"
            elif smi in seen:
                status[i] = "duplicate"
            else:
                seen.add(smi)
                fps[i] = fp
        ok = status == "ok"

        # Surrogate: cold start when there are too few labels
        sur = al["surrogate"]
        cold = len(ref_y) < int(sur["min_train"])
        summary.update(cold_start=bool(cold), n_train=int(len(ref_y)))
        if len(ref_y):
            dens = PropertyDensity(ref_y, bandwidth=al["density"]["bandwidth"], logp_floor=al["density"]["logp_floor"])
            prop_scale = dens.scale
        else:
            dens, prop_scale = None, np.ones(len(cols))
        req = np.array([[by_id[t].props[p] for p in sp_names] if t in by_id else [np.nan] * len(cols) for t in tids])
        if cold:
            mu[ok], sigma[ok] = req[ok], np.tile(prop_scale, (int(ok.sum()), 1))
        else:
            take = np.arange(len(ref_y))
            if len(take) > int(sur["max_train"]):
                take = rng.choice(take, size=int(sur["max_train"]), replace=False)
            model = build_surrogate(sur["backend"], sur["n_members"], sur["n_trees"], seed).fit(ref_fps[take], ref_y[take])
            if ok.any():
                mu[ok], sigma[ok] = model.predict(fps[ok])
        if dens is None:
            dens = PropertyDensity(mu[ok] if ok.any() else np.zeros((1, len(cols))))

        ref_cap = int(acq["novelty_ref_max"])
        existing = ref_fps if len(ref_fps) <= ref_cap else ref_fps[rng.choice(len(ref_fps), ref_cap, replace=False)]
        for tid, share in shares.items():
            sel = np.flatnonzero(tids == tid)
            t = by_id[tid]
            target = np.array([t.props[p] for p in sp_names], dtype=float)
            delta = threshold * np.abs(target)
            m = np.where(ok[sel, None], mu[sel], np.nan)
            chosen, d = select_batch(mu=m, sigma=sigma[sel], target=target, delta=delta, fps=fps[sel],
                                     existing_fps=existing, n_heavy=n_heavy[sel], budget=share, weights=acq,
                                     density=dens, prop_scale=prop_scale, mode="score", rng=rng)
            for k, arr in diag.items():
                arr[sel] = d[k]
            selected[sel[chosen]] = True
            if len(chosen):
                dens.add_points(mu[sel[chosen]])
                existing = np.vstack([existing, fps[sel[chosen]]])

    table = pd.DataFrame({"name": [f.name for f in files], "target_id": tids, "status": status})
    for j, p in enumerate(sp_names):
        table[f"mu_{p}"], table[f"sigma_{p}"] = mu[:, j], sigma[:, j]
    for k, arr in diag.items():
        table[k] = arr
    table["selected"] = selected
    table.to_csv(iter_dir / ACQUISITION_FILE, index=False)

    for f, keep in zip(files, selected):
        if not keep:
            f.unlink()
    summary.update(n_selected=int(selected.sum()), n_invalid=int(np.sum(status == "invalid")),
                   n_duplicate=int(np.sum(status == "duplicate")))
    log(f"  Acquisition kept {summary['n_selected']} of {n} candidates "
        f"(invalid {summary['n_invalid']}, duplicate {summary['n_duplicate']}, cold start {summary['cold_start']})")
    return summary
