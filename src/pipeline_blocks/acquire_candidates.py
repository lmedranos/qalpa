from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from src.components.dataset_labels import NUM_ATOMS_PROP
from src.components.candidates import canonical_id, read_candidates_meta
from src.components.trajectory import read_targets
from src.components.dataset_labels import read_xyz
from src.pipeline_blocks.plan_targets import allocate


def tanimoto_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(len(A), len(B)) Tanimoto similarity between boolean fingerprint matrices."""
    return _tanimoto_f32(A.astype(np.float32), B.astype(np.float32))


def _tanimoto_f32(Af: np.ndarray, Bf: np.ndarray) -> np.ndarray:
    inter = Af @ Bf.T
    union = Af.sum(1)[:, None] + Bf.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def max_tanimoto(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Highest similarity of each row of A to any row of B (0 if B is empty)."""
    if B is None or len(B) == 0:
        return np.zeros(len(A))
    Af, Bf = A.astype(np.float32), B.astype(np.float32)
    return np.concatenate([_tanimoto_f32(Af[s:s + chunk], Bf).max(axis=1) for s in range(0, len(Af), chunk)])


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, dtype=float) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def select_batch(*, fps, existing_fps, n_heavy, budget: int, weights: dict,
                 mode: str = "score", rng=None) -> tuple:
    """Pick `budget` candidates for one target.

    Returns (selected indices into the inputs, diagnostics dict of per-candidate arrays; NaN for dropped rows).
    """
    n = len(fps)
    idx = np.arange(n)
    budget = int(min(budget, len(idx)))
    nan = np.full(n, np.nan)
    diag = {k: nan.copy() for k in ("cost", "div", "score")}
    selected = np.zeros(n, dtype=bool)
    if budget == 0:
        return np.array([], dtype=int), dict(diag, selected=selected)

    rng = rng if rng is not None else np.random.default_rng(0)
    if mode == "random":
        chosen = rng.choice(idx, size=budget, replace=False)
        selected[chosen] = True
        return np.sort(chosen), dict(diag, selected=selected)

    f = fps[idx]
    cost = n_heavy[idx] / max(float(np.max(n_heavy[idx])), 1.0)
    mindist = 1.0 - max_tanimoto(f, existing_fps)
    div0 = mindist.copy()

    def total(nov, div):
        return weights["w_div"] * div - weights["w_cost"] * cost

    init_score = total(div0, div0)
    chosen_local, pick_scores = [], []
    free = np.ones(len(idx), dtype=bool)
    for _ in range(budget):
        score = np.where(free, total(mindist, mindist), -np.inf)
        i = int(np.argmax(score))
        chosen_local.append(i)
        pick_scores.append(score[i])
        free[i] = False
        mindist = np.minimum(mindist, 1.0 - tanimoto_similarity(f, f[i:i + 1])[:, 0])

    for name, arr in (("cost", cost), ("div", div0), ("score", init_score)):
        diag[name][idx] = arr
    diag["score"][idx[chosen_local]] = pick_scores
    chosen = idx[chosen_local]
    selected[chosen] = True
    return np.sort(chosen), dict(diag, selected=selected)


ACQUISITION_FILE = "acquisition.csv"

ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi"
).split()


def _atoms_to_xyz_block(symbols, positions) -> str:
    lines = [str(len(symbols)), ""]
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, positions)]
    return "\n".join(lines) + "\n"


def atoms_to_smiles(symbols, positions) -> Optional[str]:
    """Canonical SMILES from coordinates via Open Babel bond perception."""
    try:
        from openbabel import pybel
        from rdkit import Chem

        mol = pybel.readstring("xyz", _atoms_to_xyz_block(symbols, positions))
        mol.OBMol.ConnectTheDots()
        mol.OBMol.PerceiveBondOrders()
        smi = mol.write("smi").split()[0]
        rd = Chem.MolFromSmiles(smi)
        if rd is None or "*" in smi:
            return None
        return Chem.MolToSmiles(rd)
    except Exception:
        return None


def smiles_to_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=bool)
    except Exception:
        return None


class MolFeaturizer:
    """SMILES and Morgan fingerprints cached by molecule key."""

    def __init__(self, cache_path=None, n_bits: int = 2048, radius: int = 2):
        self.cache_path = Path(cache_path) if cache_path else None
        self.n_bits, self.radius = n_bits, radius
        self._cache: dict = {}
        self._dirty = False
        if self.cache_path is not None and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=True)
            fps = np.unpackbits(data["fps"], axis=1)[:, : self.n_bits].astype(bool)
            for k, s, fp, ok in zip(data["keys"], data["smiles"], fps, data["ok"]):
                self._cache[str(k)] = (str(s), fp) if ok else (None, None)

    def features(self, key: str, atoms_fn: Callable, use_cache: bool = True) -> tuple:
        if use_cache and key in self._cache:
            return self._cache[key]
        symbols, positions = atoms_fn()
        smi = atoms_to_smiles(symbols, positions)
        fp = smiles_to_fp(smi, self.n_bits, self.radius) if smi else None
        result = (smi, fp) if fp is not None else (None, None)
        if use_cache:
            self._cache[key] = result
            self._dirty = True
        return result

    def save(self) -> None:
        if self.cache_path is None or not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        ok = np.array([self._cache[k][1] is not None for k in keys], dtype=bool)
        fps = np.zeros((len(keys), self.n_bits), dtype=bool)
        for i, k in enumerate(keys):
            if ok[i]:
                fps[i] = self._cache[k][1]
        smiles = np.array([self._cache[k][0] or "" for k in keys], dtype=object)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, keys=np.array(keys, dtype=object), smiles=smiles, ok=ok,
                            fps=np.packbits(fps, axis=1))
        tmp.replace(self.cache_path)
        self._dirty = False


def hdf5_atoms(h5, key: str) -> tuple:
    """Return symbols and positions for an HDF5 molecule entry."""
    grp = h5[key]
    z = np.asarray(grp["atNUM"]).astype(int)
    return [ELEMENT_SYMBOLS[int(i) - 1] for i in z], np.asarray(grp["atXYZ"], dtype=float).tolist()


def _shares(budget: int, targets: list, available: dict) -> dict:
    """Budget share per target id, proportional to the planned samples, limited to targets with candidates."""
    live = [t for t in targets if available.get(t.id, 0) > 0]
    if not live:
        return {}
    minimum = 1 if budget >= len(live) else 0
    return dict(zip([t.id for t in live], allocate(budget, [t.n_samples for t in live], minimum)))


def run_acquire(*, al: dict, prop_names: list, query_dir, iter_dir, hdf5_path, cache_dir,
                tolerances: dict | None = None, threshold: float | None = None,
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

    sp_names = [p for p in prop_names if p != NUM_ATOMS_PROP]
    diag = {k: np.full(n, np.nan) for k in ("cost", "div", "score")}
    selected = np.zeros(n, dtype=bool)
    summary = {"n_candidates": n}

    available = {t.id: int(np.sum((tids == t.id))) for t in targets}
    shares = _shares(int(al["budget"]), targets, available)

    if acq["mode"] == "random":
        for tid, share in shares.items():
            idx = np.flatnonzero((tids == tid) & (status == "ok"))
            pick = rng.choice(idx, size=min(share, len(idx)), replace=False)
            selected[pick] = True
    else:
        featurizer = featurizer or MolFeaturizer(Path(cache_dir) / "fingerprints.npz")
        fps = np.zeros((n, featurizer.n_bits), dtype=bool)
        n_heavy = np.zeros(n)
        seen = set()
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

        for tid, share in shares.items():
            sel = np.flatnonzero(tids == tid)
            chosen, d = select_batch(
                fps=fps[sel], existing_fps=np.zeros((0, featurizer.n_bits)),
                n_heavy=n_heavy[sel], budget=share, weights=acq, mode="score", rng=rng
            )
            for k, arr in diag.items():
                arr[sel] = d[k]
            selected[sel[chosen]] = True

    table = pd.DataFrame({"name": [f.name for f in files], "target_id": tids, "status": status})
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
        f"(invalid {summary['n_invalid']}, duplicate {summary['n_duplicate']})")
    return summary
