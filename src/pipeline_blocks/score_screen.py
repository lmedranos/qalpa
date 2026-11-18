"""score_filter with reference values from the initial dataset: QED and SA score screening.

query.score_reference: initial_mean
    A generated molecule passes if  QED > mean QED  AND  SA score < mean SA score  of the initial dataset.
    Both scores are computed the same way for the initial dataset and the generated molecules: geometry ->
    OpenBabel (connectivity + bond orders) -> SMILES -> RDKit (QED.qed, Contrib sascorer). The reference means
    are computed once and cached next to the input HDF5 (<input stem>_score_reference.json).
    query.score_relax: {QED: r_qed, SAScore: r_sa} shifts the thresholds by a fraction of the mean, in the
    direction that lets more molecules pass (negative = stricter):
        QED threshold = mean QED * (1 - r_qed)      (higher QED is better -> lower bar)
        SA  threshold = mean SA  * (1 + r_sa)       (lower SA is better  -> higher ceiling)
query.score_reference: trajectory (default)
    The original per-iteration thresholds from the trajectory CSV (MolecularFilter.score_filter).
"""

from __future__ import annotations

import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REFERENCE_INITIAL = "initial_mean"
REFERENCE_TRAJECTORY = "trajectory"
_ELEMENTS = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def score_reference_mode(cfg) -> str:
    mode = str(cfg.query.get("score_reference", REFERENCE_TRAJECTORY))
    if mode not in (REFERENCE_INITIAL, REFERENCE_TRAJECTORY):
        raise ValueError(f"query.score_reference must be '{REFERENCE_INITIAL}' or '{REFERENCE_TRAJECTORY}', got {mode!r}")
    return mode


def trajectory_score_names(cfg) -> list:
    """Score columns to read from the trajectory CSV: none when the reference comes from the initial dataset."""
    return [] if score_reference_mode(cfg) == REFERENCE_INITIAL else list(cfg.query.scores)


def relaxed_thresholds(reference: dict, relax=None) -> dict:
    """QED and SA thresholds from the initial-dataset means and query.score_relax (see the module docstring)."""
    relax = dict(relax or {})
    unknown = set(relax) - {"QED", "SAScore"}
    if unknown:
        raise ValueError(f"query.score_relax accepts QED and SAScore, got {sorted(unknown)}")
    r_qed, r_sa = float(relax.get("QED", 0.0) or 0.0), float(relax.get("SAScore", 0.0) or 0.0)
    for name, r in (("QED", r_qed), ("SAScore", r_sa)):
        if not -1.0 < r < 1.0:
            raise ValueError(f"query.score_relax.{name} must be a fraction between -1 and 1, got {r}")
    return {"qed_min": float(reference["qed_mean"]) * (1.0 - r_qed), "sa_max": float(reference["sa_mean"]) * (1.0 + r_sa),
            "relax": {"QED": r_qed, "SAScore": r_sa}}


def smiles_from_xyz_text(xyz_text: str):
    """Same conversion as calculate_sascore.xyz_to_smiles, from xyz text instead of a file."""
    from openbabel import pybel
    try:
        mol = pybel.readstring("xyz", xyz_text)
        mol.OBMol.ConnectTheDots()
        mol.OBMol.PerceiveBondOrders()
        mol.addh()
        smiles = mol.write("smi").split()
        return smiles[0] if smiles else None
    except Exception:
        return None


def score_smiles(smiles):
    """(SA score, QED) of a SMILES string, or (None, None) if RDKit cannot parse it."""
    if not smiles:
        return None, None
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED
    from src.pipeline_blocks.calculate_sascore import sascorer
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return float(sascorer.calculateScore(mol)), float(QED.qed(mol))
    except Exception:
        return None, None


def _score_xyz_text(xyz_text: str) -> tuple:
    smiles = smiles_from_xyz_text(xyz_text)
    sa, qed = score_smiles(smiles)
    return smiles, sa, qed


def _xyz_text(z, xyz) -> str:
    lines = [str(len(z)), ""]
    for a, (x, y, w) in zip(np.asarray(z, dtype=int), np.asarray(xyz, dtype=float)):
        lines.append(f"{_ELEMENTS.get(int(a), str(int(a)))} {x:.6f} {y:.6f} {w:.6f}")
    return "\n".join(lines) + "\n"


def _workers() -> int:
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))


def _map(texts: list) -> list:
    if len(texts) < 200 or _workers() == 1:
        return [_score_xyz_text(t) for t in texts]
    with Pool(min(_workers(), 32)) as pool:
        return pool.map(_score_xyz_text, texts, chunksize=64)


def initial_reference(input_hdf5, log=print) -> dict:
    """Mean SA score and QED over the initial dataset (all structures with a valid SMILES); cached per file."""
    import h5py
    path = Path(input_hdf5)
    if not path.exists():
        raise FileNotFoundError(f"Initial dataset {path} not found; query.score_reference: initial_mean needs it.")
    stat = path.stat()
    key = {"file": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache = path.with_name(f"{path.stem}_score_reference.json")
    if cache.exists():
        try:
            saved = json.loads(cache.read_text())
            if saved.get("key") == key:
                log(f"  Score reference loaded from {cache.name}: mean SA {saved['sa_mean']:.4f}, "
                    f"mean QED {saved['qed_mean']:.4f} ({saved['n_valid']} of {saved['n_total']} structures)")
                return saved
        except (ValueError, KeyError):
            pass

    texts = []
    with h5py.File(path, "r") as h5:
        for top in h5.values():
            for grp in top.values():
                if isinstance(grp, h5py.Group) and "atNUM" in grp and "atXYZ" in grp:
                    texts.append(_xyz_text(grp["atNUM"][()], grp["atXYZ"][()]))
    log(f"  Computing SA score and QED of the initial dataset ({len(texts)} structures, {_workers()} CPUs) ...")
    results = _map(texts)
    sa = np.array([r[1] for r in results if r[1] is not None and r[2] is not None])
    qed = np.array([r[2] for r in results if r[1] is not None and r[2] is not None])
    if not len(sa):
        raise ValueError(f"No structure in {path} could be scored (SMILES conversion failed for all).")
    ref = {"key": key, "n_total": len(texts), "n_valid": int(len(sa)),
           "sa_mean": float(sa.mean()), "sa_std": float(sa.std()),
           "qed_mean": float(qed.mean()), "qed_std": float(qed.std())}
    cache.write_text(json.dumps(ref, indent=2))
    log(f"  Score reference: mean SA {ref['sa_mean']:.4f} (std {ref['sa_std']:.4f}), mean QED {ref['qed_mean']:.4f} "
        f"(std {ref['qed_std']:.4f}) from {ref['n_valid']} of {ref['n_total']} structures -> {cache.name}")
    return ref


def screen_query(query_dir, out_dir, reference: dict, relax=None, log=print) -> dict:
    """Keep molecules with QED > QED threshold and SA < SA threshold; move the others to <out_dir>/rejected.

    Thresholds: the initial-dataset means, shifted by `relax` (query.score_relax, see relaxed_thresholds).

    Molecules whose SMILES cannot be built or scored fail. Per-molecule scores: <out_dir>/scores.tsv.
    """
    query_dir, out_dir = Path(query_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in query_dir.iterdir() if f.is_file() and f.suffix == ".xyz")
    results = _map([f.read_text() for f in files])
    thr = relaxed_thresholds(reference, relax)
    sa_ref, qed_ref = thr["sa_max"], thr["qed_min"]

    rejected = out_dir / "rejected"
    counts = {"total": len(files), "passed": 0, "unscored": 0, "qed_fail": 0, "sa_fail": 0}
    with (out_dir / "scores.tsv").open("w") as f:
        f.write("molecule\tsmiles\tSAScore\tQED\tpassed\n")
        for path, (smiles, sa, qed) in zip(files, results):
            if sa is None or qed is None:
                ok = False
                counts["unscored"] += 1
            else:
                qed_ok, sa_ok = qed > qed_ref, sa < sa_ref
                counts["qed_fail"] += int(not qed_ok)
                counts["sa_fail"] += int(not sa_ok)
                ok = qed_ok and sa_ok
            counts["passed"] += int(ok)
            f.write(f"{path.name}\t{smiles}\t{sa}\t{qed}\t{int(ok)}\n")
            if not ok:
                rejected.mkdir(exist_ok=True)
                shutil.move(str(path), str(rejected / path.name))
    summary = {"reference": {"sa_mean": float(reference["sa_mean"]), "qed_mean": float(reference["qed_mean"]),
                             "n_valid": reference.get("n_valid")},
               "relax": thr["relax"], "thresholds": {"qed_min": qed_ref, "sa_max": sa_ref}, "counts": counts}
    (out_dir / "score_screen_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"  Score screening (QED > {qed_ref:.4f} = mean {float(reference['qed_mean']):.4f} x {1 - thr['relax']['QED']:g}, "
        f"SA < {sa_ref:.4f} = mean {float(reference['sa_mean']):.4f} x {1 + thr['relax']['SAScore']:g}): "
        f"{counts['passed']} of {counts['total']} "
        f"passed ({counts['qed_fail']} QED too low, {counts['sa_fail']} SA too high, {counts['unscored']} unscored)")
    return summary
