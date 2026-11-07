from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from src.configs.config import tolerance_delta
from src.components.candidates import canonical_id, read_candidates_meta
from src.components.dataset_labels import read_json_props, read_xyz_symbols, valid_qm_result
from src.components.trajectory import read_targets


def is_hit(achieved: dict, requested: dict, tolerances, prop_names,
           dataset: dict | None = None) -> bool:
    """Return whether achieved properties fall within configured tolerances."""
    from src.components.dataset_labels import dataset_values

    num_atoms_prop = dataset_values(dataset)["num_atoms_property"]
    names = [name for name in prop_names if name != num_atoms_prop]
    if isinstance(tolerances, (int, float)):
        tolerances = {name: {"relative": float(tolerances), "absolute": 0.0} for name in names}
    target = np.array([requested[name] for name in names], dtype=float)
    delta = tolerance_delta(target, names, tolerances)
    return all(
        abs(float(achieved[name]) - float(requested[name])) <= allowed
        for name, allowed in zip(names, delta)
    )


def build_al_meta(query_dir, prop_dir, targets: list, cand_meta: dict, prop_names: list,
                  tolerances, iteration: int, dataset: dict | None = None) -> tuple:
    """Validate results and attach active-learning target/hit metadata."""
    by_id = {target.id: target for target in targets}
    metadata, rejected = {}, []
    for xyz in sorted(Path(query_dir).iterdir()):
        if not (xyz.is_file() and xyz.suffix == ".xyz"):
            continue
        cid = canonical_id(xyz.name)
        json_path = Path(prop_dir) / f"{xyz.stem}.json"
        ok, reason = valid_qm_result(json_path, xyz, prop_names, dataset=dataset)
        if not ok:
            rejected.append((xyz.name, reason))
            continue
        achieved = read_json_props(
            json_path, prop_names, num_atoms=len(read_xyz_symbols(xyz)), dataset=dataset
        )
        target = by_id.get(cand_meta.get(cid, {}).get("target_id", -1))
        if target is None:
            metadata[cid] = {
                "iteration": iteration,
                "target_id": -1,
                "requested": [math.nan] * len(prop_names),
                "hit": -1,
                "achieved": achieved,
            }
            continue
        metadata[cid] = {
            "iteration": iteration,
            "target_id": int(target.id),
            "requested": [float(target.props[name]) for name in prop_names],
            "hit": int(is_hit(achieved, target.props, tolerances, prop_names, dataset=dataset)),
            "achieved": achieved,
        }
    return metadata, rejected


def prepare_al_labels(*, query_dir, prop_dir, iter_dir, prop_names: list, iteration: int,
                      tolerances=None, threshold: float | None = None,
                      keep_off_target: bool, dataset: dict | None = None, log=print) -> dict:
    """Validate QM results and build the AL metadata for to_hdf5.

    Molecules without a valid QM result (and, if keep_off_target is false, misses) are moved to
    `<iter_dir>/rejected`. Returns the metadata keyed by canonical ID.
    """
    query_dir, iter_dir = Path(query_dir), Path(iter_dir)
    plan = read_targets(iter_dir)
    targets = plan[1] if plan else []
    if tolerances is None:
        tolerances = {p: {"relative": 0.1 if threshold is None else threshold, "absolute": 0.0}
                      for p in prop_names if p != "atNUM"}
    al_meta, rejected = build_al_meta(query_dir, prop_dir, targets, read_candidates_meta(iter_dir),
                                      prop_names, tolerances, iteration, dataset=dataset)
    drop = {name: reason for name, reason in rejected}
    if not keep_off_target:
        for cid, m in al_meta.items():
            if m["hit"] != 1:
                drop[cid] = "off target"
        al_meta = {cid: m for cid, m in al_meta.items() if m["hit"] == 1}

    if drop:
        out = iter_dir / "rejected"
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(query_dir.iterdir()):
            if f.is_file() and f.suffix == ".xyz" and (f.name in drop or canonical_id(f.name) in drop):
                shutil.move(str(f), str(out / f.name))
        log(f"  {len(drop)} molecules kept out of the HDF5 file (see {out})")
    return al_meta
