from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from src.active_learning.constants import (
    NUM_ATOMS_PROP,
    PROP_JSON_KEYS,
    REQUIRED_JSON_KEYS,
    SUPPORTED_SYMBOLS,
    get_json_value,
    is_finite_number,
)
from src.active_learning.config import tolerance_delta
from src.active_learning.ids import canonical_id


def read_xyz_symbols(path) -> list:
    """Element symbols of an xyz file (first line is the atom count, second a comment)."""
    with Path(path).open("r") as f:
        n = int(f.readline().split()[0])
        f.readline()
        return [f.readline().split()[0] for _ in range(n)]


def read_xyz(path) -> tuple:
    """(symbols, positions as list of [x, y, z]) from an xyz file."""
    with Path(path).open("r") as f:
        n = int(f.readline().split()[0])
        f.readline()
        symbols, pos = [], []
        for _ in range(n):
            parts = f.readline().split()
            symbols.append(parts[0])
            pos.append([float(x) for x in parts[1:4]])
    return symbols, pos


def read_json_props(json_path, prop_names, num_atoms: Optional[int] = None) -> Optional[dict]:
    """Achieved property values from a QM JSON; None if the file is missing or any value is absent/non-finite."""
    jp = Path(json_path)
    if not jp.exists():
        return None
    try:
        with jp.open("r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    out = {}
    for p in prop_names:
        if p == NUM_ATOMS_PROP:
            if num_atoms is None:
                return None
            out[p] = float(num_atoms)
            continue
        value = get_json_value(data, PROP_JSON_KEYS[p]) if p in PROP_JSON_KEYS else None
        if not is_finite_number(value):
            return None
        out[p] = float(value)
    return out


def valid_qm_result(json_path, xyz_path, prop_names) -> tuple:
    """(ok, reason): the QM JSON exists, holds every value to_hdf5 reads, and the elements are supported."""
    jp = Path(json_path)
    if not jp.exists():
        return False, "missing QM json"
    try:
        symbols = read_xyz_symbols(xyz_path)
    except (OSError, ValueError, IndexError):
        return False, "unreadable xyz"
    if not set(symbols) <= SUPPORTED_SYMBOLS:
        return False, f"unsupported elements {sorted(set(symbols) - SUPPORTED_SYMBOLS)}"
    try:
        with jp.open("r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False, "unreadable QM json"
    for keys in REQUIRED_JSON_KEYS:
        if not is_finite_number(get_json_value(data, keys)):
            return False, f"missing or non-finite {'/'.join(keys)}"
    if read_json_props(jp, prop_names, num_atoms=len(symbols)) is None:
        return False, "missing target property"
    return True, ""


def is_hit(achieved: dict, requested: dict, tolerances, prop_names) -> bool:
    """Return whether all non-atNUM properties are within their configured tolerances."""
    names = [p for p in prop_names if p != NUM_ATOMS_PROP]
    if isinstance(tolerances, (int, float)):
        tolerances = {p: {"relative": float(tolerances), "absolute": 0.0} for p in names}
    target = np.array([requested[p] for p in names], dtype=float)
    delta = tolerance_delta(target, names, tolerances)
    for p, allowed in zip(names, delta):
        if p == NUM_ATOMS_PROP:
            continue
        if abs(float(achieved[p]) - float(requested[p])) > allowed:
            return False
    return True


def build_al_meta(query_dir, prop_dir, targets: list, cand_meta: dict, prop_names: list,
                  tolerances, iteration: int) -> tuple:
    """Validate QM results and attach target/hit metadata.

    Returns (al_meta keyed by canonical ID, rejected list of (file name, reason)).
    """
    by_id = {t.id: t for t in targets}
    al_meta, rejected = {}, []
    for xyz in sorted(Path(query_dir).iterdir()):
        if not (xyz.is_file() and xyz.suffix == ".xyz"):
            continue
        cid = canonical_id(xyz.name)
        ok, reason = valid_qm_result(Path(prop_dir) / f"{xyz.stem}.json", xyz, prop_names)
        if not ok:
            rejected.append((xyz.name, reason))
            continue
        achieved = read_json_props(Path(prop_dir) / f"{xyz.stem}.json", prop_names, num_atoms=len(read_xyz_symbols(xyz)))
        tid = cand_meta.get(cid, {}).get("target_id", -1)
        target = by_id.get(tid)
        if target is None:
            al_meta[cid] = {"iteration": iteration, "target_id": -1, "requested": [math.nan] * len(prop_names),
                            "hit": -1, "achieved": achieved}
            continue
        al_meta[cid] = {
            "iteration": iteration,
            "target_id": int(tid),
            "requested": [float(target.props[p]) for p in prop_names],
            "hit": int(is_hit(achieved, target.props, tolerances, prop_names)),
            "achieved": achieved,
        }
    return al_meta, rejected
