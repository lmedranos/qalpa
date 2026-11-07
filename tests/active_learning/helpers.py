from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.components.trajectory import Trajectory

PROPS = ["HLgap", "eMBD", "atNUM"]


def write_trajectory(path: Path, rows=None) -> Trajectory:
    rows = rows or [(4.5, -1.9, 61), (3.4, -1.9, 59), (2.5, -1.9, 59), (3.1, -2.3, 71)]
    lines = ["HLgap,eMBD,atNUM,SAScore"] + [f"{a},{b},{c},5.0" for a, b, c in rows]
    path.write_text("\n".join(lines) + "\n")
    return Trajectory(path, prop_names=list(PROPS), score_names=["SAScore"])


def write_mol_group(grp, name, hlgap, embd, n_atoms=10, al=None, props=PROPS):
    m = grp.create_group(name)
    m.create_dataset("atNUM", data=np.full(n_atoms, 6.0))
    m.create_dataset("atXYZ", data=np.zeros((n_atoms, 3)))
    m.create_dataset("HLgap", data=np.array([hlgap]))
    m.create_dataset("eMBD", data=np.array([embd]))
    if al is not None:
        m.create_dataset("al_iteration", data=np.array([al["iteration"]], dtype=np.int64))
        m.create_dataset("al_target_id", data=np.array([al["target_id"]], dtype=np.int64))
        m.create_dataset("al_requested", data=np.array(al["requested"], dtype=float))
        m.create_dataset("al_hit", data=np.array([al["hit"]], dtype=np.int8))
        m.attrs["al_props"] = ",".join(props)
    return m


def write_h5(path: Path, n_base=60, seed=0, records=()):
    """Base dataset around HLgap~3, eMBD~-1.9 plus optional AL records [(name, hlgap, embd, al dict)]."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        base = f.create_group("base")
        for i in range(n_base):
            write_mol_group(base, f"m{i}", rng.normal(3.0, 0.4), rng.normal(-1.9, 0.1), n_atoms=int(rng.integers(8, 20)))
        if records:
            g = f.create_group("G-active-loop-0")
            for name, hl, mb, al in records:
                write_mol_group(g, name, hl, mb, al=al)
    return path


def write_json_result(path: Path, hlgap=3.0, embd=-1.9, dip=1.0, total=-100.0):
    path.write_text(json.dumps({
        "Total energy (eV):": total, "SUM": {"MBD": embd}, "Absolute Total Dipole (eAng):": dip,
        "HOMO-LUMO gap": hlgap, "HOMO energy": -5.0, "LUMO energy": -5.0 + hlgap,
    }))


def write_xyz(path: Path, symbols=("C", "H", "H", "H", "H")):
    pos = [(0.0, 0.0, 0.0), (0.63, 0.63, 0.63), (-0.63, -0.63, 0.63), (-0.63, 0.63, -0.63), (0.63, -0.63, -0.63)]
    lines = [str(len(symbols)), "mol"] + [f"{s} {x} {y} {z}" for s, (x, y, z) in zip(symbols, pos)]
    path.write_text("\n".join(lines) + "\n")
