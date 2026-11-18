"""Labelled molecules and their active-learning provenance, read back from the cumulative HDF5 file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.al_config import NUM_ATOMS_PROP
from src.utils.candidates import read_json, write_json


@dataclass
class ALState:
    """One row per molecule in the HDF5 file."""

    prop_names: list
    keys: np.ndarray        # "<top group>/<molecule>"
    props: np.ndarray       # (n, P) achieved values, NaN if unavailable
    iteration: np.ndarray   # -1 for molecules without AL metadata (e.g. the base dataset)
    target_id: np.ndarray   # -1 without metadata
    requested: np.ndarray   # (n, P) requested target, NaN without metadata
    hit: np.ndarray         # 1 / 0, -1 unknown

    @classmethod
    def empty(cls, prop_names) -> "ALState":
        p = len(prop_names)
        return cls(list(prop_names), np.empty(0, dtype=object), np.empty((0, p)), np.empty(0, dtype=int),
                   np.empty(0, dtype=int), np.empty((0, p)), np.empty(0, dtype=int))

    def __len__(self) -> int:
        return len(self.keys)

    def labelled_mask(self) -> np.ndarray:
        return np.isfinite(self.props).all(axis=1) if len(self) else np.zeros(0, dtype=bool)

    def record_mask(self) -> np.ndarray:
        """Molecules generated for a known target with a known hit flag."""
        if not len(self):
            return np.zeros(0, dtype=bool)
        return np.isfinite(self.requested).all(axis=1) & np.isfinite(self.props).all(axis=1) & (self.hit >= 0)

    def labelled_props(self) -> np.ndarray:
        return self.props[self.labelled_mask()]


def load_state(hdf5_path, prop_names) -> ALState:
    """Read achieved properties (prop_names, atNUM = atom count) and AL metadata from the HDF5 file."""
    import h5py

    prop_names = list(prop_names)
    p = len(prop_names)
    path = Path(hdf5_path)
    if not path.exists():
        return ALState.empty(prop_names)

    keys, props, its, tids, reqs, hits = [], [], [], [], [], []
    with h5py.File(path, "r") as f:
        for top in f:
            top_grp = f[top]
            if not isinstance(top_grp, h5py.Group):
                continue
            for name in top_grp:
                mol = top_grp[name]
                if not isinstance(mol, h5py.Group):
                    continue
                row = np.full(p, np.nan)
                for j, prop in enumerate(prop_names):
                    if prop not in mol:
                        continue
                    if prop == NUM_ATOMS_PROP:
                        row[j] = float(mol[prop].shape[0])
                    else:
                        row[j] = float(np.asarray(mol[prop]).reshape(-1)[0])
                req = np.full(p, np.nan)
                it, tid, hit = -1, -1, -1
                if "al_requested" in mol:
                    saved = np.asarray(mol["al_requested"], dtype=float)
                    saved_names = str(mol.attrs["al_props"]).split(",") if "al_props" in mol.attrs else prop_names
                    for j, prop in enumerate(prop_names):
                        if prop in saved_names and saved_names.index(prop) < len(saved):
                            req[j] = saved[saved_names.index(prop)]
                    it = int(np.asarray(mol["al_iteration"]).reshape(-1)[0])
                    tid = int(np.asarray(mol["al_target_id"]).reshape(-1)[0])
                    hit = int(np.asarray(mol["al_hit"]).reshape(-1)[0])
                keys.append(f"{top}/{name}")
                props.append(row)
                its.append(it)
                tids.append(tid)
                reqs.append(req)
                hits.append(hit)

    if not keys:
        return ALState.empty(prop_names)
    return ALState(prop_names, np.array(keys, dtype=object), np.vstack(props), np.array(its, dtype=int),
                   np.array(tids, dtype=int), np.vstack(reqs), np.array(hits, dtype=int))


def read_al_state_file(path) -> dict:
    return read_json(path, default={}) or {}


def write_al_state_file(path, data: dict) -> None:
    write_json(path, data)
