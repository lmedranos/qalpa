from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

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
    """Canonical SMILES from coordinates via openbabel bond perception (the same route as the SAScore block)."""
    try:
        from openbabel import pybel
        from rdkit import Chem

        mol = pybel.readstring("xyz", _atoms_to_xyz_block(symbols, positions))
        mol.OBMol.ConnectTheDots()
        mol.OBMol.PerceiveBondOrders()
        smi = mol.write("smi").split()[0]
        rd = Chem.MolFromSmiles(smi)
        if rd is None or "*" in smi:  # '*' marks an element openbabel could not identify
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
    """SMILES + Morgan fingerprint per molecule, optionally cached on disk by molecule key."""

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
    """(symbols, positions) of a "<top>/<molecule>" entry in an open HDF5 file."""
    grp = h5[key]
    z = np.asarray(grp["atNUM"]).astype(int)
    return [ELEMENT_SYMBOLS[int(i) - 1] for i in z], np.asarray(grp["atXYZ"], dtype=float).tolist()
