"""Dedup block: drop generated molecules with the same connectivity, and those already in the initial dataset.

Connectivity key: bonds are perceived from the 3D coordinates with covalent radii (RDKit DetermineConnectivity,
no bond orders), every bond is treated as single and every H is an explicit atom; the key is the canonical SMILES
of that graph. Two molecules share a key exactly when their atom-bond graphs are identical, regardless of bond
orders, charges, stereochemistry or conformation.

Runs at the end of the forward step, right after `generate` (experiment_config.yaml -> al.dedup); `generate`
then leaves query/ empty, so query/ only ever receives the surviving molecules:
  - scope: all    -> one copy per connectivity across the anchor and all additional targets (lowest target id wins)
  - scope: target -> one copy per connectivity within each target
  - against_initial: true -> also drop molecules whose connectivity occurs in the initial dataset (input HDF5)
Input: every molecule generated at the anchor and the additional targets (<iter_dir>/generated/t<id>/, via
candidates_meta.json). The survivors of all targets are stored in <iter_dir>/deduplicated/ and copied to query/
(emptied first) for the backward step (score_filter onwards); removed molecules go to <iter_dir>/dedup_removed/, and every
molecule is listed in <iter_dir>/dedup_report.csv.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.candidates import read_candidates_meta

REPORT_FILE = "dedup_report.csv"
REMOVED_DIR = "dedup_removed"
KEPT_DIR = "deduplicated"
_TARGET_RE = re.compile(r"^t(\d+)_")
_INDEX_RE = re.compile(r"(\d+)$")


def connectivity_key(atomic_numbers, positions, cov_factor: float = 1.3) -> Optional[str]:
    """Canonical SMILES of the bond graph perceived from 3D coordinates (all bonds single, explicit H), or None."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    try:
        z = [int(round(float(v))) for v in atomic_numbers]
        pos = np.asarray(positions, dtype=float).reshape(len(z), 3)
        mol = Chem.RWMol()
        for zi in z:
            atom = Chem.Atom(zi)
            atom.SetNoImplicit(True)          # hydrogens are real atoms; never add implicit ones
            mol.AddAtom(atom)
        conf = Chem.Conformer(len(z))
        for i, (x, y, w) in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(w)))
        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False, charge=0, covFactor=float(cov_factor))
        out = mol.GetMol()
        for bond in out.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        out.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(out)
        return Chem.MolToSmiles(out, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def read_xyz(path) -> tuple:
    """(atomic numbers, positions) from an xyz file."""
    from rdkit import Chem

    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    table = Chem.GetPeriodicTable()
    z, pos = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        sym = parts[0]
        z.append(int(sym) if sym.isdigit() else table.GetAtomicNumber(sym.capitalize()))
        pos.append([float(v) for v in parts[1:4]])
    return z, np.array(pos)


def _xyz_key(args) -> Optional[str]:
    path, cov_factor = args
    try:
        z, pos = read_xyz(path)
    except Exception:
        return None
    return connectivity_key(z, pos, cov_factor)


def _array_key(args) -> Optional[str]:
    z, pos, cov_factor = args
    return connectivity_key(z, pos, cov_factor)


def _n_workers() -> int:
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except AttributeError:
        return max(os.cpu_count() or 1, 1)


def _parallel(fn, items: list) -> list:
    if len(items) < 200 or _n_workers() == 1:
        return [fn(x) for x in items]
    with Pool(_n_workers()) as pool:
        return pool.map(fn, items, chunksize=256)


def initial_keys(reference_hdf5, cov_factor: float = 1.3, log=print) -> set:
    """Connectivity keys of every molecule in the initial dataset (cached next to the HDF5 file)."""
    import h5py

    path = Path(reference_hdf5)
    stat = path.stat()
    cache = path.with_name(path.stem + ".connectivity.json")
    tag = {"source": str(path.resolve()), "size": stat.st_size, "mtime": int(stat.st_mtime),
           "cov_factor": float(cov_factor)}
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("tag") == tag:
                log(f"  Initial dataset: {len(data['keys'])} distinct connectivities (cached in {cache.name})")
                return set(data["keys"])
        except (OSError, ValueError):
            pass

    items = []
    with h5py.File(path, "r") as f:
        for top in f:
            if not isinstance(f[top], h5py.Group):
                continue
            for name in f[top]:
                mol = f[top][name]
                if isinstance(mol, h5py.Group) and "atNUM" in mol and "atXYZ" in mol:
                    items.append((np.asarray(mol["atNUM"]), np.asarray(mol["atXYZ"]), float(cov_factor)))
    log(f"  Computing connectivities of the initial dataset ({len(items)} molecules, {_n_workers()} workers) ...")
    keys = _parallel(_array_key, items)
    failed = sum(k is None for k in keys)
    distinct = sorted({k for k in keys if k is not None})
    log(f"  Initial dataset: {len(distinct)} distinct connectivities ({failed} molecules without a key)")
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"tag": tag, "keys": distinct}))
    tmp.replace(cache)
    return set(distinct)


def _target_of(name: str, meta: dict) -> int:
    if name in meta and "target_id" in meta[name]:
        return int(meta[name]["target_id"])
    m = _TARGET_RE.match(name)
    return int(m.group(1)) if m else -1


def _order(name: str, tid: int) -> tuple:
    m = _INDEX_RE.search(name)
    return (tid, int(m.group(1)) if m else 0, name)


def collect_generated(query_dir, iter_dir, meta: dict, log=print) -> dict:
    """{candidate name: xyz path} for all molecules generated at the anchor and the additional targets.

    Read from <iter_dir>/generated/t<id>/ through candidates_meta.json, so a re-run starts again from the full
    generated set; falls back to the xyz files in query_dir if that mapping is missing or incomplete.
    """
    gen_dir = Path(iter_dir) / "generated"
    files = {}
    for name, info in meta.items():
        tid = info.get("target_id")
        prefix = f"t{tid}_"
        path = gen_dir / f"t{tid}" / f"{name[len(prefix):] if name.startswith(prefix) else name}.xyz"
        if tid is None or not path.exists():
            files = {}
            break
        files[name] = path
    if files:
        return files
    log(f"  No complete generated/t<id>/ set listed in candidates_meta.json; using the molecules in {query_dir}")
    return {f.stem: f for f in Path(query_dir).iterdir() if f.is_file() and f.suffix == ".xyz"}


def _refill(directory: Path, files: list, rename: dict) -> None:
    """Empty `directory` (files only) and copy `files` into it under their candidate names."""
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.iterdir():
        if f.is_file():
            f.unlink()
    for f in files:
        shutil.copy(f, directory / f"{rename[f]}.xyz")


def run_dedup(*, query_dir, iter_dir, reference_hdf5=None, scope: str = "all", against_initial: bool = True,
              cov_factor: float = 1.3, log=print) -> dict:
    """Deduplicate the molecules generated at the anchor and the additional targets; return {status: count}.

    The surviving molecules are stored in <iter_dir>/deduplicated/ and copied to query_dir (which is emptied first)
    for score_filter and the later blocks; removed ones are stored in <iter_dir>/dedup_removed/.
    """
    if scope not in ("all", "target"):
        raise ValueError(f"al.dedup.scope must be 'all' or 'target', got {scope!r}")
    query_dir, iter_dir = Path(query_dir), Path(iter_dir)
    meta = read_candidates_meta(iter_dir)
    by_name = collect_generated(query_dir, iter_dir, meta, log)
    names = sorted(by_name, key=lambda s: _order(s, _target_of(s, meta)))   # anchor (target 0) first
    files = [by_name[s] for s in names]
    rename = {by_name[s]: s for s in names}
    tids = {s: _target_of(s, meta) for s in names}
    keys = _parallel(_xyz_key, [(str(f), float(cov_factor)) for f in files])

    known = set()
    if against_initial:
        if reference_hdf5 is not None and Path(reference_hdf5).exists():
            known = initial_keys(reference_hdf5, cov_factor, log)
        else:
            log(f"  Initial dataset {reference_hdf5} not found; skipping the comparison with it.")

    seen = {}            # (group, key) -> first molecule with that connectivity
    rows, kept, removed = [], [], []
    for name, f, key in zip(names, files, keys):
        tid = tids[name]
        group = tid if scope == "target" else 0
        status, dup_of = "kept", ""
        if key is None:
            status = "no_key"            # bonds could not be perceived: kept, it cannot be compared
        elif key in known:
            status = "in_initial"
        elif (group, key) in seen:
            status, dup_of = "duplicate", seen[(group, key)]
        else:
            seen[(group, key)] = name
        (removed if status in ("in_initial", "duplicate") else kept).append(f)
        rows.append({"name": name, "target_id": tid, "status": status, "duplicate_of": dup_of, "connectivity": key or ""})

    # store both sets, then hand the survivors of all targets to the next blocks through query/
    _refill(iter_dir / KEPT_DIR, kept, rename)
    _refill(iter_dir / REMOVED_DIR, removed, rename)
    for f in query_dir.iterdir() if query_dir.exists() else []:
        if f.is_file():
            f.unlink()
    shutil.copytree(iter_dir / KEPT_DIR, query_dir, dirs_exist_ok=True)
    with (iter_dir / REPORT_FILE).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "target_id", "status", "duplicate_of", "connectivity"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r["status"] for r in rows)
    per_target = defaultdict(Counter)
    for r in rows:
        per_target[r["target_id"]][r["status"]] += 1
    log(f"  Dedup ({scope}): {len(rows)} molecules -> {counts['kept'] + counts['no_key']} kept, "
        f"{counts['duplicate']} duplicates, {counts['in_initial']} in the initial dataset, "
        f"{counts['no_key']} without perceivable bonds (kept)")
    for tid in sorted(per_target):
        c = per_target[tid]
        log(f"    target {tid}: {sum(c.values())} -> {c['kept'] + c['no_key']} kept "
            f"({c['duplicate']} duplicates, {c['in_initial']} in initial)")
    log(f"  {len(kept)} molecules stored in {iter_dir / KEPT_DIR} and copied to {query_dir}; "
        f"removed ones in {iter_dir / REMOVED_DIR}; report {iter_dir / REPORT_FILE}")
    return dict(counts)
