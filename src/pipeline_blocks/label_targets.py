"""Label each computed molecule as hit / miss against the target it was generated for (ported from qalpa2).

The result is written into the HDF5 file by run_to_hdf5 (al_iteration, al_target_id, al_requested, al_hit)
so that the next plan step can see where the generator lands and how often it hits.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from functools import reduce
from pathlib import Path

import numpy as np

from src.utils.al_config import NUM_ATOMS_PROP, PROPERTY_JSON_KEYS, REQUIRED_JSON_KEYS, tolerance_delta
from src.utils.candidates import canonical_id, read_candidates_meta
from src.utils.trajectory import read_targets


def _get(data, keys):
    return reduce(lambda d, k: d.get(k) if isinstance(d, dict) else None, keys, data)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _num_atoms(xyz_path) -> int:
    with Path(xyz_path).open("r") as f:
        return int(f.readline().split()[0])


def read_achieved(json_path, xyz_path, prop_names) -> tuple:
    """(achieved {prop: value} or None, reason) from a prop_calc JSON; atNUM is the atom count of the xyz."""
    path = Path(json_path)
    if not path.exists():
        return None, "missing QM json"
    try:
        with path.open("r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None, "unreadable QM json"
    for keys in REQUIRED_JSON_KEYS:
        if not _finite(_get(data, keys)):
            return None, f"missing or non-finite {'/'.join(keys)}"
    achieved = {}
    for name in prop_names:
        if name == NUM_ATOMS_PROP:
            try:
                achieved[name] = float(_num_atoms(xyz_path))
            except (OSError, ValueError, IndexError):
                return None, "unreadable xyz"
            continue
        value = _get(data, PROPERTY_JSON_KEYS[name])
        if not _finite(value):
            return None, f"missing {name}"
        achieved[name] = float(value)
    return achieved, ""


def is_hit(achieved: dict, requested: dict, tolerances: dict) -> bool:
    """All conditioning properties within tolerance (atNUM is not part of the hit rule)."""
    names = list(tolerances)
    delta = tolerance_delta([requested[n] for n in names], names, tolerances)
    return all(abs(float(achieved[n]) - float(requested[n])) <= allowed for n, allowed in zip(names, delta))


PROP_FILTER_REPORT = "prop_filter_report.json"


def _reject(query_dir: Path, iter_dir: Path, drop: dict) -> None:
    """Move the dropped molecules to <iter_dir>/rejected; reasons.json keeps the reasons of every step."""
    if not drop:
        return
    out = iter_dir / "rejected"
    out.mkdir(parents=True, exist_ok=True)
    reasons_file = out / "reasons.json"
    reasons = {}
    if reasons_file.exists():
        try:
            reasons = json.loads(reasons_file.read_text())
        except ValueError:
            reasons = {}
    for name, reason in drop.items():
        shutil.move(str(query_dir / name), str(out / name))
        reasons[name] = reason
    reasons_file.write_text(json.dumps(reasons, indent=2))


PROP_FILTER_CSV = "prop_filter_properties.csv"
ACCURACY_FILTERED_DIR = "accuracy_filtered"     # kept molecules (copy; same folder as the non-AL prop_filter)
_EXTRA_JSON_KEYS = {"Etot": ("Total energy (eV):",)}   # reported in the CSV besides PROPERTY_JSON_KEYS


def _all_props(json_path) -> dict:
    """Every property of PROPERTY_JSON_KEYS (+ total energy) found in a prop_calc JSON; missing ones are None."""
    names = {**PROPERTY_JSON_KEYS, **_EXTRA_JSON_KEYS}
    try:
        data = json.loads(Path(json_path).read_text())
    except (OSError, ValueError):
        return {n: None for n in names}
    return {n: (float(_get(data, k)) if _finite(_get(data, k)) else None) for n, k in names.items()}


def _fmt(value) -> str:
    return "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.8g}"


def filter_by_own_target(*, query_dir, prop_dir, iter_dir, prop_names: list, tolerances: dict, iteration: int,
                         log=print) -> dict:
    """prop_filter for active learning: keep only molecules that hit the target they were generated for.

    Same rule as the hit/miss label (is_hit): every conditioning property within
    max(relative * |target|, absolute) of that molecule's own target (atNUM excluded); with the default
    tolerances this is the original prop_filter rule (query.filter_threshold * |target|), applied per target
    instead of against the anchor row. Misses and molecules without a valid QM result are moved to
    <iter_dir>/rejected. Molecules without a planned target are kept (nothing to compare with).

    Outputs:
      <iter_dir>/accuracy_filtered/: a copy of the kept molecules (they also stay in query/);
      <iter_dir>/prop_filter_properties.csv: one row per molecule (kept and rejected) with its target, the
        computed properties, and per conditioning property the target value, error, allowed error and pass/fail;
      <iter_dir>/prop_filter_report.json: hit statistics over all evaluated molecules (misses included), since
        the HDF5 file afterwards only holds the hits.
    """
    query_dir, iter_dir, prop_dir = Path(query_dir), Path(iter_dir), Path(prop_dir)
    plan = read_targets(iter_dir)
    by_id = {t.id: t for t in (plan[1] if plan else [])}
    cand_meta = read_candidates_meta(iter_dir)
    cond = list(tolerances)                                     # conditioning properties (atNUM excluded)
    has_atnum = NUM_ATOMS_PROP in prop_names
    extra = [n for n in {**PROPERTY_JSON_KEYS, **_EXTRA_JSON_KEYS} if n not in cond]

    drop, per_target, n_no_target, rows = {}, {}, 0, []
    for xyz in sorted(query_dir.iterdir()):
        if not (xyz.is_file() and xyz.suffix == ".xyz"):
            continue
        json_path = prop_dir / f"{xyz.stem}.json"
        values = _all_props(json_path)
        target = by_id.get(cand_meta.get(canonical_id(xyz.name), {}).get("target_id", -1))
        row = {"molecule": xyz.stem, "target_id": int(target.id) if target is not None else -1}
        if has_atnum:
            try:
                n_atoms = float(_num_atoms(xyz))
            except (OSError, ValueError, IndexError):
                n_atoms = None
            t_atoms = float(target.props[NUM_ATOMS_PROP]) if target is not None else None
            row.update({"target_atNUM": t_atoms, "atNUM": n_atoms,
                        "d_atNUM": n_atoms - t_atoms if None not in (n_atoms, t_atoms) else None})
        if target is not None:
            allowed = dict(zip(cond, tolerance_delta([target.props[c] for c in cond], cond, tolerances)))
        for c in cond:
            v, t = values.get(c), (float(target.props[c]) if target is not None else None)
            err = abs(v - t) if None not in (v, t) else None
            row.update({f"target_{c}": t, c: v, f"abserr_{c}": err,
                        f"relerr_pct_{c}": 100.0 * err / abs(t) if err is not None and abs(t) > 0 else None,
                        f"allowed_{c}": float(allowed[c]) if target is not None else None,
                        f"within_{c}": int(err <= allowed[c]) if err is not None else None})
        row.update({e: values.get(e) for e in extra})

        achieved, reason = read_achieved(json_path, xyz, prop_names)
        if achieved is None:
            drop[xyz.name] = reason
            row.update(status="no_valid_qm", hit=None, reason=reason)
        elif target is None:
            n_no_target += 1
            row.update(status="kept_no_target", hit=None, reason="")
        else:
            hit = is_hit(achieved, target.props, tolerances)
            stats = per_target.setdefault(int(target.id), {"evaluated": 0, "hits": 0})
            stats["evaluated"] += 1
            stats["hits"] += int(hit)
            row.update(status="kept" if hit else "off_target", hit=int(hit), reason="")
            if not hit:
                drop[xyz.name] = "off target (prop_filter)"
        rows.append(row)

    _reject(query_dir, iter_dir, drop)
    # copy of the kept molecules (they also stay in query/ for the next blocks)
    kept_dir = iter_dir / ACCURACY_FILTERED_DIR
    kept_dir.mkdir(parents=True, exist_ok=True)
    for f in kept_dir.glob("*.xyz"):
        f.unlink()
    for f in sorted(query_dir.glob("*.xyz")):
        shutil.copy(f, kept_dir / f.name)
    columns = ["molecule", "target_id", "status", "hit", "reason"]
    if has_atnum:
        columns += ["target_atNUM", "atNUM", "d_atNUM"]
    for c in cond:
        columns += [f"target_{c}", c, f"abserr_{c}", f"relerr_pct_{c}", f"allowed_{c}", f"within_{c}"]
    columns += extra
    with (iter_dir / PROP_FILTER_CSV).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(k) if k in ("molecule", "status", "reason") else
                             (row.get(k) if isinstance(row.get(k), int) else _fmt(row.get(k))) for k in columns])

    for tid, st in sorted(per_target.items()):
        st["hit_rate"] = st["hits"] / st["evaluated"] if st["evaluated"] else float("nan")
        log(f"    prop_filter target {tid}: {st['hits']}/{st['evaluated']} within tolerance "
            f"({100.0 * st['hit_rate']:.1f}%)")
    n_eval = sum(st["evaluated"] for st in per_target.values())
    n_hits = sum(st["hits"] for st in per_target.values())
    report = {"iteration": int(iteration), "evaluated": n_eval, "hits": n_hits,
              "hit_rate": n_hits / n_eval if n_eval else None, "no_target_kept": n_no_target,
              "no_valid_qm": sum(r != "off target (prop_filter)" for r in drop.values()),
              "tolerances": tolerances, "per_target": {str(k): v for k, v in sorted(per_target.items())}}
    (iter_dir / PROP_FILTER_REPORT).write_text(json.dumps(report, indent=2))
    log(f"  prop_filter (per target): kept {n_hits} of {n_eval} evaluated molecules; "
        f"{len(drop)} moved to {iter_dir / 'rejected'}; properties of all molecules in {PROP_FILTER_CSV}")
    return report


def read_prop_filter_report(iter_dir) -> dict | None:
    path = Path(iter_dir) / PROP_FILTER_REPORT
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def prepare_al_labels(*, query_dir, prop_dir, iter_dir, prop_names: list, tolerances: dict, iteration: int,
                      keep_off_target: bool, log=print) -> dict:
    """Build the AL metadata for run_to_hdf5, keyed by canonical molecule ID.

    Molecules without a valid QM result (and misses, if keep_off_target is false) are moved to
    <iter_dir>/rejected so that they are not written to the HDF5 files.
    """
    query_dir, iter_dir, prop_dir = Path(query_dir), Path(iter_dir), Path(prop_dir)
    plan = read_targets(iter_dir)
    by_id = {t.id: t for t in (plan[1] if plan else [])}
    cand_meta = read_candidates_meta(iter_dir)

    al_meta, drop = {}, {}
    for xyz in sorted(query_dir.iterdir()):
        if not (xyz.is_file() and xyz.suffix == ".xyz"):
            continue
        cid = canonical_id(xyz.name)
        achieved, reason = read_achieved(prop_dir / f"{xyz.stem}.json", xyz, prop_names)
        if achieved is None:
            drop[xyz.name] = reason
            continue
        target = by_id.get(cand_meta.get(cid, {}).get("target_id", -1))
        if target is None:
            al_meta[cid] = {"iteration": iteration, "target_id": -1, "requested": [math.nan] * len(prop_names),
                            "hit": -1, "achieved": achieved}
            continue
        hit = int(is_hit(achieved, target.props, tolerances))
        if hit != 1 and not keep_off_target:
            drop[xyz.name] = "off target"
            continue
        al_meta[cid] = {"iteration": iteration, "target_id": int(target.id),
                        "requested": [float(target.props[p]) for p in prop_names], "hit": hit, "achieved": achieved}

    _reject(query_dir, iter_dir, drop)
    hits = [m["hit"] for m in al_meta.values() if m["hit"] >= 0]
    log(f"  Labelled {len(al_meta)} molecules ({sum(h == 1 for h in hits)} hits of {len(hits)} with a target); "
        f"{len(drop)} kept out of the HDF5 file")
    return al_meta
