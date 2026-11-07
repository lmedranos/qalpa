from __future__ import annotations

import shutil
from pathlib import Path

from src.active_learning.ids import canonical_id, read_candidates_meta, read_targets
from src.active_learning.labels import build_al_meta


def prepare_al_labels(*, query_dir, prop_dir, iter_dir, prop_names: list, iteration: int,
                      tolerances=None, threshold: float | None = None,
                      keep_off_target: bool, log=print) -> dict:
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
                                      prop_names, tolerances, iteration)
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
