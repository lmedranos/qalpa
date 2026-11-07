from __future__ import annotations

import json
from pathlib import Path
CANDIDATES_META_FILE = "candidates_meta.json"


def canonical_id(name: str) -> str:
    """File name or stem -> ID shared by generated, relaxed and QM files (drops extension and `opt-`)."""
    stem = Path(str(name)).name
    for ext in (".xyz", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem[4:] if stem.startswith("opt-") else stem


def candidate_name(target_id: int, stem: str) -> str:
    return f"t{int(target_id)}_{stem}"


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def write_candidates_meta(iter_dir, meta: dict) -> None:
    write_json(Path(iter_dir) / CANDIDATES_META_FILE, meta)


def read_candidates_meta(iter_dir) -> dict:
    return read_json(Path(iter_dir) / CANDIDATES_META_FILE, default={}) or {}
