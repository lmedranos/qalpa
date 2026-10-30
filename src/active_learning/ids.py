from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.active_learning.constants import NUM_ATOMS_PROP

TARGETS_FILE = "targets.json"
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


@dataclass
class Target:
    id: int
    props: dict
    n_samples: int
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> Optional[int]:
        v = self.props.get(NUM_ATOMS_PROP)
        return None if v is None else int(round(float(v)))

    def to_dict(self) -> dict:
        return {"id": int(self.id), "props": {k: float(v) for k, v in self.props.items()},
                "n_samples": int(self.n_samples), "num_nodes": self.num_nodes, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(id=int(d["id"]), props=dict(d["props"]), n_samples=int(d["n_samples"]), meta=dict(d.get("meta", {})))


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


def write_targets(iter_dir, iteration: int, strategy: str, prop_names: list, targets: list) -> None:
    write_json(Path(iter_dir) / TARGETS_FILE, {
        "iteration": int(iteration), "strategy": strategy, "prop_names": list(prop_names),
        "targets": [t.to_dict() for t in targets],
    })


def read_targets(iter_dir) -> Optional[tuple]:
    """(prop_names, [Target]) for an iteration, or None when no plan was written."""
    data = read_json(Path(iter_dir) / TARGETS_FILE)
    if data is None:
        return None
    return list(data["prop_names"]), [Target.from_dict(t) for t in data["targets"]]


def write_candidates_meta(iter_dir, meta: dict) -> None:
    write_json(Path(iter_dir) / CANDIDATES_META_FILE, meta)


def read_candidates_meta(iter_dir) -> dict:
    return read_json(Path(iter_dir) / CANDIDATES_META_FILE, default={}) or {}
