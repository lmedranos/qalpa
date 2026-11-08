from __future__ import annotations

import json
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path

from src.components.dataset_labels import NUM_ATOMS_PROP


TARGETS_FILE = "targets.json"


@dataclass
class Target:
    """One planned point in property space and its generation allocation."""

    id: int
    props: dict
    n_samples: int
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self):
        value = self.props.get(NUM_ATOMS_PROP)
        return None if value is None else int(round(float(value)))

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "props": {key: float(value) for key, value in self.props.items()},
            "n_samples": int(self.n_samples),
            "num_nodes": self.num_nodes,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Target":
        return cls(
            id=int(data["id"]),
            props=dict(data["props"]),
            n_samples=int(data["n_samples"]),
            meta=dict(data.get("meta", {})),
        )


def _write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(obj, stream, indent=2)
    tmp.replace(path)


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r") as stream:
        return json.load(stream)


def write_targets(iter_dir, iteration: int, strategy: str, prop_names: list, targets: list) -> None:
    """Persist the planned trajectory targets for one iteration."""
    _write_json(Path(iter_dir) / TARGETS_FILE, {
        "iteration": int(iteration),
        "strategy": strategy,
        "prop_names": list(prop_names),
        "targets": [target.to_dict() for target in targets],
    })


def read_targets(iter_dir):
    """Read ``(property_names, targets)`` for an iteration, if planned."""
    data = _read_json(Path(iter_dir) / TARGETS_FILE)
    if data is None:
        return None
    return list(data["prop_names"]), [Target.from_dict(target) for target in data["targets"]]


class Trajectory:
    def __init__(self, csv_path, prop_names, score_names):
        self.prop_names = prop_names
        self.score_names = score_names
        self.names = prop_names + score_names
        csv_path = Path(csv_path)

        if not csv_path.exists():
            # try resolving a glob pattern (e.g. trajectory_*.csv)
            parent = csv_path.parent
            matches = sorted(parent.glob(csv_path.name))
            if not matches:
                raise FileNotFoundError(f"No trajectory CSV file found at {csv_path}")
            csv_path = matches[0]

        # read only requested columns
        df = pd.read_csv(csv_path, usecols=self.names)

        # store axes in two groups
        self._props = {}
        self._scores = {}

        for name in self.prop_names:
            col = df[name].to_numpy()
            if "NUM" in name.upper():
                col = col.astype(int)
            self._props[name] = col

        for name in self.score_names:
            col = df[name].to_numpy()
            self._scores[name] = col.astype(float)

    def __getitem__(self, name):
        if name in self._props:
            return self._props[name]
        if name in self._scores:
            return self._scores[name]
        raise KeyError(f"No axis named {name}")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'Trajectory' object has no attribute '{name}'")
        props = self.__dict__.get("_props", {})
        scores = self.__dict__.get("_scores", {})
        if name in props:
            return props[name]
        if name in scores:
            return scores[name]
        raise AttributeError(f"'Trajectory' object has no attribute '{name}'")

    def __len__(self):
        # length is consistent across all axes
        return len(next(iter(self._props.values()))) if self._props else len(next(iter(self._scores.values())))

    @property
    def properties(self):
        return self._props

    @property
    def scores(self):
        return self._scores

    @property
    def has_properties(self):
        return len(self._props) > 0

    @property
    def has_scores(self):
        return len(self._scores) > 0

    def get_target(self, index):
        """
        Returns a flattened list like:
        [--prop1, value1, --prop2, value2, ...]
        for all property axes at the given row index.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for trajectory with length {len(self)}")
        return self.target_args({p: self._props[p][index] for p in self.prop_names})

    def target_args(self, values):
        """CLI arguments for one target given as {prop name: value}; atNUM maps to --num-nodes."""
        context = [values[p] for p in self.prop_names if p != "atNUM"]
        num_nodes = values["atNUM"] if "atNUM" in self.prop_names else None
        row_list = []

        if num_nodes is not None:
            row_list.append("--num-nodes")
            row_list.append(str(int(round(float(num_nodes)))))
        if context:
            row_list.append("--fixed-context")
            row_list.extend(str(c) for c in context)
            # row_list.append("--calculate-props")
            # row_list.append("--calculate-relax-change")

        return row_list
