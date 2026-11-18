import json
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from pathlib import Path


NUM_NODES_COLUMN = "atNUM"


class Trajectory:
    """
    Per-iteration targets read from a trajectory CSV.

    - prop_names: conditioning properties (passed to generation as --fixed-context and
      used by the accuracy filter). Must match the properties the model was trained on.
    - score_names: score targets (e.g. SAScore) used by the score filter.
    - atNUM: if the CSV has an ``atNUM`` column (and use_num_nodes is True), it is read
      automatically as the per-iteration atom count and passed as --num-nodes. It is never
      treated as a conditioning property, so it does not need to be listed in prop_names.
    """

    def __init__(self, csv_path, prop_names, score_names, use_num_nodes=True):
        # atNUM is not a conditioning property; drop it if a config still lists it.
        self.prop_names = [p for p in prop_names if p.lower() != NUM_NODES_COLUMN.lower()]
        self.score_names = list(score_names)
        self.names = self.prop_names + self.score_names
        csv_path = Path(csv_path)

        if not csv_path.exists():
            # try resolving a glob pattern (e.g. trajectory_*.csv)
            parent = csv_path.parent
            matches = sorted(parent.glob(csv_path.name))
            if not matches:
                raise FileNotFoundError(f"No trajectory CSV file found at {csv_path}")
            csv_path = matches[0]

        df = pd.read_csv(csv_path)
        missing = [c for c in self.names if c not in df.columns]
        if missing:
            raise KeyError(f"Trajectory CSV {csv_path} is missing columns: {missing}")

        # optional fixed atom count per iteration
        num_nodes_col = next((c for c in df.columns if c.lower() == NUM_NODES_COLUMN.lower()), None)
        if use_num_nodes and num_nodes_col is not None:
            self._num_nodes = df[num_nodes_col].round().astype(int).to_numpy()
        else:
            self._num_nodes = None

        # store axes in two groups
        self._props = {}
        self._scores = {}

        for name in self.prop_names:
            self._props[name] = df[name].to_numpy(dtype=float)

        for name in self.score_names:
            self._scores[name] = df[name].to_numpy(dtype=float)

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
        for axis in (*self._props.values(), *self._scores.values()):
            return len(axis)
        return len(self._num_nodes) if self._num_nodes is not None else 0

    @property
    def properties(self):
        return self._props

    @property
    def scores(self):
        return self._scores

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def has_properties(self):
        return len(self._props) > 0

    @property
    def has_num_nodes(self):
        return self._num_nodes is not None

    @property
    def has_scores(self):
        return len(self._scores) > 0

    @property
    def plan_names(self):
        """Axes of the property space used by target planning: prop_names, plus atNUM when present."""
        return self.prop_names + ([NUM_NODES_COLUMN] if self._num_nodes is not None else [])

    @property
    def plan_properties(self):
        """{axis name: per-row values} for plan_names."""
        out = dict(self._props)
        if self._num_nodes is not None:
            out[NUM_NODES_COLUMN] = self._num_nodes.astype(float)
        return out

    def get_target(self, index):
        """
        Returns the generation CLI arguments for the given row index, e.g.
        ["--num-nodes", "20", "--fixed-context", "5.0", "-0.5"].
        --num-nodes comes from the atNUM column (if present); --fixed-context holds the
        conditioning properties in prop_names order.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for trajectory with length {len(self)}")
        return self.target_args({p: values[index] for p, values in self.plan_properties.items()})

    def target_args(self, values):
        """Generation CLI arguments for one target given as {axis name: value}; atNUM maps to --num-nodes."""
        context = [values[p] for p in self.prop_names]
        num_nodes = values.get(NUM_NODES_COLUMN)
        row_list = []

        if num_nodes is not None:
            row_list.append("--num-nodes")
            row_list.append(str(int(round(float(num_nodes)))))
        if context:
            row_list.append("--fixed-context")
            row_list.extend(str(c) for c in context)

        return row_list


# ---------------------------------------------------------------------------
# Planned targets (active learning): one iteration may request several points
# in property space, each with its own share of the generation budget.
# ---------------------------------------------------------------------------

TARGETS_FILE = "targets.json"


@dataclass
class Target:
    """One planned point in property space and its generation allocation."""

    id: int
    props: dict
    n_samples: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "props": {key: float(value) for key, value in self.props.items()},
            "n_samples": int(self.n_samples),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Target":
        return cls(id=int(data["id"]), props=dict(data["props"]), n_samples=int(data["n_samples"]),
                   meta=dict(data.get("meta", {})))


def write_targets(iter_dir, iteration: int, strategy: str, prop_names: list, targets: list) -> None:
    """Persist the planned targets of one iteration to <iter_dir>/targets.json."""
    path = Path(iter_dir) / TARGETS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as stream:
        json.dump({"iteration": int(iteration), "strategy": strategy, "prop_names": list(prop_names),
                   "targets": [t.to_dict() for t in targets]}, stream, indent=2)
    tmp.replace(path)


def read_targets(iter_dir):
    """Return (prop_names, targets) for an iteration, or None if it was not planned."""
    path = Path(iter_dir) / TARGETS_FILE
    if not path.exists():
        return None
    with path.open("r") as stream:
        data = json.load(stream)
    return list(data["prop_names"]), [Target.from_dict(t) for t in data["targets"]]
