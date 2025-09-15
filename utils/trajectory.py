import pandas as pd
import numpy as np
from pathlib import Path


class Trajectory:
    def __init__(self, directory, prop_names, score_names):
        self.prop_names = prop_names
        self.score_names = score_names
        self.names = prop_names + score_names
        directory = Path(directory)

        # find the trajectory file
        csv_files = list(directory.glob("trajectory_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No trajectory CSV files found in {directory}")
        csv_path = csv_files[0]

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
        if name in self._props:
            return self._props[name]
        if name in self._scores:
            return self._scores[name]
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
        
        context = [self._props[p][index] for p in self.prop_names if p != "atNUM"]
        num_nodes = self._props["atNUM"][index] if "atNUM" in self.prop_names else None
        row_list = []

        if num_nodes is not None:
            row_list.append(f"--num-nodes")
            row_list.append(str(num_nodes))
        if context:
            row_list.append(f"--fixed-context")
            row_list.extend(str(c) for c in context)
            # row_list.append("--calculate-props")
            # row_list.append("--calculate-relax-change")
            
        return row_list
