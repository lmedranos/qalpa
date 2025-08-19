import pandas as pd
import numpy as np


class Trajectory:
    def __init__(self, csv_path, child_names):
        self.names = child_names
        df = pd.read_csv(csv_path)
        self._axes = {}
        for i, name in enumerate(child_names):
            self._axes[name] = df.iloc[:, i].to_numpy()

    def __getitem__(self, name):
        return self._axes[name]
    
    def __getattr__(self, name):
        if name in self._axes:
            return self._axes[name]
        raise AttributeError(f"'Trajectory' object has no attribute '{name}'")
    
    def __len__(self):
        return len(next(iter(self._axes.values())))

    def get_target(self, index, prefix="--"):
        """
        Returns a flattened list like [--axis1, value1, --axis2, value2, ...]
        for all axes at the given row index.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for trajectory with length {len(self)}")
    
        row_list = []
        for name in self.names:
            row_list.append(f"{prefix}{name}")
            row_list.append(str(self._axes[name][index]))
        row_list.append("--fixed-context")
        row_list.extend(self.names)
        return row_list

   