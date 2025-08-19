import pandas as pd
import numpy as np
from pathlib import Path

class Trajectory:
    def __init__(self, directory, child_names):
        self.names = child_names
        directory = Path(directory)

        # find the trajectory file
        csv_files = list(directory.glob("trajectory_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No trajectory CSV files found in {directory}")
        csv_path = csv_files[0]

        # read only requested columns
        df = pd.read_csv(csv_path, usecols=self.names)

        # store axes
        self._axes = {}
        for name in self.names:
            col = df[name].to_numpy()
            if "NUM" in name.upper():  # cast integer-like axes
                col = col.astype(int)
            self._axes[name] = col

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
