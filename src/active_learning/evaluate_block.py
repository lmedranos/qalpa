from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.active_learning.acquire_block import ACQUISITION_FILE
from src.active_learning.metrics import append_metrics, compute_metrics, early_stop
from src.active_learning.query_planner import Corridor
from src.active_learning.state import load_state


def run_evaluate(*, traj, al: dict, hdf5_path, iter_dir, iteration: int, seed: int, metrics_file, log=print) -> tuple:
    """Append this iteration's metrics; returns (row, stop, reason) where stop reflects the early-stop rule."""
    state = load_state(hdf5_path, traj.prop_names)
    acq_path = Path(iter_dir) / ACQUISITION_FILE
    acq = pd.read_csv(acq_path) if acq_path.exists() else None
    corridor = None
    if len(traj) >= 2:
        corridor = Corridor.from_trajectory(traj, al["corridor_pad"], al["corridor_lookahead"], iteration)
    row = compute_metrics(state, iteration, acq, corridor, al["min_neff"], seed)
    append_metrics(metrics_file, row)
    log(f"  Hit rate {row['hit_rate']}, surrogate nMAE {row['surrogate_nmae']}, coverage {row['coverage']}")

    stop, reason = False, ""
    es = al["early_stop"]
    if es["enabled"]:
        stop, reason = early_stop(metrics_file, int(es["patience"]), float(es["min_delta"]))
    return row, stop, reason
