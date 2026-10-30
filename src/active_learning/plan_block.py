from __future__ import annotations

from pathlib import Path

from src.active_learning.ids import read_targets, write_targets
from src.active_learning.query_planner import plan_targets, trajectory_box
from src.active_learning.state import ensure_validation_ids, load_state


def run_plan(*, traj, al: dict, hdf5_path, iter_dir, iteration: int, total_samples: int, seed: int,
             state_file, validation_file, log=print) -> list:
    """Write targets.json for the iteration (reused if it already exists) and fix the validation set once."""
    existing = read_targets(iter_dir)
    if existing is not None:
        log(f"  Reusing the plan in {Path(iter_dir) / 'targets.json'} (delete it to re-plan)")
        return existing[1]

    state = load_state(hdf5_path, traj.prop_names)
    if not Path(validation_file).exists():
        lo, hi = trajectory_box(traj, al["corridor_pad"])
        ids = ensure_validation_ids(validation_file, state, lo, hi, int(al["validation"]["size"]), seed)
        log(f"  Fixed validation set: {len(ids)} molecules")

    targets = plan_targets(traj, state, al, iteration, total_samples, seed, state_file)
    write_targets(iter_dir, iteration, al["query_strategy"], traj.prop_names, targets)
    for t in targets:
        log(f"  Target {t.id}: {t.props} x {t.n_samples}")
    return targets
