import numpy as np
import pandas as pd

from src.active_learning.metrics import append_metrics, compute_metrics, early_stop
from src.active_learning.query_planner import Corridor
from src.active_learning.state import ALState
from tests.active_learning.helpers import PROPS, write_trajectory


def _state():
    n = 20
    rng = np.random.default_rng(0)
    props = np.column_stack([rng.normal(3, 0.3, n), rng.normal(-1.9, 0.05, n), np.full(n, 10.0)])
    req = np.tile([3.0, -1.9, 10.0], (n, 1))
    keys = np.array([f"G-active-loop-1/opt-t{i % 2}_mol_{i}" for i in range(n)], dtype=object)
    return ALState(list(PROPS), keys, props, np.full(n, 1), np.arange(n) % 2, req, (np.arange(n) % 4 == 0).astype(int))


def test_hit_rate_overall_and_by_target():
    row = compute_metrics(_state(), 1, None, None, 5, 0)
    assert row["n_labelled"] == 20 and np.isclose(row["hit_rate"], 0.25)
    assert row["hit_rate_by_target"].startswith("0:")


def test_surrogate_errors_and_calibration_from_acquisition_table():
    st = _state()
    names = [f"t{i % 2}_mol_{i}.xyz" for i in range(20)]
    mu = st.props[:, :2] + 0.1
    acq = pd.DataFrame({"name": names, "mu_HLgap": mu[:, 0], "sigma_HLgap": 0.2, "mu_eMBD": mu[:, 1],
                        "sigma_eMBD": 0.2, "score": np.arange(20.0), "selected": True})
    row = compute_metrics(st, 1, acq, None, 5, 0)
    assert row["n_selected"] == 20 and np.isclose(row["mean_acq_score"], 9.5)
    assert row["surrogate_nmae"] > 0 and row["calib_1sigma"] == 1.0 and row["calib_2sigma"] == 1.0


def test_coverage_uses_requested_density(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv", rows=[(3.0, -1.9, 10), (3.1, -1.9, 10)])
    corridor = Corridor.from_trajectory(traj, 0.05, None, 0)
    row = compute_metrics(_state(), 1, None, corridor, 5, 0)
    assert 0.0 <= row["coverage"] <= 1.0


def test_append_metrics_and_early_stop(tmp_path):
    path = tmp_path / "metrics.csv"
    hits = [0.1, 0.3, 0.31, 0.31, 0.31]
    errs = [1.0, 0.5, 0.5, 0.5, 0.5]
    for i, (h, e) in enumerate(zip(hits, errs)):
        append_metrics(path, {"iteration": i, "hit_rate": h, "surrogate_nmae": e})
        stop, _ = early_stop(path, patience=3, min_delta=0.05)
        assert stop == (i >= 4)
    stop, reason = early_stop(path, patience=3, min_delta=0.05)
    assert stop and "below" in reason


def test_no_early_stop_while_improving(tmp_path):
    path = tmp_path / "metrics.csv"
    for i in range(6):
        append_metrics(path, {"iteration": i, "hit_rate": 0.1 * (i + 1), "surrogate_nmae": 1.0 / (i + 1)})
    assert early_stop(path, patience=3, min_delta=0.05)[0] is False
    assert early_stop(tmp_path / "none.csv", 3, 0.05) == (False, "")
