import numpy as np
import pytest

from src.active_learning.config import DEFAULTS, _merge
from src.active_learning.query_planner import (
    Corridor,
    allocate,
    plan_frontier,
    plan_kde_map,
    plan_targets,
    plan_trajectory,
    trajectory_box,
)
from src.active_learning.state import ALState
from tests.active_learning.helpers import PROPS, write_trajectory


def al_cfg(**over):
    return _merge(DEFAULTS, {"enabled": True, "targets_per_iter": 3, **over})


def make_state(prop_rows, requested=None, hit=None, iteration=None):
    props = np.array(prop_rows, dtype=float)
    n = len(props)
    req = np.full((n, 3), np.nan) if requested is None else np.array(requested, dtype=float)
    return ALState(list(PROPS), np.array([f"g/m{i}" for i in range(n)], dtype=object), props,
                   np.full(n, -1) if iteration is None else np.array(iteration),
                   np.full(n, -1), req, np.full(n, -1) if hit is None else np.array(hit))


def dense_state_with_hole(rng, hole_lo=3.6):
    """Labelled data covering HLgap 2.4-3.4 and 4.0-4.6 but not the gap in between."""
    hl = np.concatenate([rng.uniform(2.4, 3.4, 300), rng.uniform(4.0, 4.6, 300)])
    rows = np.column_stack([hl, rng.normal(-1.9, 0.02, 600), np.full(600, 60.0)])
    return make_state(rows)


def test_trajectory_strategy_matches_row_and_clamps(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv")
    t = plan_trajectory(traj, 1, 100)[0]
    assert t.props["HLgap"] == 3.4 and t.n_samples == 100
    assert plan_trajectory(traj, 9, 100)[0].props["HLgap"] == 3.1


def test_corridor_needs_two_rows(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv", rows=[(3.0, -1.9, 60)])
    with pytest.raises(ValueError, match="at least 2"):
        Corridor.from_trajectory(traj, 0.1, None, 0)
    lo, hi = trajectory_box(traj, 0.1)
    assert np.all(lo < hi)


def test_corridor_samples_inside_padded_box_and_snaps_atoms(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv")
    c = Corridor.from_trajectory(traj, 0.1, None, 0)
    pts = c.sample(500, np.random.default_rng(0))
    lo, hi = c.bounds()
    assert np.all(pts >= lo - 1e-9) and np.all(pts <= hi + 1e-9)
    assert np.all(pts[:, 2] == np.round(pts[:, 2])) and pts[:, 2].min() >= 1


def test_lookahead_limits_corridor(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv")
    c = Corridor.from_trajectory(traj, 0.1, 1, 0)
    assert len(c.points) == 2


def test_allocate_sums_and_minimum():
    a = allocate(100, [3.0, 1.0, 0.0001])
    assert a.sum() == 100 and a.min() >= 1
    with pytest.raises(ValueError):
        allocate(2, [1, 1, 1])


def test_kde_map_prefers_empty_region(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv", rows=[(2.5, -1.9, 60), (4.5, -1.9, 60)])
    rng = np.random.default_rng(0)
    state = dense_state_with_hole(rng)
    # some past records so kde_map is active; all hits near the dense parts
    state.requested[:20] = np.array([[3.0, -1.9, 60]] * 20)
    state.hit[:20] = 1
    state.iteration[:20] = 0
    targets = plan_kde_map(traj, state, al_cfg(targets_per_iter=1), 0, 100, np.random.default_rng(1))
    assert 3.4 < targets[0].props["HLgap"] < 4.0, targets[0].props


def test_kde_map_drops_unreachable_but_keeps_unexplored(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv", rows=[(2.5, -1.9, 60), (4.5, -1.9, 60)])
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.uniform(2.4, 4.6, 400), rng.normal(-1.9, 0.02, 400), np.full(400, 60.0)])
    state = make_state(rows)
    # 100 molecules requested at HLgap=4.4 that never hit
    state.requested[:100] = np.array([[4.4, -1.9, 60]] * 100)
    state.hit[:100] = 0
    state.iteration[:100] = 0
    out = plan_kde_map(traj, state, al_cfg(targets_per_iter=3, density=dict(min_reach=0.2)), 1, 90,
                       np.random.default_rng(2))
    assert all(abs(t.props["HLgap"] - 4.4) > 0.05 for t in out)


def test_kde_map_deterministic_distinct_and_sums(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv", rows=[(2.5, -1.9, 60), (4.5, -1.9, 60)])
    state = dense_state_with_hole(np.random.default_rng(0))
    state.requested[:20] = np.array([[3.0, -1.9, 60]] * 20)
    state.hit[:20] = 1
    state.iteration[:20] = 0
    a = plan_kde_map(traj, state, al_cfg(), 1, 120, np.random.default_rng(5))
    b = plan_kde_map(traj, state, al_cfg(), 1, 120, np.random.default_rng(5))
    assert [t.props for t in a] == [t.props for t in b]
    assert sum(t.n_samples for t in a) == 120 and len(a) == 3
    gaps = sorted(t.props["HLgap"] for t in a)
    assert gaps[0] != gaps[1] != gaps[2]


def test_plan_targets_iteration_zero_falls_back_to_trajectory(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv")
    state = make_state(np.column_stack([np.linspace(2, 4, 50), np.full(50, -1.9), np.full(50, 60.0)]))
    out = plan_targets(traj, state, al_cfg(), 0, 100, seed=42)
    assert len(out) == 1 and out[0].meta["strategy"] == "trajectory"


def test_frontier_advances_only_above_threshold(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv")
    sf = tmp_path / "al_state.json"
    row0 = [4.5, -1.9, 61]
    state = make_state([[4.5, -1.9, 61]] * 10, requested=[row0] * 10, hit=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       iteration=[0] * 10)
    t0 = plan_frontier(traj, state, al_cfg(advance_hit_rate=0.3), 1, 50, sf)[0]
    assert t0.meta["row"] == 0
    state.hit[:] = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    t1 = plan_frontier(traj, state, al_cfg(advance_hit_rate=0.3), 2, 50, sf)[0]
    assert t1.meta["row"] == 1
    # re-planning the same iteration is idempotent
    assert plan_frontier(traj, state, al_cfg(advance_hit_rate=0.3), 2, 50, sf)[0].meta["row"] == 1


def test_frontier_never_passes_last_row(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv", rows=[(4.5, -1.9, 61), (3.4, -1.9, 59)])
    sf = tmp_path / "al_state.json"
    last = [3.4, -1.9, 59]
    state = make_state([last] * 5, requested=[last] * 5, hit=[1] * 5, iteration=[0] * 5)
    sf.write_text('{"frontier": {"row": 1, "iteration": 0}}')
    assert plan_frontier(traj, state, al_cfg(advance_hit_rate=0.3), 1, 50, sf)[0].meta["row"] == 1
