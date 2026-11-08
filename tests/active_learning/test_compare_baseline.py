import pytest

pytest.importorskip("sklearn")

from src.active_learning.compare_baseline import compare


def test_kde_map_with_score_beats_trajectory_with_random_on_a_metric():
    res = compare(rounds=6, seeds=3)
    base, new = res["trajectory+random"], res["kde_map+score"]
    assert (new["bins_covered"] > base["bins_covered"] or new["hit_rate"] > base["hit_rate"]
            or new["surrogate_rmse"] < base["surrogate_rmse"])
