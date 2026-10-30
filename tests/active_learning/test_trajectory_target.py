import pytest

from src.utils.trajectory import Trajectory
from tests.active_learning.helpers import write_trajectory


def legacy_get_target(self, index):
    """Verbatim copy of Trajectory.get_target before the refactor."""
    if index < 0 or index >= len(self):
        raise IndexError(f"Index {index} out of range for trajectory with length {len(self)}")
    context = [self._props[p][index] for p in self.prop_names if p != "atNUM"]
    num_nodes = self._props["atNUM"][index] if "atNUM" in self.prop_names else None
    row_list = []
    if num_nodes is not None:
        row_list.append("--num-nodes")
        row_list.append(str(num_nodes))
    if context:
        row_list.append("--fixed-context")
        row_list.extend(str(c) for c in context)
    return row_list


@pytest.mark.parametrize("props", [["HLgap", "eMBD", "atNUM"], ["HLgap", "eMBD"], ["atNUM"], ["eMBD", "atNUM", "HLgap"]])
def test_refactored_get_target_matches_legacy(tmp_path, props):
    csv = tmp_path / "t.csv"
    write_trajectory(csv)
    traj = Trajectory(csv, prop_names=props, score_names=["SAScore"])
    for i in range(len(traj)):
        assert traj.get_target(i) == legacy_get_target(traj, i)


def test_matches_legacy_on_navigator_csv():
    from pathlib import Path

    csv = Path(__file__).resolve().parents[2] / "Navigator" / "trajectory_HLgap_eMBD_atNUM.csv"
    traj = Trajectory(csv, prop_names=["HLgap", "eMBD", "atNUM"], score_names=["SAScore"])
    assert len(traj) >= 4
    for i in range(len(traj)):
        assert traj.get_target(i) == legacy_get_target(traj, i)


def test_target_args_accepts_float_atom_count_and_out_of_range_raises(tmp_path):
    traj = write_trajectory(tmp_path / "t.csv")
    args = traj.target_args({"HLgap": 3.3, "eMBD": -2.0, "atNUM": 60.4})
    assert args == ["--num-nodes", "60", "--fixed-context", "3.3", "-2.0"]
    with pytest.raises(IndexError):
        traj.get_target(len(traj))
