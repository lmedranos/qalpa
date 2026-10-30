import numpy as np
import h5py

from src.active_learning.state import (
    ALState,
    ensure_validation_ids,
    load_state,
    select_validation_ids,
)
from tests.active_learning.helpers import PROPS, write_h5


def test_legacy_file_without_al_fields(tmp_path):
    path = write_h5(tmp_path / "legacy.hdf5", n_base=20)
    st = load_state(path, PROPS)
    assert len(st) == 20 and st.labelled_mask().all() and not st.record_mask().any()
    assert (st.iteration == -1).all() and np.isnan(st.requested).all()
    assert st.props[:, 2].min() >= 8  # atNUM comes from the atom count


def test_new_schema_records_and_no_top_level_metadata_groups(tmp_path):
    rec = [("t0_mol_1", 3.1, -1.9, {"iteration": 0, "target_id": 2, "requested": [3.0, -1.9, 10.0], "hit": 1}),
           ("t0_mol_2", 4.0, -1.8, {"iteration": 0, "target_id": 2, "requested": [3.0, -1.9, 10.0], "hit": 0})]
    path = write_h5(tmp_path / "new.hdf5", n_base=5, records=rec)
    with h5py.File(path, "r") as f:
        assert set(f) == {"base", "G-active-loop-0"}
    st = load_state(path, PROPS)
    assert st.record_mask().sum() == 2
    assert st.hit[st.record_mask()].tolist() == [1, 0] and (st.target_id[st.record_mask()] == 2).all()
    assert np.allclose(st.requested[st.record_mask()][0], [3.0, -1.9, 10.0])


def test_requested_columns_follow_saved_prop_names(tmp_path):
    rec = [("a", 3.0, -1.9, {"iteration": 1, "target_id": 0, "requested": [3.0, -1.9, 10.0], "hit": 1})]
    path = write_h5(tmp_path / "x.hdf5", n_base=2, records=rec)
    st = load_state(path, ["eMBD", "HLgap", "atNUM"])
    row = st.requested[st.record_mask()][0]
    assert np.allclose(row, [-1.9, 3.0, 10.0])


def test_missing_file_gives_empty_state(tmp_path):
    st = load_state(tmp_path / "nope.hdf5", PROPS)
    assert len(st) == 0 and st.last_iteration() == -1


def test_validation_ids_inside_box_and_fixed_after_first_call(tmp_path):
    path = write_h5(tmp_path / "d.hdf5", n_base=80)
    st = load_state(path, PROPS)
    lo, hi = np.array([2.9, -2.5, 0]), np.array([3.4, -1.5, 100])
    vf = tmp_path / "validation_ids.json"
    ids = ensure_validation_ids(vf, st, lo, hi, 10, seed=1)
    assert 0 < len(ids) <= 10
    idx = {str(k): i for i, k in enumerate(st.keys)}
    assert all(lo[0] <= st.props[idx[k], 0] <= hi[0] for k in ids)
    # later calls with different inputs return the saved ids unchanged
    assert ensure_validation_ids(vf, ALState.empty(PROPS), lo, hi, 5, seed=9) == ids
    assert select_validation_ids(st, lo, hi, 10, 1) == ids
