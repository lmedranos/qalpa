import math

import h5py
import numpy as np
import pytest

from src.active_learning.hdf5_block import prepare_al_labels
from src.active_learning.ids import Target, canonical_id, write_candidates_meta, write_targets
from src.active_learning.labels import is_hit, valid_qm_result
from tests.active_learning.helpers import PROPS, write_json_result, write_xyz


def _setup(tmp_path):
    q, p, it = tmp_path / "query", tmp_path / "prop_calc", tmp_path / "iter_0"
    for d in (q, p, it):
        d.mkdir()
    targets = [Target(0, {"HLgap": 3.0, "eMBD": -1.9, "atNUM": 5}, 10), Target(1, {"HLgap": 4.0, "eMBD": -1.9, "atNUM": 5}, 10)]
    write_targets(it, 0, "kde_map", PROPS, targets)
    meta = {}
    cases = {"t0_mol_0": (0, 3.05), "t0_mol_1": (0, 3.9), "t1_mol_0": (1, 4.0), "t1_mol_1": (1, None)}
    for cid, (tid, hl) in cases.items():
        write_xyz(q / f"opt-{cid}.xyz")
        if hl is not None:
            write_json_result(p / f"opt-{cid}.json", hlgap=hl)
        meta[cid] = {"target_id": tid}
    write_candidates_meta(it, meta)
    return q, p, it


def test_canonical_id_strips_prefix_and_extension():
    assert canonical_id("opt-t0_mol_3.xyz") == "t0_mol_3" == canonical_id("t0_mol_3.json")


def test_valid_qm_result_cases(tmp_path):
    xyz, js = tmp_path / "m.xyz", tmp_path / "m.json"
    write_xyz(xyz)
    assert valid_qm_result(js, xyz, PROPS)[0] is False          # missing json
    write_json_result(js)
    assert valid_qm_result(js, xyz, PROPS) == (True, "")
    write_json_result(js, hlgap=math.nan)
    ok, reason = valid_qm_result(js, xyz, PROPS)
    assert not ok and "non-finite" in reason
    write_json_result(js)
    write_xyz(xyz, symbols=("C", "H", "H", "H", "Br"))
    ok, reason = valid_qm_result(js, xyz, PROPS)
    assert not ok and "unsupported elements" in reason


def test_is_hit_uses_relative_tolerance_and_ignores_atnum():
    req = {"HLgap": 3.0, "eMBD": -2.0, "atNUM": 10}
    assert is_hit({"HLgap": 3.2, "eMBD": -2.1, "atNUM": 99}, req, 0.1, PROPS)
    assert not is_hit({"HLgap": 3.5, "eMBD": -2.0, "atNUM": 10}, req, 0.1, PROPS)


def test_keep_off_target_true_labels_everything_valid(tmp_path):
    q, p, it = _setup(tmp_path)
    meta = prepare_al_labels(query_dir=q, prop_dir=p, iter_dir=it, prop_names=PROPS, threshold=0.1, iteration=0,
                             keep_off_target=True, log=lambda *_: None)
    assert set(meta) == {"t0_mol_0", "t0_mol_1", "t1_mol_0"}      # t1_mol_1 has no QM result
    assert meta["t0_mol_0"]["hit"] == 1 and meta["t0_mol_1"]["hit"] == 0
    assert meta["t1_mol_0"]["target_id"] == 1 and meta["t1_mol_0"]["requested"][0] == 4.0
    assert not (q / "opt-t1_mol_1.xyz").exists() and (it / "rejected" / "opt-t1_mol_1.xyz").exists()
    assert (q / "opt-t0_mol_1.xyz").exists()


def test_keep_off_target_false_keeps_hits_only(tmp_path):
    q, p, it = _setup(tmp_path)
    meta = prepare_al_labels(query_dir=q, prop_dir=p, iter_dir=it, prop_names=PROPS, threshold=0.1, iteration=0,
                             keep_off_target=False, log=lambda *_: None)
    assert set(meta) == {"t0_mol_0", "t1_mol_0"}
    assert sorted(f.name for f in q.glob("*.xyz")) == ["opt-t0_mol_0.xyz", "opt-t1_mol_0.xyz"]


def test_run_to_hdf5_writes_al_fields_and_skips_bad_molecules(tmp_path):
    pytest.importorskip("ase")
    from src.pipeline_blocks.to_hdf5 import run_to_hdf5

    q, p, it = _setup(tmp_path)
    write_xyz(q / "opt-t0_mol_9.xyz", symbols=("C", "H", "H", "H", "Br"))      # unsupported element
    write_json_result(p / "opt-t0_mol_9.json")
    write_xyz(q / "opt-t0_mol_8.xyz")                                         # no json at all
    meta = prepare_al_labels(query_dir=q, prop_dir=p, iter_dir=it, prop_names=PROPS, threshold=0.1, iteration=0,
                             keep_off_target=True, log=lambda *_: None)
    h5 = tmp_path / "out.hdf5"
    with h5py.File(h5, "w") as f:
        f.create_group("base")
    run_to_hdf5(q, p, True, h5, True, None, None, None, 0, al_meta=meta, al_props=PROPS)
    with h5py.File(h5, "r") as f:
        assert set(f) == {"base", "G-active-loop-0"}
        grp = f["G-active-loop-0"]
        assert set(grp) == {"opt-t0_mol_0", "opt-t0_mol_1", "opt-t1_mol_0"}   # bad ones leave no partial group
        m = grp["opt-t0_mol_1"]
        assert int(m["al_hit"][0]) == 0 and int(m["al_target_id"][0]) == 0
        assert m.attrs["al_props"] == ",".join(PROPS)
        assert float(m["HLgap"][0]) == 3.9
    assert not (tmp_path / "out.hdf5.bak").exists()


def test_run_to_hdf5_restores_backup_on_failure(tmp_path, monkeypatch):
    pytest.importorskip("ase")
    import src.pipeline_blocks.to_hdf5 as mod

    q, p, it = _setup(tmp_path)
    h5 = tmp_path / "out.hdf5"
    with h5py.File(h5, "w") as f:
        f.create_group("base").create_dataset("x", data=np.arange(3))

    def boom(*a, **k):
        with h5py.File(h5, "a") as f:
            f.create_group("half-written")
        raise RuntimeError("killed")

    monkeypatch.setattr(mod, "_write_hdf5", boom)
    with pytest.raises(RuntimeError):
        mod.run_to_hdf5(q, p, True, h5, True, None, None, None, 0)
    with h5py.File(h5, "r") as f:
        assert set(f) == {"base"}
