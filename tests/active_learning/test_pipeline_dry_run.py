"""Dry run of the forward and backward scripts with the heavy steps (srun, QM, relaxation) replaced by fakes."""
import importlib
import json
import sys
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

pytest.importorskip("hydra")
pytest.importorskip("sklearn")
pytest.importorskip("ase")

from tests.active_learning.helpers import write_h5, write_json_result, write_trajectory, write_xyz  # noqa: E402

N_SAMPLES, OVERSAMPLE, BUDGET = 20, 3, 12


class FakeFeaturizer:
    n_bits = 64

    def __init__(self, *a, **k):
        pass

    def features(self, key, atoms_fn, use_cache=True):
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        return key, rng.random(self.n_bits) < 0.3

    def save(self):
        pass


def make_cfg(tmp_path, al_enabled=True, keep_off_target=True, surrogate_props=None):
    exp = tmp_path / "Experiments" / "exp"
    (exp / "configs").mkdir(parents=True)
    write_trajectory(tmp_path / "traj.csv")
    # the 3-property helper writes atNUM too; the scripts here use two properties
    write_h5(tmp_path / "base.hdf5", n_base=80)
    cfg = {
        "exp_config_path": str(exp / "configs" / "experiment_config.yaml"),
        "base_dataset_path": str(tmp_path / "base.hdf5"),
        "data": {"exp_data_path": str(tmp_path / "Data" / "exp")},
        "hdf5": {"appending_flag": True, "hdf5_iter_flag": False, "extra_properties_flag": False},
        "query": {"props": ["HLgap", "eMBD"], "scores": ["SAScore"], "trajectory_csv": str(tmp_path / "traj.csv"),
                  "filter_scheme": [], "filter_threshold": 0.1},
        "loop": {"num_iterations": 5, "current_iter": 0},
        "pipeline": {"start_at": "split", "blocks": ["split", "train", "generate", "score_filter", "relax_geo",
                                                    "prop_calc", "prop_filter", "hdf5", "compress"]},
        "system": {"seed": 42},
        "EQdiffParam": {"num_samples": N_SAMPLES, "batch_size": 10, "num_epochs": 1, "properties_list": ["HLgap", "eMBD"],
                        "context_mapping": True, "num_context_features": 2,
                        "dataset_root": str(tmp_path / "Data" / "exp" / "diffusion_wdir" / "data"),
                        "model_root": str(tmp_path / "Data" / "exp" / "diffusion_wdir" / "model_registry"),
                        "load_ckpt_from_pretrained": None, "dataset": "aqm"},
        "EQdiff": {"split": "split.py", "train": "train.py", "generate": "generate.py"},
        "code_paths": {"configs": str(tmp_path / "configs")},
        "geometry": {"optimizer": "DFTB+", "calculate_forces": False, "model": None},
        "properties": {"calculator": "DFTB+", "model": None},
        "pipeline_blocks": {"GeoRelax": "relax.py", "PropCalc": "prop.py"},
        "max_parallel_tasks": 2, "processing_timeout_hours": 1,
        "al": {"enabled": al_enabled, "targets_per_iter": 3, "oversample": OVERSAMPLE, "budget": BUDGET,
               "keep_off_target": keep_off_target,
               "surrogate": {"n_members": 2, "n_trees": 10, "min_train": 20,
                             "props": surrogate_props or ["HLgap", "eMBD"]},
               "density": {"n_candidates": 300}, "validation": {"size": 10}},
    }
    c = OmegaConf.create(cfg)
    OmegaConf.set_struct(c, True)
    return c


def run_forward(cfg, monkeypatch):
    forward = importlib.import_module("src.scripts.forward")
    cmds = []

    def fake_run(cmd, check=True, cwd=None, env=None):
        cmds.append(list(cmd))
        save = Path(cmd[cmd.index("--save-dir") + 1]) if "--save-dir" in cmd else None
        if any(str(c).endswith("generate.py") for c in cmd):
            save.mkdir(parents=True, exist_ok=True)
            for i in range(int(cmd[cmd.index("--ngraphs") + 1])):
                write_xyz(save / f"mol_{i}.xyz")
        if any(str(c).endswith("train.py") for c in cmd):
            Path(save).mkdir(parents=True, exist_ok=True)
            (Path(save) / "best_mol_stab.ckpt").write_text("ckpt")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(forward.subprocess, "run", fake_run)
    forward.main(cfg)
    return cmds


def run_backward(cfg, monkeypatch):
    for name, attrs in {"calculate_sascore": {"sascore": lambda *a, **k: None},
                        "propcalc_preprocess": {"preprocess_fhi": lambda *a, **k: None},
                        "propcalc_postprocess": {"prop_postprocess": lambda *a, **k: None}}.items():
        monkeypatch.setitem(sys.modules, f"src.pipeline_blocks.{name}", types.SimpleNamespace(**attrs))
    sys.modules.pop("src.scripts.backward", None)
    backward = importlib.import_module("src.scripts.backward")
    monkeypatch.setattr("src.active_learning.acquire_block.MolFeaturizer", FakeFeaturizer)

    def fake_parallel(script, inp, out, *a, **k):
        out.mkdir(parents=True, exist_ok=True)
        if "relax" in str(script):
            for f in inp.iterdir():
                if f.suffix == ".xyz":
                    (out / f"opt-{f.name}").write_text(f.read_text())
        return []

    def fake_post(f, prop_calc_path, calc):
        h = zlib.crc32(f.stem.encode())
        write_json_result(Path(prop_calc_path) / f"{f.stem}.json", hlgap=2.5 + (h % 200) / 100, embd=-1.9)

    monkeypatch.setattr(backward, "node_parallelize", fake_parallel)
    monkeypatch.setattr(backward, "prop_postprocess", fake_post)
    backward.main(cfg)


def data_dir(cfg):
    return Path(cfg.data.exp_data_path)


def test_forward_disabled_keeps_single_trajectory_generation(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, al_enabled=False)
    cmds = run_forward(cfg, monkeypatch)
    gens = [c for c in cmds if any(str(x).endswith("generate.py") for x in c)]
    assert len(gens) == 1 and gens[0][gens[0].index("--fixed-context") + 1:][:2] == ["4.5", "-1.9"]
    split = next(c for c in cmds if any(str(x).endswith("split.py") for x in c))
    assert "--seed" not in split and "--validation-ids" not in split
    assert not (data_dir(cfg) / "iter_0" / "targets.json").exists()
    assert len(list((data_dir(cfg) / "iter_0" / "query").glob("*.xyz"))) == N_SAMPLES


def test_two_iterations_with_active_learning(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    exp = Path(cfg.exp_config_path).parent.parent

    # iteration 0: forward
    cmds = run_forward(cfg, monkeypatch)
    train = next(c for c in cmds if any(str(x).endswith("train.py") for x in c))
    i = train.index("--properties-list")
    assert train[i:i + 3] == ["--properties-list", "HLgap", "eMBD"] and "--context-mapping" in train
    assert "--num-context-features" in train
    split = next(c for c in cmds if any(str(x).endswith("split.py") for x in c))
    assert "--seed" in split and "--validation-ids" in split and (exp / "validation_ids.json").exists()
    it0 = data_dir(cfg) / "iter_0"
    plan = json.loads((it0 / "targets.json").read_text())
    assert len(plan["targets"]) == 1 and plan["targets"][0]["n_samples"] == N_SAMPLES * OVERSAMPLE
    meta = json.loads((it0 / "candidates_meta.json").read_text())
    assert len(meta) == N_SAMPLES * OVERSAMPLE and all(k.startswith("t0_mol_") for k in meta)
    assert len(list((it0 / "query").glob("*.xyz"))) == N_SAMPLES * OVERSAMPLE

    # iteration 0: backward
    run_backward(cfg, monkeypatch)
    assert cfg.loop.current_iter == 1
    acq = pd.read_csv(it0 / "acquisition.csv")
    assert acq["selected"].sum() == BUDGET and len(acq) == N_SAMPLES * OVERSAMPLE
    assert len(list((it0 / "query").glob("*.xyz"))) == BUDGET
    out = next((data_dir(cfg) / "hdf5").glob("*-output.hdf5"))
    with h5py.File(out, "r") as f:
        assert set(f) == {"base", "G-active-loop-0"}
        grp = f["G-active-loop-0"]
        assert len(grp) == BUDGET
        hits = [int(grp[k]["al_hit"][0]) for k in grp]
        assert 0 in hits, "off-target molecules must be kept"
        assert all(k.startswith("opt-t0_mol_") for k in grp)
    metrics = pd.read_csv(exp / "metrics.csv")
    assert len(metrics) == 1 and metrics.loc[0, "n_labelled"] == BUDGET

    # iteration 1: forward plans several targets from the data
    cmds = run_forward(cfg, monkeypatch)
    it1 = data_dir(cfg) / "iter_1"
    plan = json.loads((it1 / "targets.json").read_text())
    assert len(plan["targets"]) == 3 and sum(t["n_samples"] for t in plan["targets"]) == N_SAMPLES * OVERSAMPLE
    gens = [c for c in cmds if any(str(x).endswith("generate.py") for x in c)]
    assert len(gens) == 3
    queries = sorted(p.name for p in (it1 / "query").glob("*.xyz"))
    assert len(queries) == N_SAMPLES * OVERSAMPLE and {q.split("_")[0] for q in queries} == {"t0", "t1", "t2"}

    run_backward(cfg, monkeypatch)
    assert cfg.loop.current_iter == 2
    assert pd.read_csv(it1 / "acquisition.csv")["selected"].sum() <= BUDGET
    assert len(pd.read_csv(exp / "metrics.csv")) == 2


def test_keep_off_target_false_keeps_only_hits(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, keep_off_target=False)
    run_forward(cfg, monkeypatch)
    run_backward(cfg, monkeypatch)
    out = next((data_dir(cfg) / "hdf5").glob("*-output.hdf5"))
    with h5py.File(out, "r") as f:
        grp = f["G-active-loop-0"]
        assert len(grp) > 0 and all(int(grp[k]["al_hit"][0]) == 1 for k in grp)


def test_surrogate_can_use_subset_of_query_properties(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, surrogate_props=["HLgap"])
    run_forward(cfg, monkeypatch)
    run_backward(cfg, monkeypatch)
    acq = pd.read_csv(data_dir(cfg) / "iter_0" / "acquisition.csv")
    assert acq["mu_HLgap"].notna().any()
    assert acq["mu_eMBD"].isna().all()


def test_loop_stop_flag_makes_both_scripts_exit_cleanly(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    with __import__("omegaconf").open_dict(cfg.loop):
        cfg.loop.stop = True
    assert run_forward(cfg, monkeypatch) == []
    run_backward(cfg, monkeypatch)
    assert cfg.loop.current_iter == 0 and not (data_dir(cfg) / "iter_0").exists()
