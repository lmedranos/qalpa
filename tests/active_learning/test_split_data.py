import ast
import json
import os
import random
from pathlib import Path

SPLIT = Path(__file__).resolve().parents[2] / "src/eqdiff/e3mol_aqm/experiments/data/aqm/split_data.py"


def _load_partition():
    """Extract `partition` from split_data.py without importing torch/rdkit."""
    tree = ast.parse(SPLIT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "partition")
    ns = {"random": random, "os": os, "json": json}
    exec(compile(ast.Module([fn], []), str(SPLIT), "exec"), ns)
    return ns["partition"]


def _opt_props():
    tree = ast.parse(SPLIT.read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "optional_properties":
            return ast.literal_eval(n.value)


def test_seeded_split_is_reproducible_and_unseeded_path_unchanged():
    partition = _load_partition()
    keys = [f"g/m{i}" for i in range(100)]
    a = partition(list(range(100)), keys, seed=7)
    b = partition(list(range(100)), keys, seed=7)
    assert a == b and len(a[0]) == 82 and len(a[1]) == 10 and len(a[2]) == 8
    assert sorted(a[0] + a[1] + a[2]) == list(range(100))


def test_validation_ids_never_in_train_or_test(tmp_path):
    partition = _load_partition()
    keys = [f"g/m{i}" for i in range(100)]
    ids = tmp_path / "validation_ids.json"
    ids.write_text(json.dumps([keys[i] for i in (3, 5, 8, 13)]))
    for seed in (1, 2, 3):
        train, val, test = partition(list(range(100)), keys, seed=seed, validation_ids=str(ids))
        assert sorted(val) == [3, 5, 8, 13]
        assert not {3, 5, 8, 13} & set(train + test)
        assert len(train) + len(val) + len(test) == 100


def test_missing_validation_file_falls_back_to_plain_split(tmp_path):
    partition = _load_partition()
    keys = [f"g/m{i}" for i in range(50)]
    train, val, test = partition(list(range(50)), keys, seed=1, validation_ids=str(tmp_path / "missing.json"))
    assert len(train) + len(val) + len(test) == 50 and len(val) == 5


def test_optional_properties_are_the_dftb_gaps():
    assert _opt_props() == {"eKIN", "eXC", "eXX"}
