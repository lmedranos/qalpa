import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("openbabel")

from src.active_learning.features import MolFeaturizer, atoms_to_smiles, smiles_to_fp
from src.active_learning.labels import read_xyz
from tests.active_learning.helpers import write_xyz


def test_methane_xyz_to_smiles_and_fingerprint(tmp_path):
    write_xyz(tmp_path / "m.xyz")
    symbols, pos = read_xyz(tmp_path / "m.xyz")
    assert atoms_to_smiles(symbols, pos) == "C"
    fp = smiles_to_fp("CCO")
    assert fp.dtype == bool and fp.shape == (2048,) and fp.any()


def test_featurizer_cache_round_trip_and_failures(tmp_path):
    write_xyz(tmp_path / "m.xyz")
    atoms = lambda: read_xyz(tmp_path / "m.xyz")  # noqa: E731
    cache = tmp_path / "al_cache" / "fingerprints.npz"
    f = MolFeaturizer(cache)
    smi, fp = f.features("base/m1", atoms)
    bad_smi, bad_fp = f.features("base/bad", lambda: (["Xx"], [[0, 0, 0]]))
    assert bad_smi is None and bad_fp is None
    f.save()
    g = MolFeaturizer(cache)
    smi2, fp2 = g.features("base/m1", lambda: pytest.fail("cache miss"))
    assert smi2 == smi == "C" and np.array_equal(fp, fp2)
    assert g.features("base/bad", lambda: pytest.fail("cache miss")) == (None, None)
