import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.active_learning.surrogate import FingerprintRF, build_surrogate


def _data(n, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 128)) < 0.2).astype(float)
    w = np.linspace(-1, 1, 128)
    y = X @ w + shift
    return X, np.column_stack([y, 0.5 * y])


def test_error_decreases_with_more_data():
    Xt, Yt = _data(200, seed=99)
    errs = []
    for n in (30, 400):
        X, Y = _data(n)
        mu, _ = FingerprintRF(n_members=2, n_trees=20, seed=0).fit(X, Y).predict(Xt)
        errs.append(np.mean(np.abs(mu - Yt)))
    assert errs[1] < errs[0]


def test_sigma_larger_out_of_distribution():
    X, Y = _data(300)
    model = FingerprintRF(n_members=3, n_trees=30, seed=0).fit(X, Y)
    rng = np.random.default_rng(5)
    ood = (rng.random((100, 128)) < 0.8).astype(float)
    _, s_in = model.predict(X[:100])
    _, s_out = model.predict(ood)
    assert s_out.mean() > s_in.mean()


def test_predict_shapes_and_backend_factory():
    X, Y = _data(60)
    mu, sigma = build_surrogate("fingerprint_rf", 2, 10, 0).fit(X, Y).predict(X[:7])
    assert mu.shape == sigma.shape == (7, 2) and np.all(sigma >= 0)
    with pytest.raises(NotImplementedError):
        build_surrogate("xtb_fidelity", 1, 1, 0)
    with pytest.raises(ValueError):
        build_surrogate("nope", 1, 1, 0)
