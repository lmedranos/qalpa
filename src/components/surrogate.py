from __future__ import annotations

import numpy as np


class FingerprintRF:
    """Bagged random-forest ensemble on fingerprints; sigma is the spread over all trees of all members."""

    def __init__(self, n_members: int = 5, n_trees: int = 30, seed: int = 0):
        self.n_members, self.n_trees, self.seed = int(n_members), int(n_trees), int(seed)
        self.members: list = []
        self.y_mean = self.y_std = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "FingerprintRF":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError("scikit-learn is required for the fingerprint_rf surrogate; add it to the environment") from exc
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=float)
        self.y_mean = Y.mean(axis=0)
        self.y_std = np.where(Y.std(axis=0) > 1e-12, Y.std(axis=0), 1.0)
        Yn = (Y - self.y_mean) / self.y_std
        train_y = Yn[:, 0] if Yn.shape[1] == 1 else Yn
        self.members = []
        for m in range(self.n_members):
            rf = RandomForestRegressor(n_estimators=self.n_trees, max_features="sqrt", min_samples_leaf=2,
                                       bootstrap=True, n_jobs=1, random_state=self.seed + m)
            self.members.append(rf.fit(X, train_y))
        return self

    def predict(self, X: np.ndarray) -> tuple:
        X = np.asarray(X, dtype=np.float32)
        per_tree = np.stack([t.predict(X).reshape(len(X), -1) for rf in self.members for t in rf.estimators_])
        mu = per_tree.mean(axis=0) * self.y_std + self.y_mean
        sigma = per_tree.std(axis=0) * self.y_std
        return mu, sigma


def build_surrogate(backend: str, n_members: int, n_trees: int, seed: int):
    if backend == "fingerprint_rf":
        return FingerprintRF(n_members, n_trees, seed)
    if backend == "xtb_fidelity":
        raise NotImplementedError("The xtb_fidelity surrogate backend is not implemented; use fingerprint_rf.")
    raise ValueError(f"Unknown surrogate backend {backend!r}")
