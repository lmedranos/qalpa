from __future__ import annotations

import numpy as np

_TINY = 1e-300


def robust_center_scale(Y: np.ndarray) -> tuple:
    """Mean and MAD-based scale per column (falls back to std, then 1)."""
    Y = np.asarray(Y, dtype=float)
    center = Y.mean(axis=0)
    mad = np.mean(np.abs(Y - center), axis=0) * 1.2533  # mean absolute deviation -> std for a Gaussian
    std = Y.std(axis=0)
    scale = np.where(mad > 1e-12, mad, std)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def _gauss(Zq: np.ndarray, Z: np.ndarray, h: np.ndarray, weights: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Sum_i w_i N(zq; z_i, diag(h^2)) for every query row, evaluated in chunks."""
    d = Z.shape[1]
    norm = (2.0 * np.pi) ** (-d / 2.0) / np.prod(h)
    out = np.empty(len(Zq))
    Zs, Zqs = Z / h, Zq / h
    for s in range(0, len(Zq), chunk):
        q = Zqs[s:s + chunk]
        d2 = np.zeros((len(q), len(Zs)))
        for j in range(d):
            d2 += (q[:, j:j + 1] - Zs[None, :, j]) ** 2
        out[s:s + chunk] = np.exp(-0.5 * d2) @ weights
    return norm * out


class PropertyDensity:
    """Weighted Gaussian KDE in standardized property space with a per-axis bandwidth floor."""

    def __init__(self, Y, weights=None, bandwidth="scott", min_bandwidth=None, logp_floor: float = -20.0):
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        w = np.ones(len(Y)) if weights is None else np.asarray(weights, dtype=float)
        keep = np.isfinite(Y).all(axis=1) & np.isfinite(w) & (w > 0)
        Y, w = Y[keep], w[keep]
        if len(Y) == 0:
            raise ValueError("PropertyDensity needs at least one finite point")
        self.center, self.scale = robust_center_scale(Y)
        self.Z = (Y - self.center) / self.scale
        self.w = w
        self.logp_floor = float(logp_floor)
        d = Y.shape[1]
        n_eff = w.sum() ** 2 / np.sum(w ** 2)
        h = n_eff ** (-1.0 / (d + 4)) if bandwidth == "scott" else float(bandwidth)
        h = np.full(d, max(h, 1e-3))
        if min_bandwidth is not None:
            h = np.maximum(h, np.asarray(min_bandwidth, dtype=float) / self.scale)
        self.h = h

    @property
    def total_weight(self) -> float:
        return float(self.w.sum())

    def standardize(self, Y) -> np.ndarray:
        return (np.atleast_2d(np.asarray(Y, dtype=float)) - self.center) / self.scale

    def kernel_sum(self, Y) -> np.ndarray:
        return _gauss(self.standardize(Y), self.Z, self.h, self.w)

    def density(self, Y) -> np.ndarray:
        return self.kernel_sum(Y) / self.total_weight

    def logp(self, Y) -> np.ndarray:
        return np.maximum(np.log(self.density(Y) + _TINY), self.logp_floor)

    def add_points(self, Y, w=1.0) -> None:
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        wv = np.full(len(Y), float(w)) if np.isscalar(w) else np.asarray(w, dtype=float)
        self.Z = np.vstack([self.Z, self.standardize(Y)])
        self.w = np.concatenate([self.w, wv])


class CandidateDensity:
    """Density at fixed query points that can absorb newly selected points without re-evaluating the dataset."""

    def __init__(self, dens: PropertyDensity, Yq):
        self.dens = dens
        self.Zq = dens.standardize(Yq)
        self.total = dens.kernel_sum(Yq)
        self.W = dens.total_weight

    def add(self, y, w: float = 1.0) -> None:
        z = self.dens.standardize(y)
        self.total = self.total + w * _gauss(self.Zq, z, self.dens.h, np.ones(1))
        self.W += w

    def logp(self) -> np.ndarray:
        return np.maximum(np.log(self.total / self.W + _TINY), self.dens.logp_floor)


def kernel_regress(X_req, values, X_query, center, scale, bandwidth=None, prior: float = 0.0,
                   min_bandwidth: float = 0.15, chunk: int = 256) -> tuple:
    """Nadaraya-Watson estimate of `values` at X_query plus n_eff (sample count within the kernel).

    Rows of X_req that repeat are merged, so many molecules requested at one target act as one kernel centre.
    """
    X_req = np.atleast_2d(np.asarray(X_req, dtype=float))
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(X_req) == 0:
        return np.full(len(X_query), prior), np.zeros(len(X_query))
    Zr = (X_req - center) / scale
    uniq, inv = np.unique(np.round(Zr, 9), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, values)
    d = uniq.shape[1]
    h = max(len(uniq) ** (-1.0 / (d + 4)), min_bandwidth) if bandwidth is None else float(bandwidth)
    Zq = (X_query - center) / scale
    est = np.empty(len(Zq))
    n_eff = np.empty(len(Zq))
    for s in range(0, len(Zq), chunk):
        diff = (Zq[s:s + chunk, None, :] - uniq[None, :, :]) / h
        k = np.exp(-0.5 * np.einsum("qnd,qnd->qn", diff, diff))
        den = k @ counts
        n_eff[s:s + chunk] = den
        num = k @ sums
        est[s:s + chunk] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), prior)
    return est, n_eff


def repulsion(chosen, cand, scale, h: float = 0.5) -> np.ndarray:
    """Multiplicative factor in [0, 1] that is 0 at `chosen` and rises to 1 away from it."""
    d = (np.atleast_2d(cand) - np.asarray(chosen, dtype=float)) / scale
    return 1.0 - np.exp(-0.5 * np.sum(d * d, axis=1) / (h * h))
