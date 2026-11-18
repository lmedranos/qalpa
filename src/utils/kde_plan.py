"""Gaussian kernel density / kernel regression in property space (ported from qalpa2)."""

from __future__ import annotations

import numpy as np

_TINY = 1e-300


def robust_center_scale(Y: np.ndarray) -> tuple:
    """Mean and MAD-based scale per column (falls back to std, then 1)."""
    Y = np.asarray(Y, dtype=float)
    center = Y.mean(axis=0)
    mad = np.mean(np.abs(Y - center), axis=0) * 1.2533
    std = Y.std(axis=0)
    scale = np.where(mad > 1e-12, mad, std)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def _gauss(Zq: np.ndarray, Z: np.ndarray, h: np.ndarray, weights: np.ndarray, chunk: int = 64) -> np.ndarray:
    d = Z.shape[1]
    norm = (2.0 * np.pi) ** (-d / 2.0) / np.prod(h)
    out = np.empty(len(Zq))
    Zs, Zqs = Z / h, Zq / h
    for start in range(0, len(Zq), chunk):
        query = Zqs[start:start + chunk]
        d2 = np.zeros((len(query), len(Zs)))
        for axis in range(d):
            d2 += (query[:, axis:axis + 1] - Zs[None, :, axis]) ** 2
        out[start:start + chunk] = np.exp(-0.5 * d2) @ weights
    return norm * out


class PropertyDensity:
    """Weighted Gaussian KDE in standardized property space."""

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
        self.h = np.full(d, max(h, 1e-3))
        if min_bandwidth is not None:
            # e.g. atom counts: keep the kernel at least one atom wide
            self.h = np.maximum(self.h, np.asarray(min_bandwidth, dtype=float) / self.scale)

    def standardize(self, Y) -> np.ndarray:
        return (np.atleast_2d(np.asarray(Y, dtype=float)) - self.center) / self.scale

    def density(self, Y) -> np.ndarray:
        return _gauss(self.standardize(Y), self.Z, self.h, self.w) / float(self.w.sum())

    def logp(self, Y) -> np.ndarray:
        return np.maximum(np.log(self.density(Y) + _TINY), self.logp_floor)

    def reference_density(self) -> np.ndarray:
        """Leave-one-out density of every fitted point (O(n^2); computed once per reference set).

        Without leaving each point out, an isolated molecule would see its own kernel and look dense.
        """
        d = self.Z.shape[1]
        self_kernel = (2.0 * np.pi) ** (-d / 2.0) / np.prod(self.h) * self.w
        total = _gauss(self.Z, self.Z, self.h, self.w, chunk=256) - self_kernel
        rest = float(self.w.sum()) - self.w
        return np.where(rest > 0, np.maximum(total, 0.0) / np.maximum(rest, _TINY), 0.0)

    def density_band(self, percentiles) -> tuple:
        """(low, high) density values at the given percentiles of the fitted points' own density."""
        ref = self.reference_density()
        lo, hi = np.percentile(ref, [float(percentiles[0]), float(percentiles[1])])
        return float(lo), float(hi)


def kernel_regress(X_req, values, X_query, center, scale, bandwidth=None, prior: float = 0.0,
                   min_bandwidth: float = 0.15, chunk: int = 256) -> tuple:
    """Nadaraya-Watson estimate of `values` at X_query, and the kernel-weighted count n_eff of nearby X_req."""
    X_req = np.atleast_2d(np.asarray(X_req, dtype=float))
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(X_req) == 0:
        return np.full(len(X_query), prior), np.zeros(len(X_query))
    # many molecules share the same requested target: collapse them into unique points with counts
    Zr = (X_req - center) / scale
    uniq, inv = np.unique(np.round(Zr, 9), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, values)
    d = uniq.shape[1]
    h = max(len(uniq) ** (-1.0 / (d + 4)), min_bandwidth) if bandwidth is None else float(bandwidth)
    Zq = (X_query - center) / scale
    est, n_eff = np.empty(len(Zq)), np.empty(len(Zq))
    for start in range(0, len(Zq), chunk):
        diff = (Zq[start:start + chunk, None, :] - uniq[None, :, :]) / h
        k = np.exp(-0.5 * np.einsum("qnd,qnd->qn", diff, diff))
        den = k @ counts
        n_eff[start:start + chunk] = den
        est[start:start + chunk] = np.where(den > 1e-12, (k @ sums) / np.maximum(den, 1e-12), prior)
    return est, n_eff


def estimate_coverage(requested: np.ndarray, candidates: np.ndarray, center: np.ndarray,
                      scale: np.ndarray, min_neff: float) -> float:
    """Fraction of candidate points with at least `min_neff` kernel-weighted past requests nearby."""
    _, n_eff = kernel_regress(requested, np.zeros(len(requested)), candidates, center, scale)
    return float(np.mean(n_eff >= min_neff))


def repulsion(chosen, cand, scale, h: float = 0.5) -> np.ndarray:
    """Factor that is zero at the chosen point and rises to one with distance (spreads the targets out)."""
    d = (np.atleast_2d(cand) - np.asarray(chosen, dtype=float)) / scale
    return 1.0 - np.exp(-0.5 * np.sum(d * d, axis=1) / (h * h))
