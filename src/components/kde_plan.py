from __future__ import annotations

import numpy as np

from src.base import AcquiScorer, CoverageEstimator, Planner, PlanningRequest

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


def _gauss(Zq: np.ndarray, Z: np.ndarray, h: np.ndarray,
           weights: np.ndarray, chunk: int = 64) -> np.ndarray:
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

    def __init__(self, Y, weights=None, bandwidth="scott",
                 min_bandwidth=None, logp_floor: float = -20.0):
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        w = np.ones(len(Y)) if weights is None else np.asarray(weights, dtype=float)
        keep = np.isfinite(Y).all(axis=1) & np.isfinite(w) & (w > 0)
        Y, w = Y[keep], w[keep]
        if len(Y) == 0:
            raise ValueError("PropertyDensity needs at least one finite point")
        self.reference = Y.copy()
        self.center, self.scale = robust_center_scale(Y)
        self.Z = (Y - self.center) / self.scale
        self.w = w
        self.logp_floor = float(logp_floor)
        d = Y.shape[1]
        n_eff = w.sum() ** 2 / np.sum(w ** 2)
        h = n_eff ** (-1.0 / (d + 4)) if bandwidth == "scott" else float(bandwidth)
        self.h = np.full(d, max(h, 1e-3))
        if min_bandwidth is not None:
            self.h = np.maximum(self.h, np.asarray(min_bandwidth, dtype=float) / self.scale)

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
        weights = np.full(len(Y), float(w)) if np.isscalar(w) else np.asarray(w, dtype=float)
        self.Z = np.vstack([self.Z, self.standardize(Y)])
        self.w = np.concatenate([self.w, weights])


class CandidateDensity:
    """KDE values at fixed query points with incremental updates."""

    def __init__(self, dens: PropertyDensity, Yq):
        self.dens = dens
        self.Zq = dens.standardize(Yq)
        self.total = dens.kernel_sum(Yq)
        self.W = dens.total_weight

    def add(self, y, w: float = 1.0) -> None:
        z = self.dens.standardize(y)
        self.total += w * _gauss(self.Zq, z, self.dens.h, np.ones(1))
        self.W += w

    def logp(self) -> np.ndarray:
        return np.maximum(np.log(self.total / self.W + _TINY), self.dens.logp_floor)

    def score(self) -> np.ndarray:
        return -self.logp()


def kernel_regress(X_req, values, X_query, center, scale, bandwidth=None,
                   prior: float = 0.0, min_bandwidth: float = 0.15,
                   chunk: int = 256) -> tuple:
    """Nadaraya-Watson estimate and effective sample count."""
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
    est, n_eff = np.empty(len(Zq)), np.empty(len(Zq))
    for start in range(0, len(Zq), chunk):
        diff = (Zq[start:start + chunk, None, :] - uniq[None, :, :]) / h
        k = np.exp(-0.5 * np.einsum("qnd,qnd->qn", diff, diff))
        den = k @ counts
        n_eff[start:start + chunk] = den
        est[start:start + chunk] = np.where(den > 1e-12, (k @ sums) / np.maximum(den, 1e-12), prior)
    return est, n_eff


def estimate_coverage(requested: np.ndarray, candidates: np.ndarray,
                      center: np.ndarray, scale: np.ndarray,
                      min_neff: float) -> float:
    _, n_eff = kernel_regress(
        requested, np.zeros(len(requested)), candidates, center, scale
    )
    return float(np.mean(n_eff >= min_neff))


def repulsion(chosen, cand, scale, h: float = 0.5) -> np.ndarray:
    """Return a factor that is zero at chosen points and rises with distance."""
    d = (np.atleast_2d(cand) - np.asarray(chosen, dtype=float)) / scale
    return 1.0 - np.exp(-0.5 * np.sum(d * d, axis=1) / (h * h))


class KDEPlanner(Planner):
    """Target planner using Gaussian KDE and kernel regression."""

    def plan(self, request: PlanningRequest) -> list:
        from src.pipeline_blocks.plan_targets import plan_kde_map

        return plan_kde_map(
            request.trajectory,
            request.state,
            request.config,
            request.iteration,
            request.total_samples,
            request.rng,
        )


class KDEScorer(AcquiScorer):
    def __init__(self, density: PropertyDensity, query: np.ndarray):
        self._density = CandidateDensity(density, query)

    def score(self) -> np.ndarray:
        return self._density.score()

    def add(self, point: np.ndarray, weight: float = 1.0) -> None:
        self._density.add(point, weight)


class KDECoverageEstimator(CoverageEstimator):
    def estimate(
        self,
        requested: np.ndarray,
        candidates: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        minimum_effective_count: float,
    ) -> float:
        return estimate_coverage(requested, candidates, center, scale, minimum_effective_count)
