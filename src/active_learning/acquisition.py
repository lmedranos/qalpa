from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from src.active_learning.density import CandidateDensity, PropertyDensity


def p_hit(mu: np.ndarray, sigma: np.ndarray, target: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """P(|y - t| < delta) per candidate, assuming independent Gaussian predictions per property."""
    s = np.maximum(sigma, 1e-9)
    per_prop = ndtr((target + delta - mu) / s) - ndtr((target - delta - mu) / s)
    return np.prod(per_prop, axis=1)


def tanimoto_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(len(A), len(B)) Tanimoto similarity between boolean fingerprint matrices."""
    return _tanimoto_f32(A.astype(np.float32), B.astype(np.float32))


def _tanimoto_f32(Af: np.ndarray, Bf: np.ndarray) -> np.ndarray:
    inter = Af @ Bf.T
    union = Af.sum(1)[:, None] + Bf.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def max_tanimoto(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Highest similarity of each row of A to any row of B (0 if B is empty)."""
    if B is None or len(B) == 0:
        return np.zeros(len(A))
    Af, Bf = A.astype(np.float32), B.astype(np.float32)
    return np.concatenate([_tanimoto_f32(Af[s:s + chunk], Bf).max(axis=1) for s in range(0, len(Af), chunk)])


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, dtype=float) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def select_batch(*, mu, sigma, target, delta, fps, existing_fps, n_heavy, budget: int, weights: dict,
                 density: PropertyDensity, prop_scale, mode: str = "score", rng=None) -> tuple:
    """Pick `budget` candidates for one target.

    Returns (selected indices into the inputs, diagnostics dict of per-candidate arrays; NaN for dropped rows).
    """
    n = len(mu)
    valid = np.isfinite(mu).all(axis=1) & np.isfinite(sigma).all(axis=1)
    idx = np.flatnonzero(valid)
    budget = int(min(budget, len(idx)))
    nan = np.full(n, np.nan)
    diag = {k: nan.copy() for k in ("p_hit", "unc", "nov", "cost", "div", "score")}
    selected = np.zeros(n, dtype=bool)
    if budget == 0:
        return np.array([], dtype=int), dict(diag, selected=selected)

    rng = rng if rng is not None else np.random.default_rng(0)
    if mode == "random":
        chosen = rng.choice(idx, size=budget, replace=False)
        selected[chosen] = True
        return np.sort(chosen), dict(diag, selected=selected)

    m, s, f = mu[idx], sigma[idx], fps[idx]
    ph = p_hit(m, s, target, delta)
    unc = _norm01(np.mean(s / prop_scale, axis=1))
    cost = n_heavy[idx] / max(float(np.max(n_heavy[idx])), 1.0)
    cd = CandidateDensity(density, m)
    nov_raw = -cd.logp()
    lo, hi = float(nov_raw.min()), float(nov_raw.max())
    span = max(hi - lo, 1e-12)
    mindist = 1.0 - max_tanimoto(f, existing_fps)
    div0 = mindist.copy()
    nov0 = np.clip((nov_raw - lo) / span, 0.0, 1.0)

    def total(nov, div):
        return (weights["w_hit"] * ph + weights["w_unc"] * unc + weights["w_nov"] * nov
                - weights["w_cost"] * cost + weights["w_div"] * div)

    init_score = total(nov0, div0)
    chosen_local, pick_scores = [], []
    free = np.ones(len(idx), dtype=bool)
    for _ in range(budget):
        nov = np.clip((-cd.logp() - lo) / span, 0.0, 1.0)
        score = np.where(free, total(nov, mindist), -np.inf)
        i = int(np.argmax(score))
        chosen_local.append(i)
        pick_scores.append(score[i])
        free[i] = False
        mindist = np.minimum(mindist, 1.0 - tanimoto_similarity(f, f[i:i + 1])[:, 0])
        cd.add(m[i])

    for name, arr in (("p_hit", ph), ("unc", unc), ("nov", nov0), ("cost", cost), ("div", div0), ("score", init_score)):
        diag[name][idx] = arr
    diag["score"][idx[chosen_local]] = pick_scores
    chosen = idx[chosen_local]
    selected[chosen] = True
    return np.sort(chosen), dict(diag, selected=selected)
