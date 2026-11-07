"""Synthetic comparison of planners (trajectory vs kde_map) and acquisition modes (random vs score).

Usage: python -m src.benchmark.compare_baseline [--rounds 6] [--seeds 3]

The world is a toy generator whose accuracy is poor for low HLgap targets; no QM or GPU is involved.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.pipeline_blocks.acquire_candidates import select_batch
from src.configs.config import DEFAULTS, _merge
from src.components.kde_plan import PropertyDensity
from src.pipeline_blocks.plan_targets import plan_targets
from src.components.al_state import ALState
from src.components.surrogate import FingerprintRF

PROPS = ["HLgap", "eMBD"]
ROWS = np.array([[4.5, -1.9], [3.8, -2.1], [3.1, -2.3], [2.4, -2.5]])
LO, HI = np.array([2.0, -3.0]), np.array([5.0, -1.0])
THRESHOLD = 0.1


class _Traj:
    prop_names = PROPS
    properties = {p: ROWS[:, j] for j, p in enumerate(PROPS)}

    def __len__(self):
        return len(ROWS)


def _generate(targets, n_total, weights, rng):
    """Candidates (requested, achieved, features, target ids); the generator is noisier for low HLgap."""
    share = np.maximum(1, np.round(n_total * np.asarray(weights) / np.sum(weights)).astype(int))
    req, ach, tid = [], [], []
    for k, (t, n) in enumerate(zip(targets, share)):
        sigma = np.where(t[0] < 3.2, 0.5, 0.12)
        ach.append(t + rng.normal(0, 1, (n, 2)) * sigma * np.array([1.0, 0.1]) + np.array([-0.05, 0.0]))
        req.append(np.tile(t, (n, 1)))
        tid.append(np.full(n, k))
    ach, req, tid = np.vstack(ach), np.vstack(req), np.concatenate(tid)
    return req, ach, _features(ach, rng), tid


def _features(y, rng):
    noisy = y + rng.normal(0, 0.05, y.shape)
    return np.hstack([noisy, rng.normal(0, 1, (len(y), 3))])


def _bits(x):
    thresholds = np.linspace(-3.0, 5.0, 16)
    return (x[:, :2, None] > thresholds).reshape(len(x), -1)


def run_simulation(planner: str, acq_mode: str, rounds: int = 6, seed: int = 0, pool: int = 200, budget: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    al = _merge(DEFAULTS, {"enabled": True, "query_strategy": planner, "targets_per_iter": 4,
                           "density": {"n_candidates": 400}, "min_neff": 5})
    traj = _Traj()
    base = np.column_stack([rng.normal(3.5, 0.3, 60), rng.normal(-2.0, 0.1, 60)])
    props, reqs, hits, its = [base], [np.full((60, 2), np.nan)], [np.full(60, -1)], [np.full(60, -1)]
    feats = [_features(base, rng)]
    new_props, new_hits = [], []
    sf = None

    for r in range(rounds):
        state = ALState(PROPS, np.arange(sum(map(len, props))).astype(object), np.vstack(props), np.concatenate(its),
                        np.full(sum(map(len, props)), -1), np.vstack(reqs), np.concatenate(hits))
        targets = plan_targets(traj, state, al, r, pool, seed, sf)
        tv = [np.array([t.props[p] for p in PROPS]) for t in targets]
        req, ach, x, tid = _generate(tv, pool, [t.n_samples for t in targets], rng)

        if acq_mode == "random":
            chosen = rng.choice(len(ach), size=budget, replace=False)
        else:
            X_all, Y_all = np.vstack(feats), np.vstack(props)
            model = FingerprintRF(2, 20, seed).fit(X_all[-2000:], Y_all[-2000:])
            mu, sigma = model.predict(x)
            dens = PropertyDensity(Y_all)
            existing = _bits(X_all)
            chosen = []
            for k, t in enumerate(tv):
                idx = np.flatnonzero(tid == k)
                share = max(1, int(round(budget * len(idx) / len(ach))))
                sel, _ = select_batch(mu=mu[idx], sigma=sigma[idx], target=t, delta=THRESHOLD * np.abs(t),
                                      fps=_bits(x[idx]), existing_fps=existing, n_heavy=np.full(len(idx), 10.0),
                                      budget=share, weights=al["acquisition"], density=dens, prop_scale=dens.scale)
                chosen.extend(idx[sel].tolist())
                dens.add_points(mu[idx[sel]])
            chosen = np.array(chosen[:budget])

        hit = np.all(np.abs(ach[chosen] - req[chosen]) <= THRESHOLD * np.abs(req[chosen]), axis=1).astype(int)
        props.append(ach[chosen]); reqs.append(req[chosen]); hits.append(hit)
        its.append(np.full(len(chosen), r)); feats.append(x[chosen])
        new_props.append(ach[chosen]); new_hits.append(hit)

    new = np.vstack(new_props)
    bins = np.floor((new - LO) / (HI - LO) * 10).astype(int)
    inside = np.all((bins >= 0) & (bins < 10), axis=1)
    covered = len({tuple(b) for b in bins[inside]})

    X_all, Y_all = np.vstack(feats), np.vstack(props)
    model = FingerprintRF(2, 20, seed).fit(X_all, Y_all)
    test_y = rng.uniform(LO, HI, (500, 2))
    mu, _ = model.predict(_features(test_y, rng))
    rmse = float(np.sqrt(np.mean(((mu - test_y) / (HI - LO)) ** 2)))
    return {"bins_covered": covered, "hit_rate": float(np.mean(np.concatenate(new_hits))), "surrogate_rmse": rmse}


def compare(rounds: int, seeds: int) -> dict:
    combos = {"trajectory+random": ("trajectory", "random"), "kde_map+score": ("kde_map", "score"),
              "trajectory+score": ("trajectory", "score"), "kde_map+random": ("kde_map", "random")}
    out = {}
    for name, (planner, mode) in combos.items():
        runs = [run_simulation(planner, mode, rounds, seed) for seed in range(seeds)]
        out[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    res = compare(args.rounds, args.seeds)
    print(f"{'combination':<20}{'bins covered':>14}{'hit rate':>10}{'surrogate RMSE':>16}")
    for name, m in res.items():
        print(f"{name:<20}{m['bins_covered']:>14.1f}{m['hit_rate']:>10.3f}{m['surrogate_rmse']:>16.4f}")
    base, new = res["trajectory+random"], res["kde_map+score"]
    wins = [new["bins_covered"] > base["bins_covered"], new["hit_rate"] > base["hit_rate"],
            new["surrogate_rmse"] < base["surrogate_rmse"]]
    print(f"kde_map+score beats trajectory+random on {sum(wins)} of 3 metrics")
    return 0 if any(wins) else 1


if __name__ == "__main__":
    raise SystemExit(main())
