import numpy as np

from src.active_learning.acquisition import max_tanimoto, p_hit, select_batch, tanimoto_similarity
from src.active_learning.density import PropertyDensity

W = {"w_hit": 1.0, "w_unc": 0.5, "w_nov": 0.5, "w_div": 0.5, "w_cost": 0.1}


def _fps(n, bits=64, seed=0):
    return np.random.default_rng(seed).random((n, bits)) < 0.3


def _density(dim=1):
    return PropertyDensity(np.random.default_rng(0).normal(0, 1, (300, dim)))


def _run(n=40, budget=10, mode="score", mu=None, sigma=None, fps=None, seed=0, existing=None):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, (n, 1)) if mu is None else mu
    sigma = np.full((n, 1), 0.3) if sigma is None else sigma
    fps = _fps(n) if fps is None else fps
    return select_batch(mu=mu, sigma=sigma, target=np.array([0.0]), delta=np.array([0.2]), fps=fps,
                        existing_fps=existing, n_heavy=np.full(n, 8.0), budget=budget, weights=W,
                        density=_density(), prop_scale=np.ones(1), mode=mode, rng=np.random.default_rng(seed))


def test_p_hit_monotone_in_distance_and_shrinks_with_sigma():
    mu = np.array([[0.0], [0.5], [1.0], [2.0]])
    p = p_hit(mu, np.full((4, 1), 0.2), np.array([0.0]), np.array([0.3]))
    assert np.all(np.diff(p) < 0)
    wide = p_hit(np.array([[0.0]]), np.array([[3.0]]), np.array([0.0]), np.array([0.3]))
    assert wide[0] < p[0]


def test_p_hit_multiplies_properties():
    p1 = p_hit(np.array([[0.0, 0.0]]), np.array([[0.2, 0.2]]), np.zeros(2), np.full(2, 0.3))[0]
    p0 = p_hit(np.array([[0.0]]), np.array([[0.2]]), np.zeros(1), np.full(1, 0.3))[0]
    assert np.isclose(p1, p0 ** 2)


def test_budget_respected_and_no_duplicates():
    chosen, diag = _run(n=40, budget=10)
    assert len(chosen) == 10 and len(set(chosen.tolist())) == 10
    assert diag["selected"].sum() == 10


def test_budget_capped_by_valid_candidates():
    mu = np.random.default_rng(1).normal(0, 1, (6, 1))
    mu[:3] = np.nan
    chosen, _ = _run(n=6, budget=5, mu=mu)
    assert len(chosen) == 3 and not np.isin(chosen, [0, 1, 2]).any()


def test_random_mode_reproducible():
    a, _ = _run(mode="random", seed=3)
    b, _ = _run(mode="random", seed=3)
    c, _ = _run(mode="random", seed=4)
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist()


def test_prefers_likely_hits():
    n = 30
    mu = np.full((n, 1), 3.0)
    mu[:5] = 0.0
    chosen, _ = _run(n=n, budget=5, mu=mu, fps=_fps(n, seed=9))
    assert len(set(chosen.tolist()) & set(range(5))) >= 4


def test_diversity_avoids_near_duplicates():
    n = 20
    base = _fps(1, seed=5)
    fps = np.vstack([np.repeat(base, 10, axis=0), _fps(10, seed=11)])
    mu = np.zeros((n, 1))
    chosen, _ = _run(n=n, budget=3, mu=mu, fps=fps)
    assert (chosen < 10).sum() <= 1


def test_novelty_vs_existing_set_lowers_score_of_known_molecules():
    n = 12
    fps = _fps(n, seed=2)
    mu = np.zeros((n, 1))
    chosen, diag = _run(n=n, budget=1, mu=mu, fps=fps, existing=fps[:6])
    assert chosen[0] >= 6


def test_tanimoto_basics():
    a = np.array([[1, 1, 0, 0]], dtype=bool)
    b = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=bool)
    sim = tanimoto_similarity(a, b)[0]
    assert np.isclose(sim[0], 0.5) and sim[1] == 0.0
    assert max_tanimoto(a, b)[0] == 0.5 and max_tanimoto(a, np.zeros((0, 4), dtype=bool))[0] == 0.0
