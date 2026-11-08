import numpy as np

from src.active_learning.density import CandidateDensity, PropertyDensity, kernel_regress, repulsion


def _cloud(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)])


def test_density_lower_far_from_data():
    dens = PropertyDensity(_cloud())
    near, far = dens.logp([[0.0, 0.0]])[0], dens.logp([[8.0, 8.0]])[0]
    assert near > far


def test_logp_is_floored():
    dens = PropertyDensity(_cloud(), logp_floor=-12.0)
    assert dens.logp([[1e3, 1e3]])[0] == -12.0


def test_add_points_lowers_novelty_nearby():
    dens = PropertyDensity(_cloud())
    q = np.array([[4.0, 4.0], [-4.0, -4.0]])
    cd = CandidateDensity(dens, q)
    before = -cd.logp()
    cd.add(np.array([4.0, 4.0]), w=5.0)
    after = -cd.logp()
    assert after[0] < before[0]
    assert np.isclose(after[1], before[1], atol=1e-3) or after[1] <= before[1] + 1e-3


def test_candidate_density_matches_direct_add():
    dens = PropertyDensity(_cloud(n=100))
    q = np.array([[0.5, 0.5], [3.0, 1.0]])
    cd = CandidateDensity(dens, q)
    cd.add(np.array([3.0, 1.0]))
    dens.add_points([[3.0, 1.0]])
    assert np.allclose(cd.logp(), dens.logp(q), atol=1e-8)


def test_min_bandwidth_floor_for_counts():
    Y = np.column_stack([np.linspace(0, 1, 50), np.full(50, 60.0) + np.arange(50) % 2])
    dens = PropertyDensity(Y, min_bandwidth=[0.0, 1.0])
    assert dens.h[1] * dens.scale[1] >= 1.0 - 1e-9


def test_kernel_regress_recovers_error_curve():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 4, size=(600, 1))
    y = np.abs(x[:, 0] - 2.0)
    center, scale = np.array([2.0]), np.array([1.0])
    est, n_eff = kernel_regress(x, y, np.array([[0.5], [2.0], [3.5]]), center, scale, bandwidth=0.2)
    assert est[0] > est[1] and est[2] > est[1]
    assert np.all(n_eff > 1)


def test_kernel_regress_prior_and_neff_where_no_data():
    x = np.array([[0.0], [0.1]])
    est, n_eff = kernel_regress(x, np.array([1.0, 1.0]), np.array([[50.0]]), np.zeros(1), np.ones(1),
                                bandwidth=0.2, prior=7.0)
    assert est[0] == 7.0 and n_eff[0] < 1e-6


def test_kernel_regress_merges_repeated_requests():
    x = np.zeros((10, 1))
    est, n_eff = kernel_regress(x, np.arange(10.0), np.array([[0.0]]), np.zeros(1), np.ones(1))
    assert np.isclose(est[0], 4.5) and np.isclose(n_eff[0], 10.0)


def test_repulsion_zero_at_chosen_and_rises():
    cand = np.array([[0.0], [0.2], [5.0]])
    r = repulsion(np.array([0.0]), cand, np.ones(1), h=0.5)
    assert r[0] == 0.0 and r[1] < r[2] and np.isclose(r[2], 1.0, atol=1e-6)
