import numpy as np

from mtraj import DISCHARGED
from mtraj.simulate import simulate, simulate_mixture
from mtraj.synthetic import P_TRUE, PI0


def test_supplied_uniforms_are_deterministic():
    u = np.random.default_rng(0).random((1000, 31))
    a = simulate(P_TRUE, PI0, 1000, 30, uniforms=u)
    b = simulate(P_TRUE, PI0, 1000, 30, uniforms=u)
    np.testing.assert_array_equal(a, b)


def test_seeded_rng_reproducible():
    a = simulate(P_TRUE, PI0, 500, 20, np.random.default_rng(3))
    b = simulate(P_TRUE, PI0, 500, 20, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_antithetic_pairs_negatively_correlated():
    # Absorbing states are the last CDF columns, so "absorbed after one step" is
    # monotone in u[:, 1]. (Multi-step targets like LOS are not; see variance_reduction.)
    n, h = 20000, 1
    u = np.random.default_rng(1).random((n, h + 1))
    absorbed = lambda x: (x[:, 1] >= DISCHARGED).astype(float)
    a = absorbed(simulate(P_TRUE, PI0, n, h, uniforms=u))
    b = absorbed(simulate(P_TRUE, PI0, n, h, uniforms=1 - u))
    assert np.corrcoef(a, b)[0, 1] < -0.05


def test_time_varying_matches_constant():
    u = np.random.default_rng(2).random((300, 11))
    a = simulate(P_TRUE, PI0, 300, 10, uniforms=u)
    b = simulate([P_TRUE] * 10, PI0, 300, 10, uniforms=u)
    np.testing.assert_array_equal(a, b)


def test_absorbing_states_stay_absorbed():
    x = simulate(P_TRUE, PI0, 5000, 40, np.random.default_rng(4))
    absorbed = x[:, :-1] >= DISCHARGED
    assert (x[:, 1:][absorbed] == x[:, :-1][absorbed]).all()


SEVERITY = [DISCHARGED, 0, 1, 2, 4]


def test_column_order_preserves_law():
    from mtraj.analytic import occupancy
    x = simulate(P_TRUE, PI0, 100_000, 10, np.random.default_rng(5), column_order=SEVERITY)
    emp = np.stack([np.bincount(x[:, t], minlength=5) for t in range(11)]) / len(x)
    assert np.abs(emp - occupancy(P_TRUE, PI0, 10)).max() < 0.01


def test_mixture_uses_each_paths_matrix():
    stay = np.eye(5)
    leave = np.zeros((5, 5))
    leave[:, DISCHARGED] = 1
    pi0 = np.tile(np.eye(5)[0], (2, 1))
    draw = np.array([0, 1, 0, 1])
    x = simulate_mixture(np.stack([stay, leave]), pi0, draw, 3, np.random.default_rng(0))
    assert (x[draw == 0] == 0).all() and (x[draw == 1, 1:] == DISCHARGED).all()
