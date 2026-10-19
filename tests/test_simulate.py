import numpy as np

from mtraj import DISCHARGED
from mtraj.simulate import simulate
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
