import numpy as np
import pytest

from mtraj import DEAD, DISCHARGED, ICU, ICU_SUPPORT, TRANSIENT
from mtraj import analytic as A
from mtraj.simulate import simulate
from mtraj.synthetic import P_TRUE, PI0

N_SIM, H = 200_000, 400  # long horizon so (almost) every path is absorbed


@pytest.fixture(scope="module")
def paths():
    return simulate(P_TRUE, PI0, N_SIM, H, np.random.default_rng(0))


def _within_3se(x, target):
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(x.mean() - target) < 3 * se, (x.mean(), target, se)


@pytest.mark.parametrize("start", TRANSIENT)
def test_los_and_absorption_match_simulation(start):
    x = simulate(P_TRUE, np.eye(5)[start], 100_000, H, np.random.default_rng(start))
    los = (x < DISCHARGED).sum(1).astype(float)
    i = TRANSIENT.index(start)
    _within_3se(los, A.expected_los(P_TRUE)[i])
    _within_3se((x[:, -1] == DEAD).astype(float), A.absorption(P_TRUE)[i, 1])
    # variance: compare via the second moment E[L^2] = Var + tau^2
    _within_3se(los**2, A.var_los(P_TRUE)[i] + A.expected_los(P_TRUE)[i] ** 2)


def test_fundamental_matrix_identity():
    Q, _ = A.blocks(P_TRUE)
    np.testing.assert_allclose(A.fundamental(P_TRUE) @ (np.eye(3) - Q), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(A.absorption(P_TRUE).sum(1), 1)


def test_cohort_mortality(paths):
    _within_3se((paths[:, -1] == DEAD).astype(float), A.mortality(P_TRUE, PI0))


@pytest.mark.parametrize("t", [1, 3, 7, 14, 28])
def test_occupancy_matches_simulation(paths, t):
    pi = A.occupancy(P_TRUE, PI0, 28)
    for s in range(5):
        _within_3se((paths[:, t] == s).astype(float), pi[t, s])


def test_expected_icu_days(paths):
    icu = np.isin(paths[:, :28], [ICU, ICU_SUPPORT]).sum(1).astype(float)
    _within_3se(icu, A.expected_icu_days(P_TRUE, PI0, 28))


def test_quasi_stationary_is_left_eigenvector():
    Q, _ = A.blocks(P_TRUE)
    q = A.quasi_stationary(P_TRUE)
    lam = (q @ Q).sum()
    np.testing.assert_allclose(q @ Q, lam * q, atol=1e-12)
