import numpy as np
import pytest

from mtraj import ABSORBING
from mtraj import estimate as E
from mtraj import uncertainty as U
from mtraj.states import transitions
from mtraj.synthetic import generate


@pytest.fixture(scope="module")
def data():
    traj = generate(1500, np.random.default_rng(0))
    trans = transitions(traj)
    Cs, D0 = U.subject_counts(traj, trans)
    return traj, trans, Cs, D0


def test_subject_counts_sum_to_pooled(data):
    traj, trans, Cs, D0 = data
    np.testing.assert_array_equal(Cs.sum(0), E.counts(trans))
    np.testing.assert_allclose(D0.sum(0) / len(traj), E.initial_distribution(traj))


@pytest.mark.parametrize("method", ["bootstrap", "dirichlet"])
def test_draws_are_stochastic_matrices(data, method):
    _, _, Cs, D0 = data
    rng = np.random.default_rng(1)
    if method == "bootstrap":
        Ps, pi0s = U.bootstrap(Cs, D0, 200, 1.0, rng)
    else:
        Ps, pi0s = U.dirichlet(Cs.sum(0), D0.sum(0), 200, 1.0, rng)
    assert Ps.shape == (200, 5, 5) and pi0s.shape == (200, 5)
    np.testing.assert_allclose(Ps.sum(-1), 1)
    np.testing.assert_allclose(pi0s.sum(-1), 1)
    np.testing.assert_array_equal(Ps[:, ABSORBING], np.broadcast_to(np.eye(5)[ABSORBING], (200, 2, 5)))
    assert Ps.std(0).max() > 0  # draws actually vary


def test_dirichlet_mean_is_shrinkage_estimate(data):
    _, _, Cs, D0 = data
    C = Cs.sum(0)
    Ps, _ = U.dirichlet(C, D0.sum(0), 4000, 2.0, np.random.default_rng(2))
    np.testing.assert_allclose(Ps.mean(0), U.fit(C, 2.0), atol=3e-3)


def test_bootstrap_centered_on_point_estimate(data):
    _, _, Cs, D0 = data
    Ps, _ = U.bootstrap(Cs, D0, 1000, 1.0, np.random.default_rng(3))
    np.testing.assert_allclose(Ps.mean(0), U.fit(Cs.sum(0), 1.0), atol=3e-3)


def test_compare_methods_table(data):
    _, _, Cs, D0 = data
    rng = np.random.default_rng(4)
    b = U.summarize(*U.bootstrap(Cs, D0, 100, 1.0, rng), 28)
    p = U.summarize(*U.dirichlet(Cs.sum(0), D0.sum(0), 100, 1.0, rng), 28)
    tab = U.compare_methods(b, p)
    assert (tab.boot_lo <= tab.boot_median).all() and (tab.boot_median <= tab.boot_hi).all()
    assert (tab.width_ratio_boot_over_post > 0).all()
