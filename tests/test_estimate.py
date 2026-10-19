import numpy as np
import pandas as pd
import pytest

from mtraj import ABSORBING, ICU, ICU_SUPPORT, TRANSIENT
from mtraj import estimate as E
from mtraj.simulate import simulate, simulate_second_order
from mtraj.states import transitions
from mtraj.synthetic import P_TRUE, PI0, paths_to_seqs, second_order_tensor


def _trans(paths, max_days=60):
    seqs, _ = paths_to_seqs(paths, max_days)
    traj = pd.DataFrame({"subject_id": np.arange(len(seqs)).astype(str), "hadm_id": np.arange(len(seqs)),
                         "era": "early", "age_band": "<65", "sepsis": 0, "seq": seqs})
    return traj, transitions(traj)


@pytest.fixture(scope="module")
def first_order():
    return _trans(simulate(P_TRUE, PI0, 50_000, 60, np.random.default_rng(0)))


def _check_stochastic(P):
    assert np.all((P >= 0) & (P <= 1))
    np.testing.assert_allclose(P.sum(-1), 1)
    absorbing = P[..., ABSORBING, :]
    np.testing.assert_array_equal(absorbing, np.broadcast_to(np.eye(5)[ABSORBING], absorbing.shape))


def test_mle_recovers_p_true(first_order):
    _, t = first_order
    P = E.mle(E.counts(t))
    _check_stochastic(P)
    assert np.abs(P - P_TRUE).max() < 0.01


def test_initial_distribution(first_order):
    traj, _ = first_order
    assert np.abs(E.initial_distribution(traj) - PI0).max() < 0.01


def test_shrink_limits(first_order):
    _, t = first_order
    C = E.counts(t)
    m = E.pooled_target(C)
    _check_stochastic(E.shrink(C, 10.0, m))
    np.testing.assert_allclose(E.shrink(C, 1e-9, m), E.mle(C), atol=1e-9)
    np.testing.assert_allclose(E.shrink(C, 1e12, m)[TRANSIENT], m[TRANSIENT], atol=1e-6)


def test_mle_raises_on_empty_row():
    C = np.zeros((5, 5))
    C[0, 3] = 5
    with pytest.raises(ValueError, match="ICU"):
        E.mle(C)


def test_mle2_recovers_relapse_row():
    paths = simulate_second_order(second_order_tensor(P_TRUE), P_TRUE, PI0, 50_000, 60,
                                  np.random.default_rng(1))
    _, t = _trans(paths)
    P1 = E.mle(E.counts(t))
    P2 = E.mle2(E.counts2(t), P1)
    _check_stochastic(P2)
    truth = second_order_tensor(P_TRUE)
    assert np.abs(P2[ICU_SUPPORT + 1, ICU] - truth[ICU_SUPPORT, ICU]).max() < 0.02
    assert np.abs(P2[ICU + 1, ICU] - P_TRUE[ICU]).max() < 0.02
    _check_stochastic(E.shrink2(E.counts2(t), 5.0, P1))


def test_pooled_target_batched(first_order):
    _, t = first_order
    C = E.counts(t)
    batch = np.stack([C, 2 * C + 1])
    m = E.pooled_target(batch)
    np.testing.assert_allclose(m[0], E.pooled_target(C))
    np.testing.assert_allclose(m[1], E.pooled_target(2 * C + 1))
