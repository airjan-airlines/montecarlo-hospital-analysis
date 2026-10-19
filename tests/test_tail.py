import numpy as np

from mtraj import ICU
from mtraj import analytic as A
from mtraj import tail as T
from mtraj.synthetic import P_TRUE, PI0


def test_bed_days_sums_cohorts():
    paths = np.zeros((4, 4), dtype=np.int8)
    paths[0, :3] = ICU
    paths[3, :2] = ICU
    np.testing.assert_array_equal(T.bed_days(paths, K=2, horizon=3), [3, 2])


def test_plug_in_mean_matches_analytic():
    x = T.plug_in(P_TRUE, PI0, 50, 4000, 28, np.random.default_rng(0))
    assert abs(x.mean() - 50 * A.expected_icu_days(P_TRUE, PI0, 28)) < 3 * x.std() / np.sqrt(len(x))


def test_risk_measures_ordering():
    r = T.risk_measures(np.random.default_rng(1).normal(100, 10, 50_000))
    assert r["mean"] < r["VaR95"] < r["ES95"] and r["VaR95"] < r["VaR99"]
    assert abs(r["VaR95"] - (100 + 16.45)) < 0.5


def test_parameter_uncertainty_widens_tail_with_little_data():
    # Few observed transitions -> wide posterior -> predictive sd clearly above plug-in.
    C = np.round(P_TRUE * 20)
    C[3:] = 0
    d0 = PI0 * 20
    rng = np.random.default_rng(2)
    P = C.copy()
    P[:3] /= P[:3].sum(1, keepdims=True)
    P[3:] = np.eye(5)[3:]
    plug = T.plug_in(P, d0 / d0.sum(), 50, 4000, 28, rng)
    post = T.posterior_predictive(C, d0, 0.0, 50, 4000, 28, rng)
    assert post.std() > 1.3 * plug.std()
