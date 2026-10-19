import numpy as np
import pytest

from mtraj import variance_reduction as V
from mtraj.config import load_config
from mtraj.synthetic import P_TRUE, PI0, cell_matrix

H = 28


@pytest.fixture(scope="module")
def vc():
    return dict(load_config()["variance_reduction"], n_grid=[2000], R=200, n_pilot=5000, crn_n=2000)


@pytest.fixture(scope="module")
def table(vc):
    return V.run_estimators(P_TRUE, PI0, vc, H, np.random.default_rng(0))


def test_estimators_unbiased_on_known_targets(table):
    known = table[table.target.isin(["mort", "y2"])]
    assert (known.z_vs_exact.abs() < 3).all(), known[["target", "method", "z_vs_exact"]]


def test_control_variate_reduces_variance_for_y1(table):
    y1 = table[table.target == "y1"].set_index("method").vrf
    assert y1["control_variate"] > 1.5
    assert y1["antithetic_cv"] > y1["control_variate"] * 0.9


def test_severity_order_antithetic_beats_plain_for_y1(table):
    y1 = table[table.target == "y1"].set_index("method").vrf
    assert y1["antithetic"] > 1.0


def test_exact_icu_days_matches_simulation():
    paths = V.simulate(P_TRUE, PI0, 200_000, H, np.random.default_rng(1))
    icu = V.outcomes(paths, H, 7, 14)["icu_days"]
    assert abs(icu.mean() - V.exact_values(P_TRUE, PI0, H, 14)["icu_days"]) < 3 * icu.std() / np.sqrt(len(icu))


def test_crn_reduces_variance_of_contrast(vc):
    Pa, Pb = cell_matrix("early", "<65", 1), cell_matrix("early", "<65", 0)
    df = V.crn_contrast(Pa, PI0, Pb, PI0, vc, H, np.random.default_rng(2)).set_index("target")
    assert (df.vrf > 1.5).all(), df
    # both schemes estimate the same difference
    assert ((df.diff_crn - df.diff_indep).abs() < 3 * df.se_indep / np.sqrt(vc["R"])).all()
