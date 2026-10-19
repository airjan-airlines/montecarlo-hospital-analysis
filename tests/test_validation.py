import numpy as np
import pandas as pd
import pytest

from mtraj import DEAD, ICU, WARD
from mtraj import validation as Vd
from mtraj.config import load_config
from mtraj.states import transitions
from mtraj.synthetic import P_TRUE, generate


@pytest.fixture(scope="module")
def synth():
    traj = generate(1500, np.random.default_rng(0))
    trans = transitions(traj)
    first = traj.drop_duplicates("subject_id")
    return traj, trans, first, Vd.make_data(trans, pd.Index(first.subject_id))


def test_losses_known_values():
    probs = np.array([[0.5, 0.5, 0, 0, 0], [0, 0, 0, 0, 1.0]])
    ll, br = Vd.losses(probs, np.array([WARD, DEAD]), 1e-12)
    np.testing.assert_allclose(ll, [np.log(2), 0])
    np.testing.assert_allclose(br, [0.5, 0])
    ll0, _ = Vd.losses(probs, np.array([DEAD, WARD]), 1e-12)
    np.testing.assert_allclose(ll0, -np.log(1e-12))


@pytest.mark.parametrize("model", Vd.MODELS)
def test_predictions_are_distributions(synth, model):
    *_, d = synth
    p = Vd.fit(model, d, {"shrink1": 1.0, "shrink2": 5.0, "strat": 50.0})(d)
    assert p.shape == (len(d.frm), 5)
    np.testing.assert_allclose(p.sum(1), 1)
    assert (p >= 0).all()


def test_subject_folds_do_not_leak(synth):
    *_, d = synth
    fold = Vd.subject_folds(d.n_subj, 5, np.random.default_rng(1))[d.subj]
    for f in range(5):
        assert not set(d.subj[fold == f]) & set(d.subj[fold != f])


def test_walk_forward_split_is_subject_disjoint(synth):
    _, _, first, d = synth
    early = first.era.to_numpy()[d.subj] == "early"
    assert not set(d.subj[early]) & set(d.subj[~early])


def test_markov_models_beat_baselines(synth):
    _, _, first, d = synth
    tab, alphas = Vd.evaluate(d, first.era.to_numpy(), dict(load_config()["validation"], n_boot_se=100),
                              np.random.default_rng(2))
    kf = tab[tab.split == "kfold"].set_index("model").log_loss
    assert kf["shrink1"] < kf["persistence"] and kf["shrink1"] < kf["marginal"]
    assert set(alphas) == set(Vd.SHRUNK)


def test_cv_prefers_large_alpha_for_strata_when_strata_identical():
    # Homogeneous data: strata have the same P, so pooling (large alpha) should win.
    traj = generate(1500, np.random.default_rng(3), p_sepsis=0.5, p_old=0.5)
    traj["sepsis"] = np.random.default_rng(4).integers(0, 2, len(traj))  # labels carry no signal
    traj["age_band"] = np.where(np.random.default_rng(5).random(len(traj)) < 0.5, "<65", ">=65")
    trans = transitions(traj)
    d = Vd.make_data(trans, pd.Index(traj.drop_duplicates("subject_id").subject_id))
    grid = np.logspace(-2, 4, 13)
    curve = Vd.cv_curve(d, "strat", grid, 5, np.random.default_rng(6), 1e-12, alpha1=1.0)
    assert grid[curve.argmin()] >= 100


def test_observed_occupancy_carries_absorbing_forward():
    occ = Vd.observed_occupancy(pd.Series([[ICU, WARD, DEAD], [WARD] * 3]), horizon=4)
    np.testing.assert_array_equal(occ, [[ICU, WARD, DEAD, DEAD, DEAD], [WARD] * 5])


def test_calibration_on_correct_model():
    homogeneous = generate(3000, np.random.default_rng(7), p_sepsis=0, p_old=0)
    cal = Vd.calibration(P_TRUE, homogeneous, 28, 200, np.random.default_rng(8))
    inside = (cal["predicted"] >= cal["observed_lo"] - 1e-9) & (cal["predicted"] <= cal["observed_hi"] + 1e-9)
    assert inside.mean() > 0.85
    assert abs(cal["los_predicted"] - cal["los_observed"]) < 0.5
