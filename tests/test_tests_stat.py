import numpy as np
import pandas as pd
import pytest

from mtraj import estimate as E
from mtraj import tests_stat as TS
from mtraj import uncertainty as U
from mtraj.simulate import simulate
from mtraj.states import transitions
from mtraj.synthetic import P_TRUE, PI0, generate, paths_to_seqs


def _subject_level(traj):
    trans = transitions(traj)
    Cs, _ = U.subject_counts(traj, trans)
    first = traj.drop_duplicates("subject_id")
    return trans, Cs, first.era.to_numpy()


def test_g_statistic_zero_for_identical_groups():
    C = np.round(P_TRUE * 100)
    assert TS.g_statistic(np.stack([C, C, 2 * C])) == pytest.approx(0, abs=1e-9)


def test_path_counts_match_transitions_table():
    paths = simulate(P_TRUE, PI0, 3000, 20, np.random.default_rng(0))
    seqs, _ = paths_to_seqs(paths, 20)
    traj = pd.DataFrame({"subject_id": np.arange(3000).astype(str), "hadm_id": np.arange(3000),
                         "era": "early", "age_band": "<65", "sepsis": 0, "seq": seqs})
    trans = transitions(traj)
    C1, C2 = TS.path_counts(paths, 20)
    np.testing.assert_array_equal(C1, E.counts(trans))
    np.testing.assert_array_equal(C2, E.counts2(trans))


def test_order_lr_zero_when_second_order_equals_first():
    C1 = np.round(P_TRUE * 100)
    C1[3:] = 0
    C2 = np.stack([C1] * 6)
    r = TS.order_lr(C1 * 6, C2)
    assert r["LR"] == pytest.approx(0, abs=1e-8) and r["df"] == r["k2"] - r["k1"] > 0


def test_drift_table_detects_nothing_obvious_without_shift():
    traj = generate(1500, np.random.default_rng(1), p_sepsis=0, p_old=0)
    _, Cs, era = _subject_level(traj)
    tab = TS.drift_table(Cs, era, "early", "late", 300, np.random.default_rng(2))
    assert len(tab) == 15 and (tab.lo <= tab["diff"]).all() and (tab["diff"] <= tab.hi).all()


# --- slow: size and power on synthetic data ------------------------------------

@pytest.mark.slow
def test_stability_size_and_power():
    rng = np.random.default_rng(10)
    p_null, p_alt = [], []
    for _ in range(40):
        _, Cs, era = _subject_level(generate(800, rng, p_sepsis=0, p_old=0))
        p_null.append(TS.stability_test(Cs, era, 200, rng)["p_perm"])
    for _ in range(10):
        _, Cs, era = _subject_level(generate(800, rng, p_sepsis=0, p_old=0, era_shift=0.05))
        p_alt.append(TS.stability_test(Cs, era, 200, rng)["p_perm"])
    assert np.mean(np.array(p_null) < 0.05) <= 0.15   # size ~ 0.05 (binomial slack at 40 runs)
    assert np.mean(np.array(p_alt) < 0.05) >= 0.8     # power


@pytest.mark.slow
def test_order_test_size_and_power():
    rng = np.random.default_rng(11)

    def p_boot(second_order):
        traj = generate(800, rng, p_sepsis=0, p_old=0, second_order=second_order)
        trans = transitions(traj)
        return TS.order_test(E.counts(trans), E.counts2(trans), E.initial_distribution(traj),
                             len(traj), 60, 100, rng)["p_boot"]

    p_null = [p_boot(False) for _ in range(40)]
    p_alt = [p_boot(True) for _ in range(10)]
    assert np.mean(np.array(p_null) < 0.05) <= 0.15
    assert np.mean(np.array(p_alt) < 0.05) >= 0.8
