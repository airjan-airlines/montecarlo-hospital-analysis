import numpy as np
import pandas as pd
import pytest

from mtraj import DEAD, DISCHARGED, ICU, ICU_SUPPORT, WARD
from mtraj.config import load_config
from mtraj.states import DAY, build_trajectories, day_sequence, merge_points, transitions

H = 60  # minutes


def seq(los, icu=(), sup=(), died=False, max_days=60, thr=60):
    return day_sequence(los, np.array(icu, float).reshape(-1, 2),
                        np.array(sup, float).reshape(-1, 2), died, max_days, thr)


def test_ward_then_icu_with_pressor_then_death():
    # 6h ward, ICU from 6h to 50h with a vasopressor at 30h, death at 50h.
    s, trunc = seq(50 * H, icu=[(6 * H, 50 * H)], sup=[(30 * H, 30 * H)], died=True)
    assert s == [ICU, ICU_SUPPORT, ICU, DEAD]
    assert not trunc


def test_stay_under_24h():
    assert seq(5 * H)[0] == [WARD, DISCHARGED]


def test_icu_overlap_threshold():
    assert seq(DAY, icu=[(DAY - 59, DAY)])[0] == [WARD, DISCHARGED]
    assert seq(DAY, icu=[(DAY - 60, DAY)])[0] == [ICU, DISCHARGED]


def test_support_requires_icu_day():
    # support event on a day with <1h ICU does not make the day ICU_SUPPORT
    assert seq(DAY, icu=[(0, 30)], sup=[(10, 10)])[0] == [WARD, DISCHARGED]


def test_discharge_on_boundary_belongs_to_previous_block():
    assert seq(2 * DAY)[0] == [WARD, WARD, DISCHARGED]


def test_truncation():
    s, trunc = seq(70 * DAY, max_days=60)
    assert trunc and len(s) == 60 and DISCHARGED not in s


def test_merge_points():
    iv = merge_points(np.array([0, 100, 2000, 2100]), gap=1440)
    np.testing.assert_array_equal(iv, [[0, 100], [2000, 2100]])


def test_transitions_prev_state():
    traj = pd.DataFrame({"subject_id": ["a"], "hadm_id": [1], "era": ["early"],
                         "age_band": ["<65"], "sepsis": [0], "seq": [[WARD, ICU, ICU, DEAD]]})
    t = transitions(traj)
    assert t.from_state.tolist() == [WARD, ICU, ICU]
    assert t.to_state.tolist() == [ICU, ICU, DEAD]
    assert t.prev_state.tolist() == [-1, WARD, ICU]


def _toy_raw():
    # Hospital stay 10: admitted 6h before ICU; two unit stays; vasopressor in first.
    patient = pd.DataFrame({
        "patientunitstayid": [1, 2, 3],
        "patienthealthsystemstayid": [10, 10, 20],
        "uniquepid": ["p1", "p1", "p2"],
        "age": ["> 89", "> 89", "12"],
        "hospitalid": [7, 7, 7],
        "apacheadmissiondx": ["Sepsis, pulmonary", None, None],
        "hospitaladmitoffset": [-6 * H, -3 * DAY, -60],
        "hospitaldischargeyear": [2015, 2015, 2014],
        "hospitaldischargeoffset": [4 * DAY - 6 * H, DAY, DAY],
        "hospitaldischargestatus": ["Expired", "Expired", "Alive"],
        "unitvisitnumber": [1, 2, 1],
        "unitdischargeoffset": [20 * H, 12 * H, 600],
    })
    empty = lambda cols: pd.DataFrame({c: [] for c in cols})
    return {
        "patient": patient,
        "hospital": pd.DataFrame({"hospitalid": [7], "region": ["West"]}),
        "diagnosis": empty(["patientunitstayid", "diagnosisoffset", "diagnosisstring"]).astype(
            {"diagnosisstring": str}),
        "treatment": pd.DataFrame({"patientunitstayid": [1], "treatmentoffset": [60],
                                   "treatmentstring": ["cardiovascular|shock|vasopressors"]}),
        "infusiondrug": pd.DataFrame({"patientunitstayid": [1], "infusionoffset": [120],
                                      "drugname": ["Norepinephrine (mcg/min)"]}),
        "respiratoryCharting": pd.DataFrame({"patientunitstayid": [1], "respchartoffset": [90],
                                             "respchartvaluelabel": ["Vent Rate"]}),
    }


def test_build_trajectories_toy():
    cfg = load_config()
    traj = build_trajectories(_toy_raw(), cfg)
    assert traj.hadm_id.tolist() == [10]  # stay 20 dropped: age < 18
    r = traj.iloc[0]
    # day0: ward 6h + ICU 18h w/ support; day1: ICU 2h then ward; day2: ward; day3: ICU 2nd unit
    assert r.seq == [ICU_SUPPORT, ICU, WARD, ICU, DEAD]
    assert r.age_band == ">=65" and r.sepsis == 1 and r.era == "late"


@pytest.mark.parametrize("bad", ["status", "unit_missing"])
def test_build_trajectories_drops(bad):
    raw = _toy_raw()
    p = raw["patient"]
    if bad == "status":
        p.loc[0, "hospitaldischargestatus"] = None
    else:
        raw["patient"] = p[p.patientunitstayid != 1].reset_index(drop=True)
    assert len(build_trajectories(raw, load_config())) == 0
