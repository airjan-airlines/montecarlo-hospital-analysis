"""Cohort, daily state sequences, covariates and era from eICU tables.

Time axis: minutes since hospital admission. A unit stay with
``hospitaladmitoffset = h`` (negative: hospital admit precedes unit admit)
starts at ``-h`` and ends at ``-h + unitdischargeoffset``.
"""

import logging

import numpy as np
import pandas as pd

from . import DEAD, DISCHARGED, ICU, ICU_SUPPORT, WARD
from .load import support_events

log = logging.getLogger(__name__)

DAY = 1440
TRAJ_COLUMNS = ["subject_id", "hadm_id", "hospitalid", "region", "year", "era", "age_band",
                "sepsis", "sepsis_any", "truncated", "seq"]


def merge_points(times: np.ndarray, gap: float) -> np.ndarray:
    """Merge sorted event times into intervals where consecutive gaps <= gap."""
    if len(times) == 0:
        return np.empty((0, 2))
    times = np.sort(np.asarray(times, dtype=float))
    breaks = np.flatnonzero(np.diff(times) > gap)
    starts = np.r_[times[0], times[breaks + 1]]
    ends = np.r_[times[breaks], times[-1]]
    return np.column_stack([starts, ends])


def _overlap(intervals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Total overlap (minutes) of each block [lo, hi) with a set of intervals."""
    if len(intervals) == 0:
        return np.zeros(len(lo))
    s, e = intervals[:, 0][None, :], intervals[:, 1][None, :]
    return np.clip(np.minimum(e, hi[:, None]) - np.maximum(s, lo[:, None]), 0, None).sum(1)


def _touches(intervals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Whether any interval (points allowed) intersects each block [lo, hi)."""
    if len(intervals) == 0:
        return np.zeros(len(lo), dtype=bool)
    s, e = intervals[:, 0][None, :], intervals[:, 1][None, :]
    return ((s < hi[:, None]) & (e >= lo[:, None])).any(1)


def day_sequence(los_min: float, icu_iv: np.ndarray, sup_iv: np.ndarray, died: bool,
                 max_days: int, icu_overlap_min: float) -> tuple[list[int], bool]:
    """Daily state sequence for one hospital stay; returns (seq, truncated).

    Block d covers [d*DAY, (d+1)*DAY); the last block contains discharge. A
    discharge exactly on a boundary belongs to the preceding block.
    """
    n_days = max(1, int(np.ceil(los_min / DAY)))
    lo = np.arange(n_days) * DAY
    hi = lo + DAY
    icu = _overlap(np.asarray(icu_iv).reshape(-1, 2), lo, hi) >= icu_overlap_min
    sup = icu & _touches(np.asarray(sup_iv).reshape(-1, 2), lo, hi)
    seq = np.full(n_days, WARD)
    seq[icu] = ICU
    seq[sup] = ICU_SUPPORT
    if n_days > max_days:
        return seq[:max_days].tolist(), True
    return seq.tolist() + [DEAD if died else DISCHARGED], False


def parse_age(age: pd.Series) -> pd.Series:
    return pd.to_numeric(age.replace("> 89", "90"), errors="coerce")


def build_trajectories(raw: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    sc, cc = cfg["states"], cfg["covariates"]
    p = raw["patient"].copy()
    n0 = p.patienthealthsystemstayid.nunique()
    drops = {}

    def drop(mask_bad_stay: pd.Series, reason: str) -> None:
        nonlocal p
        bad = p.loc[mask_bad_stay, "patienthealthsystemstayid"].unique()
        drops[reason] = len(bad)
        p = p[~p.patienthealthsystemstayid.isin(bad)]

    drop(p.hospitaldischargestatus.isna(), "missing discharge status")
    drop(p.hospitaladmitoffset > 0, "unit admit before hospital admit")
    drop(p.hospitaldischargeoffset <= p.hospitaladmitoffset, "non-positive LOS")
    complete = p.groupby("patienthealthsystemstayid").unitvisitnumber.transform(
        lambda s: sorted(s) == list(range(1, len(s) + 1)))
    drop(~complete, "unit stays missing from demo")
    p["age_num"] = parse_age(p.age)
    first = p.unitvisitnumber == p.groupby("patienthealthsystemstayid").unitvisitnumber.transform("min")
    young = p.patienthealthsystemstayid.isin(
        p.loc[first & ~(p.age_num >= cfg["states"]["min_age"]), "patienthealthsystemstayid"])
    drop(young, f"age < {sc['min_age']} or missing")
    for reason, k in drops.items():
        log.info("dropped %d hospital stays: %s", k, reason)
    first = p.unitvisitnumber == p.groupby("patienthealthsystemstayid").unitvisitnumber.transform("min")

    # Unit stays in hospital time, clipped to the hospital stay.
    p["los"] = p.hospitaldischargeoffset - p.hospitaladmitoffset
    p["u_start"] = -p.hospitaladmitoffset
    p["u_end"] = np.minimum(p.u_start + p.unitdischargeoffset, p.los)
    log.info("unit stays ending after hospital discharge (clipped): %d",
             int((p.u_start + p.unitdischargeoffset > p.los).sum()))
    units = p.set_index("patientunitstayid")[["patienthealthsystemstayid", "u_start", "u_end"]]

    # Support events -> hospital time, restricted to their unit stay, merged into intervals.
    ev = support_events(raw, cfg).join(units, on="patientunitstayid", how="inner")
    ev["t"] = ev.u_start + ev.offset
    ev = ev[(ev.t >= ev.u_start) & (ev.t <= ev.u_end)]
    gap = sc["support_gap_fill_min"]
    sup_iv = {h: np.vstack([merge_points(g.t.to_numpy(), gap) for _, g in gu.groupby("patientunitstayid")])
              for h, gu in ev.groupby("patienthealthsystemstayid")}

    # Sepsis documented within the window after first ICU admission (no discharge look-ahead).
    dx = raw["diagnosis"].join(units, on="patientunitstayid", how="inner")
    dx["t"] = dx.u_start + dx.diagnosisoffset
    first_icu = p.groupby("patienthealthsystemstayid").u_start.min()
    dx = dx[dx.diagnosisstring.str.contains(cc["sepsis_regex"], case=False, na=False)]
    sepsis_any = set(dx.patienthealthsystemstayid)
    early = dx.t <= dx.patienthealthsystemstayid.map(first_icu) + cc["sepsis_window_min"]
    sepsis_early = set(dx.loc[early, "patienthealthsystemstayid"])
    apache = p[first & p.apacheadmissiondx.str.contains(cc["sepsis_regex"], case=False, na=False)]
    sepsis_early |= set(apache.patienthealthsystemstayid)
    sepsis_any |= set(apache.patienthealthsystemstayid)

    hosp = raw["hospital"].set_index("hospitalid").region
    rows, n_trunc = [], 0
    for h, g in p.groupby("patienthealthsystemstayid", sort=True):
        f = g.loc[g.unitvisitnumber.idxmin()]
        seq, trunc = day_sequence(
            f.los, g[["u_start", "u_end"]].to_numpy(float), sup_iv.get(h, np.empty((0, 2))),
            f.hospitaldischargestatus == "Expired", sc["max_days"], sc["icu_overlap_min"])
        n_trunc += trunc
        rows.append({
            "subject_id": f.uniquepid,
            "hadm_id": int(h),
            "hospitalid": int(f.hospitalid),
            "region": hosp.get(f.hospitalid),
            "year": int(f.hospitaldischargeyear),
            "era": "late" if f.hospitaldischargeyear >= cfg["era"]["late_from_year"] else "early",
            "age_band": "<65" if f.age_num < cc["age_cut"] else ">=65",
            "sepsis": int(h in sepsis_early),
            "sepsis_any": int(h in sepsis_any),
            "truncated": trunc,
            "seq": seq,
        })
    traj = pd.DataFrame(rows, columns=TRAJ_COLUMNS)
    log.info("hospital stays: %d raw -> %d kept (%d subjects); truncated at %d days: %d",
             n0, len(traj), traj.subject_id.nunique(), sc["max_days"], n_trunc)
    log.info("era counts: %s", traj.era.value_counts().to_dict())
    return traj


def transitions(traj: pd.DataFrame, covariates=("era", "age_band", "sepsis")) -> pd.DataFrame:
    """Long table of observed transitions with previous state (-1 on day 0)."""
    t = traj[["subject_id", "hadm_id", *covariates, "seq"]].copy()
    t["seq"] = t.seq.apply(list)  # parquet round-trips lists as numpy arrays
    t = t[t.seq.str.len() >= 2]
    t["from_state"] = t.seq.str[:-1]
    t["to_state"] = t.seq.str[1:]
    t["prev_state"] = t.seq.apply(lambda s: [-1] + s[:-2])
    t = t.drop(columns="seq").explode(["from_state", "to_state", "prev_state"])
    t["day"] = t.groupby("hadm_id").cumcount()
    for c in ("from_state", "to_state", "prev_state"):
        t[c] = t[c].astype(int)
    return t.reset_index(drop=True)[
        ["subject_id", "hadm_id", "day", "from_state", "to_state", "prev_state", *covariates]]
