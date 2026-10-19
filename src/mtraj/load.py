"""Read the eICU-CRD demo tables and extract organ-support events.

All offsets in eICU are minutes relative to the unit (ICU) admission of the
row's ``patientunitstayid``. Conversion to hospital time happens in states.py.
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TABLES = {
    "patient": [
        "patientunitstayid", "patienthealthsystemstayid", "uniquepid", "age",
        "hospitalid", "apacheadmissiondx", "hospitaladmitoffset",
        "hospitaldischargeyear", "hospitaldischargeoffset",
        "hospitaldischargestatus", "unitvisitnumber", "unitdischargeoffset",
    ],
    "hospital": ["hospitalid", "region"],
    "diagnosis": ["patientunitstayid", "diagnosisoffset", "diagnosisstring"],
    "treatment": ["patientunitstayid", "treatmentoffset", "treatmentstring"],
    "infusiondrug": ["patientunitstayid", "infusionoffset", "drugname"],
    "respiratoryCharting": ["patientunitstayid", "respchartoffset", "respchartvaluelabel"],
}


def read_table(root: Path, name: str) -> pd.DataFrame:
    path = Path(root) / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing eICU table {path}; see README for download")
    df = pd.read_csv(path, usecols=TABLES[name])
    missing = set(TABLES[name]) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    return df


def load_raw(root: Path) -> dict[str, pd.DataFrame]:
    return {name: read_table(root, name) for name in TABLES}


def _require_matches(mask: pd.Series, what: str) -> None:
    # eICU has no item dictionary to verify against, so fail loudly if a
    # pattern silently matches nothing (e.g. a renamed string).
    if not mask.any():
        raise ValueError(f"support pattern for {what} matched no rows")
    log.info("%s: %d matching rows", what, int(mask.sum()))


def support_events(raw: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Point events (patientunitstayid, offset) indicating vent or vasopressor."""
    sc = cfg["support"]

    tr = raw["treatment"]
    s = tr.treatmentstring.str.lower()
    m = pd.Series(False, index=tr.index)
    for pat in sc["treatment_include"]:
        m |= s.str.contains(pat.lower(), regex=False)
    for pat in sc["treatment_exclude"]:
        m &= ~s.str.contains(pat.lower(), regex=False)
    _require_matches(m, "treatment (vent/vasopressor)")
    ev_tr = tr.loc[m, ["patientunitstayid", "treatmentoffset"]].set_axis(
        ["patientunitstayid", "offset"], axis=1)

    inf = raw["infusiondrug"]
    m = inf.drugname.str.contains(sc["vasopressor_regex"], case=False, na=False)
    _require_matches(m, "infusiondrug (vasopressor)")
    ev_inf = inf.loc[m, ["patientunitstayid", "infusionoffset"]].set_axis(
        ["patientunitstayid", "offset"], axis=1)

    rc = raw["respiratoryCharting"]
    m = rc.respchartvaluelabel.isin(sc["vent_chart_labels"])
    _require_matches(m, "respiratoryCharting (vent settings)")
    ev_rc = rc.loc[m, ["patientunitstayid", "respchartoffset"]].set_axis(
        ["patientunitstayid", "offset"], axis=1)

    ev = pd.concat([ev_tr, ev_inf, ev_rc], ignore_index=True)
    return ev.drop_duplicates().sort_values(["patientunitstayid", "offset"], ignore_index=True)
