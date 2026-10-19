"""Synthetic trajectories from a known chain, in the same format as states.py."""

import numpy as np
import pandas as pd

from . import DEAD, DISCHARGED, ICU, ICU_SUPPORT, WARD
from .simulate import simulate, simulate_second_order

P_TRUE = np.array([
    [0.80, 0.04, 0.01, 0.145, 0.005],
    [0.20, 0.62, 0.08, 0.08, 0.02],
    [0.02, 0.15, 0.75, 0.01, 0.07],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])
PI0 = np.array([0.15, 0.60, 0.25, 0, 0])


def _move(P: np.ndarray, i: int, src: int, dst: int, amt: float) -> np.ndarray:
    """Shift probability mass amt in row i from column src to column dst."""
    P = P.copy()
    P[i, src] -= amt
    P[i, dst] += amt
    assert P[i, src] >= 0
    return P


def cell_matrix(era: str, age_band: str, sepsis: int, era_shift: float = 0.0) -> np.ndarray:
    """P_true adjusted for covariates (and the late era if era_shift > 0)."""
    P = P_TRUE
    if sepsis:
        P = _move(P, ICU, ICU, ICU_SUPPORT, 0.05)
        P = _move(P, ICU_SUPPORT, ICU_SUPPORT, DEAD, 0.03)
    if age_band == ">=65":
        for i in (WARD, ICU, ICU_SUPPORT):
            P = _move(P, i, i, DEAD, 0.01)
    if era == "late" and era_shift:
        P = _move(P, ICU, ICU, DISCHARGED, era_shift)
        P = _move(P, ICU_SUPPORT, ICU_SUPPORT, ICU, era_shift)
    return P


def second_order_tensor(P: np.ndarray) -> np.ndarray:
    """P2[prev, cur]: patients just stepped down from support relapse more."""
    S = len(P)
    P2 = np.broadcast_to(P, (S, S, S)).copy()
    P2[ICU_SUPPORT, ICU] = [0.10, 0.50, 0.25, 0.10, 0.05]
    return P2


def paths_to_seqs(paths: np.ndarray, max_days: int) -> tuple[list[list[int]], np.ndarray]:
    """Cut each path at first absorption; paths not absorbed by max_days are truncated."""
    absorbed = paths >= DISCHARGED
    hit = absorbed.any(1)
    first = np.where(hit, absorbed.argmax(1), max_days)
    seqs = [row[: k + 1].tolist() if h else row[:max_days].tolist()
            for row, k, h in zip(paths, first, hit)]
    return seqs, ~hit


def generate(n_subjects: int, rng: np.random.Generator, max_days: int = 60,
             era_shift: float = 0.0, second_order: bool = False,
             p_sepsis: float = 0.2, p_old: float = 0.5) -> pd.DataFrame:
    """Simulate subjects with 1-3 admissions each; returns trajectories like states.py."""
    n_adm = rng.integers(1, 4, n_subjects)
    subj = np.repeat(np.arange(n_subjects), n_adm)
    old = rng.random(n_subjects) < p_old
    df = pd.DataFrame({
        "subject_id": [f"s{i}" for i in subj],
        "hadm_id": np.arange(len(subj)),
        "era": np.where(rng.random(len(subj)) < 0.5, "early", "late"),
        "age_band": np.where(old[subj], ">=65", "<65"),
        "sepsis": (rng.random(len(subj)) < p_sepsis).astype(int),
    })
    df["seq"] = None
    df["truncated"] = False
    for (era, age, sep), idx in df.groupby(["era", "age_band", "sepsis"]).groups.items():
        P = cell_matrix(era, age, sep, era_shift)
        if second_order:
            paths = simulate_second_order(second_order_tensor(P), P, PI0, len(idx), max_days, rng)
        else:
            paths = simulate(P, PI0, len(idx), max_days, rng)
        seqs, trunc = paths_to_seqs(paths, max_days)
        df.loc[idx, "seq"] = pd.Series(seqs, index=idx, dtype=object)
        df.loc[idx, "truncated"] = trunc
    return df
