"""Closed-form absorbing-chain quantities (spec Section 5).

LOS counts transient days, i.e. the number of steps before absorption, which
matches len(seq) - 1 for an observed (non-truncated) trajectory.
"""

import numpy as np

from . import ABSORBING, DEAD, ICU, ICU_SUPPORT, TRANSIENT


def blocks(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return P[np.ix_(TRANSIENT, TRANSIENT)], P[np.ix_(TRANSIENT, ABSORBING)]


def fundamental(P: np.ndarray) -> np.ndarray:
    Q, _ = blocks(P)
    return np.linalg.inv(np.eye(len(Q)) - Q)


def expected_los(P: np.ndarray) -> np.ndarray:
    return fundamental(P).sum(1)


def var_los(P: np.ndarray) -> np.ndarray:
    N = fundamental(P)
    tau = N.sum(1)
    return (2 * N - np.eye(len(N))) @ tau - tau**2


def absorption(P: np.ndarray) -> np.ndarray:
    """B[i, k]: probability transient state i is absorbed in ABSORBING[k]."""
    _, R = blocks(P)
    return fundamental(P) @ R


def mortality(P: np.ndarray, pi0: np.ndarray) -> float:
    return float(pi0[TRANSIENT] @ absorption(P)[:, ABSORBING.index(DEAD)])


def occupancy(P: np.ndarray, pi0: np.ndarray, horizon: int) -> np.ndarray:
    """pi_t = pi0 P^t for t = 0..horizon, shape (horizon + 1, S)."""
    out = np.empty((horizon + 1, len(pi0)))
    out[0] = pi0
    for t in range(horizon):
        out[t + 1] = out[t] @ P
    return out


def expected_icu_days(P: np.ndarray, pi0: np.ndarray, horizon: int) -> float:
    """Sum over t < horizon of P(in ICU or ICU_SUPPORT at day t)."""
    pi = occupancy(P, pi0, horizon - 1)
    return float(pi[:, [ICU, ICU_SUPPORT]].sum())


def quasi_stationary(P: np.ndarray) -> np.ndarray:
    """Normalized leading left eigenvector of Q (descriptive only)."""
    Q, _ = blocks(P)
    w, v = np.linalg.eig(Q.T)
    x = np.abs(np.real(v[:, np.argmax(np.real(w))]))
    return x / x.sum()


def summary(P: np.ndarray, pi0: np.ndarray, horizon: int) -> dict:
    """All Section 5 scalars/vectors for one matrix (used per uncertainty draw)."""
    return {
        "expected_los": expected_los(P),
        "var_los": var_los(P),
        "absorption": absorption(P),
        "mortality": mortality(P, pi0),
        "expected_icu_days": expected_icu_days(P, pi0, horizon),
        "occupancy": occupancy(P, pi0, horizon),
        "quasi_stationary": quasi_stationary(P),
    }
