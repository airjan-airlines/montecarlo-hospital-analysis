"""Transition-matrix estimation: first/second-order MLE and Dirichlet shrinkage."""

import numpy as np
import pandas as pd

from . import ABSORBING, STATES, TRANSIENT

S = len(STATES)


def counts(trans: pd.DataFrame) -> np.ndarray:
    """C[i, j] over transient-origin transitions (absorbing rows are zero)."""
    C = np.zeros((S, S))
    np.add.at(C, (trans.from_state.to_numpy(), trans.to_state.to_numpy()), 1)
    return C


def counts2(trans: pd.DataFrame) -> np.ndarray:
    """C2[prev + 1, cur, next]; index 0 is the day-0 category prev = -1."""
    C = np.zeros((S + 1, S, S))
    np.add.at(C, (trans.prev_state.to_numpy() + 1, trans.from_state.to_numpy(),
                  trans.to_state.to_numpy()), 1)
    return C


def _with_absorbing(P: np.ndarray) -> np.ndarray:
    P[..., ABSORBING, :] = np.eye(S)[ABSORBING]
    return P


def mle(C: np.ndarray) -> np.ndarray:
    rows = C[TRANSIENT].sum(1)
    if (rows == 0).any():
        empty = [STATES[i] for i, r in zip(TRANSIENT, rows) if r == 0]
        raise ValueError(f"no transitions observed from {empty}; use shrinkage")
    P = np.zeros((S, S))
    P[TRANSIENT] = C[TRANSIENT] / rows[:, None]
    return _with_absorbing(P)


def pooled_target(C: np.ndarray) -> np.ndarray:
    """Overall next-state frequencies among transient-origin transitions, as every row.

    Accepts a batch of count matrices (..., S, S).
    """
    f = C[..., TRANSIENT, :].sum(-2)
    return np.broadcast_to((f / f.sum(-1, keepdims=True))[..., None, :], C.shape).copy()


def shrink(C: np.ndarray, alpha: float, m: np.ndarray) -> np.ndarray:
    """Dirichlet posterior mean (C + alpha m) / (rowsum + alpha) on transient rows."""
    P = np.zeros_like(C, dtype=float)
    num = C[..., TRANSIENT, :] + alpha * m[..., TRANSIENT, :]
    P[..., TRANSIENT, :] = num / num.sum(-1, keepdims=True)
    return _with_absorbing(P)


def mle2(C2: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Second-order MLE P2[prev + 1, cur]; unobserved (prev, cur) rows use fallback[cur]."""
    P2 = np.broadcast_to(fallback, C2.shape).copy()
    tot = C2.sum(-1, keepdims=True)
    obs = (tot[..., 0] > 0)
    P2[obs] = (C2 / np.where(tot == 0, 1, tot))[obs]
    return _with_absorbing(P2)


def shrink2(C2: np.ndarray, alpha: float, P1: np.ndarray) -> np.ndarray:
    """Second-order rows shrunk toward the first-order row of the current state."""
    return shrink(C2, alpha, np.broadcast_to(P1, C2.shape))


def initial_distribution(trajectories: pd.DataFrame) -> np.ndarray:
    first = trajectories.seq.str[0].to_numpy(dtype=int)
    return np.bincount(first, minlength=S) / len(first)
