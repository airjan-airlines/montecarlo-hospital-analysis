"""Parameter uncertainty: subject-level bootstrap and Dirichlet posterior draws."""

import numpy as np
import pandas as pd

from . import TRANSIENT
from . import analytic as A
from . import estimate as E

S = E.S


def subject_counts(traj: pd.DataFrame, trans: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject transition counts (n_subj, S, S) and day-0 state counts (n_subj, S)."""
    codes, subjects = pd.factorize(traj.subject_id)
    sub_of = dict(zip(subjects, range(len(subjects))))
    C = np.zeros((len(subjects), S, S))
    np.add.at(C, (trans.subject_id.map(sub_of).to_numpy(), trans.from_state.to_numpy(),
                  trans.to_state.to_numpy()), 1)
    D0 = np.zeros((len(subjects), S))
    np.add.at(D0, (codes, traj.seq.str[0].to_numpy(dtype=int)), 1)
    return C, D0


def fit(C: np.ndarray, alpha: float) -> np.ndarray:
    """Shrunk first-order fit toward the pooled row target; works on batches."""
    return E.shrink(C, alpha, E.pooled_target(C)) if alpha > 0 else E.mle(C)


def bootstrap(Cs: np.ndarray, D0s: np.ndarray, B: int, alpha: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Resample subjects with replacement B times and refit; returns (Ps, pi0s)."""
    n = len(Cs)
    W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    Cb = np.einsum("bs,sij->bij", W, Cs)
    d0 = W @ D0s
    if alpha > 0:
        Ps = fit(Cb, alpha)
    else:
        Ps = np.stack([E.mle(c) for c in Cb])
    return Ps, d0 / d0.sum(1, keepdims=True)


def dirichlet_params(C: np.ndarray, alpha: float) -> np.ndarray:
    return C + alpha * E.pooled_target(C)


def dirichlet(C: np.ndarray, d0: np.ndarray, B: int, alpha: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rows ~ Dirichlet(C + alpha m); pi0 ~ Dirichlet(day-0 counts)."""
    a = dirichlet_params(C, alpha)[TRANSIENT]
    g = rng.gamma(np.broadcast_to(a, (B, *a.shape)))
    Ps = np.broadcast_to(np.eye(S), (B, S, S)).copy()
    Ps[:, TRANSIENT] = g / g.sum(-1, keepdims=True)
    g0 = rng.gamma(np.broadcast_to(d0, (B, S)))
    return Ps, g0 / g0.sum(1, keepdims=True)


def summarize(Ps: np.ndarray, pi0s: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Section 5 quantities for every draw, stacked along axis 0."""
    rows = [A.summary(P, p, horizon) for P, p in zip(Ps, pi0s)]
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out["cohort_los"] = np.einsum("bi,bi->b", pi0s[:, TRANSIENT], out["expected_los"])
    out["death_prob"] = out["absorption"][..., 1]
    return out


SCALARS = {
    "mortality": lambda d: d["mortality"],
    "expected_icu_days": lambda d: d["expected_icu_days"],
    "cohort_los": lambda d: d["cohort_los"],
    **{f"los_from_{s}": (lambda d, i=i: d["expected_los"][:, i]) for i, s in enumerate(["WARD", "ICU", "ICU_SUPPORT"])},
    **{f"death_from_{s}": (lambda d, i=i: d["death_prob"][:, i]) for i, s in enumerate(["WARD", "ICU", "ICU_SUPPORT"])},
}


def interval(x: np.ndarray) -> dict[str, float]:
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    return {"median": float(med), "lo": float(lo), "hi": float(hi)}


def compare_methods(boot: dict, post: dict) -> pd.DataFrame:
    """Median and 95% interval per quantity for both methods, with width ratio."""
    rows = []
    for name, get in SCALARS.items():
        b, p = interval(get(boot)), interval(get(post))
        rows.append({"quantity": name,
                     **{f"boot_{k}": v for k, v in b.items()},
                     **{f"post_{k}": v for k, v in p.items()},
                     "width_ratio_boot_over_post": (b["hi"] - b["lo"]) / (p["hi"] - p["lo"])})
    return pd.DataFrame(rows)


def occupancy_bands(draws: dict) -> np.ndarray:
    """(2.5, 50, 97.5) percentiles of pi_t over draws, shape (3, H+1, S)."""
    return np.percentile(draws["occupancy"], [2.5, 50, 97.5], axis=0)
