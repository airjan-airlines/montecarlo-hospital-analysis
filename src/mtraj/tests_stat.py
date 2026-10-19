"""Hypothesis tests: parameter stability across groups (7.4) and Markov order (7.5)."""

import numpy as np
import pandas as pd
from scipy import stats

from . import ABSORBING, STATES, TRANSIENT
from . import estimate as E
from .simulate import simulate

S = E.S


def _row_normalize(C: np.ndarray) -> np.ndarray:
    tot = C.sum(-1, keepdims=True)
    return np.divide(C, tot, out=np.zeros_like(C), where=tot > 0)


def _loglik(C: np.ndarray, P: np.ndarray) -> float:
    m = C > 0
    return float((C[m] * np.log(P[m])).sum())


def _free_params(C: np.ndarray) -> int:
    """Sum over observed transient rows of (observed destinations - 1)."""
    rows = C[..., TRANSIENT, :].reshape(-1, S)
    nz = (rows > 0).sum(1)
    return int((nz[nz > 0] - 1).sum())


# --- stability ---------------------------------------------------------------

def group_counts(Cs: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    """(n_groups, S, S) counts from per-subject counts and integer group labels."""
    onehot = np.eye(n_groups)[labels]
    return np.einsum("sg,sij->gij", onehot, Cs)


def g_statistic(Cg: np.ndarray) -> float:
    """G = 2 sum_g sum_ij C_g log(P_g / P_pooled) over transient rows."""
    T = TRANSIENT
    Pg, Pp = _row_normalize(Cg[:, T]), _row_normalize(Cg[:, T].sum(0))
    m = Cg[:, T] > 0
    return float(2 * (Cg[:, T][m] * np.log(Pg[m] / np.broadcast_to(Pp, Pg.shape)[m])).sum())


def stability_test(Cs: np.ndarray, labels: np.ndarray, n_perm: int,
                   rng: np.random.Generator) -> dict:
    """G-test of homogeneity across groups, chi-square and subject-permutation p-values.

    df = (G - 1) * sum_i (d_i - 1), where d_i counts destinations observed from
    transient row i in the pooled data (structurally empty cells removed).
    """
    codes, names = pd.factorize(labels)
    G = len(names)
    Cg = group_counts(Cs, codes, G)
    g = g_statistic(Cg)
    df = (G - 1) * _free_params(Cg.sum(0))
    perm = np.array([g_statistic(group_counts(Cs, rng.permutation(codes), G)) for _ in range(n_perm)])
    return {"groups": list(map(str, names)), "n_per_group": np.bincount(codes).tolist(),
            "G": g, "df": df, "p_chi2": float(stats.chi2.sf(g, df)),
            "p_perm": float((1 + (perm >= g).sum()) / (1 + n_perm))}


def drift_table(Cs: np.ndarray, labels: np.ndarray, a: str, b: str, n_boot: int,
                rng: np.random.Generator) -> pd.DataFrame:
    """P_b - P_a per transient transition with subject-bootstrap 95% CIs, by |z|."""
    out = {}
    for name in (a, b):
        Cg = Cs[labels == name]
        W = rng.multinomial(len(Cg), np.full(len(Cg), 1 / len(Cg)), size=n_boot)
        out[name] = (_row_normalize(Cg.sum(0)), _row_normalize(np.einsum("bs,sij->bij", W, Cg)))
    point = out[b][0] - out[a][0]
    boot = out[b][1] - out[a][1]
    rows = []
    for i in TRANSIENT:
        for j in range(S):
            sd = boot[:, i, j].std(ddof=1)
            lo, hi = np.percentile(boot[:, i, j], [2.5, 97.5])
            rows.append({"from": STATES[i], "to": STATES[j], f"P_{a}": out[a][0][i, j],
                         f"P_{b}": out[b][0][i, j], "diff": point[i, j], "lo": lo, "hi": hi,
                         "z": point[i, j] / sd if sd > 0 else np.nan})
    df = pd.DataFrame(rows)
    return df.reindex(df.z.abs().sort_values(ascending=False).index).reset_index(drop=True)


# --- Markov order --------------------------------------------------------------

def path_counts(paths: np.ndarray, max_days: int) -> tuple[np.ndarray, np.ndarray]:
    """First- and second-order counts from simulated paths, mirroring states.py.

    A path not absorbed by day max_days is truncated to max_days states, so its
    last step (index max_days - 1 -> max_days) is kept only if it is absorbing.
    """
    C1, C2 = np.zeros((S, S)), np.zeros((S + 1, S, S))
    for t in range(max_days):
        frm, to = paths[:, t], paths[:, t + 1]
        ok = np.isin(frm, TRANSIENT)
        if t == max_days - 1:
            ok &= np.isin(to, ABSORBING)
        prev = paths[:, t - 1] + 1 if t else np.zeros(len(paths), dtype=int)
        np.add.at(C1, (frm[ok], to[ok]), 1)
        np.add.at(C2, (prev[ok], frm[ok], to[ok]), 1)
    return C1, C2


def order_lr(C1: np.ndarray, C2: np.ndarray) -> dict:
    """LR, AIC and BIC for first- vs second-order (prev = -1 is its own category)."""
    P1 = _row_normalize(C1)
    P2 = _row_normalize(C2)
    ll1, ll2 = _loglik(C1, P1), _loglik(C2, P2)
    k1, k2 = _free_params(C1), _free_params(C2)
    n = C1[TRANSIENT].sum()
    lr = 2 * (ll2 - ll1)
    return {"loglik1": ll1, "loglik2": ll2, "k1": k1, "k2": k2, "LR": lr, "df": k2 - k1,
            "p_chi2": float(stats.chi2.sf(lr, k2 - k1)),
            "aic1": 2 * k1 - 2 * ll1, "aic2": 2 * k2 - 2 * ll2,
            "bic1": k1 * np.log(n) - 2 * ll1, "bic2": k2 * np.log(n) - 2 * ll2}


def order_test(C1: np.ndarray, C2: np.ndarray, pi0: np.ndarray, n_adm: int, max_days: int,
               n_boot: int, rng: np.random.Generator) -> dict:
    """order_lr plus a parametric-bootstrap p-value under the fitted first-order model."""
    res = order_lr(C1, C2)
    P1 = E.mle(C1)
    null = np.empty(n_boot)
    for b in range(n_boot):
        null[b] = order_lr(*path_counts(simulate(P1, pi0, n_adm, max_days, rng), max_days))["LR"]
    res["p_boot"] = float((1 + (null >= res["LR"]).sum()) / (1 + n_boot))
    res["null_LR_mean"] = float(null.mean())
    return res
