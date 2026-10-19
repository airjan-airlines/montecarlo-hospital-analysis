"""Tail risk of total ICU bed-days for a cohort of K admissions (spec 6.5)."""

import numpy as np
import pandas as pd

from . import ICU, ICU_SUPPORT
from .simulate import simulate, simulate_mixture
from .uncertainty import dirichlet, fit


def bed_days(paths: np.ndarray, K: int, horizon: int) -> np.ndarray:
    icu = np.isin(paths[:, :horizon], [ICU, ICU_SUPPORT]).sum(1)
    return icu.reshape(-1, K).sum(1)


def plug_in(P, pi0, K: int, n_cohorts: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    return bed_days(simulate(P, pi0, K * n_cohorts, horizon, rng), K, horizon)


def predictive(Ps: np.ndarray, pi0s: np.ndarray, K: int, horizon: int,
               rng: np.random.Generator) -> np.ndarray:
    """One cohort per parameter draw: cohort k simulates all K patients under (Ps[k], pi0s[k])."""
    draw = np.repeat(np.arange(len(Ps)), K)
    return bed_days(simulate_mixture(Ps, pi0s, draw, horizon, rng), K, horizon)


def posterior_predictive(C, d0, alpha: float, K: int, n_cohorts: int, horizon: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Each cohort first draws (P, pi0) from the Dirichlet posterior, then simulates."""
    return predictive(*dirichlet(C, d0, n_cohorts, alpha, rng), K, horizon, rng)


def ratio_vs_sample_size(Cs: np.ndarray, D0s: np.ndarray, sizes, alpha: float, K: int,
                         n_cohorts: int, horizon: int, rng: np.random.Generator) -> pd.DataFrame:
    """VaR99 / ES95 posterior-predictive-to-plug-in ratios when fitting on n subjects.

    Shows how the cost of ignoring parameter uncertainty grows as data shrink.
    """
    rows = []
    for n in sizes:
        idx = rng.choice(len(Cs), size=min(n, len(Cs)), replace=False)
        C, d0 = Cs[idx].sum(0), D0s[idx].sum(0)
        plug = plug_in(fit(C, alpha), d0 / d0.sum(), K, n_cohorts, horizon, rng)
        post = posterior_predictive(C, d0, alpha, K, n_cohorts, horizon, rng)
        a, b = risk_measures(plug), risk_measures(post)
        rows.append({"n_subjects": len(idx), "n_transitions": int(C.sum()),
                     **{f"{k}_ratio": b[k] / a[k] for k in ("sd", "VaR99", "ES95")}})
    return pd.DataFrame(rows)


def risk_measures(x: np.ndarray) -> dict[str, float]:
    var95, var99 = np.quantile(x, [0.95, 0.99])
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "VaR95": float(var95),
            "VaR99": float(var99), "ES95": float(x[x >= var95].mean())}


def compare(plug: np.ndarray, post: np.ndarray) -> pd.DataFrame:
    a, b = risk_measures(plug), risk_measures(post)
    return pd.DataFrame({"plug_in": a, "posterior_predictive": b,
                         "ratio": {k: b[k] / a[k] for k in a}})
