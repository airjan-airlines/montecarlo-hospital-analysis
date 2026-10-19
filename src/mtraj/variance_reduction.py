"""Monte Carlo estimators with variance reduction (spec 6.3) and CRN contrasts (6.4).

Targets per simulated path over horizon H:
  icu_days  ICU or ICU_SUPPORT days among days 0..H-1 (the control variate)
  y1        icu_days > icu_days_threshold                (no closed form)
  y2        LOS > los_threshold, i.e. still in hospital on day los_threshold
  mort      dead by day H                                 (closed form: pi_H[DEAD])

Every estimator is linear in per-replication path means, so replications are
processed in chunks and only means are kept.
"""

import numpy as np
import pandas as pd

from . import DEAD, ICU, ICU_SUPPORT, TRANSIENT
from . import analytic as A
from .simulate import simulate

TARGETS = ["y1", "y2", "mort"]
CHUNK = 1_000_000  # paths simulated per batch
VRF_BOOT = 1000    # bootstrap reps over replications for VRF intervals


def outcomes(paths: np.ndarray, horizon: int, icu_thr: int, los_thr: int) -> dict[str, np.ndarray]:
    icu = np.isin(paths[:, :horizon], [ICU, ICU_SUPPORT]).sum(1).astype(float)
    return {
        "icu_days": icu,
        "y1": (icu > icu_thr).astype(float),
        "y2": np.isin(paths[:, los_thr], TRANSIENT).astype(float),
        "mort": (paths[:, horizon] == DEAD).astype(float),
    }


def exact_values(P: np.ndarray, pi0: np.ndarray, horizon: int, los_thr: int) -> dict[str, float]:
    """Closed-form values where they exist (y1 has none)."""
    pi = A.occupancy(P, pi0, horizon)
    return {"icu_days": A.expected_icu_days(P, pi0, horizon),
            "y2": float(pi[los_thr, TRANSIENT].sum()),
            "mort": float(pi[horizon, DEAD])}


def _uniforms(rng: np.random.Generator, r: int, n: int, width: int, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return rng.random((r * n, width))
    half = rng.random((r, n // 2, width))
    return np.concatenate([half, 1 - half], axis=1).reshape(r * n, width)


def rep_means(P, pi0, n: int, R: int, rng: np.random.Generator, vc: dict, horizon: int,
              antithetic: bool = False, order=None) -> dict[str, np.ndarray]:
    """Per-replication sample means of every outcome, shape (R,) each."""
    if antithetic and n % 2:
        raise ValueError("antithetic needs even n")
    out = {k: [] for k in ["icu_days", *TARGETS]}
    r_batch = max(1, CHUNK // n)
    for start in range(0, R, r_batch):
        r = min(r_batch, R - start)
        u = _uniforms(rng, r, n, horizon + 1, antithetic)
        paths = simulate(P, pi0, r * n, horizon, uniforms=u, column_order=order)
        for k, v in outcomes(paths, horizon, vc["icu_days_threshold"], vc["los_threshold"]).items():
            out[k].append(v.reshape(r, n).mean(1))
    return {k: np.concatenate(v) for k, v in out.items()}


def pilot_beta(P, pi0, vc: dict, horizon: int, rng: np.random.Generator,
               antithetic: bool, order=None) -> dict[str, float]:
    """CV coefficient beta = Cov(Y, C) / Var(C) from an independent pilot run.

    For antithetic sampling the regression is on pair averages.
    """
    n = vc["n_pilot"]
    u = _uniforms(rng, 1, n, horizon + 1, antithetic)
    o = outcomes(simulate(P, pi0, n, horizon, uniforms=u, column_order=order),
                 horizon, vc["icu_days_threshold"], vc["los_threshold"])
    if antithetic:
        o = {k: 0.5 * (v[: n // 2] + v[n // 2:]) for k, v in o.items()}
    c = o["icu_days"]
    return {k: float(np.cov(o[k], c)[0, 1] / c.var(ddof=1)) for k in TARGETS}


def run_estimators(P, pi0, vc: dict, horizon: int, rng: np.random.Generator) -> pd.DataFrame:
    """All estimators x n x target over R replications; one row per combination."""
    order = vc["severity_order"]
    exact = exact_values(P, pi0, horizon, vc["los_threshold"])
    beta = pilot_beta(P, pi0, vc, horizon, rng, antithetic=False)
    beta_anti = pilot_beta(P, pi0, vc, horizon, rng, antithetic=True, order=order)
    R, rows = vc["R"], []
    for n in vc["n_grid"]:
        plain = rep_means(P, pi0, n, R, rng, vc, horizon)
        naive = rep_means(P, pi0, n, R, rng, vc, horizon, antithetic=True)
        anti = rep_means(P, pi0, n, R, rng, vc, horizon, antithetic=True, order=order)
        for t in TARGETS:
            est = {
                "plain": plain[t],
                "antithetic_naive": naive[t],
                "antithetic": anti[t],
                "control_variate": plain[t] - beta[t] * (plain["icu_days"] - exact["icu_days"]),
                "antithetic_cv": anti[t] - beta_anti[t] * (anti["icu_days"] - exact["icu_days"]),
            }
            var_plain = est["plain"].var(ddof=1)
            # VRF CI: resample replications jointly (CV shares plain's streams)
            boot = rng.integers(0, R, (VRF_BOOT, R))
            var_plain_b = est["plain"][boot].var(1, ddof=1)
            for method, x in est.items():
                lo, hi = np.percentile(var_plain_b / x[boot].var(1, ddof=1), [2.5, 97.5])
                rows.append({"target": t, "n": n, "method": method, "mean": x.mean(),
                             "se": x.std(ddof=1), "vrf": var_plain / x.var(ddof=1),
                             "vrf_lo": lo, "vrf_hi": hi,
                             "exact": exact.get(t, np.nan),
                             "z_vs_exact": (x.mean() - exact[t]) / (x.std(ddof=1) / np.sqrt(R))
                             if t in exact else np.nan})
    return pd.DataFrame(rows)


def crn_contrast(Pa, pi0a, Pb, pi0b, vc: dict, horizon: int,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Difference a - b in each outcome: common random numbers vs independent streams."""
    n, R, order = vc["crn_n"], vc["R"], vc["severity_order"]

    def means(P, pi0, u):
        paths = simulate(P, pi0, len(u), horizon, uniforms=u, column_order=order)
        o = outcomes(paths, horizon, vc["icu_days_threshold"], vc["los_threshold"])
        return {k: v.reshape(-1, n).mean(1) for k, v in o.items()}

    crn, ind = {k: [] for k in ["icu_days", *TARGETS]}, {k: [] for k in ["icu_days", *TARGETS]}
    r_batch = max(1, CHUNK // n)
    for start in range(0, R, r_batch):
        r = min(r_batch, R - start)
        u = rng.random((r * n, horizon + 1))
        a, b = means(Pa, pi0a, u), means(Pb, pi0b, u)
        b_ind = means(Pb, pi0b, rng.random((r * n, horizon + 1)))
        for k in crn:
            crn[k].append(a[k] - b[k])
            ind[k].append(a[k] - b_ind[k])
    rows = []
    for k in crn:
        c, i = np.concatenate(crn[k]), np.concatenate(ind[k])
        rows.append({"target": k, "diff_crn": c.mean(), "se_crn": c.std(ddof=1),
                     "diff_indep": i.mean(), "se_indep": i.std(ddof=1),
                     "vrf": i.var(ddof=1) / c.var(ddof=1)})
    return pd.DataFrame(rows)
