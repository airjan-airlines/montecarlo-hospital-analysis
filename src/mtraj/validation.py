"""Out-of-sample scoring of next-state models (spec 7.1-7.3) and shrinkage CV (4.3).

Everything works on a ``Data`` bundle of per-transition integer arrays so that
fitting a model on any subset of subjects is a bincount, not a groupby.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import DEAD, TRANSIENT
from . import analytic as A
from . import estimate as E

S = E.S
MODELS = ["persistence", "marginal", "mle1", "shrink1", "mle2", "shrink2", "strat"]
SHRUNK = ["shrink1", "shrink2", "strat"]


@dataclass
class Data:
    subj: np.ndarray      # subject code per transition
    frm: np.ndarray
    to: np.ndarray
    prev: np.ndarray      # -1 on day 0
    stratum: np.ndarray   # age_band x sepsis code, 0..3
    n_subj: int

    def subset(self, mask: np.ndarray) -> "Data":
        return Data(self.subj[mask], self.frm[mask], self.to[mask], self.prev[mask],
                    self.stratum[mask], self.n_subj)


STRATA = [("<65", 0), ("<65", 1), (">=65", 0), (">=65", 1)]


def make_data(trans: pd.DataFrame, subjects: pd.Index) -> Data:
    code = {k: i for i, k in enumerate(STRATA)}
    stratum = np.array([code[k] for k in zip(trans.age_band, trans.sepsis)])
    return Data(subjects.get_indexer(trans.subject_id), trans.from_state.to_numpy(),
                trans.to_state.to_numpy(), trans.prev_state.to_numpy(), stratum, len(subjects))


def c1(d: Data) -> np.ndarray:
    return np.bincount(d.frm * S + d.to, minlength=S * S).reshape(S, S).astype(float)


def c2(d: Data) -> np.ndarray:
    return np.bincount(((d.prev + 1) * S + d.frm) * S + d.to,
                       minlength=(S + 1) * S * S).reshape(S + 1, S, S).astype(float)


def c_strat(d: Data) -> np.ndarray:
    return np.bincount((d.stratum * S + d.frm) * S + d.to,
                       minlength=len(STRATA) * S * S).reshape(len(STRATA), S, S).astype(float)


def shrink1(C: np.ndarray, alpha: float) -> np.ndarray:
    return E.shrink(C, alpha, E.pooled_target(C))


def fit(model: str, train: Data, alphas: dict[str, float]):
    """Return a predictor mapping Data -> (n, S) next-state probabilities."""
    C = c1(train)
    if model == "persistence":
        eps = float(np.mean(train.to != train.frm))
        P = (1 - eps) * np.eye(S) + eps / (S - 1) * (1 - np.eye(S))
        return lambda d: P[d.frm]
    if model == "marginal":
        f = C[TRANSIENT].sum(0)
        return lambda d: np.broadcast_to(f / f.sum(), (len(d.frm), S))
    if model == "mle1":
        P = E.mle(C)
        return lambda d: P[d.frm]
    P1 = shrink1(C, alphas["shrink1"])
    if model == "shrink1":
        return lambda d: P1[d.frm]
    if model == "mle2":
        P2 = E.mle2(c2(train), E.mle(C))
        return lambda d: P2[d.prev + 1, d.frm]
    if model == "shrink2":
        P2 = E.shrink2(c2(train), alphas["shrink2"], P1)
        return lambda d: P2[d.prev + 1, d.frm]
    if model == "strat":
        Ps = E.shrink(c_strat(train), alphas["strat"], np.broadcast_to(P1, (len(STRATA), S, S)))
        return lambda d: Ps[d.stratum, d.frm]
    raise ValueError(model)


def losses(probs: np.ndarray, to: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-transition multiclass log-loss and Brier score."""
    p_true = probs[np.arange(len(to)), to]
    onehot = np.eye(S)[to]
    return -np.log(np.maximum(p_true, floor)), ((probs - onehot) ** 2).sum(1)


def subject_folds(n_subj: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n_subj) % k


def alpha_grid(vc: dict) -> np.ndarray:
    g = vc["alpha_grid"]
    return np.logspace(g["log10_min"], g["log10_max"], g["n"])


def cv_curve(d: Data, model: str, grid: np.ndarray, k: int, rng: np.random.Generator,
             floor: float, alpha1: float | None = None) -> np.ndarray:
    """Subject-level K-fold mean log-loss for each alpha (pooled out-of-fold)."""
    fold = subject_folds(d.n_subj, k, rng)[d.subj]
    total = np.zeros(len(grid))
    for f in range(k):
        tr, te = d.subset(fold != f), d.subset(fold == f)
        for i, a in enumerate(grid):
            alphas = {"shrink1": a if model == "shrink1" else alpha1, model: a}
            total[i] += losses(fit(model, tr, alphas)(te), te.to, floor)[0].sum()
    return total / len(d.frm)


def select_alphas(d: Data, grid: np.ndarray, k: int, rng: np.random.Generator,
                  floor: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """CV-choose alpha for shrink1 first, then shrink2 and strat given it."""
    curves = {"shrink1": cv_curve(d, "shrink1", grid, k, rng, floor)}
    a1 = float(grid[curves["shrink1"].argmin()])
    for m in ("shrink2", "strat"):
        curves[m] = cv_curve(d, m, grid, k, rng, floor, alpha1=a1)
    return {m: float(grid[c.argmin()]) for m, c in curves.items()}, curves


def subject_bootstrap_se(loss: np.ndarray, subj: np.ndarray, n_boot: int,
                         rng: np.random.Generator) -> float:
    """SE of a per-transition mean when resampling subjects (ratio estimator)."""
    u, inv = np.unique(subj, return_inverse=True)
    tot = np.bincount(inv, weights=loss)
    cnt = np.bincount(inv).astype(float)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot)
    return float(((W @ tot) / (W @ cnt)).std(ddof=1))


def score(train: Data, test: Data, alphas: dict, floor: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {m: losses(fit(m, train, alphas)(test), test.to, floor) for m in MODELS}


def evaluate(d: Data, subj_era: np.ndarray, vc: dict, rng: np.random.Generator):
    """Walk-forward (early -> late, by subject era) and subject K-fold results.

    alpha is re-selected by inner CV on each training set, so test data never
    influence hyper-parameters. Returns (table, walk-forward alphas).
    """
    floor, k, grid, nb = vc["prob_floor"], vc["k_folds"], alpha_grid(vc), vc["n_boot_se"]
    rows = []

    early = subj_era[d.subj] == "early"
    train, test = d.subset(early), d.subset(~early)
    wf_alphas, _ = select_alphas(train, grid, k, rng, floor)
    wf = score(train, test, wf_alphas, floor)
    for m, (ll, br) in wf.items():
        rows.append(_row("walk_forward", m, ll, br, wf["mle1"][0], test.subj, nb, rng))

    fold = subject_folds(d.n_subj, k, rng)[d.subj]
    oof = {m: (np.empty(len(d.to)), np.empty(len(d.to))) for m in MODELS}
    for f in range(k):
        tr, te = d.subset(fold != f), d.subset(fold == f)
        alphas, _ = select_alphas(tr, grid, k, rng, floor)
        for m, (ll, br) in score(tr, te, alphas, floor).items():
            oof[m][0][fold == f], oof[m][1][fold == f] = ll, br
    for m, (ll, br) in oof.items():
        rows.append(_row("kfold", m, ll, br, oof["mle1"][0], d.subj, nb, rng))
    return pd.DataFrame(rows), wf_alphas


def _row(split, model, ll, br, ll_ref, subj, nb, rng) -> dict:
    """Metrics with subject-bootstrap SEs, plus the paired log-loss difference vs mle1."""
    return {"split": split, "model": model, "n_test": len(ll),
            "log_loss": ll.mean(), "log_loss_se": subject_bootstrap_se(ll, subj, nb, rng),
            "brier": br.mean(), "brier_se": subject_bootstrap_se(br, subj, nb, rng),
            "dlog_loss_vs_mle1": (ll - ll_ref).mean(),
            "dlog_loss_vs_mle1_se": subject_bootstrap_se(ll - ll_ref, subj, nb, rng)}


# --- trajectory-level calibration (7.3) --------------------------------------

def observed_occupancy(seqs: pd.Series, horizon: int) -> np.ndarray:
    """(n, horizon+1) state per admission per day; absorbing states carried forward.

    Truncated trajectories (no absorbing state) are padded with their last
    state; with max_days >= horizon this never affects days <= horizon.
    """
    out = np.empty((len(seqs), horizon + 1), dtype=np.int8)
    for r, s in enumerate(seqs):
        s = list(s)[: horizon + 1]
        out[r, : len(s)] = s
        out[r, len(s):] = s[-1]
    return out


def calibration(P: np.ndarray, test_traj: pd.DataFrame, horizon: int, n_boot: int,
                rng: np.random.Generator) -> dict:
    """Predicted vs observed occupancy, 28-day mortality and mean LOS on test admissions.

    Prediction starts from the test set's own day-0 distribution, so it checks
    the transition model rather than the case mix.
    """
    occ = observed_occupancy(test_traj.seq, horizon)
    onehot = np.eye(S)[occ]                                   # (n, H+1, S)
    pi0 = onehot[:, 0].mean(0)
    pred = A.occupancy(P, pi0, horizon)
    codes, subj = pd.factorize(test_traj.subject_id)
    per_subj = np.zeros((len(subj), horizon + 1, S))
    np.add.at(per_subj, codes, onehot)
    cnt = np.bincount(codes).astype(float)
    W = rng.multinomial(len(subj), np.full(len(subj), 1 / len(subj)), size=n_boot)
    boot = np.einsum("bs,sht->bht", W, per_subj) / (W @ cnt)[:, None, None]
    complete = ~test_traj.truncated.to_numpy(bool)
    los = test_traj.seq.str.len().to_numpy()[complete] - 1
    return {
        "observed": onehot.mean(0), "predicted": pred,
        "observed_lo": np.percentile(boot, 2.5, axis=0), "observed_hi": np.percentile(boot, 97.5, axis=0),
        "mort28_observed": float(onehot[:, horizon, DEAD].mean()),
        "mort28_predicted": float(pred[horizon, DEAD]),
        "mort28_observed_ci": [float(np.percentile(boot[:, horizon, DEAD], q)) for q in (2.5, 97.5)],
        "los_observed": float(los.mean()),
        "los_predicted": float(pi0[TRANSIENT] @ A.expected_los(P)),
    }
