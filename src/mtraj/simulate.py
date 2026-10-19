"""Vectorized absorbing-chain simulator (loop over time, never over patients)."""

import numpy as np


def _cdf(P: np.ndarray) -> np.ndarray:
    c = np.cumsum(P, axis=-1)
    c[..., -1] = 1.0  # guard against rounding so every u < 1 maps to a state
    return c


def _draw(cdf_rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draw: first column with cdf > u, row-wise."""
    return (u[:, None] >= cdf_rows).sum(axis=1)


def _uniform_source(n: int, horizon: int, rng, uniforms):
    if uniforms is None:
        rng = rng if rng is not None else np.random.default_rng()
        return lambda t: rng.random(n)
    if uniforms.shape != (n, horizon + 1):
        raise ValueError(f"uniforms must have shape {(n, horizon + 1)}")
    return lambda t: uniforms[:, t]


def simulate(P, pi0, n: int, horizon: int, rng: np.random.Generator | None = None,
             uniforms: np.ndarray | None = None, column_order=None) -> np.ndarray:
    """Simulate n paths of length horizon+1; returns int8 array (n, horizon+1).

    P is a single (S, S) matrix or a sequence of horizon matrices (P_t used for
    the step t -> t+1). ``uniforms`` of shape (n, horizon+1) replaces the RNG
    (column 0 draws the initial state), enabling antithetic and CRN schemes.
    ``column_order`` sets the order in which states are laid out on [0, 1) for
    the inverse CDF; it changes which u maps to which state (not the law), so a
    severity order makes outcomes monotone in u for antithetic/CRN coupling.
    """
    if column_order is not None:
        order = np.asarray(column_order)
        return order[simulate(np.asarray(P, float)[..., order][..., order, :], np.asarray(pi0)[order],
                              n, horizon, rng, uniforms)].astype(np.int8)
    P = np.asarray(P, dtype=float)
    time_varying = P.ndim == 3
    if time_varying and len(P) < horizon:
        raise ValueError(f"need {horizon} matrices, got {len(P)}")
    u_at = _uniform_source(n, horizon, rng, uniforms)
    cdf = _cdf(P)
    out = np.empty((n, horizon + 1), dtype=np.int8)
    out[:, 0] = _draw(np.broadcast_to(_cdf(np.asarray(pi0, float)), (n, len(pi0))), u_at(0))
    for t in range(horizon):
        c = cdf[t] if time_varying else cdf
        out[:, t + 1] = _draw(c[out[:, t]], u_at(t + 1))
    return out


def simulate_second_order(P2: np.ndarray, P: np.ndarray, pi0, n: int, horizon: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Second-order chain: P2[prev, cur] is the next-state row; day 0 uses P."""
    cdf2, cdf = _cdf(np.asarray(P2, float)), _cdf(np.asarray(P, float))
    out = np.empty((n, horizon + 1), dtype=np.int8)
    out[:, 0] = _draw(np.broadcast_to(_cdf(np.asarray(pi0, float)), (n, len(pi0))), rng.random(n))
    out[:, 1] = _draw(cdf[out[:, 0]], rng.random(n))
    for t in range(1, horizon):
        out[:, t + 1] = _draw(cdf2[out[:, t - 1], out[:, t]], rng.random(n))
    return out


def simulate_mixture(Ps: np.ndarray, pi0s: np.ndarray, draw: np.ndarray, horizon: int,
                     rng: np.random.Generator | None = None,
                     uniforms: np.ndarray | None = None) -> np.ndarray:
    """Path k follows matrix Ps[draw[k]] and initial law pi0s[draw[k]].

    Used for posterior-predictive simulation: B parameter draws, many paths each.
    """
    cdf, cdf0 = _cdf(np.asarray(Ps, float)), _cdf(np.asarray(pi0s, float))
    n = len(draw)
    u_at = _uniform_source(n, horizon, rng, uniforms)
    out = np.empty((n, horizon + 1), dtype=np.int8)
    out[:, 0] = _draw(cdf0[draw], u_at(0))
    for t in range(horizon):
        out[:, t + 1] = _draw(cdf[draw, out[:, t]], u_at(t + 1))
    return out
