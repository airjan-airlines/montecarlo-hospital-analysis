"""Time 10^6 trajectories x 60 steps (spec target: under ~5 s)."""

import time

import numpy as np

from mtraj.simulate import simulate
from mtraj.synthetic import P_TRUE, PI0


def main(n: int = 1_000_000, horizon: int = 60) -> None:
    rng = np.random.default_rng(0)
    simulate(P_TRUE, PI0, 1000, horizon, rng)  # warm up
    t0 = time.perf_counter()
    paths = simulate(P_TRUE, PI0, n, horizon, rng)
    dt = time.perf_counter() - t0
    print(f"{n:,} x {horizon} steps: {dt:.2f} s ({paths.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
