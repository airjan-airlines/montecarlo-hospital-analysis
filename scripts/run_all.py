"""processed trajectories (or --synthetic) -> results/metrics.json and printed summary."""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from mtraj import STATES
from mtraj import analytic as A
from mtraj import estimate as E
from mtraj.config import ROOT, load_config
from mtraj.states import transitions
from mtraj.synthetic import generate

log = logging.getLogger("run_all")


def load_trajectories(cfg: dict, synthetic: bool) -> pd.DataFrame:
    if synthetic:
        sc = cfg["synthetic"]
        return generate(sc["n_subjects"], np.random.default_rng(cfg["seed"]),
                        cfg["states"]["max_days"], sc["era_shift"], sc["second_order"])
    path = cfg["data"]["processed"] / "trajectories.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_states.py or pass --synthetic")
    return pd.read_parquet(path)


def show(name: str, M: np.ndarray, rows, cols) -> None:
    print(f"\n{name}")
    print(pd.DataFrame(M, index=rows, columns=cols).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    synthetic = args.synthetic or cfg["data"]["use_synthetic"]
    H = cfg["analysis"]["horizon"]
    traj = load_trajectories(cfg, synthetic)
    trans = transitions(traj)
    log.info("%s: %d admissions, %d subjects, %d transitions",
             "synthetic" if synthetic else "eICU", len(traj), traj.subject_id.nunique(), len(trans))

    P = E.mle(E.counts(trans))
    pi0 = E.initial_distribution(traj)
    s = A.summary(P, pi0, H)
    T = STATES[:3]
    show("P_hat (first-order MLE)", P, STATES, STATES)
    show("N = (I - Q)^-1", A.fundamental(P), T, T)
    show("B = N R (absorption probabilities)", s["absorption"], T, STATES[3:])
    print(f"\npi0 = {np.round(pi0, 4)}")
    print(f"expected LOS by start state = {np.round(s['expected_los'], 2)} days")
    print(f"cohort in-hospital mortality = {s['mortality']:.4f}")
    print(f"expected ICU days within {H} = {s['expected_icu_days']:.3f}")

    out = ROOT / cfg["analysis"]["results"] / ("synthetic" if synthetic else "")
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "source": "synthetic" if synthetic else "eicu-crd-demo",
        "n_admissions": len(traj), "n_subjects": int(traj.subject_id.nunique()),
        "n_transitions": len(trans),
        "P_hat": P.tolist(), "pi0": pi0.tolist(),
        "expected_los": s["expected_los"].tolist(),
        "mortality": s["mortality"], "expected_icu_days": s["expected_icu_days"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out / "metrics.json")


if __name__ == "__main__":
    main()
