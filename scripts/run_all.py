"""processed trajectories (or --synthetic) -> results/{metrics.json, tables/, figures/}."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mtraj import STATES, TRANSIENT
from mtraj import analytic as A
from mtraj import estimate as E
from mtraj import plots
from mtraj import tail as TL
from mtraj import uncertainty as U
from mtraj import variance_reduction as V
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


def write_table(df: pd.DataFrame, path: Path, digits: int = 4) -> None:
    """CSV plus a plain Markdown table (no tabulate dependency)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    fmt = lambda v: f"{v:.{digits}g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def quick(cfg: dict) -> None:
    """Shrink Monte Carlo sizes for development runs."""
    cfg["uncertainty"]["B"] = 200
    cfg["variance_reduction"].update(R=100, n_grid=[1000, 10000], n_pilot=5000, crn_n=2000)
    cfg["tail"]["n_cohorts"] = 4000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small Monte Carlo sizes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.quick:
        quick(cfg)
    synthetic = args.synthetic or cfg["data"]["use_synthetic"]
    H, alpha = cfg["analysis"]["horizon"], cfg["estimation"]["alpha"]
    rng_unc, rng_vr, rng_crn, rng_tail = (np.random.default_rng(s) for s in
                                          np.random.SeedSequence(cfg["seed"]).spawn(4))
    out = ROOT / cfg["analysis"]["results"] / ("synthetic" if synthetic else "")
    figs, tabs = out / "figures", out / "tables"

    traj = load_trajectories(cfg, synthetic)
    trans = transitions(traj)
    log.info("%s: %d admissions, %d subjects, %d transitions",
             "synthetic" if synthetic else "eICU", len(traj), traj.subject_id.nunique(), len(trans))

    # --- point estimates -------------------------------------------------------
    C = E.counts(trans)
    P_mle = E.mle(C)
    P = U.fit(C, alpha)
    pi0 = E.initial_distribution(traj)
    s = A.summary(P, pi0, H)
    T = STATES[:3]
    show("P_hat (first-order MLE)", P_mle, STATES, STATES)
    show(f"P_tilde (shrunk, alpha={alpha})", P, STATES, STATES)
    show("N = (I - Q)^-1", A.fundamental(P), T, T)
    show("B = N R (absorption probabilities)", s["absorption"], T, STATES[3:])
    print(f"\npi0 = {np.round(pi0, 4)}")
    print(f"expected LOS by start state = {np.round(s['expected_los'], 2)} days")
    print(f"cohort in-hospital mortality = {s['mortality']:.4f}")
    print(f"expected ICU days within {H} = {s['expected_icu_days']:.3f}")

    # --- parameter uncertainty -------------------------------------------------
    B = cfg["uncertainty"]["B"]
    Cs, D0s = U.subject_counts(traj, trans)
    boot = U.summarize(*U.bootstrap(Cs, D0s, B, alpha, rng_unc), H)
    post = U.summarize(*U.dirichlet(C, D0s.sum(0), B, alpha, rng_unc), H)
    unc = U.compare_methods(boot, post)
    write_table(unc, tabs / "uncertainty_methods")
    plots.occupancy(s["occupancy"], U.occupancy_bands(post), figs / "fig3_occupancy.png")
    log.info("uncertainty: bootstrap vs Dirichlet, B=%d", B)

    # --- variance reduction ----------------------------------------------------
    vc = cfg["variance_reduction"]
    vr = V.run_estimators(P, pi0, vc, H, rng_vr)
    write_table(vr, tabs / "variance_reduction")
    plots.se_vs_n(vr, figs / "fig7_se_vs_n.png")
    log.info("variance reduction: R=%d, n=%s", vc["R"], vc["n_grid"])

    # --- CRN contrast: sepsis vs no sepsis, each shrunk toward pooled P ----------
    strata = {}
    for sep in (1, 0):
        tr_s, traj_s = trans[trans.sepsis == sep], traj[traj.sepsis == sep]
        strata[sep] = (E.shrink(E.counts(tr_s), alpha, P), E.initial_distribution(traj_s))
    crn = V.crn_contrast(*strata[1], *strata[0], vc, H, rng_crn)
    write_table(crn, tabs / "crn_sepsis_contrast")
    log.info("CRN contrast: sepsis - no sepsis")

    # --- tail risk -------------------------------------------------------------
    tc = cfg["tail"]
    d0 = D0s.sum(0)
    plug = TL.plug_in(P, pi0, tc["K"], tc["n_cohorts"], H, rng_tail)
    pp = TL.posterior_predictive(C, d0, alpha, tc["K"], tc["n_cohorts"], H, rng_tail)
    risk = TL.compare(plug, pp)
    write_table(risk.reset_index(names="measure"), tabs / "tail_risk")
    sweep = TL.ratio_vs_sample_size(Cs, D0s, [50, 100, 250, 500, len(Cs)], alpha, tc["K"],
                                    tc["n_cohorts"], H, rng_tail)
    write_table(sweep, tabs / "tail_ratio_vs_sample_size")
    plots.tail_hist(plug, pp, risk, tc["K"], figs / "fig8_tail_risk.png")
    print("\nTail risk (cohort ICU bed-days)\n" + risk.round(3).to_string())

    # --- metrics ---------------------------------------------------------------
    big = vr[vr.n == max(vc["n_grid"])].set_index(["target", "method"]).vrf
    unc_i = unc.set_index("quantity")
    metrics = {
        "source": "synthetic" if synthetic else "eicu-crd-demo",
        "n_admissions": len(traj), "n_subjects": int(traj.subject_id.nunique()),
        "n_transitions": len(trans), "alpha": alpha,
        "P_hat": P_mle.tolist(), "P_shrunk": P.tolist(), "pi0": pi0.tolist(),
        "expected_los": s["expected_los"].tolist(),
        "mortality": s["mortality"],
        "mortality_ci_bootstrap": [unc_i.loc["mortality", "boot_lo"], unc_i.loc["mortality", "boot_hi"]],
        "mortality_ci_dirichlet": [unc_i.loc["mortality", "post_lo"], unc_i.loc["mortality", "post_hi"]],
        "observed_mortality": float((traj.seq.str[-1] == STATES.index("DEAD")).mean()),
        "expected_icu_days": s["expected_icu_days"],
        "cohort_los": float(pi0[TRANSIENT] @ s["expected_los"]),
        "vrf": {f"{t}/{m}": float(v) for (t, m), v in big.items()},
        "crn_vrf": dict(zip(crn.target, crn.vrf.astype(float))),
        "tail": {k: risk[k].to_dict() for k in risk.columns},
        "tail_VaR99_ratio_vs_n": dict(zip(sweep.n_subjects.astype(str), sweep.VaR99_ratio)),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out / "metrics.json")


if __name__ == "__main__":
    main()
