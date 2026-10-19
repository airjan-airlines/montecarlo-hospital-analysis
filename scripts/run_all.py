"""processed trajectories (or --synthetic) -> results/{metrics.json, tables/, figures/}."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mtraj import DEAD, STATES, TRANSIENT
from mtraj import analytic as A
from mtraj import estimate as E
from mtraj import plots
from mtraj import tail as TL
from mtraj import tests_stat as TS
from mtraj import uncertainty as U
from mtraj import validation as VD
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
    cfg["validation"]["n_boot_se"] = 100
    cfg["stat_tests"].update(n_permutations=200, n_param_boot=50)


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
    H, max_days = cfg["analysis"]["horizon"], cfg["states"]["max_days"]
    vcfg, scfg = cfg["validation"], cfg["stat_tests"]
    rng_cv, rng_val, rng_unc, rng_vr, rng_crn, rng_tail, rng_stab, rng_ord = (
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg["seed"]).spawn(8))
    out = ROOT / cfg["analysis"]["results"] / ("synthetic" if synthetic else "")
    figs, tabs = out / "figures", out / "tables"

    traj = load_trajectories(cfg, synthetic)
    trans = transitions(traj)
    first = traj.drop_duplicates("subject_id")          # subject-level labels: first admission
    subjects = pd.Index(first.subject_id)
    log.info("%s: %d admissions, %d subjects, %d transitions",
             "synthetic" if synthetic else "eICU", len(traj), len(subjects), len(trans))

    # --- shrinkage CV (4.3) ----------------------------------------------------
    data = VD.make_data(trans, subjects)
    grid = VD.alpha_grid(vcfg)
    alphas, curves = VD.select_alphas(data, grid, vcfg["k_folds"], rng_cv, vcfg["prob_floor"])
    alpha = alphas["shrink1"]
    write_table(pd.DataFrame({"alpha": grid, **curves}), tabs / "shrinkage_cv")
    plots.cv_curves(grid, curves, alphas, figs / "fig4_shrinkage_cv.png")
    log.info("CV-selected alphas: %s", alphas)

    # --- point estimates and strata (4.1, 4.4, 5) -------------------------------
    C = E.counts(trans)
    P_mle = E.mle(C)
    P = U.fit(C, alpha)
    pi0 = E.initial_distribution(traj)
    s = A.summary(P, pi0, H)
    T = STATES[:3]
    show("P_hat (first-order MLE)", P_mle, STATES, STATES)
    show(f"P_tilde (shrunk, alpha={alpha:.3g})", P, STATES, STATES)
    show("N = (I - Q)^-1", A.fundamental(P), T, T)
    show("B = N R (absorption probabilities)", s["absorption"], T, STATES[3:])
    print(f"\npi0 = {np.round(pi0, 4)}")
    print(f"expected LOS by start state = {np.round(s['expected_los'], 2)} days")
    print(f"cohort in-hospital mortality = {s['mortality']:.4f}")
    print(f"expected ICU days within {H} = {s['expected_icu_days']:.3f}")
    write_table(pd.DataFrame(P, index=STATES, columns=STATES).reset_index(names="from"), tabs / "P_pooled")
    write_table(pd.DataFrame({"start": T, "expected_los": s["expected_los"], "var_los": s["var_los"],
                              "p_discharged": s["absorption"][:, 0], "p_dead": s["absorption"][:, 1],
                              "quasi_stationary": s["quasi_stationary"]}), tabs / "analytic")

    strata = E.fit_strata(trans, P, alphas["strat"])
    rows = []
    for name, Ps in strata.items():
        mask = np.ones(len(traj), bool)
        for part in name.split(", "):
            col, val = part.split("=", 1)
            mask &= traj[col].astype(str).to_numpy() == val
        p0 = E.initial_distribution(traj[mask])
        rows.append({"stratum": name, "n_admissions": int(mask.sum()),
                     "mortality": A.mortality(Ps, p0), "cohort_los": float(p0[TRANSIENT] @ A.expected_los(Ps)),
                     "expected_icu_days": A.expected_icu_days(Ps, p0, H),
                     "observed_mortality": float((traj[mask].seq.str[-1] == DEAD).mean())})
    write_table(pd.DataFrame(rows), tabs / "strata")
    plots.state_diagram(P, figs / "fig1_state_diagram.png")
    plots.heatmaps({"pooled": P, "no sepsis": strata["sepsis=0"], "sepsis": strata["sepsis=1"]},
                   figs / "fig2_heatmaps.png")

    # --- validation (7.1-7.3) ----------------------------------------------------
    subj_era = first.era.to_numpy()
    val, wf_alphas = VD.evaluate(data, subj_era, vcfg, rng_val)
    write_table(val, tabs / "model_comparison")
    plots.model_comparison(val, figs / "fig5_model_comparison.png")
    late_subj = set(first.subject_id[first.era == "late"])
    train_wf = data.subset(subj_era[data.subj] == "early")
    P_wf = U.fit(VD.c1(train_wf), wf_alphas["shrink1"])
    cal = VD.calibration(P_wf, traj[traj.subject_id.isin(late_subj)], H, vcfg["n_boot_se"], rng_val)
    plots.calibration(cal, figs / "fig6_calibration.png")
    print("\nModel comparison\n" + val.round(4).to_string())

    # --- stability and order tests (7.4, 7.5) -------------------------------------
    Cs, D0s = U.subject_counts(traj, trans)
    stab = {"era": TS.stability_test(Cs, subj_era, scfg["n_permutations"], rng_stab)}
    if "region" in first and first.region.notna().any():
        reg = first.region.to_numpy()
        ok = pd.notna(reg)
        stab["region"] = TS.stability_test(Cs[ok], reg[ok], scfg["n_permutations"], rng_stab)
    write_table(pd.DataFrame([{"grouping": k, **{c: (str(v) if isinstance(v, list) else v)
                                                 for c, v in r.items()}} for k, r in stab.items()]),
                tabs / "stability_tests")
    drift = TS.drift_table(Cs, subj_era, "early", "late", vcfg["n_boot_se"], rng_stab)
    write_table(drift, tabs / "era_drift")
    order = TS.order_test(C, E.counts2(trans), pi0, len(traj), max_days, scfg["n_param_boot"], rng_ord)
    write_table(pd.DataFrame([order]), tabs / "order_test")
    print("\nStability\n" + pd.DataFrame(stab).T[["G", "df", "p_chi2", "p_perm"]].to_string())
    print("\nOrder test\n" + pd.Series(order).to_string())

    # --- parameter uncertainty (6.2) -----------------------------------------------
    B = cfg["uncertainty"]["B"]
    boot = U.summarize(*U.bootstrap(Cs, D0s, B, alpha, rng_unc), H)
    post = U.summarize(*U.dirichlet(C, D0s.sum(0), B, alpha, rng_unc), H)
    unc = U.compare_methods(boot, post)
    write_table(unc, tabs / "uncertainty_methods")
    plots.occupancy(s["occupancy"], U.occupancy_bands(post), figs / "fig3_occupancy.png")

    # --- variance reduction and CRN (6.3, 6.4) ---------------------------------------
    vc = cfg["variance_reduction"]
    vr = V.run_estimators(P, pi0, vc, H, rng_vr)
    write_table(vr, tabs / "variance_reduction")
    plots.se_vs_n(vr, figs / "fig7_se_vs_n.png")
    sep = {k: (strata[f"sepsis={k}"], E.initial_distribution(traj[traj.sepsis == k])) for k in (1, 0)}
    crn = V.crn_contrast(*sep[1], *sep[0], vc, H, rng_crn)
    write_table(crn, tabs / "crn_sepsis_contrast")

    # --- tail risk (6.5) ---------------------------------------------------------------
    tc = cfg["tail"]
    d0 = D0s.sum(0)
    plug = TL.plug_in(P, pi0, tc["K"], tc["n_cohorts"], H, rng_tail)
    pp = TL.posterior_predictive(C, d0, alpha, tc["K"], tc["n_cohorts"], H, rng_tail)
    risk = TL.compare(plug, pp)
    write_table(risk.reset_index(names="measure"), tabs / "tail_risk")
    sweep = TL.ratio_vs_sample_size(Cs, D0s, [50, 100, 250, 500, len(Cs)], alpha, tc["K"],
                                    tc["n_cohorts"], H, rng_tail)
    write_table(sweep, tabs / "tail_ratio_vs_sample_size")
    small = rng_tail.choice(len(Cs), size=min(50, len(Cs)), replace=False)
    C50, d50 = Cs[small].sum(0), D0s[small].sum(0)
    plug50 = TL.plug_in(U.fit(C50, alpha), d50 / d50.sum(), tc["K"], tc["n_cohorts"], H, rng_tail)
    pp50 = TL.posterior_predictive(C50, d50, alpha, tc["K"], tc["n_cohorts"], H, rng_tail)
    plots.tail_hist([(f"Fit on all {len(Cs)} patients", plug, pp),
                     (f"Fit on {len(small)} random patients", plug50, pp50)],
                    tc["K"], figs / "fig8_tail_risk.png")
    print("\nTail risk (cohort ICU bed-days)\n" + risk.round(3).to_string())

    # --- metrics -------------------------------------------------------------------------
    big = vr[vr.n == max(vc["n_grid"])].set_index(["target", "method"]).vrf
    unc_i = unc.set_index("quantity")
    v = val.set_index(["split", "model"])
    best = {sp: v.loc[sp].log_loss.idxmin() for sp in ("walk_forward", "kfold")}
    gain = {sp: float(1 - v.loc[(sp, best[sp]), "log_loss"] / v.loc[(sp, "persistence"), "log_loss"])
            for sp in best}
    metrics = {
        "source": "synthetic" if synthetic else "eicu-crd-demo",
        "n_admissions": len(traj), "n_subjects": len(subjects), "n_transitions": len(trans),
        "n_hospitals": int(traj.hospitalid.nunique()) if "hospitalid" in traj else None,
        "alphas": alphas, "alphas_walk_forward": wf_alphas,
        "P_hat": P_mle.tolist(), "P_shrunk": P.tolist(), "pi0": pi0.tolist(),
        "expected_los": s["expected_los"].tolist(),
        "mortality": s["mortality"],
        "mortality_ci_bootstrap": [unc_i.loc["mortality", "boot_lo"], unc_i.loc["mortality", "boot_hi"]],
        "mortality_ci_dirichlet": [unc_i.loc["mortality", "post_lo"], unc_i.loc["mortality", "post_hi"]],
        "observed_mortality": float((traj.seq.str[-1] == DEAD).mean()),
        "expected_icu_days": s["expected_icu_days"],
        "cohort_los": float(pi0[TRANSIENT] @ s["expected_los"]),
        "max_boot_over_post_width": float(unc.width_ratio_boot_over_post.max()),
        "best_model": best, "logloss_gain_vs_persistence": gain,
        "best_vs_mle1": {sp: [float(v.loc[(sp, best[sp]), "dlog_loss_vs_mle1"]),
                              float(v.loc[(sp, best[sp]), "dlog_loss_vs_mle1_se"])] for sp in best},
        "calibration": {k: cal[k] for k in ("mort28_observed", "mort28_predicted", "mort28_observed_ci",
                                            "los_observed", "los_predicted")},
        "stability": {k: {c: r[c] for c in ("G", "df", "p_chi2", "p_perm", "groups", "n_per_group")}
                      for k, r in stab.items()},
        "order_test": {k: float(order[k]) for k in ("LR", "df", "p_chi2", "p_boot", "aic1", "aic2", "bic1", "bic2")},
        "vrf": {f"{t}/{m}": float(x) for (t, m), x in big.items()},
        "vrf_ci": {f"{t}/{m}": [float(lo), float(hi)] for (t, m), lo, hi in
                   zip(big.index, vr[vr.n == max(vc["n_grid"])].vrf_lo, vr[vr.n == max(vc["n_grid"])].vrf_hi)},
        "vr_max_abs_z": float(vr.z_vs_exact.abs().max()),
        "crn_vrf": dict(zip(crn.target, crn.vrf.astype(float))),
        "crn_diff": dict(zip(crn.target, crn.diff_crn.astype(float))),
        "tail": {k: risk[k].to_dict() for k in risk.columns},
        "tail_VaR99_ratio_vs_n": dict(zip(sweep.n_subjects.astype(str), sweep.VaR99_ratio)),
        "tail_VaR99_excess_vs_n": dict(zip(sweep.n_subjects.astype(str), sweep.VaR99_ratio - 1)),
        "tail_VaR99_excess_full": float(risk.loc["VaR99", "ratio"] - 1),
        "P_strata": {k: Ps.tolist() for k, Ps in strata.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("wrote %s", out / "metrics.json")


if __name__ == "__main__":
    main()
