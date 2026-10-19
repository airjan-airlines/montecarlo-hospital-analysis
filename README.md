<!-- Generated from README.template.md by scripts/render_readme.py. Edit the template, not README.md. -->
# Markov models of ICU patient trajectories

Hospital stays from the open-access **eICU Collaborative Research Database demo**
(2069 hospital stays, 1758 patients, 186 US hospitals,
14049 daily transitions) modelled as a discrete-time absorbing Markov chain over
`WARD → ICU → ICU_SUPPORT → {DISCHARGED, DEAD}`, with closed-form absorbing-chain results,
subject-level uncertainty, honest out-of-sample validation, and Monte Carlo with variance reduction.

**Headline numbers.** Model-implied in-hospital mortality is **8.3%**
(subject-bootstrap 95% CI 7.1–9.5%;
observed 8.3%). Out of sample, the best model (`shrink2`)
cuts log-loss by **19.0%** vs a persistence baseline
when trained on 2014 and tested on 2015. A control variate (ICU days, whose mean is known in
closed form) cuts Monte Carlo variance for P(ICU days > 7) by **2.5×**
(95% CI 2.2–2.8), and common random numbers
cut the variance of sepsis-vs-no-sepsis contrasts by **2.5–3.7×**.
Ignoring parameter uncertainty understates cohort ICU-bed VaR99 by
**12%** when fitting on 50 patients, but makes no material difference
on the full sample.

## Method

**States and discretization.** Each hospital stay is cut into 24h blocks from hospital
admission. A day is `ICU` if ICU stays cover ≥ 60 minutes of it, and `ICU_SUPPORT` if it is an ICU day
with invasive ventilation or a vasopressor (norepinephrine, epinephrine, phenylephrine,
vasopressin, dopamine). The stay ends in `DEAD` (in-hospital death) or `DISCHARGED`.
Stays are truncated at 60 days. Organ support in eICU is charted as point events; events closer than
24h within an ICU stay are merged into one support interval. See `DECISIONS.md` for every choice.

**Estimation.** First-order MLE `P̂[i,j] = C[i,j] / C[i,·]` over transitions from transient states;
a second-order variant conditions on the previous day's state as well; Dirichlet shrinkage
`P̃ = (C + α m) / (C·1 + α)` with `α` chosen by subject-level 5-fold CV log-loss
(`α` = 5.62 pooled, 17.8 second-order, 316 stratified).
Stratified fits (age < / ≥ 65, sepsis within 24h of ICU admission) are shrunk toward the pooled `P̃`.

**Analytic results.** With `Q` the transient block and `R` the transient→absorbing block,
`N = (I − Q)⁻¹`, expected LOS `τ = N1`, `Var(LOS) = (2N − I)τ − τ∘τ`, absorption probabilities
`B = NR`, cohort mortality `π₀·B[:, DEAD]`, occupancy `π_t = π₀Pᵗ`. The stationary distribution of
`P` is not reported: with absorbing states it puts all mass on `DISCHARGED`/`DEAD` and says nothing
about the hospital stay.

**Simulation.** A vectorized inverse-CDF simulator (loop over days, never over patients; 10⁶
paths × 60 days in ~1.4 s) accepts external uniforms for antithetic and common-random-number
coupling, and a per-path matrix index for posterior-predictive simulation.

**Uncertainty and tests.** Subject-level bootstrap and a Dirichlet posterior for parameters;
G-test plus subject-permutation test for stability across years and regions; likelihood-ratio,
AIC/BIC and parametric-bootstrap test of first- vs second-order. All splitting and resampling is
by patient, never by stay or day.

## Results

![State diagram](results/figures/fig1_state_diagram.png)

**1. Transitions.** Staying put dominates (ward 0.75, ICU 0.60,
support 0.63 per day); most ICU exits step down to the ward (0.255/day)
rather than leaving hospital directly (0.099/day).

![Heatmaps](results/figures/fig2_heatmaps.png)

**2. Sepsis.** Septic patients leave the ICU straight to discharge less often
(0.066 vs 0.107 per day) and die from the ICU more often
(0.021 vs 0.009), even after shrinkage toward the pooled matrix.

![Occupancy](results/figures/fig3_occupancy.png)

**3. Occupancy.** Expected ICU days within 28 days is 3.45 per admission and
expected LOS is 6.80 days; posterior bands are tight for discharge and wider for death,
the rarer outcome.

![Shrinkage CV](results/figures/fig4_shrinkage_cv.png)

**4. Shrinkage.** With ~14k transitions the pooled first-order fit needs almost no shrinkage, but
second-order and stratified rows are sparse and the CV optimum is interior.

![Model comparison](results/figures/fig5_model_comparison.png)

**5. Model comparison.** Every Markov model beats persistence and the marginal baseline by a
wide margin; second-order with shrinkage is best. Its paired log-loss difference from first-order
MLE is -0.0204 ± 0.0028 walk-forward and
-0.0240 ± 0.0022 5-fold (negative is better; ±1 subject-bootstrap SE).

| split | model | n_test | log_loss | log_loss_se | brier | brier_se | dlog_loss_vs_mle1 | dlog_loss_vs_mle1_se |
|---|---|---|---|---|---|---|---|---|
| walk_forward | persistence | 7214 | 1.076 | 0.01841 | 0.5149 | 0.01015 | 0.1836 | 0.008039 |
| walk_forward | marginal | 7214 | 1.265 | 0.01394 | 0.6701 | 0.007284 | 0.3727 | 0.01249 |
| walk_forward | mle1 | 7214 | 0.8921 | 0.01557 | 0.4779 | 0.009304 | 0 | 0 |
| walk_forward | shrink1 | 7214 | 0.8921 | 0.01514 | 0.4779 | 0.00889 | -3.773e-05 | 3.846e-05 |
| walk_forward | mle2 | 7214 | 0.878 | 0.01448 | 0.4724 | 0.008419 | -0.01412 | 0.005527 |
| walk_forward | shrink2 | 7214 | 0.8717 | 0.01582 | 0.4722 | 0.008681 | -0.02038 | 0.002789 |
| walk_forward | strat | 7214 | 0.8871 | 0.0148 | 0.4763 | 0.009087 | -0.004964 | 0.001101 |
| kfold | persistence | 14049 | 1.073 | 0.01248 | 0.5136 | 0.007232 | 0.1811 | 0.005734 |
| kfold | marginal | 14049 | 1.275 | 0.01185 | 0.6742 | 0.005807 | 0.3824 | 0.01032 |
| kfold | mle1 | 14049 | 0.8923 | 0.0102 | 0.4771 | 0.006421 | 0 | 0 |
| kfold | shrink1 | 14049 | 0.8923 | 0.01088 | 0.4771 | 0.006498 | 8.91e-06 | 1.925e-05 |
| kfold | mle2 | 14049 | 0.8741 | 0.01193 | 0.4697 | 0.006244 | -0.01818 | 0.00386 |
| kfold | shrink2 | 14049 | 0.8683 | 0.01061 | 0.4697 | 0.005823 | -0.02397 | 0.002181 |
| kfold | strat | 14049 | 0.8886 | 0.01061 | 0.476 | 0.00605 | -0.003697 | 0.001069 |

![Calibration](results/figures/fig6_calibration.png)

**6. Calibration (train 2014 → test 2015).** Predicted 28-day mortality
8.8% vs observed 7.8%
(95% CI 6.2–9.4%);
mean LOS 7.03 vs 6.44 days. The first-order chain
discharges patients too slowly in the first week: a memoryless daily hazard cannot reproduce the early
peak of short stays, the same misfit the order test detects.

![SE vs n](results/figures/fig7_se_vs_n.png)

**7. Variance reduction.** All estimators follow the n^(−1/2) slope. At n = 10⁵ (500 replications;
95% bootstrap CIs) the variance reduction factors for P(ICU days > 7) are
0.87× (0.73–1.03) naive antithetic,
1.12× (0.96–1.32) severity-ordered antithetic,
2.46× (2.19–2.78) control variate and
2.64× (2.26–3.15) both. The control variate does
the work; antithetic pairing adds little on this chain. For targets with an exact value (P(LOS > 14),
28-day mortality) every estimator's mean is within 3 SE of it (largest |z| = 2.59).
Common random numbers cut the variance of the sepsis-vs-no-sepsis difference by 2.5× for
P(ICU days > 7) and 3.3× for 28-day mortality.

![Tail risk](results/figures/fig8_tail_risk.png)

**8. Tail risk.** For a cohort of 50 admissions, plug-in VaR99 of ICU bed-days over 28 days is
234 vs 233 posterior-predictive: on the full
sample, patient-level randomness swamps parameter uncertainty (left panel). Fitting on 50 random
patients (right panel) the posterior-predictive tail is visibly heavier. Refitting on 50 / 100 / 250 / 500 patients it grows to
1.013 / 1.031 / 1.075 /
1.123× (posterior / plug-in), so the cost of ignoring parameter uncertainty is a
small-sample problem.

**Stability and order.** Transition probabilities are stable across years (G = 15.6,
df = 12, permutation p = 0.45) but differ across US regions
(G = 107.1, df = 36, permutation p = 0.0020).
First-order is rejected against second-order (LR = 770 on 34 df,
parametric-bootstrap p = 0.0020, the smallest attainable; BIC 25156 vs
24711). The bootstrap intervals are up to 1.26× wider than
the Dirichlet ones because days from the same patient are correlated.

## What didn't work / limitations

- **Markov assumption.** First-order is rejected and the calibration plot shows why: no duration
  dependence. Second-order helps out of sample, but part of the apparent memory is probably
  unobserved heterogeneity between patients (a mixture of first-order chains looks higher-order).
  A semi-Markov or covariate-rich model is the natural next step.
- **Antithetic variates barely help** (0.87× naive, 1.12×
  severity-ordered). With states laid out in index order on [0, 1), a high uniform means both
  discharge and death, so outcomes are not monotone in the uniforms; ordering the inverse CDF by
  severity restores negative pair correlation for one step, but over 28 steps of a strongly
  persistent chain the coupling washes out. No order makes LOS monotone, since both absorbing
  states end the stay.
- **P(LOS > 14) is not simulation-only**, as the spec assumed: it equals the transient mass of
  `π₁₄`. It serves as a second unbiasedness check.
- **ICU patients only.** eICU records start from an ICU admission, so ward-only stays are missing and
  `WARD` days are those before, between or after ICU stays.
- **Daily discretization.** Sub-day moves are lost, and a day with any ICU hour above the threshold
  counts as an ICU day.
- **Organ support from point events.** Ventilation and vasopressors are charted, not timed as
  intervals; the 24h gap-filling rule is a modelling choice.
- **Sepsis look-ahead, mostly avoided.** Sepsis counts only if documented within 24h of first ICU
  admission. Using diagnoses from any time in the stay would condition on the future (discharge
  coding), which is fine for description but look-ahead bias in a real-time predictor.
- **In-hospital death only.** Death after discharge is invisible; `DISCHARGED` includes transfers
  to other hospitals and hospice.
- **Small demo sample and short era span.** Two discharge years (2014, 2015) and a demo subset
  of a much larger database; region and hospital differences cannot be separated from case mix here.

Future work: continuous-time multi-state models, SOFA-based states, more covariates, readmissions,
post-discharge mortality.

## Reproduce

```bash
uv sync                                    # Python 3.12, numpy/pandas/scipy/matplotlib/pyyaml/pyarrow
# download the eICU-CRD demo tables (open access, no credentialing)
B=https://physionet.org/files/eicu-crd-demo/2.0.1; D=data/raw/eicu-crd-demo; mkdir -p $D
for f in patient hospital diagnosis treatment infusiondrug respiratoryCharting; do
  curl -sfo $D/$f.csv.gz $B/$f.csv.gz; done
uv run python scripts/build_states.py      # raw tables -> data/processed/*.parquet
uv run python scripts/run_all.py           # -> results/ (add --quick for a ~10 s run)
uv run python scripts/run_all.py --synthetic   # same pipeline on synthetic data, no download needed
uv run python scripts/render_readme.py     # fill this README from results/metrics.json
uv run pytest                              # fast tests; `-m slow` for size/power checks
uv run python scripts/benchmark_sim.py     # 10^6 x 60-step simulation timing
```

The loader reads `data.root` from `config.yaml`; the full eICU-CRD (credentialed) uses the same schema,
so switching is a change to that path.
