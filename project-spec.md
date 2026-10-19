# SPEC: Markov Models of ICU Patient Trajectories (MIMIC-IV)

**Audience:** a coding agent implementing this project from scratch. **Goal:** a small, rigorous, fast-to-build Python project that models hospital stays as a discrete-time absorbing Markov chain, quantifies estimation uncertainty, validates out of sample, and uses Monte Carlo with variance reduction. It is a portfolio project for quant trading roles, so statistical rigor, honest validation and clean, vectorized code matter more than clinical depth.

**Time budget:** \~4 working days. Prefer simple, correct and tested over feature-rich.

---

## 0. Ground rules for the implementer

- Python 3.11+. Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `pytest`. Nothing else unless clearly justified.
- All randomness through `numpy.random.Generator` seeded from config. Every result must be reproducible.
- Simulation must be **vectorized across patients** (no Python loop over patients). Loops over time steps are fine.
- Everything must run end-to-end on **synthetic data** (Section 3) before real data is touched. Real data is a drop-in swap.
- Never print or commit raw patient-level data. `data/` is git-ignored. Only aggregate results go in `results/`.
- When something in this spec is ambiguous, choose the simplest reasonable option and record the decision in `DECISIONS.md`.

---

## 1. Data

### 1.1 Source

- **Primary (start here):** MIMIC-IV Clinical Database **Demo** on PhysioNet (`mimic-iv-demo`, open access, \~100 patients, same schema as full MIMIC-IV). Use the latest version available (2.2 at time of writing). Files live under `hosp/` and `icu/` as `.csv.gz`.
- **Later swap:** full MIMIC-IV (credentialed access). The loader must take a root path from config so switching is a one-line change.
- If the agent's environment cannot download files, the user will place them in `data/raw/mimic-iv-demo/`. Do not try to work around network restrictions.

### 1.2 Tables used

| Table | Columns used |
| --- | --- |
| `hosp/patients` | `subject_id`, `anchor_age`, `anchor_year`, `anchor_year_group` |
| `hosp/admissions` | `subject_id`, `hadm_id`, `admittime`, `dischtime`, `deathtime`, `hospital_expire_flag` |
| `hosp/diagnoses_icd` | `hadm_id`, `icd_code`, `icd_version` |
| `icu/icustays` | `hadm_id`, `stay_id`, `intime`, `outtime` |
| `icu/procedureevents` | `stay_id`, `itemid`, `starttime`, `endtime` |
| `icu/inputevents` | `stay_id`, `itemid`, `starttime`, `endtime` |
| `icu/d_items` | `itemid`, `label` (to verify item IDs) |

### 1.3 Organ-support item IDs

Verify every ID against `d_items` at load time and **fail loudly** if a label does not match expectations.

- Invasive mechanical ventilation (`procedureevents`): `225792` ("Invasive Ventilation").
- Vasopressors (`inputevents`): norepinephrine `221906`, epinephrine `221289`, phenylephrine `221749`, vasopressin `222315`, dopamine `221662`.

Non-invasive ventilation is **excluded** from "support" in v1 (record in `DECISIONS.md`).

### 1.4 Cohort

- All hospital admissions for patients with `anchor_age >= 18`.
- Each admission is one trajectory. A patient may contribute several; **all resampling and splitting is done at** `subject_id` **level** to avoid leakage between correlated trajectories.
- Drop admissions with missing `admittime`/`dischtime` or `dischtime <= admittime` (log counts dropped).

### 1.5 Era (for stability / walk-forward)

MIMIC-IV shifts dates. Approximate true admission year as:

```
true_year ≈ start_of(anchor_year_group) + (admittime.year - anchor_year) + 1
```

(`anchor_year_group` is like `"2008 - 2010"`; using the group midpoint offset is fine.) Assign each admission to an era bucket from config, default: `early = ≤2013`, `late = ≥2014`. Log the counts per era. On the demo, eras will be thin; that is expected and must be stated in the README.

---

## 2. State construction

### 2.1 States (fixed index order)

| idx | name | type |
| --- | --- | --- |
| 0 | `WARD` | transient: in hospital, not in ICU |
| 1 | `ICU` | transient: in ICU, no organ support |
| 2 | `ICU_SUPPORT` | transient: in ICU with invasive ventilation and/or vasopressor |
| 3 | `DISCHARGED` | absorbing |
| 4 | `DEAD` | absorbing (in-hospital death) |

Transient set `T = {0,1,2}`, absorbing set `A = {3,4}`.

### 2.2 Daily discretization

- Day `d` of an admission covers `[admittime + 24h·d, admittime + 24h·(d+1))`, for `d = 0 … D-1`, where block `D-1` contains `dischtime`.
- State for day `d` = most severe applicable: `ICU_SUPPORT` &gt; `ICU` &gt; `WARD`.
  - `ICU` if any ICU stay overlaps the block by ≥ 1 hour (configurable).
  - `ICU_SUPPORT` if, additionally, any support event overlaps the block (any overlap).
- After the final block, append one absorbing state: `DEAD` if `hospital_expire_flag == 1`, else `DISCHARGED`.
- Truncate trajectories at `max_days` (default 60) for estimation; log how many are truncated. Truncated trajectories contribute their observed transitions only (no absorbing step appended).

### 2.3 Outputs of this stage

- `trajectories.parquet` (or pickle) in `data/processed/` with columns: `subject_id`, `hadm_id`, `era`, `age_band`, `sepsis`, `seq` (list of ints).
- `transitions` long table: `subject_id`, `hadm_id`, `day`, `from_state`, `to_state`, `prev_state` (state at day−1, or `-1` on day 0), plus covariates.

### 2.4 Covariates (keep to two)

- `age_band`: `<65` vs `>=65` from `anchor_age`.
- `sepsis`: 1 if any ICD code in: ICD-9 `99591`, `99592`, `78552`; ICD-10 prefix `A40`, `A41`, or codes `R6520`, `R6521`. **Look-ahead caveat (must be in README):** ICD codes are assigned at discharge, so conditioning on them uses future information. This is acceptable for descriptive stratification but would be look-ahead bias in a real-time predictor. Flag it explicitly; it is a deliberate talking point.

---

## 3. Synthetic data generator (build first)

`synthetic.py`: given a known 5×5 matrix `P_true`, initial distribution `pi0`, covariate-specific variants, and era-specific variants, simulate `n_subjects` patients with 1–3 admissions each and emit tables in **the same format as Section 2.3**.

Default `P_true` (rows = from, cols = to; order WARD, ICU, ICU_SUPPORT, DISCHARGED, DEAD):

```
WARD         [0.80, 0.04, 0.01, 0.145, 0.005]
ICU          [0.20, 0.62, 0.08, 0.08,  0.02 ]
ICU_SUPPORT  [0.02, 0.15, 0.75, 0.01,  0.07 ]
DISCHARGED   [0,    0,    0,    1,     0    ]
DEAD         [0,    0,    0,    0,     1    ]
pi0 = [0.15, 0.60, 0.25, 0, 0]
```

Options: `era_shift` (perturb some rows in the late era) and `second_order=True` (next-state probabilities depend on previous state too), so the stability and order tests can be verified to detect real effects and not detect absent ones.

---

## 4. Estimation (`estimate.py`)

### 4.1 First-order MLE

Count matrix `C[i,j]` over transitions from transient `i`. `P̂[i,:] = C[i,:] / C[i,:].sum()`. Absorbing rows fixed as identity. Raise if any transient row has zero count (shrinkage handles this; see 4.3).

### 4.2 Second-order MLE

Condition on `(prev_state, from_state)`. Day-0 transitions use `prev_state = -1` as its own category.

### 4.3 Dirichlet shrinkage

Posterior mean:

```
P̃[i,j] = (C[i,j] + α · m[i,j]) / (C[i,:].sum() + α)
```

where `m[i,:]` is the pooled-row target (for stratified fits: the pooled-cohort row; for the pooled fit: overall next-state frequencies among transient-origin transitions). Choose `α` from a log-spaced grid by **subject-level K-fold CV log-loss**(Section 7). Report the chosen `α` and the CV curve.

### 4.4 Stratified fits

Fit `P` per stratum (`age_band` × `sepsis` and each marginally) with shrinkage toward the pooled `P`.

### 4.5 Initial distribution

`π̂₀` = empirical distribution of day-0 states.

---

## 5. Analytic quantities (`analytic.py`)

Given `P` with blocks `Q` (T×T) and `R` (T×A):

| Quantity | Formula |
| --- | --- |
| State occupancy | `π_t = π₀ Pᵗ`, `t = 0…H` (default H = 28) |
| Fundamental matrix | `N = (I − Q)⁻¹` |
| Expected LOS by start state | `τ = N 1` |
| Var(LOS) by start state | `(2N − I)τ − τ∘τ` |
| Absorption probabilities | `B = N R` |
| Cohort in-hospital mortality | `π₀[T] · B[:, DEAD]` |
| Expected ICU days within H | `Σ_{t<H} (π_t[ICU] + π_t[ICU_SUPPORT])` |
| Quasi-stationary distribution | leading left eigenvector of `Q`, normalized (descriptive only) |

Do **not** report the stationary distribution of `P` as a result (it is degenerate with absorbing states). Mention why in the README in one sentence.

---

## 6. Simulation and uncertainty (`simulate.py`, `uncertainty.py`, `variance_reduction.py`)

### 6.1 Vectorized simulator

`simulate(P, pi0, n, horizon, rng=None, uniforms=None) -> int array (n, horizon+1)`. Sample next states by inverse-CDF on row cumulative sums with `searchsorted` / comparison against a `(n,)` uniform vector per step. Accept externally supplied uniforms (shape `(n, horizon+1)`) so antithetic and common-random-number schemes can reuse them. Support a time-varying list of matrices (`P_t`) as well as a single `P`.

**Performance target:** 10⁶ trajectories × 60 steps in under \~5 s on a laptop. Add a benchmark script.

### 6.2 Parameter uncertainty

Two methods, both returning an array of `B` sampled matrices:

1. **Subject-level bootstrap:** resample `subject_id`s with replacement, refit (with the selected shrinkage), `B = 1000`default.
2. **Dirichlet posterior:** each transient row `~ Dirichlet(C[i,:] + α·m[i,:])`.

For each draw compute the Section 5 quantities. Report median and 2.5/97.5 percentiles. Compare the two methods' interval widths in a table.

### 6.3 Variance reduction (the centerpiece of the Monte Carlo section)

Targets that are **not** closed-form (so simulation is genuinely needed):

- `Y1` = P(ICU days within 28 &gt; 7)
- `Y2` = P(LOS &gt; 14 days)

Estimators, each evaluated over `R = 500` independent replications at several `n` (e.g. 10³, 10⁴, 10⁵):

1. **Plain MC.**
2. **Antithetic variates:** pair uniforms `u` and `1 − u`.
3. **Control variate:** control `C` = ICU days within 28 per patient, whose exact mean is known from `π_t` (Section 5). `β`estimated from an independent pilot run (not the same sample, to keep the estimator unbiased). Estimator: `Ȳ − β(C̄ − E[C])`.
4. **Antithetic + control variate.**

Report for each: mean, standard error, **variance-reduction factor vs plain MC**, and a log-log plot of SE vs `n` with a reference slope of −1/2.

Sanity check (also a test): for a target whose exact value IS known (28-day mortality), every estimator's mean must fall within 3 SE of the analytic value.

### 6.4 Common random numbers for risk-factor contrasts

To estimate differences between strata (e.g. sepsis vs no sepsis) in `Y1`, `Y2`, mortality and expected ICU days, simulate both strata with the **same uniforms**. Report the variance reduction of the difference vs independent streams.

### 6.5 Tail risk (`tail.py`)

Cohort of `K = 50` admissions drawn from `π̂₀`. Quantity: total ICU bed-days over 28 days.

- **Plug-in:** simulate with `P̂` only.
- **Posterior-predictive:** for each simulated cohort, first draw `P` from the Dirichlet posterior, then simulate.

Report mean, VaR95, VaR99, and ES95 (mean beyond VaR95) for both, and the ratio posterior/plug-in. The point to make: ignoring parameter uncertainty understates tail risk.

---

## 7. Validation and hypothesis tests (`validation.py`, `tests_stat.py`)

### 7.1 Scoring

Per-transition **multiclass log-loss** (with probability floor 1e-12) and **multiclass Brier score**, averaged over transient-origin transitions in the evaluation set.

Models compared:

| Model | Description |
| --- | --- |
| `persistence` | stay in current state with prob `1−ε`, rest spread uniformly (`ε` fit on train) |
| `marginal` | next-state frequencies ignoring current state |
| `mle1` | first-order MLE |
| `shrink1` | first-order with Dirichlet shrinkage |
| `mle2` / `shrink2` | second-order variants |
| `strat` | stratified by covariates, shrunk to pooled |

### 7.2 Splits

1. **Walk-forward (primary):** train on `early` era, test on `late` era.
2. **Subject-level K-fold (K=5)** as a secondary check, since demo eras are small.

Report a single results table: model × split × {log-loss, Brier}, with subject-level bootstrap SEs on the test metric.

### 7.3 Trajectory-level calibration

On the test set, compare predicted vs observed: 28-day in-hospital mortality, mean LOS, and the occupancy curve `π_t`(observed fractions per day with bootstrap bands). One figure.

### 7.4 Parameter stability across eras

- Likelihood-ratio (G) test of homogeneity: `G = 2 Σ_eras Σ_ij C_e[i,j] log(P̂_e[i,j] / P̂_pooled[i,j])`, with chi-square df = `(n_eras − 1) · |T| · (S − 1)` minus structurally empty cells.
- Because counts are small, also report a **permutation p-value** (permute era labels at subject level, 2000 permutations).
- Per-row breakdown: which transitions drifted most (difference and CI).

### 7.5 Markov order test

LR test of first-order vs second-order, plus AIC/BIC, plus **parametric-bootstrap p-value** (simulate from the fitted first-order model, refit both, 500 reps). Must be validated on synthetic data: correctly rejects when `second_order=True`, correctly fails to reject otherwise (check approximate size ≈ 0.05 over repeated synthetic runs, in a slow-marked test).

---

## 8. Repository layout

```
mimic-markov/
├── README.md
├── SPEC.md
├── DECISIONS.md
├── config.yaml
├── pyproject.toml
├── data/                   # git-ignored
│   ├── raw/
│   └── processed/
├── src/mtraj/
│   ├── __init__.py
│   ├── config.py
│   ├── load.py             # read MIMIC tables, verify item IDs
│   ├── states.py           # cohort + daily state sequences + covariates + era
│   ├── synthetic.py
│   ├── estimate.py
│   ├── analytic.py
│   ├── simulate.py
│   ├── uncertainty.py
│   ├── variance_reduction.py
│   ├── tail.py
│   ├── validation.py
│   ├── tests_stat.py
│   └── plots.py
├── scripts/
│   ├── build_states.py     # raw -> processed
│   ├── run_all.py          # processed -> results/ (supports --synthetic)
│   └── benchmark_sim.py
├── results/
│   ├── figures/
│   ├── tables/             # CSV/Markdown
│   └── metrics.json        # all headline numbers
└── tests/
```

`config.yaml` holds: data root, `use_synthetic`, seed, `max_days`, horizon `H`, ICU overlap threshold, era cut, `α` grid, `B`, `R`, `n`grid, cohort size `K`, item IDs.

`run_all.py --synthetic` must work with no data present.

---

## 9. Required figures

1. State diagram with edge widths ∝ transition probability (matplotlib is fine; no graphviz dependency required).
2. Heatmap of `P̂` (pooled) and side-by-side for sepsis vs no sepsis.
3. Stacked state occupancy `π_t`, days 0–28, with posterior bands on the DEAD and DISCHARGED curves.
4. Shrinkage CV curve (log-loss vs `α`).
5. Model comparison bar chart (test log-loss with SE bars).
6. Calibration: observed vs predicted occupancy curves on the test set.
7. SE vs `n` (log-log) for the four MC estimators.
8. Tail-risk histogram of cohort ICU bed-days, plug-in vs posterior-predictive, with VaR95/VaR99 marked.

Consistent style, labeled axes with units, saved as PNG at 150 dpi.

---

## 10. Tests (`pytest`)

Fast tests (default run, &lt; 30 s total):

- Estimated rows sum to 1; absorbing rows are identity; all probabilities in \[0,1\].
- MLE recovers `P_true` from 50k synthetic trajectories within tolerance (e.g. max abs error &lt; 0.01).
- `N`, `τ`, `B` match long-run simulation means within 3 SE.
- `π₀ Pᵗ` matches simulated occupancy within 3 SE at several `t`.
- Simulator with supplied uniforms is deterministic; antithetic pairing produces negatively correlated outputs for a monotone target.
- Control-variate estimator is unbiased on a known target (Section 6.3 sanity check).
- State construction on a hand-built toy admission (a few hours ward, ICU with a vasopressor, death) yields the expected sequence. Include edge cases: stay &lt; 24h, ICU overlap just under/over threshold, truncation at `max_days`.
- Subject-level splitting: no `subject_id` appears in both train and test.

Slow tests (`@pytest.mark.slow`): stability and order tests' size/power on synthetic data.

---

## 11. Milestones

| Day | Deliverable | Done when |
| --- | --- | --- |
| 1 | Synthetic generator, estimation, analytic module, simulator + tests | `pytest` passes; `run_all.py --synthetic` prints `P̂`, `N`, `B`, mortality |
| 2 | Uncertainty (bootstrap + Dirichlet), variance reduction, CRN, tail risk | Figures 3, 7, 8 generated on synthetic data; VR factors in `metrics.json` |
| 3 | Data loader + state construction on MIMIC-IV demo; validation, stability and order tests | Full pipeline runs on demo; tables in `results/tables/` |
| 4 | Shrinkage CV, stratified fits, all figures, README | Section 12 checklist complete |

---

## 12. README requirements (definition of done)

The README is what interviewers read. It must contain, in this order:

1. **One-paragraph summary** with 3–4 headline numbers pulled from `metrics.json` (e.g. cohort mortality with 95% interval, best model's out-of-sample log-loss improvement over persistence, control-variate variance-reduction factor, VaR99/plug-in ratio).
2. **Method** in ≤ 1 page: states, discretization, estimation, shrinkage, analytic results, simulation.
3. **Results:** the 8 figures with one-sentence takeaways each, and the model comparison table.
4. **What didn't work / limitations:** Markov assumption (with order-test result), daily discretization, single center, small demo sample, ICD look-ahead bias, in-hospital death only, era approximation from shifted dates.
5. **Reproduce:** exact commands (`pip install -e .`, `python scripts/build_states.py`, `python scripts/run_all.py`), and how to switch to full MIMIC-IV.

Every number in the README must come from `metrics.json` (no hand-typed results). Provide a tiny script or template step that fills them in.

---

## 13. Out of scope for v1

Continuous-time models (`msm`-style), SOFA-based states, more than two covariates, neural models, readmission modeling, post-discharge mortality, dashboards. List any of these as "future work" in the README only.
