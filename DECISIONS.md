# Decisions

Deviations from `project-spec.md` and choices made where it was ambiguous.

## Dataset: eICU-CRD Demo instead of MIMIC-IV Demo

- MIMIC-IV full access needs credentialing with a supervisor; the MIMIC demo is ~100 patients.
  The eICU-CRD Demo v2.0.1 (open access) has 2,520 unit stays / 2,174 hospital stays / 1,841
  patients across 186 hospitals, so it replaces MIMIC entirely.
- **Cohort is ICU-admitted patients only.** Every eICU record is a unit stay, so ward-only
  admissions do not exist. `WARD` days are the hospital days before/after/between ICU stays.
- **Identifiers:** `subject_id` = `uniquepid`, `hadm_id` = `patienthealthsystemstayid`.
- **Time axis:** eICU offsets are minutes relative to each unit admission. A unit stay is placed
  in hospital time at `[-hospitaladmitoffset, -hospitaladmitoffset + unitdischargeoffset]`.
- **Era:** `hospitaldischargeyear` 2014 = early, 2015 = late (no date shifting to undo).

## Cohort exclusions (counts from the demo)

| Reason | Hospital stays dropped |
| --- | --- |
| Missing `hospitaldischargestatus` | 25 |
| `hospitaladmitoffset > 0` (unit admit before hospital admit) | 14 |
| Unit visit numbers not 1..k (demo omits some unit stays, would mislabel ICU days as ward) | 58 |
| Age < 18 or missing (`"> 89"` mapped to 90) | 8 |

Result: 2,069 hospital stays, 1,758 subjects. 33 unit stays ending after hospital discharge are
clipped to discharge. 5 trajectories are truncated at 60 days.

## Organ support

eICU has no item dictionary like MIMIC's `d_items`, so "fail loudly" is implemented as: each
support pattern must match at least one row, and match counts are logged.

- Invasive ventilation: `treatment.treatmentstring` containing "mechanical ventilation" or
  "ventilator weaning", and `respiratoryCharting` labels "Vent Rate" / "Tidal Volume (set)".
- Vasopressors: `treatment.treatmentstring` containing "|vasopressors"; `infusiondrug.drugname`
  matching norepinephrine / epinephrine / phenylephrine / vasopressin / dopamine
  (dobutamine and milrinone are inotropes, excluded).
- Non-invasive ventilation excluded in v1 (as in spec).
- `respiratoryCare.ventstartoffset/ventendoffset` are unusable in the demo (`ventendoffset` is
  always 0), so not used.
- Support data are point events, not intervals. Events within the same unit stay closer than
  24h (`support_gap_fill_min`) are merged into one interval; a day is `ICU_SUPPORT` if it is an
  ICU day and touches any support interval.

## State discretization

- Day `d` covers `[24h·d, 24h·(d+1))` from hospital admission; `D = ceil(LOS / 24h)`, so a
  discharge exactly on a block boundary belongs to the preceding block.
- `ICU` needs ≥ 60 minutes of total unit-stay overlap within the block.

## Covariates

- `age_band` from the first unit stay's age.
- `sepsis` = sepsis/septic diagnosis documented **within 24h of first ICU admission**
  (`diagnosis.diagnosisoffset`), or APACHE admission diagnosis of sepsis. This avoids the
  discharge-coding look-ahead the spec warns about. `sepsis_any` (any time in the stay) is kept
  for comparison only: 315 of 340 `sepsis_any` stays are also early, so the look-ahead effect
  here is small but nonzero.
- `hospitalid` and `region` are stored for the stability analysis (across hospitals/regions).

## Storage

- Parquet via `pyarrow` (one dependency beyond the spec's list) for `data/processed/`.

## Uncertainty (spec 6.2)

- Shrinkage `alpha` is a fixed config value (1.0) until the CV selection in step 4 replaces it.
- Dirichlet posterior also draws `pi0 ~ Dirichlet(day-0 counts)` so cohort quantities include
  initial-state uncertainty, matching the bootstrap, which resamples day-0 states with subjects.
- The bootstrap gives intervals up to ~1.35x wider than the Dirichlet posterior, most for LOS. The
  Dirichlet treats transitions as independent; days from the same patient are not. The
  subject-level bootstrap is the more honest interval and is the one quoted in headlines.
- Occupancy bands in Figure 3 use the Dirichlet posterior, as the spec asks.

## Variance reduction (spec 6.3, 6.4)

- `Y2 = P(LOS > 14)` is **not** simulation-only: it equals the transient mass of `pi_14`, so it is
  closed-form. It is kept as a second unbiasedness check alongside 28-day mortality. `Y1` is
  the only target with no Section 5 closed form.
- Antithetic variates with the natural state order (WARD, ICU, SUPPORT, DISCHARGED, DEAD) can
  *increase* variance: high `u` maps to both discharge and death, so outcomes are not monotone in
  `u`. The simulator's `column_order` lays states out in severity order
  (DISCHARGED < WARD < ICU < SUPPORT < DEAD) on `[0, 1)` without changing the law. Both variants
  are reported (`antithetic_naive`, `antithetic`). No single order makes LOS monotone, since
  both absorbing states end the stay.
- Antithetic estimators use `n` total paths (`n/2` pairs) so cost matches plain MC.
- "Within 3 SE" for the unbiasedness check uses the SE of the mean over the `R` replications
  (`sd / sqrt(R)`), the stricter reading.
- CV `beta` comes from an independent pilot run; for antithetic+CV it is fit on pair averages.
- VRFs carry 95% intervals from resampling the `R` replications jointly with plain MC (the CV
  estimator reuses plain's paths, so an F-interval assuming independence would be wrong). At
  `R = 500` a VRF is only known to about ±15%: an early full run reported 1.30x for
  severity-ordered antithetic on `Y1`, six reseeded runs gave 0.98-1.23x. Only the CV gain
  (~2.5x) and the CRN gains (2.5-3.7x) are clearly real on the eICU chain.
- CRN contrast (sepsis vs no sepsis) uses stratum matrices shrunk toward the pooled `P` and
  each stratum's own `pi0`, with the severity column order so shared uniforms couple outcomes.

## Tail risk (spec 6.5)

- With ~14k transitions, parameter uncertainty barely moves cohort tail risk (VaR99 ratio ~1.01).
  To show the spec's point honestly, `tail_ratio_vs_sample_size` refits on random subsets of
  subjects: the VaR99 understatement grows as data shrink (~14% at 50 subjects).

## Shrinkage and stratified fits (spec 4.3, 4.4)

- `alpha` grid: 25 log-spaced values in `[1e-2, 1e4]` pseudo-counts per row.
- Three `alpha`s are chosen by subject-level 5-fold CV log-loss, in order: `shrink1` (rows toward
  the pooled next-state frequencies), then `shrink2` (second-order rows toward the first-order
  row of the current state) and `strat` (stratum rows toward the pooled shrunk `P`), both given
  `shrink1`'s `alpha`. The `shrink1` `alpha` is used for all downstream analysis.
- Strata: `age_band × sepsis` jointly and each marginally, all with the `strat` `alpha`.

## Validation (spec 7.1-7.3)

- **Subject-level era:** a subject's era (and region) is the one of their first admission. 52 of
  1,758 eICU subjects have admissions in both years; labelling at subject level keeps the
  walk-forward split subject-disjoint and makes the permutation unit the subject.
- `alpha`s are re-selected by inner CV on every training set (nested), so test data never
  touch hyper-parameters. The K-fold metric pools out-of-fold losses across folds.
- `persistence`: `epsilon` = fraction of training transitions that change state (its MLE).
- `mle2` rows for unseen `(prev, cur)` pairs fall back to the first-order MLE row.
- Alongside each model's SE, the table reports the **paired** log-loss difference vs `mle1` with
  its own subject-bootstrap SE: model differences (~0.01-0.02) are far smaller than the
  between-subject spread, so unpaired SEs overstate the uncertainty of a comparison.
- Figure 5 uses dots with error bars, not bars: a zero-based bar hides 0.01 differences on a
  0.9 scale.
- Calibration predicts from the test set's own day-0 distribution with the model fit on the
  training era, so it checks transition dynamics rather than case mix.

## Stability and order tests (spec 7.4, 7.5)

- G-test df = `(groups - 1) · Σ_i (d_i - 1)`, with `d_i` the destinations observed from
  transient row `i` in the pooled data (structurally empty cells removed).
- Stability is tested across eras (spec) and across US census regions (eICU-specific; 186
  hospitals are too many sparse groups, regions are not). Stays with missing region are left
  out of the region test.
- The chi-square p-value treats transitions as independent; the subject-permutation p-value
  does not. Where they disagree, the permutation p-value is the one to trust.
- Order test df counts free parameters as observed destinations minus one per observed row;
  the parametric bootstrap p-value does not depend on this approximation.
- The order test is validated on **homogeneous** synthetic data. Unobserved heterogeneity (a mix
  of patient types, each first-order) makes the pooled chain look higher-order, so a
  rejection on real data means "pooled first-order is wrong", not necessarily "patients have
  memory".
