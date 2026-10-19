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
