# results/2026-10-05_osrev: third-amendment sensitivity analyses

Index of the result packages of the third protocol amendment (5 October 2026). The amendment specified
sensitivity analyses, requested by the senior author at internal review of the complete draft, before
they were run. Every result here is post hoc with respect to the initial protocol of 18 September 2026
and its first and second amendments, is conditional on the generative model, and replaces none of the
planned analyses in `results/2026-09-18_full`.

Common conduct:

- New simulations use NumPy `SeedSequence` entropy 20261005 with the spawn-key prefixes listed below;
  no prefix overlaps an earlier run (entropy 20260918 and 20260920).
- The library of the earlier runs (`code/lib/duomaxsim/{config,model,rules,estimators,metrics,experiments}.py`)
  was not modified. Each package lives in a new module `code/lib/duomaxsim/osrev_<package>.py` and first
  reproduces stored results of the earlier runs from their original seeds (bit-identical) before it adds anything.
- Each package was audited independently: the audit recomputed the reported numbers from the stored files and
  compared a separately written simulator within Monte Carlo error. Packages and audits were written by AI
  coding agents; this is software verification, not independent scientific review.
- Each package folder has its own `README.md` (file guide, columns) and `provenance.txt` (date, host, job,
  versions, sha256 of the code that ran).

| Folder | Content | Supplement section | Code | Seeds (entropy 20261005) | Slurm job |
|---|---|---|---|---|---|
| `beats/` | Windowed against plain beat averaging at matched beats and matched ceilings; limited-sampling denominators; paired against independent interval of the window effect; serial correlation, respiratory component, span-proportional caliper error | S8.1 | `osrev_beats.py`, `20_osrev_beats_run.py`, `21_osrev_beats_tables.py`, `configs/osrev_beats.yaml`, `slurm/osrev_beats.sh` | prefix 71; audit 971 | 5301653 |
| `views/` | Error distribution of the four view rules across the 12 view-accuracy scenarios and added settings; preferred rule by worst-case criterion; decomposition of the excess of the largest view mean; span-conditional error quantiles | S8.2 | `osrev_views.py`, `20_osrev_views_run.py`, `21_osrev_views_analyse.py`, `configs/osrev_views.yaml`, `slurm/osrev_views.sh` | prefix 72; audit 972 | 5301599 |
| `thresholds/` | Assumed true-span distribution; two-by-two classification tables at the illustrative cut-offs of 7, 10 and 13 mm; misclassification near each cut-off | S8.3 | `osrev_thresholds.py`, `20_osrev_thresholds_run.py`, `21_osrev_thresholds_analyse.py`, `configs/osrev_thresholds.yaml`, `slurm/osrev_thresholds.sh` | prefix 73; audit 973 | 5301601 (replicates); `published` mode on the login node |
| `triggers/` | Pair-level and patient-level contingency tables of the cross-view triggers with patient-clustered standard errors; wider error definitions and shared offsets; reviewer-performance grid; joint AP and SL distributions | S8.4 | `osrev_triggers.py`, `20_osrev_triggers_run.py`, `21_osrev_triggers_tables.py`, `configs/osrev_triggers.yaml`, `slurm/osrev_triggers.sh` | prefix 74; audit 974 | 5301795 |
| `fig4_clustered/` | Patient-clustered standard errors for Figure 4, panels c and d; six stored cells re-simulated with their original seeds, counts asserted identical | S8.4 | `23_osrev_fig4_clustered_se.py`, `slurm/osrev_fig4_clustered.sh` | original seeds, entropy 20260918, spawn key (4, cell) | 5306853 |
| `ai_reference/` | Model ranking at matched true MAE and matched own-error SD; analytic variance check; inter-reader limits conditional on a reader pair and marginal across pairs; multi-reader designs; size and power of sentinel tests | S8.5 (model in S3.6 and S3.7) | `osrev_ai_reference.py`, `20_osrev_ai_reference_run.py`, `21_osrev_ai_reference_analyse.py`, `22_osrev_ai_reference_checks.py`, `configs/osrev_ai_reference.yaml`, `slurm/osrev_ai_reference.sh` | prefix 75; audit 975 | 5301941 (5301894 failed and was resubmitted after a fix to the runner) |
| `calibration/` | Reproduction of the percentages derived from Singh et al. 2026; geometric calibration of long-axis underestimation on the published quantity; view rules at calibrated values; follow-up of 6 October 2026 (`followup_*`) | S8.6 (derivation in S2) | `osrev_calibration.py`, `20_osrev_calibration_run.py`, `21_osrev_calibration_analyse.py`, `configs/osrev_calibration.yaml`, `slurm/osrev_calibration.sh` | prefix 76; audit 976 | 5301947 (5301846 superseded: same simulation results, lower csv precision); follow-up jobs in `environment.md` |
| `verification/` | Reruns of the three second implementations against the library; paired and independent estimates of the window effect | S4 | `code/osrev_verification.py`, `slurm/osrev_verification.sh`, `tests/independent/` | prefix 191; stored seeds for exact regeneration | 5301518 |

Further spawn-key prefix under entropy 20261005: 131 (stand-alone check of the simulated AP minus SL
difference, reported with the parameter sources). Elapsed times and core-hours of all jobs are in
`environment.md`.

Tests: `tests/test_osrev_<package>.py` (153 tests in the six packages; log of the full suite in `tests/logs/`).
Tables of manuscript v06 are built from these folders by `code/23_make_tables_v06.py`, and Supplementary
Figures S6 to S8 by `code/figures/figS6_true_span_distribution.py`, `figS7_beats_matched.py` and
`figS8_error_distribution.py`.

Deviation from the amendment text: the calibration of the sentinel variance test was run as an unpaired
permutation test (sentinel and assisted reads are different cases), not as a paired resampling calibration.
