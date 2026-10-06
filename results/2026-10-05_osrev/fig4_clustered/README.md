# results/2026-10-05_osrev/fig4_clustered

Project DUO-Max, `the project root`. Display support for Figure 4 (third amendment, 2026-10-06).

Patient-clustered standard errors of the pair-level trigger rates drawn in Figure 4, panels c and d. The stored E4 analysis tables hold pair counts without per-patient data, so the six cells behind those panels (base view accuracy, view errors on, mean SL/AP ratio 0.64, three beats, beat-to-beat variation 5% to 30%; cells 181, 187, 193, 199, 205, 211 of experiment E4 in `code/configs/base.yaml`) were re-simulated with their original seeds, `SeedSequence(20260918, spawn_key=(4, cell))`. Every count (fired pairs, fired pairs with an error, error pairs, non-error pairs) is asserted identical to `results/2026-09-18_full/analysis/E4_operating.csv`; no point estimate changed.

- Script: `code/23_osrev_fig4_clustered_se.py`; batch script `code/slurm/osrev_fig4_clustered.sh`.
- Slurm job 5306853 (MeluXina, partition cpu, qos short), 2026-10-06; accounting in `provenance.txt` and the log `logs/osrev-fig4-clustered-5306853.out`.
- `fig4_clustered_se.csv` (90 rows: 6 cells x 5 trigger thresholds x 3 measures): value, numerator, denominator, binomial standard error (pairs treated as independent), the v05 standard error (binomial x sqrt(3)), the patient-level linearization (delta method) standard error for a ratio of totals (`rates_from_patterns` in `code/lib/duomaxsim/osrev_triggers.py`), the design effect and the clustered 95% Monte Carlo interval.
- Base cell 193, 5 mm: positive predictive value 69.7% (1,835 of 2,634), clustered interval 67.9% to 71.5%, design effect 1.04, equal to `results/2026-10-05_osrev/triggers/triggers_interval_methods.csv`.
