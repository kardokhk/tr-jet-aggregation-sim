# results/2026-10-05_osrev/tables

Derived and auxiliary files for `code/23_make_tables_v06.py`, which writes `drafts/tables-2026-10-05-v06.md`
(main Table 1, Supplementary Tables S1 to S6 and S8 to S27). No simulation output is stored here.

| File | Written by | Content |
|---|---|---|
| `table_cells_v06.csv` | `code/23_make_tables_v06.py` | One row per number registered while the tables were built: table, label, source file, unrounded value, decimals, scale, printed string |
| `slurm_jobs_sacct_2026-10-06.csv` | `sacct -X -P -j <jobs> --format=JobID,JobName,Submit,Start,End,Elapsed,State,NodeList,CPUTimeRAW,AllocCPUS` on 2026-10-06 (re-exported the same day to add job 5306870; the first 16 rows are unchanged) | Accounting records of the 17 jobs listed in Supplementary Table S8 and Supplementary Methods S6 (pipe-separated) |
| `pytest_collect_2026-10-06.csv` | `python -m pytest --collect-only -q tests`, counted per file, on 2026-10-06 | Number of collected test cases per test file (Supplementary Table S26) |

Regenerate the tables: `/project/home/p201509/envs/duomax-sim/bin/python code/23_make_tables_v06.py` (login node, deterministic).
Spot check: `/project/home/p201509/envs/duomax-sim/bin/python notes/scratch/2026-10-06-tables-check.py`.

The generator reads the test counts of Supplementary Tables S8 and S26 from `logs/duomax-pytest-5306852.out` and
`logs/duomax-pytest-5306870.out`, and asserts that every number quoted in the Results of the main text is present in
Supplementary Table S27; it stops if a result file, a selection or one of those values changes.
