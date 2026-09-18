# Pilot run 2026-09-18 (login node, not confirmatory)

- Purpose: to smoke-test E1 to E6 and time them. The numbers here are for code checking only and must not be reported.
- Code: duomaxsim 0.1.0 (no git yet); config `code/configs/base.yaml` (base-2026-09-18-v01, all parameters provisional); master seed 20260918.
- Replicates: 10^4 patients per cell (E1 to E4) and 100 studies per cell (E5, E6), set with `--n-rep`.
- Cells: E1 0:6 and 1146:1152; E2 0:10 and 1430:1440; E3 0:48; E4 0:6 and 282:288; E5 0:2; E6 0:2.
- Commands and full config snapshots are in the `*.manifest.json` files.
