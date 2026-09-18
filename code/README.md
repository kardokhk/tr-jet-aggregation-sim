# duomaxsim: simulation code for the DUO-Max methods paper

Implements protocol `notes/protocol-2026-09-18-v01.md`: the generative model (section 3), estimators A1 to A7, beat rules and cross-view triggers (section 4), and experiments E1 to E6 (section 5).

## Layout

| Path | Content |
|---|---|
| `lib/duomaxsim/config.py` | YAML loading, grid-cell enumeration, per-cell `SeedSequence` |
| `lib/duomaxsim/model.py` | case mix, view errors (u, o, rho), instrument factor g, beats with RR dependence (sinus/AF), available beats, reader measurement |
| `lib/duomaxsim/rules.py` | N-beat rule with +-W acceptance window and "limited sampling" flag; cross-view warn/adjudication triggers; rule A4 with reader detection (s_det, f_rej) |
| `lib/duomaxsim/estimators.py` | A1 anchor mean, A2 mean of view means, A3 max, A5 median, A6 index beat, A7 oracle-corrected mean |
| `lib/duomaxsim/metrics.py` | streaming bias/RMSE/SD/relative-bias accumulators with Monte Carlo SE; proportions; ICC(A,1) with McGraw-Wong CI; Bland-Altman; Clark (1961) max-of-2 moments; E[max of K normals]; numerical MVN E[max] |
| `lib/duomaxsim/experiments.py` | `run_E1` ... `run_E6`, `run_cells` (long-format DataFrame) |
| `configs/base.yaml` | every parameter with `value`, `grid` (where the protocol has one), `unit`, `source` (all `provisional`) and `note`, plus the per-experiment `vary` lists |
| `01_run_experiment.py` | command-line entry point |

## Running

```bash
PY=/project/home/p201509/envs/duomax-sim/bin/python
$PY code/01_run_experiment.py --experiment E1 --config code/configs/base.yaml --list-cells
$PY code/01_run_experiment.py --experiment E1 --config code/configs/base.yaml --cells 0:96 --out results/2026-09-18_full
$PY -m pytest tests -q -p no:cacheprovider
```

- `--cells i:j` is half-open. Each invocation writes one parquet file (`E1_cells_00000-00096.parquet`) plus a `.manifest.json` holding the command, the full config and the package versions. Cells are never written one per file, because of the inode quota.
- `--n-rep` overrides the number of replicates (patients for E1 to E4, simulated studies for E5 and E6). Use it for pilots only; confirmatory runs use the config values (10^5 and 2,000).
- A cell's results do not depend on how the cells are split across invocations (tested). They do depend on `meta.chunk_size`, so keep that value fixed.

## Output schema (long format)

Each row holds one metric for one cell × estimator/variant × axis × estimand. The columns are:

- `experiment`, `cell`, `p_*` (the cell's grid values);
- variant columns where relevant: `s_det`, `f_rej`, `t_warn`, `t_adj`, `cutoff_mm`, `subset`, `ai`, `sigma_ai_mm`, `ref`, `design`, `n_double`, `overlap_frac`, `sentinel_n`, `anchor_alpha`;
- `estimator`, `axis`, `estimand`, `metric`, `value`, `mcse`, `n`;
- `seed_entropy`, `seed_spawn_key`, `chunk_size`, `cpu_s`, `wall_s`.

Estimands are `T1` (true maximal span S), `T2` (anchor-view expected span mu_anchor) and, in E2 only, `Tv` (g × mu_anchor, the view expectation under the study's instrument settings).

## Tests (`tests/`)

- `test_known_answers.py` covers the bias of max of K view means (K = 2, 3, 4) against E[max of K N(0,1)]·sigma/sqrt(N); Clark (1961) moments; numerical MVN integration for k = 3 and 4; unbiasedness of the mean estimators; the lognormal median-centring bias; and the N = 3 window probability against 2D numerical integration for lognormal and additive-Gaussian beats.
- `test_edge_cases.py` covers K = 1, N = 1, zero noise giving the truth exactly, A7 under view error only, retrospective short clips, reductions of A4 to limiting cases, the A4 detection rates, seed invariance to cell chunking, ICC CI coverage, and degenerate agreement.
- `test_regression.py` covers the fixed-seed fixture `tests/fixtures/regression_v01.csv`, built by `tests/make_fixture.py`. Regenerate it only for an intended model change, and log the reason in `notes/lab-notebook.md`.
