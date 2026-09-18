# Figures

Final numbering fixed on 2026-09-18. Every figure is regenerable from its script with the project
environment (`/project/home/p201509/envs/duomax-sim/bin/python`, login node). Each stem has a PDF
master, a 600 dpi PNG, an RGB LZW TIFF, a `_source.csv` holding the plotted values, and the QA
derivatives `_grey.png` and `_actualsize.png`. Estimator colours, markers and line styles come from
`code/figures/estimator_style.py` (A7 is "offset-corrected mean").

Captions: `drafts/figure-legends-2026-09-18-v02.md`.

| Stem | Figure | Script | Source CSV (plotted values) | Upstream tables |
|---|---|---|---|---|
| `fig1_design` | Figure 1 | `code/13_schematic_figures.py fig1` | `figures/fig1_design_source.csv` (illustrative beat values) | none (schematic) |
| `fig2_view_accuracy` | Figure 2 | `code/figures/fig2_view_accuracy.py` | `figures/fig2_view_accuracy_source.csv` | `results/2026-09-18_amend2/analysis/E1bE3b_e1_long.csv`, `E1bE3b_e1_inflation.csv`, `E1bE3b_e1_worstcase.csv` |
| `fig3_beat_rules` | Figure 3 | `code/figures/fig3_beat_rules.py` | `figures/fig3_beat_rules_source.csv` | `results/2026-09-18_full/analysis/E2_window_met.csv`, `E2_beats_acquired.csv`, `E2_error.csv`, `E2_limited.csv` |
| `fig4_triggers` | Figure 4 | `code/figures/fig4_triggers.py` | `figures/fig4_triggers_source.csv` | `results/2026-09-18_full/analysis/E4_any_trigger.csv`, `E4_operating.csv`, `E4_ellipticity_truth.csv`, `E4_ellipticity_cells.csv` |
| `fig5_reference` | Figure 5 | `code/figures/fig5_reference.py` | `figures/fig5_reference_source.csv` | `results/2026-09-18_amend2/analysis/E5bE6b_E5b_ai_agreement.csv`, `E5bE6b_E6b_precision.csv`, `E5bE6b_E6b_sentinel_power.csv` |
| `figS1_estimator_bias` | Supplementary Figure S1 | `code/figures/figS1_estimator_bias.py` | `figures/figS1_estimator_bias_source.csv` | `results/2026-09-18_full/analysis/E1E3_e1_base_slice.csv`, `E1E3_e1_a3_crossing.csv`, `E1E3_e1_a4_variants.csv` |
| `figS2_misclassification` | Supplementary Figure S2 | `code/figures/figS2_misclassification.py` | `figures/figS2_misclassification_source.csv` | `results/2026-09-18_full/analysis/E1E3_e3_base.csv` |
| `figS3_e1_sensitivity` | Supplementary Figure S3 | `code/figures/figS3_e1_sensitivity.py` | `figures/figS3_e1_sensitivity_source.csv` | `results/2026-09-18_full/analysis/E1E3_e1_rho_N.csv`, `E1E3_e1_a4_variants.csv`, `E1E3_e1_base_slice.csv` |
| `figS4_e3b` | Supplementary Figure S4 | `code/figures/figS4_e3b.py` | `figures/figS4_e3b_source.csv` | `results/2026-09-18_amend2/analysis/E1bE3b_e3_long.csv` |
| `figS5_face_validity` | Supplementary Figure S5 | `code/figures/figS5_face_validity.py` | `figures/figS5_face_validity_source.csv` | `results/2026-09-18_amend2/analysis/face_*.csv` (from `code/12_face_validity.py`) |
| `graphical_abstract` | Graphical abstract | `code/13_schematic_figures.py ga` | `figures/graphical_abstract_source.csv` | `results/2026-09-18_amend2/analysis/schematic_ga_numbers.csv`, `schematic_e1b_base_ap_bias.csv` (written by the same script) |

Caption location for every row: `drafts/figure-legends-2026-09-18-v02.md`, under the heading of the
same figure number.

## Renaming history (2026-09-18)

| Final stem | Former stem |
|---|---|
| `fig1_design` | `fig1_design` |
| `fig2_view_accuracy` | `fig2_view_accuracy` |
| `fig3_beat_rules` | `fig4_beat_rules` |
| `fig4_triggers` | `fig5_triggers` |
| `fig5_reference` | `fig6_reference` |
| `figS1_estimator_bias` | `fig2_estimator_bias` |
| `figS2_misclassification` | `fig3_misclassification` |
| `figS3_e1_sensitivity` | `figS1_e1_sensitivity` |
| `figS4_e3b` | `figS2_e3b` |
| `figS5_face_validity` | `figS_face_validity` |
| `graphical_abstract` | `graphical_abstract` |

`figures/superseded/` holds every output as it stood before this renumbering (old stems, and the
pre-revision `fig1_design`, `fig2_view_accuracy` and `graphical_abstract`). Nothing there is current.

## Run all

```
PY=/project/home/p201509/envs/duomax-sim/bin/python
for s in fig2_view_accuracy fig3_beat_rules fig4_triggers fig5_reference figS1_estimator_bias \
         figS2_misclassification figS3_e1_sensitivity figS4_e3b figS5_face_validity; do
  $PY code/figures/$s.py
done
$PY code/13_schematic_figures.py all
```
