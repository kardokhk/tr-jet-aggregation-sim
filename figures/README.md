# Figures

Numbering revised for manuscript v04 (EHJ-CVI) on 2026-09-19; see the renaming history below. Every figure is regenerable from its script with the project
environment (`/project/home/p201509/envs/duomax-sim/bin/python`, login node). Each stem has a PDF
master, a 600 dpi PNG, an RGB LZW TIFF, a `_source.csv` holding the plotted values, and the QA
derivatives `_grey.png` and `_actualsize.png`. Estimator colours, markers and line styles come from
`code/figures/estimator_style.py`. Since v04 the figures show plain rule names only (for example
"Largest view mean after review"); the codes A1 to A7, T1, T2 and E1 to E6b are mapped in Supplementary
Table S6 (`drafts/tables-2026-09-19-v04.md`).

Captions: `drafts/figure-legends-2026-10-05-v06.md` (main legends with alt text, supplementary legends S1 to S8, graphical abstract).

| Stem | Figure | Script | Source CSV (plotted values) | Upstream tables |
|---|---|---|---|---|
| `fig1_design` | Figure 1 | `code/13_schematic_figures.py fig1` | `figures/fig1_design_source.csv` (illustrative beat values) | none (schematic) |
| `fig2_beat_rules` | Figure 2 | `code/figures/fig2_beat_rules.py` | `figures/fig2_beat_rules_source.csv` | `results/2026-09-18_full/analysis/E2_window_met.csv`, `E2_beats_acquired.csv`, `E2_error.csv`, `E2_limited.csv` |
| `fig3_view_accuracy` | Figure 3 | `code/figures/fig3_view_accuracy.py` | `figures/fig3_view_accuracy_source.csv` | `results/2026-09-18_amend2/analysis/E1bE3b_e1_long.csv`, `E1bE3b_e1_inflation.csv`, `E1bE3b_e1_worstcase.csv` |
| `fig4_triggers` | Figure 4 | `code/figures/fig4_triggers.py` | `figures/fig4_triggers_source.csv` | `results/2026-09-18_full/analysis/E4_any_trigger.csv`, `E4_operating.csv`, `E4_ellipticity_truth.csv`, `E4_ellipticity_cells.csv`, and `results/2026-10-05_osrev/fig4_clustered/fig4_clustered_se.csv` (clustered standard errors, panels c and d) |
| `fig5_reference` | Figure 5 | `code/figures/fig5_reference.py` | `figures/fig5_reference_source.csv` | `results/2026-09-18_amend2/analysis/E5bE6b_E5b_ai_agreement.csv`, `E5bE6b_E6b_precision.csv`, `E5bE6b_E6b_sentinel_power.csv` |
| `figS1_estimator_bias` | Supplementary Figure S1 | `code/figures/figS1_estimator_bias.py` | `figures/figS1_estimator_bias_source.csv` | `results/2026-09-18_full/analysis/E1E3_e1_base_slice.csv`, `E1E3_e1_a3_crossing.csv`, `E1E3_e1_a4_variants.csv` |
| `figS2_misclassification` | Supplementary Figure S2 | `code/figures/figS2_misclassification.py` | `figures/figS2_misclassification_source.csv` | `results/2026-09-18_full/analysis/E1E3_e3_base.csv` |
| `figS3_e1_sensitivity` | Supplementary Figure S3 | `code/figures/figS3_e1_sensitivity.py` | `figures/figS3_e1_sensitivity_source.csv` | `results/2026-09-18_full/analysis/E1E3_e1_rho_N.csv`, `E1E3_e1_a4_variants.csv`, `E1E3_e1_base_slice.csv` |
| `figS4_e3b` | Supplementary Figure S4 | `code/figures/figS4_e3b.py` | `figures/figS4_e3b_source.csv` | `results/2026-09-18_amend2/analysis/E1bE3b_e3_long.csv` |
| `figS5_face_validity` | Supplementary Figure S5 | `code/figures/figS5_face_validity.py` | `figures/figS5_face_validity_source.csv` | `results/2026-09-18_amend2/analysis/face_*.csv` (from `code/12_face_validity.py`) |
| `figS6_true_span_distribution` | Supplementary Figure S6 | `code/figures/figS6_true_span_distribution.py` | `figures/figS6_true_span_distribution_source.csv` | `results/2026-10-05_osrev/thresholds/thresholds_true_span_density.csv`, `thresholds_true_span_distribution.csv` |
| `figS7_beats_matched` | Supplementary Figure S7 | `code/figures/figS7_beats_matched.py` | `figures/figS7_beats_matched_source.csv` | `results/2026-10-05_osrev/beats/beats_arm_errors.csv`, `beats_table_rmse_by_n_beats.csv`, `beats_table_sensitivity.csv` |
| `figS8_error_distribution` | Supplementary Figure S8 | `code/figures/figS8_error_distribution.py` | `figures/figS8_error_distribution_source.csv` | `results/2026-10-05_osrev/views/views_a_12scenarios.csv`, `views_e_conditional_quantiles.csv` |
| `graphical_abstract` | Graphical abstract | `code/13_schematic_figures.py ga` | `figures/graphical_abstract_source.csv` | `results/2026-09-18_amend2/analysis/schematic_ga_numbers.csv`, `schematic_e1b_base_ap_bias.csv` (written by the same script) |

Caption location for every row: the v06 figure legends file named above.

## Text size revision (2026-09-19, v05)

EHJ-CVI requires figure text at least 2 mm tall at print size, so every figure except the graphical
abstract was re-typeset with an 8 pt minimum (body, ticks, keys, annotations), 8.5 pt panel titles
(Figure 1: 9 pt) and 9 pt bold panel letters. The shared constants live in
`code/figures/estimator_style.py` (`FS_MIN`, `FS_TITLE`, `FS_LETTER`, `use_journal_text()`). Only text and
layout changed: every `_source.csv` is byte-identical to its pre-revision file. Final sizes (width x
height, mm): Figure 1 183 x 82, Figure 2 183 x 170, Figure 3 183 x 170, Figure 4 183 x 130,
Figure 5 183 x 170, S1 183 x 170, S2 183 x 120, S3 183 x 128, S4 183 x 170, S5 183 x 170; graphical
abstract unchanged (180 x 110). Submission copies: `submission/figures/`. Pre-revision scripts and
PNGs: `notes/scratch/2026-09-19-figs-8pt-before/` (local only).

## Renaming history (2026-09-19, v04)

| Final stem | Former stem (v03) |
|---|---|
| `fig2_beat_rules` | `fig3_beat_rules` |
| `fig3_view_accuracy` | `fig2_view_accuracy` |

All other stems unchanged. Plotted values unchanged (every `_source.csv` byte-identical to its v03
file); text relabelled to plain names, and the graphical abstract re-typeset at 9 to 11.5 pt for
18 x 11 cm. The v03 outputs are kept in `notes/scratch/2026-09-19-figs-before/` (local only).

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
for s in fig2_beat_rules fig3_view_accuracy fig4_triggers fig5_reference figS1_estimator_bias \
         figS2_misclassification figS3_e1_sensitivity figS4_e3b figS5_face_validity; do
  $PY code/figures/$s.py
done
$PY code/13_schematic_figures.py all
```

## Revision for manuscript v06 (2026-10-06)

Changes for manuscript v06; the scripts as they stood before are in
`figures/superseded/*_2026-10-06.py` (local only). No plotted point estimate changed in any figure;
the `_source.csv` files of Figures 2, 3 and 5, Supplementary Figure S5 and the graphical abstract are
byte-identical to their previous versions.

- Figure 1 (`fig1_design`): panel b now names three different
  widths (colour jet span, vena contracta, coaptation gap) besides the oblique plane; panel c draws the
  beat-consistency window as the rule works (beats measured one at a time, stop at the first set of three
  within ±15% of their mean, other beats discarded; `window_rule()` in the script reproduces the rule for
  the illustrative values); panel d says "AI evaluation". Illustrative beat 2 changed from 6.6 to 6.2 mm
  so that no set of three qualifies after three beats.
- Figure 4: bars of panels c and d are 95% Monte Carlo intervals from patient-clustered standard errors
  (formerly binomial standard error x sqrt(3)), from `code/23_osrev_fig4_clustered_se.py`, Slurm job
  5306853 (six E4 cells re-simulated with their original seeds; counts identical to the stored tables).
  In the source csv only the `mcse` and `x_mcse` columns of panels c and d changed. Axis labels of
  panel d read sensitivity and false-positive rate.
- Figure 5: key of panel c relabelled ("Spread across studies", "Mean reported CI") and the y axis names
  ICC(A,1).
- Supplementary Figure S5, panel c: the hatched band "31% to 49%" removed; a dashed line marks the one
  like-axis value (inflow plane 30.8% below the three-dimensional maximal diameter, Singh et al. 2026);
  the biplane band is drawn at 18.7% to 24.6%.
- Supplementary Figures S7 and S8 are new (third amendment); S6 was added on 2026-10-05.
- Graphical abstract: "Mean bias" on the axis and in the title of panel a; panel c labels the upper bars
  "View pairs with error > 2 mm" and the axis "Trigger fired (%)".
- Final gate, all stems: PDF at stated size with TrueType fonts only (no Type 3), TIFF RGB with LZW and
  no alpha, 600 dpi, font Nimbus Sans. Sizes (mm): Figure 1 183 x 170, S7 183 x 140, S8 183 x 100, others
  unchanged.
- Submission copies: `submission/figures/Figure1..5.{pdf,tif}`, `GraphicalAbstract.{pdf,tif}`,
  `FigureS1..S8.png`.

Additional run commands: `$PY code/figures/figS6_true_span_distribution.py`,
`$PY code/figures/figS7_beats_matched.py`, `$PY code/figures/figS8_error_distribution.py`,

## Revision after the review of manuscript v06 (2026-10-06)

No plotted value changed in Figures 2, 3, 4 and 5 or in Supplementary Figure S8: their `_source.csv`
files are byte-identical to the versions before this revision (checked with md5sum). Only the
graphical abstract source csv changed (panel d).

  165 degree plane to the septolateral axis awaits confirmation by an imaging author.** The labels in
  the image, the legend and the alt text follow the slide supplied with the image and have not been
  checked by an echocardiographer. In this example the septolateral reading (14.2 mm) exceeds the
  anteroposterior reading (11.5 mm), which the simulated orifice does not allow (the simulated
  septolateral span is 0.4 to 1.0 times the anteroposterior span); the alt text and the Limitations say
  so. The Nyquist limit and colour scale of the example are not recorded with the image and are not
  shown. Small red residues of the scanner's caliper marks remain beside the redrawn end points.
- Figure 1b (both versions): the colour jet span is drawn 0.3 units atrial to the leaflet tips
  (was 0.5), so that it is visibly wider than the coaptation gap.
- Figure 4 is greyscale by design: thresholds are encoded by grey level and marker shape, so
  `fig4_triggers.png` and `fig4_triggers_grey.png` show the same image; this is not a failed colour
  export. Panel titles are plain text ("Warning threshold, 2 to 4 mm"; "Adjudication threshold, 5 and
  6 mm"), the key title is "Threshold", the labels "CV 5%" and "CV 15%" of panel d were moved clear of
  the neighbouring series, the x axis of panel d is described in words (linear to 0.5, logarithmic
  above), and each axis has a small margin beyond its first and last tick so that no marker is drawn
  across or outside an axis.
- Figure 5: wording of keys and axes as in the text ("Truth", "Error unrelated to images", "Inherits
  systematic error of reads", "Reader-offset SD", "Double-read subset of 500 cases", "Probability of
  detection", "False-positive rate", "Sentinel cases", "Drift towards AI proposal"); the F test series
  has open markers at full ink (formerly 30% opacity).
- Supplementary Figure S8a: the label "equal to RMSE" sits below its line.
- Graphical abstract: panel d is drawn against a single read with the models of the Results (share of
  image error reproduced 0% and 100%; amend2 E5 cells 9 and 11), so that its values are those of the
  main text (1.60 to 2.23 mm and 2.08 to 1.89 mm; asserted in `ga_numbers()`), and the key names the
  reference. Panel a states "12 view-accuracy scenarios per rule" in the panel; no plus signs; all text
  10 pt or larger; "Root-mean-square error" spelt out; category label "Patients when no view had an
  error".
- Supplementary Figure S5e: the label "Sugiura 2021" was checked against the reference record
  (Clin Res Cardiol 2021;110:451-9) and is unchanged.
- Final gate for the changed stems (`fig1_design`, `fig4_triggers`, `fig5_reference`,
  `figS8_error_distribution`, `graphical_abstract`): PDF at the stated size with embedded TrueType
  Nimbus Sans only, TIFF RGB with LZW and no alpha at 600 dpi. Submission copies refreshed:
  `submission/figures/Figure1`, `Figure4`, `Figure5` (pdf and tif), `GraphicalAbstract` (pdf and tif),
  `FigureS8.png`.


## 2026-10-06: manuscript Figure 1 is the schematic

The manuscript uses the all-schematic `fig1_design` (three panels: a jet span and sources of error, b beat-consistency window, c view rules and outputs).
