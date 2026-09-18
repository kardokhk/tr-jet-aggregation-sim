# Parameter table (base case v02, reconciled 2026-09-18)

Sources were retrieved and re-verified by independent agents on 2026-09-18 (workflow wf_46c202a1-a89; full notes in notes/scratch/2026-09-18-params-*.md). Construct match: direct = TEE colour-jet span at coaptation level; close = VC width or coaptation gap; proxy = VC area, jet area, EROA, PISA; weak = other valve or measurement. No source measures TEE colour-jet span directly, so every empirical parameter is also varied over a grid.

| Parameter | Base | Grid | Construct | Source (DOI) | Note |
|---|---|---|---|---|---|
| Case mix, median true AP span | 10 mm | fixed | close | 10.1007/s00392-020-01784-w; 10.3389/fcvm.2024.1452446 | VC width medians 9.5 to 13 mm in T-TEER cohorts |
| Case mix, log-SD | 0.40 | fixed | close | derived from IQR 7.2 to 12.3 mm (10.1007/s00392-020-01784-w) | 5th to 95th percentile about 5.2 to 19 mm |
| Mean SL/AP ratio | 0.64 | 0.53, 0.64, 0.8, 0.9 | close | Song 2011 10.1016/j.echo.2011.01.005 (AP minus SL 3.9 mm); Singh 2026 10.21037/acs-2025-1-72-tvd (3D min/max 0.53) | 0.64 calibrated to the 3.9 mm difference |
| Anchor-view underestimation scale | 0.06 | 0.03 (low scenario) | assumed | none | en-face anchor assumed nearly unbiased |
| Long-axis underestimation scale | 0.25 (mean 20%) | 0.10 (low scenario) | close | Singh 2026 Table 3 (biplane VCW 19 to 25% below 3D average; single-plane 31 to 49% below 3D maximum); bRIGHT 10.1016/j.echo.2023.12.002 (TG-SAX central gap 8.1 vs ME inflow-outflow 4.6 to 5.2 mm) | Singh differences are 3D minus 2D (sign corrected by verifier) |
| Probability of overestimation per view | 0.05 to 0.20 | on/off | assumed | weak in-vitro proxy 10.1016/j.echo.2005.03.021 | no clinical source |
| Overestimation magnitude (median) | 15% | fixed | assumed | none | |
| Inter-view correlation | 0.3 | 0, 0.3, 0.6, 0.9 | assumed | none | |
| Beat-to-beat CV, sinus | 15% | 5 to 30% | close | Moraldo 2013 10.1016/j.ijcard.2012.11.059 (PISA distance 15.5%); Wong 1987 10.1016/0002-9149(87)91035-6 (jet area 14 to 22%, proxy) | |
| RR CV in AF | 20% | 10, 20, 30% | assumed | none retrieved | |
| Extra AF beat CV | 5% | fixed | assumed | Wong 1987 calls AF effect on jet area "marginal" | |
| Log span on log RR slope | 0.2 | fixed | assumed | direction from Sumida 2003 10.1016/s0894-7317(03)00275-x (aortic) | |
| Caliper SD per beat | 1.0 mm | fixed (E1 to E4) | close | Hauptmann 2026 10.1001/jamacardio.2026.3803 (per-reading 0.69 mm, TTE VCW); Singh 2026 (within-reader about 1.2 mm, derived) | |
| Reader bias SD | 0.75 mm | 0, 0.75, 2.0 (E6) | close | Singh 2026 (inter minus intra variance, n = 10); Alexander 2022 10.1053/j.jvca.2022.03.025 for the unstandardised upper scenario | imprecise |
| Instrument factor SD(log) | 0.10 | fixed | proxy | Fan 1994 10.1161/01.cir.89.5.2141 (6 machines, sqrt area conversion 0.09 to 0.13) | width conversion assumes isotropic scaling |

Guideline and practice anchors for relevance (notes/scratch/2026-09-18-relevance.md): beat averaging 3 in sinus and at least 5 in AF (ASE/EACVI 2015 chamber quantification 10.1016/j.echo.2014.10.003; applied to TR VCW by Dreyfus 2026 10.1093/eurheartj/ehag214); EACVI 2022 "at least two-three beats" for TR VCW (10.1093/ehjci/jeab253); TVARC biplane VCW averaging (10.1093/eurheartj/ehad653); maximal gap in transgastric short axis in device screening (ASE 2022 TEE standards 10.1016/j.echo.2021.07.006); no public source for fixed-mm cross-view triggers.

Open verification items: Liu 2021 full text (LoA convention) not accessible; Song 2011 absolute AP and SL means not accessible; Retraction Watch database not queried (Europe PMC flags only). Resolve at citation-integrity pass.
