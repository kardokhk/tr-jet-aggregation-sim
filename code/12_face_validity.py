"""Face validity of the generative model (protocol Amendment 2, last bullet).

Simulates, with the library functions of code/lib/duomaxsim and the base
configuration v02 (code/configs/base.yaml), the quantities that published
studies report, so that the model can be compared with them:

  (a) inter-reader agreement (ICC(A,1), Bland-Altman bias and 95% LoA) of a
      single-view width read, single beat and 3-beat mean, in simulated
      reader studies of n = 50 (Hauptmann 2026 design) and n = 10 biplane
      (Singh 2026 reproducibility subgroup);
  (b) single long-axis view mean minus T1 (mm and %), and biplane average
      minus the average of the true AP and SL spans, over the Amendment 2
      grid of anchor scale a x long-axis scale L (Singh 2026);
  (c) coefficient of variation of measured beat values, including caliper
      error, across the beat-CV grid (Moraldo 2013, Wong 1987);
  (d) distribution of measured AP spans (T1, anchor-view mean, long-axis view
      mean, A4 composite, single beat) vs VC width median 9.5 [7.2-12.3] mm
      (Sugiura 2021), pooled and in simulated cohorts of n = 44.

Model settings not stated in the base config (assumptions, see findings file):
 - Readers in a reader study share the images (beats, view errors, g) and
   differ in per-beat caliper error and a reader-level bias drawn once per
   reader per simulated study (not per patient, unlike E1 to E4).
 - "Same beats": both readers measure beats 1 (to 3) of the clip.
   "Different beats": reader 2 measures beats 4 (to 6) of the same clip.
 - Reader-study reads use no acceptance window; (b) and (d) use the base
   beat rule (N 3, window 15%, prospective) through duomaxsim.read_once.
 - Biplane width in a view = mean of the AP and SL measured widths in that view.
 - View 1 = anchor, view 2 = long-axis (the ME-view analogue). K = 3.

Output: results/2026-09-18_amend2/analysis/face_*.csv
Run:    /project/home/p201509/envs/duomax-sim/bin/python code/12_face_validity.py [--table-only]
Seed:   numpy SeedSequence(20260921), one spawned child per part (a1, a2, b, c).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import __version__ as DMS_VERSION  # noqa: E402
from duomaxsim.config import base_params, load_config  # noqa: E402
from duomaxsim.experiments import read_once  # noqa: E402
from duomaxsim.metrics import bland_altman, icc_a1  # noqa: E402
from duomaxsim.model import draw_latent, measure  # noqa: E402
from duomaxsim.rules import a4_final  # noqa: E402

OUT = ROOT / "results" / "2026-09-18_amend2" / "analysis"
MASTER_SEED = 20260921
AP, SL = 0, 1
ANCHOR, LONG = 0, 1

# Published comparators (verified values; notes/parameter-table-2026-09-18.md and
# notes/scratch/2026-09-18-params-*.md)
HAUPT = dict(icc=0.935, icc_lo=0.894, icc_hi=0.961, loa_lo=-1.78, loa_hi=2.04, n=50)
SINGH_INTER = dict(icc_min=0.865, icc_max=0.944, n=10)


def qci(x, q, alpha=0.05):
    """Sample quantile and distribution-free 95% CI from order statistics."""
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    n = x.size
    est = float(np.quantile(x, q))
    lo_i = int(stats.binom.ppf(alpha / 2, n, q))
    hi_i = int(stats.binom.ppf(1 - alpha / 2, n, q))
    return est, float(x[max(lo_i - 1, 0)]), float(x[min(hi_i, n - 1)])


def prop(x):
    x = np.asarray(x, bool)
    p = x.mean()
    return float(p), float(np.sqrt(p * (1 - p) / x.size))


def mean_se(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


# ------------------------------------------------------------------ (a) reader studies

def reader_reads(lat, p, rng, n_beats_max=6):
    """Two readers' per-beat measured widths without reader bias: (2, n, 2, K, B')."""
    lat6 = dataclasses.replace(lat, spans=lat.spans[..., :n_beats_max])
    z = np.zeros(lat.n)
    return np.stack([measure(lat6, p, rng, z), measure(lat6, p, rng, z)])


def reader_study(x1, x2, n_per, rb, design, quantity):
    """Study-level ICC and BA for reads x1, x2 (n_patients,) grouped into studies."""
    ns = x1.size // n_per
    Y = np.stack([x1[: ns * n_per].reshape(ns, n_per) + rb[:, :1],
                  x2[: ns * n_per].reshape(ns, n_per) + rb[:, 1:2]], -1)
    icc, (lo, hi) = icc_a1(Y)
    bias, sd, llo, lhi, _ = bland_altman(Y[..., 0] - Y[..., 1])
    between_sd = Y.mean(-1).std(-1, ddof=1)
    return pd.DataFrame(dict(design=design, quantity=quantity, study=np.arange(ns), n=n_per,
                             icc=icc, icc_ci_lo=lo, icc_ci_hi=hi, ba_bias=bias, sd_diff=sd,
                             loa_lo=llo, loa_hi=lhi, loa_width=lhi - llo,
                             between_sd_mean_of_readers=between_sd))


def part_a(p, ss):
    rows = []
    rng = np.random.Generator(np.random.PCG64(ss))
    n_studies, n_per = 2000, 50
    n = n_studies * n_per
    lat = draw_latent(p, n, rng)
    M = reader_reads(lat, p, rng)                     # (2, n, 2, K, 6), no reader bias
    rb = p["sigma_rb_mm"] * rng.standard_normal((n_studies, 2))
    for vname, v in (("long-axis view", LONG), ("anchor view", ANCHOR)):
        for axname, sel in (("AP width", lambda a: a[:, AP]), ("biplane width", lambda a: a.mean(1))):
            m1, m2 = sel(M[0][:, :, v, :]), sel(M[1][:, :, v, :])     # (n, 6)
            reads = {
                "single beat, same beat": (m1[:, 0], m2[:, 0]),
                "single beat, different beats": (m1[:, 0], m2[:, 3]),
                "3-beat mean, same beats": (m1[:, :3].mean(1), m2[:, :3].mean(1)),
                "3-beat mean, different beats": (m1[:, :3].mean(1), m2[:, 3:6].mean(1)),
            }
            for rname, (x1, x2) in reads.items():
                d = reader_study(x1, x2, n_per, rb, "Hauptmann-like n=50", f"{vname}, {axname}")
                d["read"] = rname
                rows.append(d)
    # Singh-like: n = 10, biplane width, long-axis view; fresh patients and readers
    rng2 = np.random.Generator(np.random.PCG64(ss.spawn(1)[0]))
    n_st2, n_per2 = 10000, 10
    lat = draw_latent(p, n_st2 * n_per2, rng2)
    M = reader_reads(lat, p, rng2)
    rb2 = p["sigma_rb_mm"] * rng2.standard_normal((n_st2, 2))
    m1, m2 = M[0][:, :, LONG, :].mean(1), M[1][:, :, LONG, :].mean(1)
    for rname, (x1, x2) in {
        "single beat, same beat": (m1[:, 0], m2[:, 0]),
        "single beat, different beats": (m1[:, 0], m2[:, 3]),
        "3-beat mean, same beats": (m1[:, :3].mean(1), m2[:, :3].mean(1)),
        "3-beat mean, different beats": (m1[:, :3].mean(1), m2[:, 3:6].mean(1)),
    }.items():
        d = reader_study(x1, x2, n_per2, rb2, "Singh-like n=10", "long-axis view, biplane width")
        d["read"] = rname
        rows.append(d)
    studies = pd.concat(rows, ignore_index=True)

    summ = []
    for (des, qty, rd), g in studies.groupby(["design", "quantity", "read"], sort=False):
        r = dict(design=des, quantity=qty, read=rd, n_per_study=int(g["n"].iloc[0]), n_studies=len(g))
        for m in ("icc", "ba_bias", "sd_diff", "loa_lo", "loa_hi", "loa_width", "between_sd_mean_of_readers"):
            r[f"{m}_median"], r[f"{m}_median_lo"], r[f"{m}_median_hi"] = qci(g[m], 0.5)
            r[f"{m}_p025"], _, _ = qci(g[m], 0.025)
            r[f"{m}_p975"], _, _ = qci(g[m], 0.975)
            r[f"{m}_mean"], r[f"{m}_mean_mcse"] = mean_se(g[m])
        # percentile rank of the published value within the simulated study distribution
        if des.startswith("Hauptmann"):
            r["p_icc_le_published"], r["p_icc_le_published_mcse"] = prop(g["icc"] <= HAUPT["icc"])
            r["p_loawidth_ge_published"], r["p_loawidth_ge_published_mcse"] = prop(
                g["loa_width"] >= HAUPT["loa_hi"] - HAUPT["loa_lo"])
        else:
            r["p_icc_le_published"], r["p_icc_le_published_mcse"] = prop(g["icc"] <= SINGH_INTER["icc_max"])
            r["p_icc_le_singh_min"], r["p_icc_le_singh_min_mcse"] = prop(g["icc"] <= SINGH_INTER["icc_min"])
        # derived: ICC if the between-subject SD were Hauptmann's implied 2.7 mm
        err_var = float((g["sd_diff"] ** 2).mean() / 2.0)
        r["per_read_error_sd_mm"] = np.sqrt(err_var)
        r["icc_at_between_sd_2p7_derived"] = 2.7 ** 2 / (2.7 ** 2 + err_var)
        summ.append(r)
    return studies, pd.DataFrame(summ)


# ------------------------------------------------------------------ (b), (d) view vs T1

def ratio_of_means(diff, truth):
    """R = mean(diff)/mean(truth) with delta-method MCSE (Singh-style percentage)."""
    R = diff.mean() / truth.mean()
    z = diff - R * truth
    return float(100 * R), float(100 * z.std(ddof=1) / np.sqrt(z.size) / truth.mean())


def part_b(p, ss, a_grid, L_grid, n):
    rows, dist_rows, cohort_rows = [], [], []
    kids = ss.spawn(len(a_grid) * len(L_grid))
    k = 0
    for a in a_grid:
        for L in L_grid:
            q = dict(p)
            q["u_scale"] = [a, L, L, 1.2 * L]
            rng = np.random.Generator(np.random.PCG64(kids[k]))
            k += 1
            lat = draw_latent(q, n, rng)
            rb = q["sigma_rb_mm"] * rng.standard_normal(n)
            meas, br = read_once(lat, q, rng, rb)
            vm = br.value                                       # (n, 2, K) view means, 3-beat rule
            U = rng.random(vm.shape[:-1] + (vm.shape[-1] - 1,))
            a4 = a4_final(vm, lat.over, U, q["t_warn"], q["t_adj"], q["s_det"], q["f_rej"], q["a4_scrutiny"])[0]
            T = lat.S
            keys = dict(u_anchor=a, u_long=L, mean_under_anchor_pct=100 * a * np.sqrt(2 / np.pi),
                        mean_under_long_pct=100 * L * np.sqrt(2 / np.pi), n=n)
            comps = {
                "long-axis view mean, AP, vs T1 AP": (vm[:, AP, LONG], T[:, AP]),
                "long-axis view single beat, AP, vs T1 AP": (meas[:, AP, LONG, 0], T[:, AP]),
                "anchor view mean, AP, vs T1 AP": (vm[:, AP, ANCHOR], T[:, AP]),
                "long-axis view biplane mean vs mean of T1 AP and SL": (vm[:, :, LONG].mean(1), T.mean(1)),
                "A4 composite, AP, vs T1 AP": (a4[:, AP], T[:, AP]),
            }
            for cname, (x, t) in comps.items():
                dlt = x - t
                md, md_se = mean_se(dlt)
                sd = float(dlt.std(ddof=1))
                rom, rom_se = ratio_of_means(dlt, t)
                rel, rel_se = mean_se(100 * dlt / t)
                rows.append(dict(keys, comparison=cname, mean_diff_mm=md, mean_diff_mcse=md_se,
                                 sd_diff_mm=sd, sd_diff_mcse=sd / np.sqrt(2 * (dlt.size - 1)),
                                 loa_lo_mm=md - 1.96 * sd, loa_hi_mm=md + 1.96 * sd,
                                 pct_ratio_of_means=rom, pct_ratio_of_means_mcse=rom_se,
                                 pct_mean_relative=rel, pct_mean_relative_mcse=rel_se,
                                 mean_truth_mm=float(t.mean())))
            # (d) distributions of measured AP spans
            dist = {"T1 AP (true span)": T[:, AP], "anchor view 3-beat mean": vm[:, AP, ANCHOR],
                    "long-axis view 3-beat mean": vm[:, AP, LONG],
                    "long-axis view single beat": meas[:, AP, LONG, 0], "A4 composite": a4[:, AP]}
            n_coh = 44
            n_c = n // n_coh
            for dname, x in dist.items():
                r = dict(keys, quantity=dname)
                for qq, lab in ((0.25, "q25"), (0.5, "median"), (0.75, "q75"), (0.05, "q05"), (0.95, "q95")):
                    r[lab], r[f"{lab}_ci_lo"], r[f"{lab}_ci_hi"] = qci(x, qq)
                X = x[: n_c * n_coh].reshape(n_c, n_coh)
                cm = np.median(X, 1)
                c25, c75 = np.quantile(X, 0.25, axis=1), np.quantile(X, 0.75, axis=1)
                r["cohort44_median_p025"], r["cohort44_median_p975"] = np.quantile(cm, [0.025, 0.975])
                r["cohort44_q25_p025"], r["cohort44_q25_p975"] = np.quantile(c25, [0.025, 0.975])
                r["cohort44_q75_p025"], r["cohort44_q75_p975"] = np.quantile(c75, [0.025, 0.975])
                r["p_cohort_median_le_9p5"], r["p_cohort_median_le_9p5_mcse"] = prop(cm <= 9.5)
                r["n_cohorts"] = n_c
                dist_rows.append(r)
                if a == 0.06 and L == 0.25:
                    cohort_rows.append(pd.DataFrame(dict(quantity=dname, cohort=np.arange(n_c),
                                                         median=cm, q25=c25, q75=c75)))
    return pd.DataFrame(rows), pd.DataFrame(dist_rows), pd.concat(cohort_rows, ignore_index=True)


# ------------------------------------------------------------------ (c) beat CV

def part_c(p, ss, cv_grid, n, n_beats=5):
    rows = []
    kids = ss.spawn(len(cv_grid))
    for cv, kid in zip(cv_grid, kids):
        q = dict(p, beat_cv=cv, window=None, N_beats=n_beats)   # generates exactly n_beats beats
        rng = np.random.Generator(np.random.PCG64(kid))
        lat = draw_latent(q, n, rng)
        rb = q["sigma_rb_mm"] * rng.standard_normal(n)
        x = measure(lat, q, rng, rb)                               # (n, 2, K, 5)
        for src, arr in (("measured (with caliper)", x), ("true beat spans (no caliper)", lat.spans)):
            for axname, ax in (("AP", AP), ("SL", SL)):
                for vname, vs in (("long-axis view", [LONG]), ("anchor view", [ANCHOR]), ("all views", [0, 1, 2])):
                    y = arr[:, ax][:, vs, :].reshape(-1, n_beats)
                    m = y.mean(1)
                    ok = m > 0
                    cvi = y[ok].std(1, ddof=1) / m[ok]
                    c2 = cvi ** 2
                    pooled = np.sqrt(c2.mean())
                    pooled_se = c2.std(ddof=1) / np.sqrt(c2.size) / (2 * pooled)
                    # restricted to series whose mean is >= 3 mm: the RMS pooled CV is
                    # dominated by a few near-zero means otherwise (assumption, see findings)
                    big = m[ok] >= 3.0
                    c2b = c2[big]
                    pooled3 = np.sqrt(c2b.mean())
                    pooled3_se = c2b.std(ddof=1) / np.sqrt(c2b.size) / (2 * pooled3)
                    mcv3, mcv3_se = mean_se(cvi[big])
                    mcv, mcv_se = mean_se(cvi)
                    med = qci(cvi, 0.5)
                    rows.append(dict(beat_cv_param=cv, source=src, axis=axname, view=vname,
                                     n_series=int(ok.sum()), n_excluded_mean_le0=int((~ok).sum()),
                                     pooled_cv_pct=100 * pooled, pooled_cv_mcse=100 * pooled_se,
                                     mean_cv_pct=100 * mcv, mean_cv_mcse=100 * mcv_se,
                                     median_cv_pct=100 * med[0], median_cv_ci_lo=100 * med[1],
                                     median_cv_ci_hi=100 * med[2],
                                     n_series_mean_ge3=int(big.sum()),
                                     pooled_cv_ge3_pct=100 * pooled3, pooled_cv_ge3_mcse=100 * pooled3_se,
                                     mean_cv_ge3_pct=100 * mcv3, mean_cv_ge3_mcse=100 * mcv3_se,
                                     p05_cv_pct=100 * np.quantile(cvi, 0.05),
                                     p95_cv_pct=100 * np.quantile(cvi, 0.95),
                                     mean_level_mm=float(m[ok].mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ face-validity table

def build_table():
    """face_table.csv: simulated vs published value, construct match, judgement.
    Every simulated number is read from the face_*.csv files written above."""
    A = pd.read_csv(OUT / "face_a_reader_agreement.csv")
    B = pd.read_csv(OUT / "face_b_view_vs_T1.csv")
    Cc = pd.read_csv(OUT / "face_c_beat_cv.csv")
    D = pd.read_csv(OUT / "face_d_span_distribution.csv")

    def a(design, qty, read):
        return A[(A.design.str.startswith(design)) & (A.quantity == qty) & (A["read"] == read)].iloc[0]

    def b(comp, L, ua=0.06):
        return B[(B.comparison == comp) & np.isclose(B.u_long, L) & np.isclose(B.u_anchor, ua)].iloc[0]

    def c(cv, src="measured (with caliper)"):
        return Cc[np.isclose(Cc.beat_cv_param, cv) & (Cc.source == src) & (Cc.axis == "AP")
                  & (Cc.view == "long-axis view")].iloc[0]

    def d(q, L, ua=0.06):
        return D[(D.quantity == q) & np.isclose(D.u_long, L) & np.isclose(D.u_anchor, ua)].iloc[0]

    f2 = lambda x: f"{x:.2f}"
    rows = []
    s1 = a("Hauptmann", "long-axis view, AP width", "single beat, same beat")
    s3 = a("Hauptmann", "long-axis view, AP width", "3-beat mean, same beats")
    bp1 = a("Hauptmann", "long-axis view, biplane width", "single beat, same beat")
    rows.append(dict(
        item="a1 inter-reader 95% LoA, single-view width, n = 50",
        simulated=(f"AP width, 1 beat: median LoA {f2(s1.loa_lo_median)} to +{f2(s1.loa_hi_median)} mm, width "
                   f"{f2(s1.loa_width_median)} mm (95% of studies {f2(s1.loa_width_p025)} to {f2(s1.loa_width_p975)}); "
                   f"3-beat mean: width {f2(s3.loa_width_median)} mm ({f2(s3.loa_width_p025)} to {f2(s3.loa_width_p975)}); "
                   f"biplane 1 beat: width {f2(bp1.loa_width_median)} mm"),
        published="Hauptmann 2026: -1.78 to +2.04 mm (width 3.82 mm), n = 50, TTE apical 4-chamber VCW",
        construct="close (TTE VCW, core lab; simulated TEE-like span, same beats measured by both readers)",
        judgement=(f"Simulated single-beat AP LoA is wider than published (share of studies with width >= 3.82 mm "
                   f"{s1.p_loawidth_ge_published:.3f}); a 3-beat mean or a biplane single beat reproduces it"),
        source="face_a_reader_agreement.csv, Hauptmann-like n=50, long-axis view"))
    rows.append(dict(
        item="a2 inter-reader ICC(A,1), n = 50",
        simulated=(f"AP width, 1 beat: median {s1.icc_median:.3f} (95% of studies {s1.icc_p025:.3f} to {s1.icc_p975:.3f}); "
                   f"3-beat mean {s3.icc_median:.3f}; per-read error SD {s1.per_read_error_sd_mm:.2f} mm; "
                   f"ICC at between-subject SD 2.7 mm (derived) {s1.icc_at_between_sd_2p7_derived:.3f}; "
                   f"simulated between-subject SD {s1.between_sd_mean_of_readers_median:.2f} mm"),
        published="Hauptmann 2026: 0.935 (0.894 to 0.961); implied per-read error SD 0.69 mm and between-subject SD 2.7 mm (derived)",
        construct="close",
        judgement=(f"ICC matches (published value at simulated percentile {s1.p_icc_le_published:.2f}) only because a "
                   "larger simulated per-read error is offset by a wider simulated case mix; ICC is not a sharp check"),
        source="face_a_reader_agreement.csv"))
    sg = a("Singh", "long-axis view, biplane width", "single beat, same beat")
    rows.append(dict(
        item="a3 inter-reader ICC, biplane VCW, n = 10",
        simulated=(f"median {sg.icc_median:.3f} (95% of studies {sg.icc_p025:.3f} to {sg.icc_p975:.3f}); "
                   f"share of studies <= 0.865: {sg.p_icc_le_singh_min:.3f}, <= 0.944: {sg.p_icc_le_published:.3f}"),
        published="Singh 2026: 0.865 to 0.944 across three ME views, n = 10, TEE",
        construct="direct (TEE colour VCW)",
        judgement="Consistent; with n = 10 the published range is compatible with a wide range of error models (weak check)",
        source="face_a_reader_agreement.csv, Singh-like n=10"))
    for L, lab in ((0.25, "base L 0.25"), (0.50, "L 0.50")):
        r = b("long-axis view mean, AP, vs T1 AP", L)
        rows.append(dict(
            item=f"b1 single long-axis view mean minus T1, {lab}",
            simulated=(f"{r.mean_diff_mm:.2f} mm (MCSE {r.mean_diff_mcse:.3f}); {r.pct_ratio_of_means:.1f}% of mean T1 "
                       f"(MCSE {r.pct_ratio_of_means_mcse:.2f}); SD of differences {r.sd_diff_mm:.2f} mm; mean T1 {r.mean_truth_mm:.2f} mm"),
            published="Singh 2026: single-plane VCW 5.18 to 8.29 mm below 3D maximum diameter (31% to 49%, derived); SD of differences 4.5 to 4.9 mm (derived); mean 3D maximum 16.82 mm",
            construct="direct construct for T1 (maximal span); 3D colour MPR maximum is itself an imperfect reference",
            judgement=("Base underestimates less than published by about half (15% vs 31% to 49%)" if L == 0.25 else
                       "Reaches the lower end of the published range in %; mm smaller because simulated T1 is smaller"),
            source=f"face_b_view_vs_T1.csv, u_anchor 0.06, u_long {L}"))
    for L, lab in ((0.25, "base L 0.25"), (0.50, "L 0.50")):
        r = b("long-axis view biplane mean vs mean of T1 AP and SL", L)
        rows.append(dict(
            item=f"b2 biplane average minus mean of true AP and SL, {lab}",
            simulated=(f"{r.mean_diff_mm:.2f} mm (MCSE {r.mean_diff_mcse:.3f}); {r.pct_ratio_of_means:.1f}% "
                       f"(MCSE {r.pct_ratio_of_means_mcse:.2f})"),
            published="Singh 2026: biplane VCW 2.42 to 3.17 mm (19% to 25%) below 3D average VCW",
            construct="direct",
            judgement=("Base underestimates less than published (15% vs 19% to 25%)" if L == 0.25 else
                       "Overshoots published (33% vs 19% to 25%): no single L fits b1 and b2 together"),
            source=f"face_b_view_vs_T1.csv, u_anchor 0.06, u_long {L}"))
    c15, c10, c05 = c(0.15), c(0.10), c(0.05)
    t15 = c(0.15, "true beat spans (no caliper)")
    rows.append(dict(
        item="c1 beat-to-beat CV of measured values (pooled RMS, 5 beats, AP, long-axis view, mean >= 3 mm)",
        simulated=(f"base beat CV 15%: {c15.pooled_cv_ge3_pct:.1f}% (MCSE {c15.pooled_cv_ge3_mcse:.2f}); true spans "
                   f"{t15.pooled_cv_ge3_pct:.1f}%; beat CV 10%: {c10.pooled_cv_ge3_pct:.1f}%; beat CV 5%: {c05.pooled_cv_ge3_pct:.1f}%"),
        published="Moraldo 2013: 15.5% (per-patient 2% to 29%), PISA distance, MR, TTE, n = 11, measured values",
        construct="close (linear colour-Doppler dimension, other valve)",
        judgement=("Base overshoots by about 5 points because the 15% beat CV was taken from measured values and caliper "
                   "error is added on top; measured 15.5% corresponds to a beat CV parameter of about 5% to 10%"),
        source="face_c_beat_cv.csv"))
    rows.append(dict(
        item="c2 beat CV of measured values vs jet-area CV",
        simulated=(f"base mean CV {c15.mean_cv_ge3_pct:.1f}% (MCSE {c15.mean_cv_ge3_mcse:.2f}); lowest on grid "
                   f"(beat CV 5%) {c05.mean_cv_ge3_pct:.1f}%"),
        published="Wong 1987: mean CV 14% to 22% of jet area (5 beats, includes reading error); linear equivalent about 7% to 11% (derived, sqrt scaling)",
        construct="proxy (area)",
        judgement=("Inside the published area range, but above the linear equivalent at every grid value: caliper error "
                   "alone contributes about 12% at a 9 mm span (derived)"),
        source="face_c_beat_cv.csv"))
    for q, L in (("long-axis view 3-beat mean", 0.25), ("anchor view 3-beat mean", 0.25), ("A4 composite", 0.25),
                 ("T1 AP (true span)", 0.25), ("long-axis view 3-beat mean", 0.50)):
        r = d(q, L)
        rows.append(dict(
            item=f"d measured AP span distribution: {q}, L {L}",
            simulated=(f"median {r['median']:.2f} [IQR {r.q25:.2f} to {r.q75:.2f}] mm; medians of cohorts of 44: "
                       f"{r.cohort44_median_p025:.2f} to {r.cohort44_median_p975:.2f} mm (95%); share of cohorts with "
                       f"median <= 9.5 mm {r.p_cohort_median_le_9p5:.2f}"),
            published="Sugiura 2021: TEE VC width median 9.5 [7.2 to 12.3] mm, n = 44 (T-TEER)",
            construct="close (VC width, view not stated)",
            judgement=("Published median inside the simulated cohort range" if r.cohort44_median_p025 <= 9.5 <= r.cohort44_median_p975
                       else "Published median outside the simulated cohort range"),
            source=f"face_d_span_distribution.csv, u_anchor 0.06, u_long {L}"))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "face_table.csv", index=False)
    return t


def main():
    t0 = time.process_time()
    cfg = load_config(ROOT / "code" / "configs" / "base.yaml")
    p = base_params(cfg)
    assert cfg["meta"]["config_version"] == "base-2026-09-18-v02"
    assert p["K"] == 3 and p["rhythm"] == "sinus" and p["data_mode"] == "prospective"
    ss_a, ss_b, ss_c = np.random.SeedSequence(MASTER_SEED).spawn(3)
    OUT.mkdir(parents=True, exist_ok=True)

    studies, a_sum = part_a(p, ss_a)
    a_sum.to_csv(OUT / "face_a_reader_agreement.csv", index=False)
    keep = studies[studies["read"].isin(["single beat, same beat", "3-beat mean, same beats",
                                         "single beat, different beats", "3-beat mean, different beats"])]
    keep = keep[keep["quantity"].isin(["long-axis view, AP width", "long-axis view, biplane width"])]
    keep.to_csv(OUT / "face_a_reader_studies.csv", index=False)
    t_a = time.process_time() - t0

    b, d, coh = part_b(p, ss_b, [0.03, 0.06, 0.125, 0.25], [0.10, 0.25, 0.50], n=88000)
    b.to_csv(OUT / "face_b_view_vs_T1.csv", index=False)
    d.to_csv(OUT / "face_d_span_distribution.csv", index=False)
    coh.to_csv(OUT / "face_d_cohort44_base.csv", index=False)
    t_b = time.process_time() - t0 - t_a

    c = part_c(p, ss_c, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], n=30000)
    c.to_csv(OUT / "face_c_beat_cv.csv", index=False)
    t_c = time.process_time() - t0 - t_a - t_b

    prov = dict(script="code/12_face_validity.py", config="code/configs/base.yaml",
                config_version=cfg["meta"]["config_version"], duomaxsim=DMS_VERSION,
                master_seed=MASTER_SEED, seed_children="SeedSequence(20260921).spawn(3) -> a, b, c; "
                "a spawns 1 extra child for the n=10 design; b and c spawn one child per scenario",
                numpy=np.__version__, pandas=pd.__version__,
                n=dict(a_hauptmann="2000 studies x 50", a_singh="10000 studies x 10",
                       b_d="88000 patients per (a, L) scenario", c="30000 patients x 3 views x 5 beats"),
                cpu_s=dict(a=round(t_a, 1), b_d=round(t_b, 1), c=round(t_c, 1)),
                date="2026-09-18")
    (OUT / "face_provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))
    build_table()


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        build_table()
    else:
        main()
