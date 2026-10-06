"""Protocol amendment 3, A3-5 (post hoc sensitivity analyses requested by the senior author):
AI validation and reference sets.

Nothing in the existing library is modified. The draw functions below repeat the draw order of
experiments.run_E5 and experiments.run_E6 and return the per-patient arrays that those runners
discard; with the new options off they consume the random-number stream exactly as the runners do,
so the arrays are bit-identical for the same SeedSequence (asserted in
tests/test_osrev_ai_reference.py).

Contents
  e5_draws, reference_designs, e5_library_metrics      AI-validation draws and the run_E5 summaries
  e6_draws, sentinel_tests, e6_library_metrics          double-read and sentinel draws, run_E6 tests
  perm_variance_ratio_pvalues                           two-sample permutation test of the variance ratio
  reader_panel_reads                                    reader panels reading repeated case samples
  crossed_allocation, rotating_allocation,
  between_reader_mom                                    multi-reader designs and their estimator
  oneway_components                                     one-way variance decomposition
  folded_normal_mean, predicted_apparent,
  match_sigma_for_mae, study_agreement                  analytic checks and model matching
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import optimize, stats
from scipy.special import ndtr

from .experiments import _agreement, _chunks, _gen, _n_chunks, ai_shared, image_composite, protocol_read
from .metrics import bland_altman, brown_forsythe, icc_a1, mean_mcse
from .model import AXES, draw_latent

Z975 = 1.96   # the library's limits of agreement use 1.96, not 1.959964


# ====================================================================== E5: AI validation

def e5_draws(p: dict, ss: np.random.SeedSequence, n_studies: int, chunk: int,
             extra_noise: bool = False) -> dict:
    """Per-patient arrays of experiments.run_E5, in the same draw order.

    Returns arrays shaped (n_studies, n_val): T1 (true span of the study axis), R_img (composite
    from the images, reader 1's review decisions), r1, r2, r3 (three reads by three distinct
    readers of a pool of n_readers), b1, b2, b3 (the reader offset applied to each read) and z
    (standard-normal AI noise, shared by every AI model of run_E5); plus a_int and b_slope (the
    inherited-bias calibration).

    extra_noise=False (default) consumes exactly the random numbers of run_E5. extra_noise=True
    draws a second standard-normal array z2 after z in each chunk (run_E5 draws nothing after z,
    so all other arrays are unchanged).
    """
    ax = AXES.index(p["study_axis"])
    nv = int(p["n_val"])
    R = int(p["n_readers"])
    spc = max(1, chunk // nv)
    kids = ss.spawn(1 + _n_chunks(n_studies, spc))
    rng = _gen(kids[0])
    lat = draw_latent(p, int(p["calib_n"]), rng)
    r = protocol_read(lat, p, rng, p["sigma_rb_mm"] * rng.standard_normal(lat.n))[:, ax]
    b_slope, a_int = np.polyfit(lat.S[:, ax], r, 1)
    names = ["T1", "R_img", "r1", "r2", "r3", "b1", "b2", "b3", "z"] + (["z2"] if extra_noise else [])
    keep = {k: [] for k in names}
    for s_c, rng in _chunks(kids[1:], n_studies, spc):
        n = s_c * nv
        lat = draw_latent(p, n, rng)
        pool = p["sigma_rb_mm"] * rng.standard_normal((s_c, R))
        perm = np.argsort(rng.random((n, R)), axis=1)[:, :3]
        study = np.repeat(np.arange(s_c), nv)
        drift = p["drift_mm_per_100"] * np.tile(np.arange(nv), s_c) / 100.0
        for j in range(3):
            bj = pool[study, perm[:, j]] + drift
            v, U = protocol_read(lat, p, rng, bj, return_u=True)
            keep[f"r{j + 1}"].append(v[:, ax].reshape(s_c, nv))
            keep[f"b{j + 1}"].append(bj.reshape(s_c, nv))
            if j == 0:
                U_r1 = U
        keep["T1"].append(lat.S[:, ax].reshape(s_c, nv))
        keep["z"].append(rng.standard_normal((s_c, nv)))
        keep["R_img"].append(image_composite(lat, p, U_r1)[:, ax].reshape(s_c, nv))
        if extra_noise:
            keep["z2"].append(rng.standard_normal((s_c, nv)))
    out = {k: np.concatenate(v, axis=0) for k, v in keep.items()}
    out["a_int"] = float(a_int)
    out["b_slope"] = float(b_slope)
    return out


def reference_designs(r1, r2, r3, tols, adjudication_value: str = "adjudicator") -> dict:
    """Reference designs of run_E5 from three reads (same arithmetic as the runner).

    single = r1; mean2 = (r1 + r2) / 2; adj_tol<t> = r3 where |r1 - r2| > t, else (r1 + r2) / 2
    ('adjudicator'), or the mean of the closest pair of the three reads ('closest_pair_mean').
    """
    refs = {"single": r1, "mean2": 0.5 * (r1 + r2)}
    dis = np.abs(r1 - r2)
    for tol in tols:
        if adjudication_value == "adjudicator":
            adjv = r3
        elif adjudication_value == "closest_pair_mean":
            cand = np.stack([0.5 * (r1 + r2), 0.5 * (r1 + r3), 0.5 * (r2 + r3)], -1)
            gaps = np.stack([np.abs(r1 - r2), np.abs(r1 - r3), np.abs(r2 - r3)], -1)
            adjv = np.take_along_axis(cand, np.argmin(gaps, -1)[..., None], -1)[..., 0]
        else:
            raise ValueError(adjudication_value)
        refs[f"adj_tol{tol:g}"] = np.where(dis > tol, adjv, 0.5 * (r1 + r2))
    return refs


def e5_library_metrics(d: dict, p: dict, sigmas, tols, lam: float | None = None) -> dict:
    """The run_E5 summaries recomputed from e5_draws output.

    Returns {(ai, sigma, ref, metric): (mean over studies, MCSE)} for the apparent_*, true_* and
    diff_* metrics, {("ref", ref, metric): ...} for the reference-quality rows and
    {("paired", ai, sigma, ref): proportion of studies with lower MAE than the independent AI}
    when lam is given, with the same arithmetic as the runner.
    """
    T1, z = d["T1"], d["z"]
    refs = reference_designs(d["r1"], d["r2"], d["r3"], tols, p["adjudication_value"])
    out = {}
    if lam is not None:
        q = _agreement(d["R_img"], T1)
        for m in ("mae", "ba_bias"):
            out[("ref", "R_img", f"ref_{m}_vs_T1")] = tuple(float(v) for v in mean_mcse(q[m]))
    for rn, ref in refs.items():
        q = _agreement(ref, T1)
        for m in ("mae", "ba_bias"):
            out[("ref", rn, f"ref_{m}_vs_T1")] = tuple(float(v) for v in mean_mcse(q[m]))
    for sig in sigmas:
        ais = {"independent": T1 + sig * z, "inherited": d["a_int"] + d["b_slope"] * T1 + sig * z}
        if lam is not None:
            ais["shared"] = ai_shared(T1, d["R_img"], lam, sig, z)
        per = {}
        for an, ai in ais.items():
            tru = _agreement(ai, T1)
            per[(an, "T1")] = tru["mae"]
            for rn, ref in refs.items():
                app = _agreement(ai, ref)
                per[(an, rn)] = app["mae"]
                for m in app:
                    out[(an, float(sig), rn, "apparent_" + m)] = tuple(float(v) for v in mean_mcse(app[m]))
                    out[(an, float(sig), rn, "true_" + m)] = tuple(float(v) for v in mean_mcse(tru[m]))
                    out[(an, float(sig), rn, "diff_" + m)] = tuple(float(v) for v in mean_mcse(app[m] - tru[m]))
        for an in ais:
            if an == "independent":
                continue
            for rn in list(refs) + ["T1"]:
                dm = per[(an, rn)] - per[("independent", rn)]
                out[("paired", an, float(sig), rn)] = float(np.mean(dm < 0))
    return out


def study_agreement(ai: np.ndarray, ref: np.ndarray) -> dict:
    """Per-study MAE, bias (mean of ai - ref), SD (ddof 1) and 95% limits-of-agreement width
    (2 x 1.96 x SD) of arrays shaped (n_studies, n)."""
    dd = ai - ref
    sd = dd.std(axis=-1, ddof=1)
    return {"mae": np.abs(dd).mean(-1), "bias": dd.mean(-1), "sd": sd, "loa_width": 2 * Z975 * sd,
            "mse": (dd * dd).mean(-1)}


def folded_normal_mean(mu, sd):
    """E|X| for X ~ N(mu, sd^2): sd sqrt(2/pi) exp(-mu^2 / (2 sd^2)) + mu (1 - 2 Phi(-mu / sd)).
    With mu = 0 this is sqrt(2/pi) sd. sd = 0 gives |mu|."""
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = sd * np.sqrt(2.0 / np.pi) * np.exp(-mu ** 2 / (2.0 * sd ** 2)) + mu * (1.0 - 2.0 * ndtr(-mu / sd))
    return np.where(sd > 0, val, np.abs(mu))


def predicted_apparent(var_ai, var_ref, cov, bias_ai=0.0, bias_ref=0.0) -> dict:
    """Second-moment prediction for the difference d = model - reference.

    Var(d) = Var(e_model) + Var(e_ref) - 2 Cov(e_model, e_ref); bias(d) = bias_model - bias_ref;
    MSE(d) = Var(d) + bias(d)^2; LoA width = 2 x 1.96 sqrt(Var(d)); MAE under normality is the
    folded-normal mean. cov = rho sd_model sd_ref for errors with correlation rho.
    """
    var_d = var_ai + var_ref - 2.0 * cov
    bias_d = bias_ai - bias_ref
    sd_d = np.sqrt(np.maximum(var_d, 0.0))
    return {"var": var_d, "bias": bias_d, "mse": var_d + bias_d ** 2, "loa_width": 2 * Z975 * sd_d,
            "mae_normal": folded_normal_mean(bias_d, sd_d)}


def match_sigma_for_mae(e_sys: np.ndarray, z: np.ndarray, target_mae: float, hi: float = 10.0) -> float:
    """Own-error SD s such that mean |e_sys + s z| = target_mae on the calibration sample (e_sys, z).

    Raises ValueError if the systematic part alone already exceeds the target (no solution).
    """
    e_sys = np.ravel(e_sys)
    z = np.ravel(z)
    f = lambda s: float(np.abs(e_sys + s * z).mean()) - target_mae
    if f(0.0) > 0:
        raise ValueError("systematic error alone exceeds the target MAE")
    if f(0.0) == 0:
        return 0.0
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-10, rtol=1e-12))


# ====================================================================== E6: double reads and sentinel sets

def e6_draws(p: dict, ss: np.random.SeedSequence, n_studies: int, chunk: int,
             skip_double: bool = False) -> dict:
    """Per-case arrays of experiments.run_E6, in the same draw order.

    Returns r1, r2 (n_studies, ng): double reads by the two readers of each study; b_pair
    (n_studies, 2); man (n_studies, production_n): manual reads by one production reader; ai: the
    AI draft, T1 + ai_draft_bias_mm + ai_draft_sigma_mm x noise; T1; b_prod.

    skip_double=False (default) consumes exactly the random numbers of run_E6. skip_double=True
    omits the double-read part (new option; the sentinel arrays then come from a different part
    of the stream and are not comparable with a run_E6 result for the same seed).
    """
    ax = AXES.index(p["study_axis"])
    gold = [int(x) for x in p["gold_n"]]
    prodn = int(p["production_n"])
    ovl = [float(f) for f in p["overlap_frac"]]
    sizes = sorted(set(gold) | {int(round(prodn * f)) for f in ovl})
    ng = max(sizes)
    per_study = ng + prodn
    spc = max(1, chunk // per_study)
    kids = ss.spawn(_n_chunks(n_studies, spc))
    keep = {k: [] for k in ("r1", "r2", "b_pair", "man", "ai", "T1", "b_prod")}
    for s_c, rng in _chunks(kids, n_studies, spc):
        if not skip_double:
            lat = draw_latent(p, s_c * ng, rng)
            b = p["sigma_rb_mm"] * rng.standard_normal((s_c, 2))
            drift = p["drift_mm_per_100"] * np.tile(np.arange(ng), s_c) / 100.0
            study = np.repeat(np.arange(s_c), ng)
            keep["r1"].append(protocol_read(lat, p, rng, b[study, 0] + drift)[:, ax].reshape(s_c, ng))
            keep["r2"].append(protocol_read(lat, p, rng, b[study, 1] + drift)[:, ax].reshape(s_c, ng))
            keep["b_pair"].append(b)
        lat = draw_latent(p, s_c * prodn, rng)
        b = p["sigma_rb_mm"] * rng.standard_normal(s_c)
        study = np.repeat(np.arange(s_c), prodn)
        drift = p["drift_mm_per_100"] * np.tile(np.arange(prodn), s_c) / 100.0
        man = protocol_read(lat, p, rng, b[study] + drift)[:, ax].reshape(s_c, prodn)
        T1 = lat.S[:, ax].reshape(s_c, prodn)
        ai = T1 + p["ai_draft_bias_mm"] + p["ai_draft_sigma_mm"] * rng.standard_normal((s_c, prodn))
        keep["man"].append(man)
        keep["ai"].append(ai)
        keep["T1"].append(T1)
        keep["b_prod"].append(b)
    out = {k: np.concatenate(v, axis=0) for k, v in keep.items() if v}
    out["sizes"] = sizes
    return out


def sentinel_tests(dman: np.ndarray, ns: int, w: float) -> dict:
    """p-values of the run_E6 sentinel tests (same arithmetic as the runner).

    dman: (n_studies, production_n) manual read minus AI draft. Sentinel sample: the first ns
    cases (manual). Assisted sample: the remaining cases with the difference shrunk to
    (1 - w) x dman. Returns welch (two-sided, means), f (one-sided, H1 sentinel variance larger),
    bf_two and bf_one (Brown-Forsythe; one-sided H1 sentinel more dispersed).
    """
    ds = dman[:, :ns]
    da = (1.0 - w) * dman[:, ns:]
    m1, m2 = ds.mean(-1), da.mean(-1)
    v1, v2 = ds.var(-1, ddof=1), da.var(-1, ddof=1)
    n1, n2 = ds.shape[-1], da.shape[-1]
    se2 = v1 / n1 + v2 / n2
    t = (m1 - m2) / np.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p_t = 2 * stats.t.sf(np.abs(t), df)
    p_f = stats.f.sf(v1 / v2, n1 - 1, n2 - 1)
    p_bf2, p_bf1 = brown_forsythe(ds, da)
    return {"welch": p_t, "f": p_f, "bf_two": p_bf2, "bf_one": p_bf1}


def e6_library_metrics(d: dict, p: dict) -> dict:
    """The run_E6 summaries recomputed from e6_draws output (skip_double=False).

    Returns {("double", n, metric): value} with metric in loa_hi_empirical_sd,
    loa_ci_width_each_mean, icc_mean, icc_empirical_sd, and {("sentinel", ns, w, test): rejection
    rate} for the tests of run_E6 (names as stored: reject_welch_mean, reject_f_variance,
    reject_bf, reject_bf_onesided).
    """
    out = {}
    lvl = float(p["test_level"])
    for nn in d["sizes"]:
        r1, r2 = d["r1"][:, :nn], d["r2"][:, :nn]
        icc, (lo, hi) = icc_a1(np.stack([r1, r2], -1))
        bias, sd, llo, lhi, hw = bland_altman(r1 - r2)
        out[("double", nn, "icc_mean")] = float(np.nanmean(icc))
        out[("double", nn, "icc_empirical_sd")] = float(np.nanstd(icc, ddof=1))
        out[("double", nn, "icc_ci_width_mean")] = float(np.nanmean(hi - lo))
        out[("double", nn, "loa_hi_mean")] = float(np.nanmean(lhi))
        out[("double", nn, "loa_hi_empirical_sd")] = float(np.nanstd(lhi, ddof=1))
        out[("double", nn, "loa_ci_width_each_mean")] = float(np.nanmean(2 * hw))
    dman = d["man"] - d["ai"]
    for ns in p["sentinel_n"]:
        for al in p["anchor_alpha"]:
            pv = sentinel_tests(dman, int(ns), float(al))
            for name, key in (("reject_welch_mean", "welch"), ("reject_f_variance", "f"),
                              ("reject_bf", "bf_two"), ("reject_bf_onesided", "bf_one")):
                out[("sentinel", int(ns), float(al), name)] = float((pv[key] < lvl).astype(float).mean())
    return out


def perm_variance_ratio_pvalues(d: np.ndarray, ns_list, w_list, n_perm: int,
                                rng: np.random.Generator | None = None,
                                perms: np.ndarray | None = None, study_chunk: int = 100) -> dict:
    """One-sided two-sample permutation test of the variance ratio (sentinel over assisted).

    d: (n_studies, N) manual-minus-draft differences in case order. For a sentinel size ns and an
    anchoring fraction w the two samples are x1 = d[:, :ns] and x2 = (1 - w) d[:, ns:], as in
    run_E6. Statistic: T = s1^2 / s2^2, each variance about its own sample mean (ddof 1).
    Reference distribution: T recomputed after reassigning the N pooled observations to groups of
    sizes ns and N - ns by n_perm random permutations, drawn independently for every study.
    p = (1 + #{T* >= T (1 - 1e-12)}) / (n_perm + 1), which is exact (size <= level) when the N observations
    are exchangeable under the null hypothesis.

    For a given study the same permutations serve every ns (the first ns positions of the
    permuted order form the sentinel group) and every w. perms (n_studies, n_perm, N) may be
    supplied instead of rng (testing). Returns {(ns, w): p-values (n_studies,)}.
    """
    d = np.asarray(d, dtype=float)
    S, N = d.shape
    ns_list = [int(n) for n in ns_list]
    w_list = [float(w) for w in w_list]
    if max(ns_list) >= N - 1 or min(ns_list) < 2:
        raise ValueError("each group needs at least two observations")
    if any(w >= 1.0 for w in w_list):
        raise ValueError("w must be below 1 (the assisted variance would be zero)")
    nmax = max(ns_list)
    pv = {(ns, w): np.empty(S) for ns in ns_list for w in w_list}
    base = np.arange(N, dtype=np.int16 if N < 32767 else np.int64)
    for a in range(0, S, study_chunk):
        dc = d[a:a + study_chunk]
        sc = dc.shape[0]
        if perms is None:
            idx = rng.permuted(np.broadcast_to(base, (sc * n_perm, N)).copy(), axis=1)[:, :nmax]
            idx = idx.reshape(sc, n_perm, nmax).astype(np.int64)
        else:
            idx = np.asarray(perms[a:a + study_chunk, :, :nmax], dtype=np.int64)
        vals = np.take_along_axis(dc[:, None, :], idx, axis=2)          # (sc, n_perm, nmax)
        vals2 = vals * vals
        csum = np.cumsum(dc, axis=1)
        csum2 = np.cumsum(dc * dc, axis=1)
        for ns in ns_list:
            n2 = N - ns
            own = idx[:, :, :ns] < ns                                   # drawn from the sentinel block
            v, v2 = vals[:, :, :ns], vals2[:, :, :ns]
            A1 = np.where(own, v, 0.0).sum(-1)
            A2 = np.where(own, v2, 0.0).sum(-1)
            B1 = v.sum(-1) - A1
            B2 = v2.sum(-1) - A2
            s_sent, q_sent = csum[:, ns - 1], csum2[:, ns - 1]
            s_ass, q_ass = csum[:, -1] - s_sent, csum2[:, -1] - q_sent
            for w in w_list:
                c = 1.0 - w
                tot1 = s_sent + c * s_ass
                tot2 = q_sent + c * c * q_ass
                # observed
                v1o = (q_sent - s_sent ** 2 / ns) / (ns - 1)
                v2o = (c * c * q_ass - (c * s_ass) ** 2 / n2) / (n2 - 1)
                # permuted
                s1 = A1 + c * B1
                q1 = A2 + c * c * B2
                s2 = tot1[:, None] - s1
                q2 = tot2[:, None] - q1
                v1p = (q1 - s1 ** 2 / ns) / (ns - 1)
                v2p = (q2 - s2 ** 2 / n2) / (n2 - 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    # T* >= T without dividing; relative tolerance 1e-12 so that a rearrangement
                    # reproducing the observed split counts as a tie despite rounding of the sums
                    ge = (v1p * v2o[:, None]) >= (v1o[:, None] * v2p) * (1.0 - 1e-12)
                pv[(ns, w)][a:a + sc] = (1.0 + ge.sum(-1)) / (n_perm + 1.0)
    return pv


# ====================================================================== reader panels

def reader_panel_reads(p: dict, rng: np.random.Generator, n_panels: int, n_rep: int, n_cases: int,
                       n_readers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Reads of n_rep independent case samples of n_cases by each of n_panels reader panels.

    A panel is n_readers readers with offsets b ~ N(0, sigma_rb_mm^2), fixed across the panel's
    case samples; every reader reads every case with rule p['reference_estimator'] (each read
    has its own caliper errors and review decisions). Returns reads (n_panels, n_rep, n_cases,
    n_readers) of the study axis and b (n_panels, n_readers).

    Draw order: cases, then offsets, then the readers in turn. With n_rep = 1 and n_readers = 2
    this is the double-read part of experiments.run_E6 (first chunk) and the arrays are
    bit-identical for the same generator state.
    """
    ax = AXES.index(p["study_axis"])
    n = n_panels * n_rep * n_cases
    lat = draw_latent(p, n, rng)
    b = p["sigma_rb_mm"] * rng.standard_normal((n_panels, n_readers))
    drift = p["drift_mm_per_100"] * np.tile(np.arange(n_cases), n_panels * n_rep) / 100.0
    panel = np.repeat(np.arange(n_panels), n_rep * n_cases)
    reads = np.empty((n_panels, n_rep, n_cases, n_readers))
    for j in range(n_readers):
        reads[..., j] = protocol_read(lat, p, rng, b[panel, j] + drift)[:, ax].reshape(n_panels, n_rep, n_cases)
    return reads, b


def crossed_allocation(n_readers: int, total_reads: int) -> np.ndarray:
    """Fully crossed design: every reader reads every case; n_cases = total_reads // n_readers.
    Returns a boolean mask (n_cases, n_readers)."""
    if n_readers < 2:
        raise ValueError("at least two readers")
    n_cases = total_reads // n_readers
    return np.ones((n_cases, n_readers), dtype=bool)


def rotating_allocation(n_readers: int, total_reads: int) -> np.ndarray:
    """Rotating double-read design: n_cases = total_reads // 2 and each case is read by two readers.

    The n_readers (n_readers - 1) / 2 reader pairs are listed in lexicographic order ((0, 1),
    (0, 2), ..., (R-2, R-1)) and case i (0-based, acquisition order) is read by pair number
    i mod n_pairs. Returns a boolean mask (n_cases, n_readers). With two readers this equals the
    crossed design.
    """
    if n_readers < 2:
        raise ValueError("at least two readers")
    pairs = list(itertools.combinations(range(n_readers), 2))
    n_cases = total_reads // 2
    mask = np.zeros((n_cases, n_readers), dtype=bool)
    for i in range(n_cases):
        j, k = pairs[i % len(pairs)]
        mask[i, j] = mask[i, k] = True
    return mask


def between_reader_mom(reads: np.ndarray, mask: np.ndarray) -> dict:
    """Method-of-moments estimate of between-reader and within-pair variation from paired differences.

    reads: (..., n_cases, R); mask (n_cases, R): which reader read which case. For every reader
    pair (j, k), j < k, with n_jk >= 2 jointly read cases: mean difference m_jk and variance
    s2_jk (ddof 1) of reads_j - reads_k. Under reads = case effect + reader offset b + noise,
    E[m_jk^2] = 2 sigma_b^2 + sigma_w^2 / n_jk and E[s2_jk] = sigma_w^2 (sigma_w^2: variance of
    the difference between two reads of the same case by a fixed pair). Estimators:
      tau2    = mean over pairs of (m_jk^2 - s2_jk / n_jk)        estimates 2 sigma_b^2 (may be < 0)
      sigw2   = sum (n_jk - 1) s2_jk / sum (n_jk - 1)
      sd_b    = sqrt(max(tau2, 0) / 2)                            between-reader SD
      loa_marginal = 1.96 sqrt(max(tau2, 0) + sigw2)              upper 95% limit of agreement for a
                                                                  randomly chosen reader pair (mean
                                                                  difference 0 by exchangeability)
    Returns arrays over the leading axes, plus n_pairs.
    """
    mask = np.asarray(mask, dtype=bool)
    R = mask.shape[1]
    t_sum = 0.0
    num = 0.0
    den = 0
    used = 0
    for j, k in itertools.combinations(range(R), 2):
        both = mask[:, j] & mask[:, k]
        n = int(both.sum())
        if n < 2:
            continue
        dd = reads[..., both, j] - reads[..., both, k]
        m = dd.mean(-1)
        s2 = dd.var(-1, ddof=1)
        t_sum = t_sum + (m * m - s2 / n)
        num = num + (n - 1) * s2
        den += n - 1
        used += 1
    if used == 0:
        raise ValueError("no reader pair shares two or more cases")
    tau2 = t_sum / used
    sigw2 = num / den
    return {"tau2": tau2, "sigw2": sigw2, "sd_b": np.sqrt(np.maximum(tau2, 0.0) / 2.0),
            "loa_marginal": Z975 * np.sqrt(np.maximum(tau2, 0.0) + sigw2), "n_pairs": used}


def oneway_components(x: np.ndarray) -> dict:
    """One-way random-effects decomposition of x (n_groups, n_rep), balanced.

    within = mean within-group variance (ddof 1); between = (MSB - within) / n_rep with
    MSB = n_rep x variance of group means (ddof 1); total = between + within. The between
    component is not truncated at zero.
    """
    x = np.asarray(x, dtype=float)
    G, M = x.shape
    if M < 2 or G < 2:
        raise ValueError("need at least two groups and two replicates")
    within = float(x.var(axis=1, ddof=1).mean())
    msb = float(M * x.mean(axis=1).var(ddof=1))
    between = (msb - within) / M
    return {"within_var": within, "between_var": between, "total_var": between + within, "msb": msb,
            "grand_mean": float(x.mean())}
