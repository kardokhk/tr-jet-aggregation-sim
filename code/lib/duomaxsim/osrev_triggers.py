"""Protocol amendment 3, package A3-4 (triggers and review): post hoc sensitivity extensions.

Added 2026-10-05 in response to the senior author's review of manuscript v05
(comments C49, C186, C188 and the AP/SL part of C324). Nothing in the
pre-specified library is changed. This module adds

  cohort_chunks            one read of a cohort in exactly the random-number order of
                           experiments.run_E1 / run_E3 / run_E4, returning the latent
                           state and the view means instead of summary rows;
  view_error_mm            signed systematic error of every view in mm;
  pair_event_definitions   the published pair-level error definition (D0) and the
                           alternative definitions of amendment A3-4 (b);
  pair_partition           a mutually exclusive classification of anchor-other pairs;
  pattern_hist / rates_from_patterns / bootstrap_from_patterns
                           the pair-level 2 by 2 table kept as a histogram of
                           per-patient (TP, FP, FN) counts, so that every rate, its
                           patient-clustered standard error and an exact cluster
                           bootstrap follow from one table of unrounded counts;
  two_by_two / rates_2x2   plain 2 by 2 counts and rates (unit of analysis = row);
  reviewed_rule_grid       the reviewed rule (rules.a4_final) over a grid of reviewer
                           probabilities on common decision uniforms;
  error_metrics / paired_contrast
                           bias, RMSE, MAE, sensitivity and specificity with Monte
                           Carlo standard errors, and paired (common random number)
                           contrasts between two rules read on the same patients;
  ratio_map / rescale_sl / axis_prob_ge
                           alternative joint AP/SL distributions obtained by a
                           quantile transformation of the coded SL/AP ratio, the
                           exact propagation of a new ratio through the latent
                           state, and a quadrature check of P(|AP - SL| >= tau).

Every function reduces to the library when its new options are off:
cohort_chunks reproduces the library runners bit for bit; reviewed_rule_grid
calls rules.a4_final unchanged, and with probabilities (0, 0) returns the
largest view mean exactly; ratio_map and rescale_sl return their input
objects untouched when the ratio model is the coded one.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats
from scipy.special import ndtri

from .experiments import read_once
from .model import Latent, draw_latent
from .rules import a4_final, cross_view_flags

Z95 = 1.959964

# ------------------------------------------------------------------ cohort


def gen(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


def n_chunks(n_total: int, chunk: int) -> int:
    return -(-int(n_total) // int(chunk))


def read_chunk(p: dict, n: int, rng: np.random.Generator) -> dict:
    """One chunk of a cohort: latent state, one read, A4 decision uniforms.

    Random-number order is that of experiments.run_E1 and run_E3 (draw_latent,
    reader bias, measure, decision uniforms). run_E4 stops before the decision
    uniforms, so everything it uses is reproduced as well.
    """
    K = int(p["K"])
    lat = draw_latent(p, n, rng)
    rb = p["sigma_rb_mm"] * rng.standard_normal(n)
    meas, br = read_once(lat, p, rng, rb)
    U = rng.random((n, 2, K - 1))
    return dict(lat=lat, rb=rb, meas=meas, br=br, vm=br.value, U=U)


def cohort_chunks(p: dict, ss: np.random.SeedSequence, n_total: int, chunk: int, only: int | None = None):
    """Yield (chunk index, read_chunk dict) for a cohort seeded like a library cell.

    `ss.spawn(n_chunks)` gives one child per chunk, as in the library runners;
    `only` restricts the generator to one chunk (for chunk-level parallel work)
    without changing any chunk's random numbers.
    """
    kids = ss.spawn(n_chunks(n_total, chunk))
    for i, ch in enumerate(kids):
        if only is not None and i != only:
            continue
        yield i, read_chunk(p, min(chunk, n_total - i * chunk), gen(ch))


# ------------------------------------------------------------------ error definitions


def view_error_mm(lat: Latent) -> dict:
    """Systematic errors of each view in mm at the study's instrument settings, shape (n, 2, K).

    over   g * S * o    gross overestimation (the quantity in the published definition)
    under  g * S * u    gross underestimation
    net    g * (mu - S) = over - under: signed plane-related error of the view
    total  g * mu - S   = net + (g - 1) * S: signed error against the true maximal span
                          T1, including the instrument factor g shared by all views
    All are errors of the view's median beat span; measurement noise is excluded.
    """
    gS = (lat.g[:, None] * lat.S)[..., None]
    over = gS * lat.o
    under = gS * lat.u
    net = over - under
    total = lat.g[:, None, None] * lat.mu - lat.S[..., None]
    return dict(over=over, under=under, net=net, total=total)


def pair_event_definitions(lat: Latent, thr: float, tau: float | None = None) -> dict:
    """Boolean error events per (anchor, other view) pair, shape (n, 2, K-1).

    D0_published           g*S*o_other > thr or g*S*u_anchor > thr (experiments.run_E4).
    D0a_other_over         g*S*o_other > thr (component of D0)
    D0b_anchor_under       g*S*u_anchor > thr (component of D0)
    D1_plus_other_under    D0 or g*S*u_other > thr
    D2_net_either_view     |net| > thr in the anchor or in the other view
    D3_total_either_view   |total| > thr in the anchor or in the other view
    D4_expected_excess     g*(mu_other - mu_anchor) > thr: the other view truly reads
                           higher than the anchor, the only error a one-sided
                           between-view trigger responds to in expectation
    D5_shared_total        both views wrong by more than thr in the same direction (total)
    D5n_shared_net         as D5 with the net (plane-related) error
    With `tau` (the trigger threshold):
    X_expected_excess_ge_tau   g*(mu_other - mu_anchor) >= tau
    H_shared_total_agree       D5 and |g*(mu_other - mu_anchor)| < tau: both views wrong
                               while their expected values agree within the trigger
    Hn_shared_net_agree        the same with the net error
    """
    e = view_error_mm(lat)
    o_v, u_v = e["over"][..., 1:], e["under"][..., 1:]
    u_a = e["under"][..., :1]
    shape = o_v.shape
    net_a, net_v = e["net"][..., :1], e["net"][..., 1:]
    tot_a, tot_v = e["total"][..., :1], e["total"][..., 1:]
    dexp = net_v - net_a                      # = g * (mu_other - mu_anchor)
    d0a = o_v > thr
    d0b = np.broadcast_to(u_a > thr, shape)
    d0 = d0a | d0b
    same_t = (np.abs(tot_a) > thr) & (np.abs(tot_v) > thr) & (np.sign(tot_a) == np.sign(tot_v))
    same_n = (np.abs(net_a) > thr) & (np.abs(net_v) > thr) & (np.sign(net_a) == np.sign(net_v))
    out = {
        "D0_published": d0,
        "D0a_other_over": d0a,
        "D0b_anchor_under": d0b.copy(),
        "D1_plus_other_under": d0 | (u_v > thr),
        "D2_net_either_view": (np.abs(net_a) > thr) | (np.abs(net_v) > thr),
        "D3_total_either_view": (np.abs(tot_a) > thr) | (np.abs(tot_v) > thr),
        "D4_expected_excess": dexp > thr,
        "D5_shared_total": same_t,
        "D5n_shared_net": same_n,
    }
    if tau is not None:
        out["X_expected_excess_ge_tau"] = dexp >= tau
        out["H_shared_total_agree"] = same_t & (np.abs(dexp) < tau)
        out["Hn_shared_net_agree"] = same_n & (np.abs(dexp) < tau)
    return out


PARTITION_LABELS = (
    "both views within threshold",
    "anchor only, too low",
    "anchor only, too high",
    "other view only, too low",
    "other view only, too high",
    "both too low",
    "both too high",
    "anchor too low, other too high",
    "anchor too high, other too low",
)


def pair_partition(lat: Latent, thr: float, kind: str = "total") -> np.ndarray:
    """Mutually exclusive class 0..8 (PARTITION_LABELS) of each pair, shape (n, 2, K-1).

    A view is 'too low' when its signed error (`kind`: 'total' or 'net', see
    view_error_mm) is below -thr and 'too high' when it is above +thr.
    """
    e = view_error_mm(lat)[kind]
    a = np.broadcast_to(e[..., :1], e[..., 1:].shape)
    v = e[..., 1:]
    sa = np.where(a < -thr, -1, np.where(a > thr, 1, 0))
    sv = np.where(v < -thr, -1, np.where(v > thr, 1, 0))
    table = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4,
             (-1, -1): 5, (1, 1): 6, (-1, 1): 7, (1, -1): 8}
    code = np.empty(v.shape, dtype=np.int8)
    for (x, y), c in table.items():
        code[(sa == x) & (sv == y)] = c
    return code


def trigger_fired(vm: np.ndarray, tau: float) -> np.ndarray:
    """One-sided between-view trigger: other view mean minus anchor mean >= tau, (..., K-1).

    Uses rules.cross_view_flags; with t_warn = t_adj = tau the adjudication flag
    is exactly diff >= tau.
    """
    return cross_view_flags(vm, tau, tau)[2]


# ------------------------------------------------------------------ 2 by 2 tables


def two_by_two(fired, event) -> dict:
    """Unrounded counts of a 2 by 2 table; the unit of analysis is the array element."""
    f = np.asarray(fired, dtype=bool).ravel()
    e = np.asarray(event, dtype=bool).ravel()
    if f.shape != e.shape:
        raise ValueError("shape mismatch")
    return dict(tp=int((f & e).sum()), fp=int((f & ~e).sum()),
                fn=int((~f & e).sum()), tn=int((~f & ~e).sum()))


def _prop(k: float, n: float):
    if n <= 0:
        return np.nan, np.nan
    v = k / n
    return v, float(np.sqrt(v * (1.0 - v) / n))


def rates_2x2(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Rates of one 2 by 2 table with binomial (independent-unit) standard errors.

    Returns measure -> (value, se, numerator, denominator). An empty
    denominator gives NaN.
    """
    n = tp + fp + fn + tn
    spec = {
        "prevalence": (tp + fn, n),
        "fired": (tp + fp, n),
        "sensitivity": (tp, tp + fn),
        "false_positive_rate": (fp, fp + tn),
        "specificity": (tn, fp + tn),
        "ppv": (tp, tp + fp),
        "npv": (tn, tn + fn),
    }
    out = {}
    for m, (k, d) in spec.items():
        v, se = _prop(k, d)
        out[m] = (v, se, int(k), int(d))
    return out


def pattern_hist(fired: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Histogram of per-patient pair counts.

    fired, event: (n, m) booleans for the m anchor-other pairs of each patient.
    Returns H with shape (m+1, m+1, m+1); H[a, b, c] is the number of patients
    with a true-positive, b false-positive and c false-negative pairs (and
    m - a - b - c true-negative pairs). H is additive over chunks and cohorts.
    """
    fired = np.asarray(fired, dtype=bool)
    event = np.asarray(event, dtype=bool)
    if fired.shape != event.shape or fired.ndim != 2:
        raise ValueError("fired and event must both have shape (n, m)")
    m = fired.shape[1]
    tp = (fired & event).sum(1)
    fp = (fired & ~event).sum(1)
    fn = (~fired & event).sum(1)
    H = np.zeros((m + 1,) * 3, dtype=np.int64)
    np.add.at(H, (tp, fp, fn), 1)
    return H


def _pattern_axes(H: np.ndarray):
    m = H.shape[0] - 1
    tp, fp, fn = np.meshgrid(np.arange(m + 1), np.arange(m + 1), np.arange(m + 1), indexing="ij")
    tn = m - tp - fp - fn
    return m, tp, fp, fn, tn


_RATIOS = {
    "prevalence": ("ev", "all"),
    "fired": ("fi", "all"),
    "sensitivity": ("tp", "ev"),
    "false_positive_rate": ("fp", "ne"),
    "specificity": ("tn", "ne"),
    "ppv": ("tp", "fi"),
    "npv": ("tn", "nf"),
}


def _pattern_terms(H):
    m, tp, fp, fn, tn = _pattern_axes(H)
    if (H[tn < 0] != 0).any():
        raise ValueError("impossible pattern (more than m pairs)")
    t = dict(tp=tp, fp=fp, fn=fn, tn=tn, ev=tp + fn, ne=fp + tn, fi=tp + fp, nf=fn + tn,
             all=np.full(tp.shape, m))
    return m, t


def rates_from_patterns(H: np.ndarray) -> dict:
    """Pair-level rates, with three standard errors, from a pattern histogram.

    Each rate is a ratio R = sum_i num_i / sum_i den_i over patients i.
      se_iid       binomial SE treating the pairs as independent: sqrt(R(1-R)/sum den)
      se_sqrt_m    se_iid * sqrt(m): the manuscript's 'conservative' SE, the bound
                   reached if the m pairs of a patient were perfectly dependent and
                   every patient contributed m pairs to the denominator
      se_cluster   patient-level linearization (delta method) for a ratio of totals:
                   sqrt(n/(n-1) * sum_i (num_i - R den_i)^2) / sum_i den_i,
                   which respects the clustering of pairs within a patient
    Returns measure -> dict(value, num, den, se_iid, se_sqrt_m, se_cluster, deff)
    and the key 'counts' -> dict(tp, fp, fn, tn, n_patients, pairs_per_patient).
    """
    H = np.asarray(H)
    m, t = _pattern_terms(H)
    n = int(H.sum())
    out = {"counts": dict(tp=int((H * t["tp"]).sum()), fp=int((H * t["fp"]).sum()),
                          fn=int((H * t["fn"]).sum()), tn=int((H * t["tn"]).sum()),
                          n_patients=n, pairs_per_patient=m)}
    for name, (a, b) in _RATIOS.items():
        num = float((H * t[a]).sum())
        den = float((H * t[b]).sum())
        if den <= 0:
            out[name] = dict(value=np.nan, num=int(num), den=int(den), se_iid=np.nan,
                             se_sqrt_m=np.nan, se_cluster=np.nan, deff=np.nan)
            continue
        R = num / den
        se_iid = float(np.sqrt(R * (1 - R) / den))
        resid2 = float((H * (t[a] - R * t[b]) ** 2).sum())
        se_cl = float(np.sqrt(n / (n - 1) * resid2) / den) if n > 1 else np.nan
        out[name] = dict(value=R, num=int(num), den=int(den), se_iid=se_iid,
                         se_sqrt_m=se_iid * float(np.sqrt(m)), se_cluster=se_cl,
                         deff=(se_cl / se_iid) ** 2 if se_iid > 0 else np.nan)
    return out


def bootstrap_from_patterns(H: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Cluster (patient) bootstrap of the pair-level rates.

    Resampling n patients with replacement is, for statistics that depend on a
    patient only through (TP, FP, FN), exactly a multinomial draw of the
    pattern counts. Returns measure -> (se_boot, lo2.5, hi97.5).
    """
    H = np.asarray(H)
    m, t = _pattern_terms(H)
    n = int(H.sum())
    draws = rng.multinomial(n, H.ravel() / n, size=int(n_boot)).astype(float)
    out = {}
    for name, (a, b) in _RATIOS.items():
        num = draws @ t[a].ravel()
        den = draws @ t[b].ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / den
        r = r[np.isfinite(r)]
        if r.size < 2:
            out[name] = (np.nan, np.nan, np.nan)
        else:
            lo, hi = np.quantile(r, [0.025, 0.975])
            out[name] = (float(r.std(ddof=1)), float(lo), float(hi))
    return out


# ------------------------------------------------------------------ reviewed rule


def reviewed_rule_grid(vm, over, U, t_warn: float, t_adj: float, grid, scrutiny: str = "triggered") -> dict:
    """Largest view mean after review for each (s_det, f_rej) in `grid`.

    s_det: probability that a reviewed view with true overestimation is
    excluded; f_rej: probability that a reviewed view without overestimation
    is excluded. All variants share the decision uniforms U (common random
    numbers), and each is rules.a4_final unchanged. Returns
    (s, f) -> dict(value, reviewed, reject, reject_over, reject_valid); the last
    four are (..., K-1) booleans per other view.
    """
    over = np.asarray(over, dtype=bool)
    out = {}
    for s, f in grid:
        val, info = a4_final(vm, over, U, t_warn, t_adj, float(s), float(f), scrutiny)
        rej = info["reject"]
        rev = info["warn"] | info["adj"]
        if scrutiny == "adj_only":
            rev = info["adj"]
        elif scrutiny == "all_higher":
            rev = (vm[..., 1:] - vm[..., :1]) > 0
        out[(float(s), float(f))] = dict(value=val, reviewed=rev, reject=rej,
                                         reject_over=rej & over[..., 1:], reject_valid=rej & ~over[..., 1:])
    return out


def error_metrics(est: np.ndarray, truth: np.ndarray, cutoffs=()) -> dict:
    """Bias, RMSE, MAE (mm) and, per cut-off c, sensitivity and specificity for truth >= c.

    Returns metric -> (value, mcse, n, k) where k is the event count behind a
    proportion (NaN otherwise). MCSE: bias and MAE, SD/sqrt(n); RMSE, delta
    method SD(e^2)/(2 RMSE sqrt(n)); proportions, binomial.
    """
    est = np.asarray(est, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    e = est - truth
    n = e.size
    mse = float(np.mean(e * e))
    rmse = float(np.sqrt(mse))
    ae = np.abs(e)
    out = {
        "bias_mm": (float(e.mean()), float(e.std(ddof=1) / np.sqrt(n)), n, np.nan),
        "rmse_mm": (rmse, float((e * e).std(ddof=1) / np.sqrt(n) / (2 * rmse)) if rmse > 0 else 0.0, n, np.nan),
        "mae_mm": (float(ae.mean()), float(ae.std(ddof=1) / np.sqrt(n)), n, np.nan),
    }
    for c in cutoffs:
        ev = truth >= c
        pos = est >= c
        ne, nn = int(ev.sum()), int((~ev).sum())
        tp, tn = int((pos & ev).sum()), int((~pos & ~ev).sum())
        v, se = _prop(tp, ne)
        out[f"sensitivity_{c:g}mm"] = (v, se, ne, tp)
        v, se = _prop(tn, nn)
        out[f"specificity_{c:g}mm"] = (v, se, nn, tn)
    return out


def paired_contrast(est_a: np.ndarray, est_b: np.ndarray, truth: np.ndarray, cutoffs=()) -> dict:
    """Rule a minus rule b on the same patients (common random numbers).

    Returns metric -> (difference, mcse, n). The MCSE is the SD of the
    per-patient influence values of the difference divided by sqrt(n):
    bias, e_a - e_b; MAE, |e_a| - |e_b|; RMSE, e_a^2/(2 RMSE_a) - e_b^2/(2 RMSE_b);
    sensitivity (specificity), the difference of the classification indicators
    among patients with (without) the event.
    """
    truth = np.asarray(truth, dtype=float).ravel()
    ea = np.asarray(est_a, dtype=float).ravel() - truth
    eb = np.asarray(est_b, dtype=float).ravel() - truth
    n = ea.size

    def mean_se(x):
        return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan)

    ra, rb_ = np.sqrt(np.mean(ea * ea)), np.sqrt(np.mean(eb * eb))
    out = {}
    d, se = mean_se(ea - eb)
    out["bias_mm"] = (d, se, n)
    d, se = mean_se(np.abs(ea) - np.abs(eb))
    out["mae_mm"] = (d, se, n)
    infl = ea * ea / (2 * ra) - eb * eb / (2 * rb_)
    out["rmse_mm"] = (float(ra - rb_), float(infl.std(ddof=1) / np.sqrt(n)), n)
    for c in cutoffs:
        ev = truth >= c
        pa = (ea + truth) >= c
        pb = (eb + truth) >= c
        x = pa[ev].astype(float) - pb[ev].astype(float)
        d, se = mean_se(x) if x.size else (np.nan, np.nan)
        out[f"sensitivity_{c:g}mm"] = (d, se, int(ev.sum()))
        x = (~pa[~ev]).astype(float) - (~pb[~ev]).astype(float)
        d, se = mean_se(x) if x.size else (np.nan, np.nan)
        out[f"specificity_{c:g}mm"] = (d, se, int((~ev).sum()))
    return out


# ------------------------------------------------------------------ joint AP/SL distribution

S_AP_MEAN_FACTOR = "exp(S_log_sd^2 / 2)"


def mean_ap_mm(p: dict) -> float:
    """E[S_AP] of the coded lognormal case mix."""
    return float(p["S_median_mm"] * np.exp(p["S_log_sd"] ** 2 / 2.0))


def beta_shapes(r_mean: float, lo: float, hi: float, conc: float):
    m = (r_mean - lo) / (hi - lo)
    if not 0.0 < m < 1.0:
        raise ValueError("r_mean must lie strictly inside (r_lower, r_upper)")
    return m * conc, (1.0 - m) * conc


def ratio_spec_from_params(p: dict) -> dict:
    """The coded ratio model (model.draw_latent, section 3.1) as a ratio spec."""
    return dict(family="beta", r_mean=float(p["r_mean"]), r_lower=float(p["r_lower"]),
                r_upper=float(p["r_upper"]), r_concentration=float(p["r_concentration"]))


def _same_spec(a: dict, b: dict) -> bool:
    return a.get("family", "beta") == b.get("family", "beta") == "beta" and all(
        float(a[k]) == float(b[k]) for k in ("r_mean", "r_lower", "r_upper", "r_concentration"))


def ratio_map(r: np.ndarray, p: dict, spec: dict | None) -> np.ndarray:
    """Transform coded SL/AP ratios to another ratio distribution by quantile mapping.

    r was drawn by model.draw_latent as r_lower + (r_upper - r_lower) * Beta(a, b).
    Its probability integral transform v = F_coded(r) is uniform, so
    r_new = F_new^{-1}(v) is an exact draw from the new distribution that is a
    monotone function of the old one (common random numbers across ratio
    models). `spec`:
      None, or family 'beta' with the coded parameters: r is returned unchanged
            (the same object);
      family 'beta':      r_lower + (r_upper - r_lower) * Beta(m c, (1 - m) c);
      family 'lognormal': r = exp(mu_log + sd_log * z), unbounded above, so that
            (ln S_AP, ln S_SL) is bivariate normal with correlation
            S_log_sd / sqrt(S_log_sd^2 + sd_log^2);
      family 'constant':  r = value for every patient.
    """
    base = ratio_spec_from_params(p)
    if spec is None or _same_spec(spec, base):
        return r
    fam = spec["family"]
    if fam == "constant":
        return np.full(r.shape, float(spec["value"]))
    a0, b0 = beta_shapes(base["r_mean"], base["r_lower"], base["r_upper"], base["r_concentration"])
    x = (np.asarray(r, dtype=float) - base["r_lower"]) / (base["r_upper"] - base["r_lower"])
    v = np.clip(stats.beta.cdf(x, a0, b0), 1e-15, 1.0 - 1e-15)
    if fam == "beta":
        a1, b1 = beta_shapes(float(spec["r_mean"]), float(spec["r_lower"]), float(spec["r_upper"]),
                             float(spec["r_concentration"]))
        return float(spec["r_lower"]) + (float(spec["r_upper"]) - float(spec["r_lower"])) * stats.beta.ppf(v, a1, b1)
    if fam == "lognormal":
        return np.exp(float(spec["mu_log"]) + float(spec["sd_log"]) * ndtri(v))
    raise ValueError(fam)


def rescale_sl(lat: Latent, r_new: np.ndarray) -> Latent:
    """Latent state with the SL/AP ratio replaced by r_new, everything else unchanged.

    In model.draw_latent the SL axis is S_SL = r * S_AP, mu_SL = S_SL * (1 - u + o)
    and spans_SL = g * mu_SL * exp(beat terms): every SL quantity is proportional
    to r and no other draw depends on r. Multiplying S, mu and spans of the SL
    axis by r_new / r is therefore the state that draw_latent would have
    produced with r_new, on the same random numbers. The AP axis is untouched.
    If r_new is lat.r (the same object) lat itself is returned.
    """
    if r_new is lat.r:
        return lat
    r_new = np.asarray(r_new, dtype=float)
    if r_new.shape != lat.r.shape:
        raise ValueError("r_new must have shape (n,)")
    if (r_new <= 0).any():
        raise ValueError("ratios must be positive")
    f = r_new / lat.r
    S = lat.S.copy()
    S[:, 1] = lat.S[:, 0] * r_new
    mu = lat.mu.copy()
    mu[:, 1] *= f[:, None]
    spans = lat.spans.copy()
    spans[:, 1] *= f[:, None, None]
    return Latent(S=S, r=r_new, u=lat.u, o=lat.o, over=lat.over, mu=mu, g=lat.g,
                  logrr=lat.logrr, spans=spans, avail=lat.avail)


def spec_for_mean_difference(p: dict, family: str, mean_diff_mm: float, *, r_concentration: float | None = None,
                             log_corr: float | None = None) -> dict:
    """Ratio spec with E[S_AP - S_SL] = mean_diff_mm (r independent of S_AP, so E[r] = 1 - diff / E[S_AP]).

    'beta': coded bounds, concentration `r_concentration`. 'lognormal':
    sd_log chosen so that corr(ln S_AP, ln S_SL) = log_corr.
    """
    r_mean = 1.0 - float(mean_diff_mm) / mean_ap_mm(p)
    if family == "beta":
        return dict(family="beta", r_mean=r_mean, r_lower=float(p["r_lower"]), r_upper=float(p["r_upper"]),
                    r_concentration=float(p["r_concentration"] if r_concentration is None else r_concentration))
    if family == "lognormal":
        s = float(p["S_log_sd"])
        sd_log = s * float(np.sqrt(1.0 / float(log_corr) ** 2 - 1.0))
        return dict(family="lognormal", mu_log=float(np.log(r_mean) - sd_log ** 2 / 2.0), sd_log=sd_log)
    raise ValueError(family)


def axis_prob_ge(p: dict, spec: dict | None, tau: float) -> float:
    """P(|S_AP - S_SL| >= tau) by quadrature over the ratio distribution (no simulation).

    S_AP ~ lognormal(ln S_median, S_log_sd), r independent of S_AP;
    |S_AP - S_SL| = S_AP * |1 - r|.
    """
    spec = spec or ratio_spec_from_params(p)
    sap = stats.lognorm(s=float(p["S_log_sd"]), scale=float(p["S_median_mm"]))
    tau = float(tau)
    fam = spec.get("family", "beta")
    if fam == "constant":
        d = abs(1.0 - float(spec["value"]))
        return float(sap.sf(tau / d)) if d > 0 else (1.0 if tau <= 0 else 0.0)

    def tail(r):
        d = abs(1.0 - r)
        return sap.sf(tau / d) if d > 0 else 0.0

    if fam == "beta":
        lo, hi = float(spec["r_lower"]), float(spec["r_upper"])
        a, b = beta_shapes(float(spec["r_mean"]), lo, hi, float(spec["r_concentration"]))
        # substitute z = F(x) so that integrable end-point singularities of the Beta density vanish
        f = lambda v: tail(lo + (hi - lo) * stats.beta.ppf(v, a, b))
        return float(integrate.quad(f, 0.0, 1.0, limit=400)[0])
    if fam == "lognormal":
        mu, sd = float(spec["mu_log"]), float(spec["sd_log"])
        f = lambda z: stats.norm.pdf(z) * tail(float(np.exp(mu + sd * z)))
        zc = -mu / sd                                   # r = 1, where the tail has a kink
        pts = [zc] if -10 < zc < 10 else None
        return float(integrate.quad(f, -10, 10, limit=400, points=pts)[0])
    raise ValueError(fam)
