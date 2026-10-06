"""Protocol amendment 3, package A3-3 (thresholds): post hoc reporting extensions.

Added 2026-10-05 in response to the senior author's review of manuscript v05
(comment C201). Nothing in the pre-specified library is changed. This module

  1. regenerates the patients of experiment E3 (`experiments.run_E3`) and keeps the
     per-patient true spans and estimates instead of only the summary rows
     (`simulate_e3_patients`). Random-number consumption is identical to `run_E3`,
     so the same SeedSequence gives the same patients, and `e3_rows` rebuilds the
     rows of `run_E3` bit for bit from them;
  2. tabulates full 2 by 2 tables, predictive values, reclassification against a
     comparator rule, near-threshold bands and misclassification by distance of the
     true span from the cut-off;
  3. gives the true-span distribution implied by the configuration in closed form
     (AP axis, lognormal) or by one-dimensional quadrature (SL axis, lognormal
     times a scaled Beta ratio).

The cut-offs are illustrative values on the simulated span scale. They are not
clinical eligibility criteria and nothing here should be labelled as such.

Conventions: an "event" is a true maximal span T1 >= c on the analysed axis; a
"positive" is an estimate >= c (same inequality as `run_E3`). The signed distance
is d = T1 - c, so d >= 0 for events.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

from .estimators import standard_estimators
from .experiments import _chunks, _n_chunks, read_once
from .model import AXES, draw_latent
from .rules import a4_final

E3_NAMES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
VIEW_RULES = ("A1", "A2", "A3", "A4")
Z95 = 1.959964


# ------------------------------------------------------------------ patients

def simulate_e3_patients(p: dict, ss: np.random.SeedSequence, n_total: int, chunk: int) -> dict:
    """Per-patient output of experiment E3.

    Same calls in the same order as `experiments.run_E3` (draw_latent, reader bias,
    one read, A4 decision uniforms), chunk by chunk, so the generated patients are
    those of `run_E3` for the same `ss`, `n_total` and `chunk`.

    Returns {"S": (n, 2) true maximal spans T1, "est": {name: (n, 2)}} with the
    seven E3 estimators, patients in generation order.
    """
    K = int(p["K"])
    S_parts = []
    est_parts = {en: [] for en in E3_NAMES}
    kids = ss.spawn(_n_chunks(n_total, chunk))
    for n, rng in _chunks(kids, n_total, chunk):
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas, br = read_once(lat, p, rng, rb)
        vm = br.value
        ests = standard_estimators(vm, lat, meas, br.used)
        U = rng.random((n, 2, K - 1))
        ests["A4"] = a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"],
                              p["a4_scrutiny"])[0]
        S_parts.append(lat.S)
        for en in E3_NAMES:
            est_parts[en].append(ests[en])
    return {"S": np.concatenate(S_parts, axis=0),
            "est": {en: np.concatenate(v, axis=0) for en, v in est_parts.items()}}


# ------------------------------------------------------------------ counts

def confusion_counts(truth: np.ndarray, est: np.ndarray, cut: float, ref: np.ndarray | None = None,
                     mask: np.ndarray | None = None) -> dict:
    """2 by 2 counts for "span >= cut", optionally with reclassification against `ref`.

    truth, est, ref: (n,) arrays; mask: optional (n,) bool selecting patients.
    Keys: n, tp, fp, fn, tn and, when `ref` is given, ev_up, ev_dn, ne_up, ne_dn
    (events or non-events that `est` classifies positive while `ref` classifies
    negative: up; the reverse: dn). All integers.
    """
    truth = np.asarray(truth, dtype=float)
    est = np.asarray(est, dtype=float)
    if mask is None:
        mask = np.ones(truth.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    ev = (truth >= cut) & mask
    ne = (truth < cut) & mask
    pos = est >= cut
    out = {"n": int(mask.sum()),
           "tp": int((pos & ev).sum()), "fp": int((pos & ne).sum()),
           "fn": int((~pos & ev).sum()), "tn": int((~pos & ne).sum())}
    if ref is not None:
        rpos = np.asarray(ref, dtype=float) >= cut
        out.update(ev_up=int((pos & ~rpos & ev).sum()), ev_dn=int((~pos & rpos & ev).sum()),
                   ne_up=int((pos & ~rpos & ne).sum()), ne_dn=int((~pos & rpos & ne).sum()))
    return out


def _prop(k, n):
    """Proportion k/n with binomial Monte Carlo SE; NaN when n = 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(n > 0, k / n, np.nan)
        se = np.where(n > 0, np.sqrt(pr * (1 - pr) / n), np.nan)
    return pr, se


def classification_metrics(tp, fp, fn, tn) -> dict:
    """Sensitivity, specificity, PPV, NPV, prevalence and accuracy from counts.

    Each metric m is returned as m (proportion), m_mcse (binomial Monte Carlo SE,
    sqrt(p(1-p)/n)), m_lo95 and m_hi95 (m -+ 1.959964 MCSE, clipped to [0, 1]) and
    m_n (denominator). Vectorised; a zero denominator gives NaN.
    """
    tp, fp, fn, tn = (np.asarray(a, dtype=float) for a in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    spec = {"sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp), "ppv": (tp, tp + fp),
            "npv": (tn, tn + fn), "prevalence": (tp + fn, n), "accuracy": (tp + tn, n)}
    out = {}
    for m, (k, d) in spec.items():
        pr, se = _prop(k, d)
        out[m] = pr
        out[m + "_mcse"] = se
        out[m + "_lo95"] = np.clip(pr - Z95 * se, 0.0, 1.0)
        out[m + "_hi95"] = np.clip(pr + Z95 * se, 0.0, 1.0)
        out[m + "_n"] = d
    return out


def reclassification_metrics(ev_up, ev_dn, ne_up, ne_dn, n_ev, n_ne) -> dict:
    """Net reclassification of a rule against a comparator rule on the same patients.

    nri_events    = (ev_up - ev_dn) / n_ev   = sensitivity(rule) - sensitivity(comparator)
    nri_nonevents = (ne_dn - ne_up) / n_ne   = specificity(rule) - specificity(comparator)
    nri           = nri_events + nri_nonevents  (two-category NRI = difference in Youden index)
    net_correct   = ((ev_up - ev_dn) + (ne_dn - ne_up)) / (n_ev + n_ne)
                  = accuracy(rule) - accuracy(comparator), the prevalence-weighted net
                    proportion of patients moved to the correct side of the cut-off.
    Monte Carlo SEs are paired (multinomial on moved-up, moved-down, unchanged), the
    same formulae as `experiments.run_E3` for the first three.
    """
    ev_up, ev_dn, ne_up, ne_dn, n_ev, n_ne = (np.asarray(a, dtype=float)
                                              for a in (ev_up, ev_dn, ne_up, ne_dn, n_ev, n_ne))
    with np.errstate(divide="ignore", invalid="ignore"):
        pu, pd_ = ev_up / n_ev, ev_dn / n_ev
        qu, qd = ne_up / n_ne, ne_dn / n_ne
        nri_e = pu - pd_
        nri_n = qd - qu
        se_e = np.sqrt((pu + pd_ - nri_e ** 2) / n_ev)
        se_n = np.sqrt((qu + qd - nri_n ** 2) / n_ne)
        n = n_ev + n_ne
        good = (ev_up + ne_dn) / n
        bad = (ev_dn + ne_up) / n
        net = good - bad
        se_net = np.sqrt((good + bad - net ** 2) / n)
    return {"nri_events": nri_e, "nri_events_mcse": se_e,
            "nri_nonevents": nri_n, "nri_nonevents_mcse": se_n,
            "nri": nri_e + nri_n, "nri_mcse": np.sqrt(se_e ** 2 + se_n ** 2),
            "net_correct": net, "net_correct_mcse": se_net}


def near_band_counts(truth: np.ndarray, est: np.ndarray, cut: float, half_width: float) -> dict:
    """2 by 2 counts among patients with |truth - cut| <= half_width (closed band)."""
    truth = np.asarray(truth, dtype=float)
    return confusion_counts(truth, est, cut, mask=np.abs(truth - cut) <= half_width)


def distance_bin_edges(bin_width: float, max_dist: float) -> np.ndarray:
    nb = int(round(max_dist / bin_width))
    return bin_width * np.arange(-nb, nb + 1)


def distance_bin_counts(truth: np.ndarray, est: np.ndarray, cut: float, bin_width: float = 0.5,
                        max_dist: float = 5.0) -> dict:
    """Misclassification by signed distance d = truth - cut.

    Bins are [lo, lo + bin_width) for lo = -max_dist, ..., max_dist - bin_width, so a
    true span exactly at the cut-off (an event) falls in the first non-negative bin
    and every bin lies wholly on one side of the cut-off. Patients with d outside
    [-max_dist, max_dist) are not counted. Returns arrays over bins: lo, hi, n,
    n_pos (estimate >= cut) and n_mis (estimate on the wrong side of the cut-off:
    false positives where lo < 0, false negatives where lo >= 0).
    """
    truth = np.asarray(truth, dtype=float)
    est = np.asarray(est, dtype=float)
    nb = int(round(max_dist / bin_width))
    k = np.floor((truth - cut) / bin_width).astype(np.int64) + nb
    ok = (k >= 0) & (k < 2 * nb)
    pos = est >= cut
    n = np.bincount(k[ok], minlength=2 * nb)
    n_pos = np.bincount(k[ok & pos], minlength=2 * nb)
    edges = distance_bin_edges(bin_width, max_dist)
    lo, hi = edges[:-1], edges[1:]
    n_mis = np.where(lo < 0, n_pos, n - n_pos)
    return {"lo": lo, "hi": hi, "n": n, "n_pos": n_pos, "n_mis": n_mis}


def e3_rows(S: np.ndarray, est: dict, cutoffs) -> list[dict]:
    """Rows of `experiments.run_E3` rebuilt from per-patient arrays (same order,
    same arithmetic), used to assert exact agreement with the library."""
    rows = []

    def prop(k_, n_):
        pr = k_ / n_ if n_ else np.nan
        return pr, (np.sqrt(pr * (1 - pr) / n_) if n_ else np.nan)

    for d in range(2):
        for c in [float(c) for c in cutoffs]:
            for en in E3_NAMES:
                g = confusion_counts(S[:, d], est[en][:, d], c, ref=est["A1"][:, d])
                ne, nn = g["tp"] + g["fn"], g["tn"] + g["fp"]
                keys = dict(estimator=en, axis=AXES[d], estimand="T1", cutoff_mm=c)
                se, se_se = prop(g["tp"], ne)
                sp, sp_se = prop(g["tn"], nn)
                ppv, ppv_se = prop(g["tp"], g["tp"] + g["fp"])
                prev, prev_se = prop(ne, ne + nn)
                pu, pd_ = g["ev_up"] / ne, g["ev_dn"] / ne
                qu, qd = g["ne_up"] / nn, g["ne_dn"] / nn
                nri_e = pu - pd_
                nri_n = qd - qu
                se_e = np.sqrt((pu + pd_ - nri_e ** 2) / ne)
                se_n = np.sqrt((qu + qd - nri_n ** 2) / nn)
                for m, v, s_, nn_ in (("sensitivity", se, se_se, ne), ("specificity", sp, sp_se, nn),
                                      ("ppv", ppv, ppv_se, g["tp"] + g["fp"]),
                                      ("prevalence", prev, prev_se, ne + nn),
                                      ("nri_events_vs_A1", nri_e, se_e, ne),
                                      ("nri_nonevents_vs_A1", nri_n, se_n, nn),
                                      ("nri_vs_A1", nri_e + nri_n, np.sqrt(se_e ** 2 + se_n ** 2), ne + nn)):
                    rows.append(dict(**keys, metric=m, value=float(v), mcse=float(s_), n=int(nn_)))
    return rows


def count_tables(S: np.ndarray, est: dict, cutoffs, rules=VIEW_RULES, ref: str = "A1",
                 bands=(1.0, 2.0), bin_width: float = 0.5, max_dist: float = 5.0,
                 hist_width: float = 0.25, hist_max: float = 60.0) -> dict:
    """All count tables of the package for one set of patients, as lists of row dicts.

    "overall": axis, cutoff_mm, rule, n, tp, fp, fn, tn, ev_up, ev_dn, ne_up, ne_dn
    "band":    axis, cutoff_mm, half_width_mm, rule, n, tp, fp, fn, tn
    "bins":    axis, cutoff_mm, rule, d_lo_mm, d_hi_mm, n, n_pos, n_mis
    "hist":    axis, lo_mm, hi_mm, n  (true spans; the last bin is open-ended)
    """
    out = {"overall": [], "band": [], "bins": [], "hist": []}
    cutoffs = [float(c) for c in cutoffs]
    nh = int(round(hist_max / hist_width))
    for d in range(2):
        T = S[:, d]
        k = np.minimum(np.floor(T / hist_width).astype(np.int64), nh)
        h = np.bincount(k, minlength=nh + 1)
        for i in range(nh + 1):
            out["hist"].append(dict(axis=AXES[d], lo_mm=i * hist_width,
                                    hi_mm=(i + 1) * hist_width if i < nh else np.inf, n=int(h[i])))
        for c in cutoffs:
            for r in rules:
                e = est[r][:, d]
                key = dict(axis=AXES[d], cutoff_mm=c, rule=r)
                out["overall"].append({**key, **confusion_counts(T, e, c, ref=est[ref][:, d])})
                for hw in bands:
                    out["band"].append({**key, "half_width_mm": float(hw), **near_band_counts(T, e, c, hw)})
                b = distance_bin_counts(T, e, c, bin_width, max_dist)
                for i in range(len(b["n"])):
                    out["bins"].append({**key, "d_lo_mm": float(b["lo"][i]), "d_hi_mm": float(b["hi"][i]),
                                        "n": int(b["n"][i]), "n_pos": int(b["n_pos"][i]),
                                        "n_mis": int(b["n_mis"][i])})
    return out


# ------------------------------------------------------------------ true-span distribution

def _ratio_spec(p: dict):
    """(kind, lo, hi, a, b) of the SL/AP ratio r as drawn in `model.draw_latent`."""
    lo, hi = float(p["r_lower"]), float(p["r_upper"])
    m = (p["r_mean"] - lo) / (hi - lo) if hi > lo else 1.0
    if m >= 1.0:
        return "point", hi, hi, np.nan, np.nan
    if m <= 0.0:
        return "point", lo, lo, np.nan, np.nan
    c = float(p["r_concentration"])
    return "beta", lo, hi, m * c, (1 - m) * c


_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)
_GL_T = 0.5 * (_GL_X + 1.0)          # nodes on (0, 1)
_GL_W = 0.5 * _GL_W


def _ratio_nodes(p: dict):
    """Quadrature nodes r_j and weights w_j (sum 1) for expectations over r."""
    kind, lo, hi, a, b = _ratio_spec(p)
    if kind == "point":
        return np.array([lo]), np.array([1.0])
    w = _GL_W * stats.beta.pdf(_GL_T, a, b)
    return lo + (hi - lo) * _GL_T, w / w.sum()


def span_cdf(p: dict, axis: str, x) -> np.ndarray:
    """P(T1 <= x) on the given axis for the case mix of `model.draw_latent`.

    AP: T1 ~ lognormal(ln S_median_mm, S_log_sd). SL: T1 = AP * r, r independent of
    AP, so P(SL <= x) = E_r[Phi((ln(x / r) - ln S_median_mm) / S_log_sd)].
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    mu, s = np.log(float(p["S_median_mm"])), float(p["S_log_sd"])
    with np.errstate(divide="ignore"):
        lx = np.log(np.maximum(x, 0.0))
    if axis == "AP":
        return special.ndtr((lx - mu) / s)
    if axis != "SL":
        raise ValueError(axis)
    r, w = _ratio_nodes(p)
    return (special.ndtr((lx[:, None] - np.log(r)[None, :] - mu) / s) * w[None, :]).sum(axis=1)


def span_pdf(p: dict, axis: str, x) -> np.ndarray:
    """Density of T1 (per mm) on the given axis; 0 for x <= 0."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    mu, s = np.log(float(p["S_median_mm"])), float(p["S_log_sd"])
    out = np.zeros_like(x)
    ok = x > 0
    lx = np.log(x[ok])
    if axis == "AP":
        z = (lx - mu) / s
        out[ok] = np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * s * x[ok])
        return out
    if axis != "SL":
        raise ValueError(axis)
    r, w = _ratio_nodes(p)
    z = (lx[:, None] - np.log(r)[None, :] - mu) / s
    out[ok] = (np.exp(-0.5 * z * z) * w[None, :]).sum(axis=1) / (np.sqrt(2 * np.pi) * s * x[ok])
    return out


def span_quantile(p: dict, axis: str, q) -> np.ndarray:
    """Quantiles of T1 on the given axis (closed form for AP, root finding for SL)."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    mu, s = np.log(float(p["S_median_mm"])), float(p["S_log_sd"])
    if axis == "AP":
        return np.exp(mu + s * special.ndtri(q))
    hi = float(np.exp(mu + 12 * s))
    return np.array([optimize.brentq(lambda x: span_cdf(p, axis, x)[0] - qi, 1e-9, hi, xtol=1e-12)
                     for qi in q])


def span_mean_sd(p: dict, axis: str) -> tuple[float, float]:
    """Mean and SD of T1 on the given axis (exact moments)."""
    mu, s = np.log(float(p["S_median_mm"])), float(p["S_log_sd"])
    m1 = np.exp(mu + s * s / 2)
    m2 = np.exp(2 * mu + 2 * s * s)
    if axis == "SL":
        r, w = _ratio_nodes(p)
        m1 = m1 * float((r * w).sum())
        m2 = m2 * float((r * r * w).sum())
    elif axis != "AP":
        raise ValueError(axis)
    return float(m1), float(np.sqrt(m2 - m1 * m1))


def span_distribution(p: dict, axis: str, cutoffs=(7.0, 10.0, 13.0)) -> dict:
    """Summary of the true-span distribution as implemented, for one axis."""
    qs = span_quantile(p, axis, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    mean, sd = span_mean_sd(p, axis)
    out = {"axis": axis, "S_median_mm": float(p["S_median_mm"]), "S_log_sd": float(p["S_log_sd"]),
           "mean_mm": mean, "sd_mm": sd}
    for name, v in zip(("p2_5", "p5", "p25", "median", "p75", "p95", "p97_5"), qs):
        out[name + "_mm"] = float(v)
    for c in cutoffs:
        out[f"prop_ge_{c:g}mm"] = float(1.0 - span_cdf(p, axis, c)[0])
    return out
