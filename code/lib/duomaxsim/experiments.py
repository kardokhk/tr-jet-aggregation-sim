"""Experiments E1 to E6 (protocol section 5). Each run_* returns a list of row dicts
(long format: one row per estimator/variant/axis/estimand/metric) for one grid cell."""
from __future__ import annotations

import itertools
import time
from collections import defaultdict

import numpy as np
from scipy import stats

from . import config as C
from .estimators import standard_estimators
from .metrics import ErrAcc, MeanAcc, PropAcc, bland_altman, brown_forsythe, icc_a1, mean_mcse
from .model import AXES, Latent, draw_latent, measure
from .rules import a4_final, beat_rule, cross_view_flags


# ------------------------------------------------------------------ helpers

def _gen(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


def _chunks(children, n_total: int, chunk: int):
    for i, ch in enumerate(children):
        yield min(chunk, n_total - i * chunk), _gen(ch)


def _n_chunks(n_total, chunk):
    return -(-n_total // chunk)


def read_once(lat: Latent, p: dict, rng, reader_bias):
    meas = measure(lat, p, rng, reader_bias)
    br = beat_rule(meas, lat.avail[:, None, :], p["N_beats"], p["window"], p["limited_value"])
    return meas, br


def _final_value(vm, lat: Latent, p: dict, U, est: str):
    if est == "A1":
        return vm[..., 0]
    if est == "A4":
        return a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])[0]
    raise ValueError(est)


def protocol_read(lat: Latent, p: dict, rng, reader_bias, estimator: str | None = None,
                  return_u: bool = False):
    """One reader's final value per axis, shape (n, 2).

    With return_u=True also returns the A4 decision uniforms U (n, 2, K-1) of
    this read. Random-number consumption is identical either way.
    """
    est = estimator or p["reference_estimator"]
    _, br = read_once(lat, p, rng, reader_bias)
    vm = br.value
    K = vm.shape[-1]
    U = rng.random(vm.shape[:-1] + (K - 1,))
    val = _final_value(vm, lat, p, U, est)
    return (val, U) if return_u else val


def image_composite(lat: Latent, p: dict, U, estimator: str | None = None):
    """R_img (amendment 2, E5b): the protocol composite computed from the images
    alone, shape (n, 2).

    Same beats, view errors, instrument factor g and available beats as the
    reads, with caliper error and reader bias set to 0 (the optional
    measure_floor_mm is still applied, as in `measure`). The beat rule is
    applied as for a read; A4 decisions reuse the decision uniforms `U` of a
    given read (in E5: reader 1), so R_img consumes no random numbers.
    """
    est = estimator or p["reference_estimator"]
    x = lat.spans
    floor = p.get("measure_floor_mm")
    if floor is not None:
        x = np.maximum(x, floor)
    br = beat_rule(x, lat.avail[:, None, :], p["N_beats"], p["window"], p["limited_value"])
    return _final_value(br.value, lat, p, U, est)


def ai_shared(T1, R_img, lam: float, sigma: float, z):
    """Amendment 2 AI: T1 + lambda * (R_img - T1) + sigma * z. lambda = 0 gives
    exactly the independent AI T1 + sigma * z (same z)."""
    return T1 + lam * (R_img - T1) + sigma * z


def _row(**kw):
    return kw


def _err_rows(acc: ErrAcc, **keys):
    out = []
    if acc.n < 2:
        return out
    for metric, (v, se) in acc.summary().items():
        out.append(_row(**keys, metric=metric, value=float(v), mcse=float(se), n=acc.n))
    return out


def _prop_row(acc: PropAcc, metric, **keys):
    v, se, n = acc.summary()
    return _row(**keys, metric=metric, value=float(v), mcse=float(se), n=int(n))


def _mean_row(acc: MeanAcc, metric, **keys):
    v, se, n = acc.summary()
    return _row(**keys, metric=metric, value=float(v), mcse=float(se), n=int(n))


# ------------------------------------------------------------------ E1

def run_E1(p, grids, ss, n_total, chunk):
    K = int(p["K"])
    variants = list(itertools.product(grids["s_det"], grids["f_rej"]))
    accs = defaultdict(ErrAcc)
    lim_anchor = [PropAcc(), PropAcc()]
    sel_infl = bool(p.get("report_selection_inflation", False))   # amendment 2 (E1b)
    infl = [MeanAcc(), MeanAcc()]
    kids = ss.spawn(_n_chunks(n_total, chunk))
    for n, rng in _chunks(kids, n_total, chunk):
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas, br = read_once(lat, p, rng, rb)
        vm = br.value
        ests = standard_estimators(vm, lat, meas, br.used)
        U = rng.random((n, 2, K - 1))
        for s, f in variants:
            ests[("A4", s, f)] = a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], s, f, p["a4_scrutiny"])[0]
        truths = {"T1": lat.S, "T2": lat.mu[..., 0]}
        for key, est in ests.items():
            for d in range(2):
                for tn, tv in truths.items():
                    accs[(key, d, tn)].add(est[:, d], tv[:, d])
        for d in range(2):
            lim_anchor[d].add(br.limited[:, d, 0])
            if sel_infl:
                infl[d].add(ests["A3"][:, d] - ests["A1"][:, d])
    rows = []
    for (key, d, tn), acc in accs.items():
        if isinstance(key, tuple):
            ek = dict(estimator="A4", s_det=key[1], f_rej=key[2], t_warn=p["t_warn"], t_adj=p["t_adj"])
        else:
            ek = dict(estimator=key)
        rows += _err_rows(acc, **ek, axis=AXES[d], estimand=tn)
    for d in range(2):
        rows.append(_prop_row(lim_anchor[d], "p_limited_anchor", estimator="rule", axis=AXES[d]))
    if sel_infl:
        for d in range(2):
            rows.append(_mean_row(infl[d], "selection_inflation_mm", estimator="A3-A1", axis=AXES[d]))
    return rows


# ------------------------------------------------------------------ E2

def run_E2(p, grids, ss, n_total, chunk):
    accs = defaultdict(ErrAcc)
    props = defaultdict(PropAcc)
    means = defaultdict(MeanAcc)
    kids = ss.spawn(_n_chunks(n_total, chunk))
    for n, rng in _chunks(kids, n_total, chunk):
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas, br = read_once(lat, p, rng, rb)
        ests = standard_estimators(br.value, lat, meas, br.used)
        truths = {"T1": lat.S, "T2": lat.mu[..., 0], "Tv": lat.g[:, None] * lat.mu[..., 0]}
        for d in range(2):
            acc_d = br.accepted[:, d, 0]
            subsets = {"all": np.ones(n, bool), "accepted": acc_d, "limited": ~acc_d}
            for en in ("A1", "A6"):
                for sn, sm in subsets.items():
                    if sm.any():
                        for tn, tv in truths.items():
                            accs[(en, d, sn, tn)].add(ests[en][sm, d], tv[sm, d])
            props[("p_window_met_first_N", d)].add(br.met_first[:, d, 0])
            props[("p_limited", d)].add(br.limited[:, d, 0])
            means[("beats_acquired_mean", d, "all")].add(br.used[:, d, 0])
            means[("beats_acquired_mean", d, "accepted")].add(br.used[:, d, 0], where=acc_d)
    rows = []
    for (en, d, sn, tn), acc in accs.items():
        rows += _err_rows(acc, estimator=en, subset=sn, axis=AXES[d], estimand=tn)
    for (m, d), acc in props.items():
        rows.append(_prop_row(acc, m, estimator="rule", axis=AXES[d]))
    for (m, d, sn), acc in means.items():
        rows.append(_mean_row(acc, m, estimator="rule", subset=sn, axis=AXES[d]))
    return rows


# ------------------------------------------------------------------ E3

def run_E3(p, grids, ss, n_total, chunk):
    K = int(p["K"])
    cut = [float(c) for c in p["cutoffs_mm"]]
    names = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
    cnt = defaultdict(int)
    kids = ss.spawn(_n_chunks(n_total, chunk))
    for n, rng in _chunks(kids, n_total, chunk):
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas, br = read_once(lat, p, rng, rb)
        vm = br.value
        ests = standard_estimators(vm, lat, meas, br.used)
        U = rng.random((n, 2, K - 1))
        ests["A4"] = a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])[0]
        for d in range(2):
            for c in cut:
                ev = lat.S[:, d] >= c
                ref = ests["A1"][:, d] >= c
                for en in names:
                    pos = ests[en][:, d] >= c
                    k = (en, d, c)
                    cnt[k + ("tp",)] += int((pos & ev).sum())
                    cnt[k + ("fn",)] += int((~pos & ev).sum())
                    cnt[k + ("tn",)] += int((~pos & ~ev).sum())
                    cnt[k + ("fp",)] += int((pos & ~ev).sum())
                    cnt[k + ("ev_up",)] += int((pos & ~ref & ev).sum())
                    cnt[k + ("ev_dn",)] += int((~pos & ref & ev).sum())
                    cnt[k + ("ne_up",)] += int((pos & ~ref & ~ev).sum())
                    cnt[k + ("ne_dn",)] += int((~pos & ref & ~ev).sum())
    rows = []

    def prop(k_, n_):
        pr = k_ / n_ if n_ else np.nan
        return pr, (np.sqrt(pr * (1 - pr) / n_) if n_ else np.nan)

    for d in range(2):
        for c in cut:
            for en in names:
                g = lambda s: cnt[(en, d, c, s)]
                ne, nn = g("tp") + g("fn"), g("tn") + g("fp")
                keys = dict(estimator=en, axis=AXES[d], estimand="T1", cutoff_mm=c)
                se, se_se = prop(g("tp"), ne)
                sp, sp_se = prop(g("tn"), nn)
                ppv, ppv_se = prop(g("tp"), g("tp") + g("fp"))
                prev, prev_se = prop(ne, ne + nn)
                pu, pd_ = g("ev_up") / ne, g("ev_dn") / ne
                qu, qd = g("ne_up") / nn, g("ne_dn") / nn
                nri_e = pu - pd_
                nri_n = qd - qu
                se_e = np.sqrt((pu + pd_ - nri_e ** 2) / ne)
                se_n = np.sqrt((qu + qd - nri_n ** 2) / nn)
                for m, v, s_, nn_ in (("sensitivity", se, se_se, ne), ("specificity", sp, sp_se, nn),
                                      ("ppv", ppv, ppv_se, g("tp") + g("fp")), ("prevalence", prev, prev_se, ne + nn),
                                      ("nri_events_vs_A1", nri_e, se_e, ne), ("nri_nonevents_vs_A1", nri_n, se_n, nn),
                                      ("nri_vs_A1", nri_e + nri_n, np.sqrt(se_e ** 2 + se_n ** 2), ne + nn)):
                    rows.append(_row(**keys, metric=m, value=float(v), mcse=float(s_), n=int(nn_)))
    return rows


# ------------------------------------------------------------------ E4

def run_E4(p, grids, ss, n_total, chunk):
    pairs = [(tw, ta) for tw in grids["t_warn"] for ta in grids["t_adj"] if tw < ta]
    thr = float(p["error_threshold_mm"])
    props = defaultdict(PropAcc)
    kids = ss.spawn(_n_chunks(n_total, chunk))
    for n, rng in _chunks(kids, n_total, chunk):
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        _, br = read_once(lat, p, rng, rb)
        vm = br.value                                     # (n, 2, K)
        gS = lat.g[:, None] * lat.S                       # (n, 2)
        event = (gS[..., None] * lat.o[..., 1:] > thr) | (gS * lat.u[..., 0] > thr)[..., None]
        axdiff = np.abs(vm[:, 0, 0] - vm[:, 1, 0])
        truediff = np.abs(lat.S[:, 0] - lat.S[:, 1])
        for tw, ta in pairs:
            _, warn, adj = cross_view_flags(vm, tw, ta)
            trig = warn | adj
            for d in range(2):
                k = lambda m: (m, tw, ta, AXES[d])
                props[k("p_any_trigger")].add(trig[:, d].any(-1))
                props[k("p_any_warn")].add(warn[:, d].any(-1))
                props[k("p_any_adj")].add(adj[:, d].any(-1))
                props[k("p_error_event_pair")].add(event[:, d])
                props[k("false_alarm_trigger_pair")].add(trig[:, d], where=~event[:, d])
                props[k("false_alarm_adj_pair")].add(adj[:, d], where=~event[:, d])
                props[k("miss_trigger_pair")].add(~trig[:, d], where=event[:, d])
                props[k("miss_adj_pair")].add(~adj[:, d], where=event[:, d])
            props[("p_anchor_axis_diff_ge_twarn", tw, ta, "AP-SL")].add(axdiff >= tw)
            props[("p_anchor_axis_diff_ge_tadj", tw, ta, "AP-SL")].add(axdiff >= ta)
            props[("p_true_axis_diff_ge_twarn", tw, ta, "AP-SL")].add(truediff >= tw)
    rows = []
    for (m, tw, ta, ax), acc in props.items():
        if acc.n == 0:
            continue
        rows.append(_prop_row(acc, m, estimator="trigger", t_warn=tw, t_adj=ta, axis=ax))
    return rows


# ------------------------------------------------------------------ E5

def _agreement(ai, ref):
    """Per-study agreement of `ai` against `ref`; arrays (S, n)."""
    d = ai - ref
    bias, sd, lo, hi, _ = bland_altman(d)
    icc = icc_a1(np.stack([ai, ref], axis=-1), ci=False)
    return {"mae": np.abs(d).mean(-1), "ba_bias": bias, "loa_lo": lo, "loa_hi": hi, "icc": icc}


def run_E5(p, grids, ss, n_studies, chunk):
    ax = AXES.index(p["study_axis"])
    nv = int(p["n_val"])
    R = int(p["n_readers"])
    spc = max(1, chunk // nv)
    lam = p.get("ai_shared_lambda")
    lam = None if lam is None else float(lam)
    kids = ss.spawn(1 + _n_chunks(n_studies, spc))
    # calibration of the inherited-bias AI: regression of a single protocol read on T1
    rng = _gen(kids[0])
    lat = draw_latent(p, int(p["calib_n"]), rng)
    r = protocol_read(lat, p, rng, p["sigma_rb_mm"] * rng.standard_normal(lat.n))[:, ax]
    b_slope, a_int = np.polyfit(lat.S[:, ax], r, 1)
    store = defaultdict(list)
    for s_c, rng in _chunks(kids[1:], n_studies, spc):
        n = s_c * nv
        lat = draw_latent(p, n, rng)
        pool = p["sigma_rb_mm"] * rng.standard_normal((s_c, R))
        perm = np.argsort(rng.random((n, R)), axis=1)[:, :3]
        study = np.repeat(np.arange(s_c), nv)
        drift = p["drift_mm_per_100"] * np.tile(np.arange(nv), s_c) / 100.0
        reads = []
        for j in range(3):
            v, U = protocol_read(lat, p, rng, pool[study, perm[:, j]] + drift, return_u=True)
            reads.append(v[:, ax].reshape(s_c, nv))
            if j == 0:
                U_r1 = U
        T1 = lat.S[:, ax].reshape(s_c, nv)
        z = rng.standard_normal((s_c, nv))
        if lam is not None:   # amendment 2 (E5b); no random numbers consumed
            R_img = image_composite(lat, p, U_r1)[:, ax].reshape(s_c, nv)
            q = _agreement(R_img, T1)
            for m in ("mae", "ba_bias"):
                store[("ref", "R_img", f"ref_{m}_vs_T1")].append(q[m])
        r1, r2, r3 = reads
        refs = {"single": r1, "mean2": 0.5 * (r1 + r2)}
        dis = np.abs(r1 - r2)
        for tol in grids["adj_tol_mm"]:
            if p["adjudication_value"] == "adjudicator":
                adjv = r3
            else:
                cand = np.stack([0.5 * (r1 + r2), 0.5 * (r1 + r3), 0.5 * (r2 + r3)], -1)
                gaps = np.stack([np.abs(r1 - r2), np.abs(r1 - r3), np.abs(r2 - r3)], -1)
                adjv = np.take_along_axis(cand, np.argmin(gaps, -1)[..., None], -1)[..., 0]
            refs[f"adj_tol{tol:g}"] = np.where(dis > tol, adjv, 0.5 * (r1 + r2))
            store[("ref", f"adj_tol{tol:g}", "p_adjudicated")].append((dis > tol).mean(-1))
        for rn, ref in refs.items():
            q = _agreement(ref, T1)
            for m in ("mae", "ba_bias"):
                store[("ref", rn, f"ref_{m}_vs_T1")].append(q[m])
        for sig in grids["sigma_ai_mm"]:
            ais = {"independent": T1 + sig * z, "inherited": a_int + b_slope * T1 + sig * z}
            if lam is not None:
                ais["shared"] = ai_shared(T1, R_img, lam, sig, z)
            per = {}
            for an, ai in ais.items():
                tru = _agreement(ai, T1)
                per[(an, "T1")] = tru["mae"]
                for rn, ref in refs.items():
                    app = _agreement(ai, ref)
                    per[(an, rn)] = app["mae"]
                    for m in app:
                        store[(an, sig, rn, "apparent_" + m)].append(app[m])
                        store[(an, sig, rn, "true_" + m)].append(tru[m])
                        store[(an, sig, rn, "diff_" + m)].append(app[m] - tru[m])
            if lam is not None:
                # paired per-study comparison with the unbiased (independent) AI
                for an in ("shared", "inherited"):
                    for rn in list(refs) + ["T1"]:
                        dm = per[(an, rn)] - per[("independent", rn)]
                        store[("paired", an, sig, rn, "mae_minus_independent")].append(dm)
    rows = []
    for key, lst in store.items():
        x = np.concatenate(lst)
        if key[0] == "paired":
            _, an, sig, rn, _ = key
            k = dict(estimator=p["reference_estimator"], ai=an, sigma_ai_mm=float(sig), ref=rn,
                     axis=p["study_axis"], comparator="independent")
            m, se = mean_mcse(x)
            sd = float(np.std(x, ddof=1))
            pl = float(np.mean(x < 0))
            rows.append(_row(**k, metric="p_lower_mae_than_independent", value=pl,
                             mcse=float(np.sqrt(pl * (1 - pl) / x.size)), n=x.size))
            rows.append(_row(**k, metric="paired_mae_diff_mean", value=float(m), mcse=float(se), n=x.size))
            rows.append(_row(**k, metric="paired_mae_diff_sd", value=sd,
                             mcse=float(sd / np.sqrt(2 * (x.size - 1))), n=x.size))
            continue
        m, se = mean_mcse(x)
        if key[0] == "ref":
            rows.append(_row(estimator=p["reference_estimator"], ref=key[1], axis=p["study_axis"],
                             metric=key[2], value=float(m), mcse=float(se), n=x.size))
        else:
            an, sig, rn, met = key
            rows.append(_row(estimator=p["reference_estimator"], ai=an, sigma_ai_mm=float(sig), ref=rn,
                             axis=p["study_axis"], metric=met, value=float(m), mcse=float(se), n=x.size))
    rows.append(_row(estimator="calibration", axis=p["study_axis"], metric="inherited_intercept_mm",
                     value=float(a_int), mcse=np.nan, n=int(p["calib_n"])))
    rows.append(_row(estimator="calibration", axis=p["study_axis"], metric="inherited_slope",
                     value=float(b_slope), mcse=np.nan, n=int(p["calib_n"])))
    if lam is not None:
        rows.append(_row(estimator="shared_ai", axis=p["study_axis"], metric="ai_shared_lambda",
                         value=lam, mcse=np.nan, n=n_studies))
    return rows


# ------------------------------------------------------------------ E6

def run_E6(p, grids, ss, n_studies, chunk):
    ax = AXES.index(p["study_axis"])
    gold = [int(x) for x in p["gold_n"]]
    prodn = int(p["production_n"])
    ovl = [float(f) for f in p["overlap_frac"]]
    sizes = sorted(set(gold) | {int(round(prodn * f)) for f in ovl})
    ng = max(sizes)
    per_study = ng + prodn
    spc = max(1, chunk // per_study)
    lvl = float(p["test_level"])
    bf = bool(p.get("e6_brown_forsythe", False))   # amendment 2 (E6b)
    store = defaultdict(list)
    kids = ss.spawn(_n_chunks(n_studies, spc))
    for s_c, rng in _chunks(kids, n_studies, spc):
        # gold / overlap double reading: two readers per study
        lat = draw_latent(p, s_c * ng, rng)
        b = p["sigma_rb_mm"] * rng.standard_normal((s_c, 2))
        drift = p["drift_mm_per_100"] * np.tile(np.arange(ng), s_c) / 100.0
        study = np.repeat(np.arange(s_c), ng)
        r1 = protocol_read(lat, p, rng, b[study, 0] + drift)[:, ax].reshape(s_c, ng)
        r2 = protocol_read(lat, p, rng, b[study, 1] + drift)[:, ax].reshape(s_c, ng)
        for nn in sizes:
            Y = np.stack([r1[:, :nn], r2[:, :nn]], -1)
            icc, (lo, hi) = icc_a1(Y)
            bias, sd, llo, lhi, hw = bland_altman(r1[:, :nn] - r2[:, :nn])
            for m, v in (("icc", icc), ("icc_ci_width", hi - lo), ("ba_bias", bias), ("loa_lo", llo),
                         ("loa_hi", lhi), ("loa_ci_width_each", 2 * hw), ("loa_width", lhi - llo)):
                store[("double", nn, m)].append(v)
        # sentinel: one production reader, manual reads, AI draft
        lat = draw_latent(p, s_c * prodn, rng)
        b = p["sigma_rb_mm"] * rng.standard_normal(s_c)
        study = np.repeat(np.arange(s_c), prodn)
        drift = p["drift_mm_per_100"] * np.tile(np.arange(prodn), s_c) / 100.0
        man = protocol_read(lat, p, rng, b[study] + drift)[:, ax].reshape(s_c, prodn)
        T1 = lat.S[:, ax].reshape(s_c, prodn)
        ai = T1 + p["ai_draft_bias_mm"] + p["ai_draft_sigma_mm"] * rng.standard_normal((s_c, prodn))
        dman = man - ai
        for ns in p["sentinel_n"]:
            ns = int(ns)
            ds = dman[:, :ns]
            for al in p["anchor_alpha"]:
                da = (1.0 - al) * dman[:, ns:]
                m1, m2 = ds.mean(-1), da.mean(-1)
                v1, v2 = ds.var(-1, ddof=1), da.var(-1, ddof=1)
                n1, n2 = ds.shape[-1], da.shape[-1]
                se2 = v1 / n1 + v2 / n2
                t = (m1 - m2) / np.sqrt(se2)
                df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
                p_t = 2 * stats.t.sf(np.abs(t), df)
                p_f = stats.f.sf(v1 / v2, n1 - 1, n2 - 1)
                store[("sentinel", ns, al, "reject_welch_mean")].append(p_t < lvl)
                store[("sentinel", ns, al, "reject_f_variance")].append(p_f < lvl)
                store[("sentinel", ns, al, "reject_either_unadjusted")].append((p_t < lvl) | (p_f < lvl))
                if bf:
                    p_bf2, p_bf1 = brown_forsythe(ds, da)
                    store[("sentinel", ns, al, "reject_bf")].append(p_bf2 < lvl)
                    store[("sentinel", ns, al, "reject_bf_onesided")].append(p_bf1 < lvl)
    rows = []
    for key, lst in store.items():
        x = np.concatenate(lst).astype(float)
        if key[0] == "double":
            _, nn, m = key
            designs = [("gold", None)] if nn in gold else []
            designs += [("overlap", f) for f in ovl if int(round(prodn * f)) == nn]
            mm, se = mean_mcse(x)
            sd = np.nanstd(x, ddof=1)
            nfin = int(np.isfinite(x).sum())
            for dz, f in designs:
                base = dict(estimator=p["reference_estimator"], design=dz, n_double=nn,
                            overlap_frac=(np.nan if f is None else f), axis=p["study_axis"])
                rows.append(_row(**base, metric=m + "_mean", value=float(mm), mcse=float(se), n=nfin))
                if m in ("icc", "loa_lo", "loa_hi", "ba_bias"):
                    rows.append(_row(**base, metric=m + "_empirical_sd", value=float(sd),
                                     mcse=float(sd / np.sqrt(2 * (nfin - 1))), n=nfin))
        else:
            _, ns, al, m = key
            pw = x.mean()
            rows.append(_row(estimator=p["reference_estimator"], design="sentinel", sentinel_n=ns,
                             anchor_alpha=float(al), axis=p["study_axis"], metric="power_" + m,
                             value=float(pw), mcse=float(np.sqrt(pw * (1 - pw) / x.size)), n=x.size))
    return rows


RUNNERS = {"E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4, "E5": run_E5, "E6": run_E6}


def run_cells(cfg: dict, exp: str, cells, n_rep: int | None = None, overrides: dict | None = None):
    """Run grid cells of one experiment; returns a pandas DataFrame (long format)."""
    import pandas as pd

    ex = cfg["experiments"][exp]
    grids = {k: v.get("grid", [v["value"]]) for k, v in cfg["parameters"].items()}
    all_cells = C.experiment_cells(cfg, exp)
    rep_key = "n_patients" if exp in ("E1", "E2", "E3", "E4") else "n_studies"
    n_rep = int(n_rep or ex[rep_key])
    chunk = int(cfg["meta"]["chunk_size"])
    out = []
    for ci in cells:
        p = C.cell_params(cfg, exp, ci, overrides)
        ss = C.cell_seed(cfg, exp, ci)
        t0 = time.process_time()
        w0 = time.perf_counter()
        rows = RUNNERS[exp](p, grids, ss, n_rep, chunk)
        cpu, wall = time.process_time() - t0, time.perf_counter() - w0
        meta = {"experiment": exp, "cell": ci, rep_key: n_rep, "chunk_size": chunk,
                "seed_entropy": str(cfg["meta"]["master_seed"]),
                "seed_spawn_key": ",".join(map(str, ss.spawn_key)),
                "cpu_s": cpu, "wall_s": wall}
        for k, v in all_cells[ci].items():
            meta["p_" + k] = v
        for k, v in (overrides or {}).items():
            meta["p_" + k] = v
        for r in rows:
            out.append({**meta, **r})
    return pd.DataFrame(out)
