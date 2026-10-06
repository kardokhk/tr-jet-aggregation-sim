"""Protocol amendment 3, package A3-2 (view rules): per-patient errors of the four view rules,
error quantiles with Monte Carlo standard errors, and the counterfactual decomposition of the
excess of the largest view mean over the anchor-view mean.

Nothing in the existing library is modified. `simulate_view_rules` consumes random numbers in
exactly the order of `experiments.run_E1` (draw_latent, reader bias, caliper error, A4 decision
uniforms), so that with the counterfactual options off (arm "full") its per-patient values
reproduce the stored E1 and E1b cells bit for bit when given the same SeedSequence and chunk
size (tests/test_osrev_views.py).

Rules (manuscript names): A1 anchor-view mean, A2 mean across views, A3 largest view mean,
A4 largest view mean after review (s_det, f_rej, t_warn, t_adj, a4_scrutiny from `p`).

Counterfactual arms (all computed from the same latent draws, caliper normals z, reader bias
and decision uniforms U, i.e. on common random numbers):
  full                     the model as published
  zero_noise               caliper error 0, reader bias 0, and every beat of a view equal to its
                           expected span g * mu_v (no residual beat variation, no RR-related
                           variation); only the between-view offsets (1 - u_v + o_v) differ
  equal_views              every view takes the anchor's systematic factor (u_v = u_0,
                           o_v = o_0, over_v = over_0), so all views have the same expected span
                           g * mu_0; beat, RR, caliper and reader noise as published
  zero_noise_equal_views   both; every view mean equals g * mu_0 exactly
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .estimators import a1_anchor, a2_mean, a3_max
from .metrics import ErrAcc, MeanAcc, PropAcc
from .model import AXES, Latent, draw_latent
from .rules import a4_final, beat_rule

RULES = ("A1", "A2", "A3", "A4")
ARMS = {  # name -> (zero_noise, equal_views)
    "full": (False, False),
    "zero_noise": (True, False),
    "equal_views": (False, True),
    "zero_noise_equal_views": (True, True),
}
SIGNED_Q = (2.5, 5.0, 25.0, 50.0, 75.0, 95.0, 97.5)
ABS_Q = (50.0, 90.0, 95.0)
ABS_THR = (1.0, 2.0, 3.0)
SPAN_EDGES = (0.0, 5.0, 7.0, 10.0, 13.0, 16.0, np.inf)   # left-closed bins of true span, mm
COND_Q = (2.5, 50.0, 97.5)
N_GROUPS = 50


def _gen(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


# ------------------------------------------------------------------ model variants

def counterfactual_latent(lat: Latent, zero_noise: bool = False, equal_views: bool = False) -> Latent:
    """Counterfactual copy of `lat`; consumes no random numbers.

    Both options off: `lat` itself is returned (bit-identical arrays).
    equal_views: mu_v := mu_0 and spans_v := spans_v * mu_0 / mu_v, which is the published model
    evaluated with u_v = u_0 and o_v = o_0 (spans are multiplicative in mu), keeping each view's
    own beat and RR noise. zero_noise: spans := g * mu for every beat.
    """
    if not zero_noise and not equal_views:
        return lat
    mu, u, o, over, spans = lat.mu, lat.u, lat.o, lat.over, lat.spans
    if equal_views:
        mu_new = np.broadcast_to(mu[..., :1], mu.shape).copy()
        spans = spans * (mu_new / mu)[..., None]
        u = np.broadcast_to(u[..., :1], u.shape).copy()
        o = np.broadcast_to(o[..., :1], o.shape).copy()
        over = np.broadcast_to(over[..., :1], over.shape).copy()
        mu = mu_new
    if zero_noise:
        spans = np.broadcast_to((lat.g[:, None, None] * mu)[..., None], lat.spans.shape).copy()
    return replace(lat, mu=mu, u=u, o=o, over=over, spans=spans)


def measure_from_z(lat: Latent, p: dict, z: np.ndarray | None, reader_bias: np.ndarray) -> np.ndarray:
    """`model.measure` with the caliper normals supplied (z = None: caliper error 0).

    With z = rng.standard_normal(lat.spans.shape) this is arithmetically identical to
    `model.measure(lat, p, rng, reader_bias)`.
    """
    if z is None:
        x = lat.spans + reader_bias[:, None, None, None]
    else:
        cal = p["sigma_cal_mm"] * z
        x = lat.spans + reader_bias[:, None, None, None] + cal
    floor = p.get("measure_floor_mm")
    if floor is not None:
        x = np.maximum(x, floor)
    return x


def view_rule_values(lat: Latent, p: dict, x: np.ndarray, U: np.ndarray):
    """Apply the beat rule and the four view rules to measured beats x (n, 2, K, B).

    Returns ({rule: (n, 2)}, vm (n, 2, K), limited (n, 2, K))."""
    br = beat_rule(x, lat.avail[:, None, :], p["N_beats"], p["window"], p["limited_value"])
    vm = br.value
    est = {
        "A1": a1_anchor(vm),
        "A2": a2_mean(vm),
        "A3": a3_max(vm),
        "A4": a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])[0],
    }
    return est, vm, br.limited


def simulate_view_rules(p: dict, ss: np.random.SeedSequence, n_total: int, chunk: int,
                        arms=("full",)) -> dict:
    """Per-patient values of rules A1 to A4 for each requested arm.

    Random-number consumption per chunk is that of `experiments.run_E1`: draw_latent, reader bias
    (n normals), caliper normals (shape of spans), A4 uniforms (n, 2, K-1). All arms reuse them.

    Returns a dict with
      S (n, 2) true spans T1; g (n,);
      est[arm][rule] (n, 2); sel[arm] (n, 2) index of the largest view mean;
      sel_true_best[arm] (n, 2) bool, the selected view has the largest expected span mu;
      sel_over[arm] (n, 2) bool, the selected view carries true overestimation;
      acc[(rule, d)] ErrAcc and infl[d] MeanAcc and lim_anchor[d] PropAcc for arm "full",
      accumulated chunk by chunk exactly as in run_E1 (estimand T1).
    """
    K = int(p["K"])
    for a in arms:
        if a not in ARMS:
            raise ValueError(a)
    kids = ss.spawn(-(-n_total // chunk))
    S, G = [], []
    est = {a: {r: [] for r in RULES} for a in arms}
    sel = {a: [] for a in arms}
    sel_best = {a: [] for a in arms}
    sel_over = {a: [] for a in arms}
    acc = {(r, d): ErrAcc() for r in RULES for d in range(2)}
    infl = [MeanAcc(), MeanAcc()]
    lim = [PropAcc(), PropAcc()]
    for i, ch in enumerate(kids):
        n = min(chunk, n_total - i * chunk)
        rng = _gen(ch)
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        z = rng.standard_normal(lat.spans.shape)
        U = rng.random((n, 2, K - 1))
        S.append(lat.S)
        G.append(lat.g)
        for a in arms:
            zero_noise, equal_views = ARMS[a]
            lc = counterfactual_latent(lat, zero_noise, equal_views)
            x = measure_from_z(lc, p, None if zero_noise else z, np.zeros(n) if zero_noise else rb)
            e, vm, limited = view_rule_values(lc, p, x, U)
            for r in RULES:
                est[a][r].append(e[r])
            j = np.argmax(vm, axis=-1)
            sel[a].append(j)
            sel_best[a].append(j == np.argmax(lc.mu, axis=-1))
            sel_over[a].append(np.take_along_axis(lc.over, j[..., None], axis=-1)[..., 0])
            if a == "full":
                for r in RULES:
                    for d in range(2):
                        acc[(r, d)].add(e[r][:, d], lat.S[:, d])
                for d in range(2):
                    lim[d].add(limited[:, d, 0])
                    infl[d].add(e["A3"][:, d] - e["A1"][:, d])
    cat = np.concatenate
    return {
        "S": cat(S), "g": cat(G),
        "est": {a: {r: cat(v) for r, v in est[a].items()} for a in arms},
        "sel": {a: cat(v) for a, v in sel.items()},
        "sel_true_best": {a: cat(v) for a, v in sel_best.items()},
        "sel_over": {a: cat(v) for a, v in sel_over.items()},
        "acc": acc, "infl": infl, "lim_anchor": lim,
    }


# ------------------------------------------------------------------ error summaries

def _group_bounds(n: int, n_groups: int):
    G = int(min(n_groups, n))
    edges = np.linspace(0, n, G + 1).astype(np.int64)
    return G, edges


def jackknife_se(reps: np.ndarray) -> np.ndarray:
    """Delete-a-group jackknife SE from leave-one-group-out replicates, shape (G, ...)."""
    reps = np.asarray(reps, dtype=float)
    G = reps.shape[0]
    if G < 2:
        return np.full(reps.shape[1:], np.nan)
    d = reps - reps.mean(axis=0)
    return np.sqrt((G - 1) / G * (d * d).sum(axis=0))


def _stat_vector(e: np.ndarray) -> np.ndarray:
    """bias, sd, rmse, mae, signed quantiles, absolute quantiles, p95abs / rmse."""
    a = np.abs(e)
    rmse = np.sqrt(np.mean(e * e))
    qs = np.percentile(e, SIGNED_Q)
    qa = np.percentile(a, ABS_Q)
    sd = e.std(ddof=1) if e.size > 1 else np.nan
    ratio = qa[-1] / rmse if rmse > 0 else np.nan
    return np.concatenate([[e.mean(), sd, rmse, a.mean()], qs, qa, [ratio]])


STAT_NAMES = (["bias_mm", "sd_err_mm", "rmse_mm", "mae_mm"]
              + [f"q{q:g}_err_mm" for q in SIGNED_Q]
              + [f"q{q:g}_abs_mm" for q in ABS_Q]
              + ["ratio_q95abs_rmse"])
JACK_KEEP = ("bias_mm", "sd_err_mm", "rmse_mm", "mae_mm", "q95_abs_mm")


def error_summary(e: np.ndarray, n_groups: int = N_GROUPS) -> tuple[list[dict], np.ndarray]:
    """Summary of a 1-d error vector (estimate minus truth).

    Returns (rows, reps): rows are dicts metric, value, mcse, mcse_method, count, n; reps is the
    (G, len(STAT_NAMES)) array of leave-one-group-out replicates (groups are consecutive blocks
    of patients in generation order; patients are independent draws).

    MCSE: bias sd/sqrt(n); RMSE delta method, sd(e^2)/sqrt(n)/(2 RMSE); MAE sd(|e|)/sqrt(n);
    proportions sqrt(p(1-p)/n); SD of error, every quantile (numpy linear interpolation) and the
    ratio q95(|e|)/RMSE by the delete-a-group jackknife.
    """
    e = np.asarray(e, dtype=float).ravel()
    n = e.size
    if n == 0:
        raise ValueError("empty error vector")
    full = _stat_vector(e)
    G, edges = _group_bounds(n, n_groups)
    reps = np.empty((G, full.size))
    for g in range(G):
        reps[g] = _stat_vector(np.concatenate([e[:edges[g]], e[edges[g + 1]:]])) if G > 1 else full
    jse = jackknife_se(reps)
    a = np.abs(e)
    sd = full[1]
    analytic = {
        "bias_mm": sd / np.sqrt(n) if n > 1 else np.nan,
        "rmse_mm": (np.sqrt(max(np.mean(e ** 4) - np.mean(e * e) ** 2, 0.0) / n) / (2 * full[2])
                    if full[2] > 0 else 0.0),
        "mae_mm": a.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
    }
    rows = []
    for k, name in enumerate(STAT_NAMES):
        if name in analytic:
            rows.append(dict(metric=name, value=float(full[k]), mcse=float(analytic[name]),
                             mcse_method="analytic", count=np.nan, n=n))
        else:
            rows.append(dict(metric=name, value=float(full[k]), mcse=float(jse[k]),
                             mcse_method="jackknife", count=np.nan, n=n))
    for t in ABS_THR:
        k_ = int((a > t).sum())
        pr = k_ / n
        rows.append(dict(metric=f"p_abs_gt{t:g}mm", value=pr, mcse=float(np.sqrt(pr * (1 - pr) / n)),
                         mcse_method="binomial", count=k_, n=n))
    return rows, reps


def conditional_summary(e: np.ndarray, s: np.ndarray, edges=SPAN_EDGES, n_groups: int = N_GROUPS) -> list[dict]:
    """Error quantiles conditional on the true span s falling in [edges[i], edges[i+1]).

    Per bin: n, bias, SD, RMSE, MAE, 2.5th, 50th and 97.5th percentiles of signed error, 95th
    percentile of absolute error and its ratio to the bin RMSE, each with a delete-a-group
    jackknife SE (groups defined on all patients, then restricted to the bin). Empty bins give
    NaN with n = 0; a final row "all" pools every patient.
    """
    e = np.asarray(e, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    n = e.size
    G, ge = _group_bounds(n, n_groups)
    gid = np.repeat(np.arange(G), np.diff(ge))

    def stats(x):
        if x.size == 0:
            return np.full(9, np.nan)
        a = np.abs(x)
        rmse = np.sqrt(np.mean(x * x))
        q = np.percentile(x, COND_Q)
        qa = np.percentile(a, 95.0)
        return np.array([x.mean(), x.std(ddof=1) if x.size > 1 else np.nan, rmse, a.mean(),
                         q[0], q[1], q[2], qa, qa / rmse if rmse > 0 else np.nan])

    names = ["bias_mm", "sd_err_mm", "rmse_mm", "mae_mm", "q2.5_err_mm", "q50_err_mm", "q97.5_err_mm",
             "q95_abs_mm", "ratio_q95abs_rmse"]
    bins = [(f"{edges[i]:g} to {edges[i + 1]:g}" if np.isfinite(edges[i + 1]) else f">= {edges[i]:g}",
             (s >= edges[i]) & (s < edges[i + 1])) for i in range(len(edges) - 1)]
    bins.append(("all", np.ones(n, bool)))
    rows = []
    for lab, m in bins:
        x, gx = e[m], gid[m]
        full = stats(x)
        if x.size > 1 and G > 1:
            reps = np.array([stats(x[gx != g]) for g in range(G)])
            se = jackknife_se(reps)
        else:
            se = np.full(full.size, np.nan)
        for k, nm in enumerate(names):
            rows.append(dict(span_bin=lab, metric=nm, value=float(full[k]), mcse=float(se[k]), n=int(x.size)))
    return rows


def paired_mean(x: np.ndarray) -> tuple[float, float]:
    """Mean and MCSE of a per-patient quantity (paired contrast on common random numbers)."""
    x = np.asarray(x, dtype=float).ravel()
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def ratio_of_means(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """mean(num) / mean(den) with delta-method MCSE for paired per-patient vectors."""
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    md = den.mean()
    if md == 0:
        return np.nan, np.nan
    r = num.mean() / md
    resid = num - r * den
    return float(r), float(resid.std(ddof=1) / np.sqrt(num.size) / abs(md))


def decomposition(sim: dict, d: int, a: str = "A3", b: str = "A1") -> dict:
    """2 x 2 factorial decomposition of E[a - b] on axis d from a four-arm `simulate_view_rules`.

    full = offset_only + noise_only - both_off + interaction, where offset_only is the
    zero_noise arm, noise_only the equal_views arm and both_off the zero_noise_equal_views arm
    (identically 0 for A3 - A1). Each term is a mean of per-patient paired differences.
    """
    ex = {arm: sim["est"][arm][a][:, d] - sim["est"][arm][b][:, d] for arm in ARMS}
    out = {}
    for arm, v in ex.items():
        out[f"excess_{arm}"], out[f"excess_{arm}_mcse"] = paired_mean(v)
    inter = ex["full"] - ex["zero_noise"] - ex["equal_views"] + ex["zero_noise_equal_views"]
    out["interaction"], out["interaction_mcse"] = paired_mean(inter)
    # contribution of offsets entered first (alone) and last (added to noise), same for noise
    last_off = ex["full"] - ex["equal_views"]
    last_noise = ex["full"] - ex["zero_noise"]
    out["offset_last"], out["offset_last_mcse"] = paired_mean(last_off)
    out["noise_last"], out["noise_last_mcse"] = paired_mean(last_noise)
    for nm, v in (("share_offset_first", ex["zero_noise"]), ("share_offset_last", last_off),
                  ("share_noise_first", ex["equal_views"]), ("share_noise_last", last_noise),
                  ("share_offset_shapley", 0.5 * (ex["zero_noise"] + last_off))):
        out[nm], out[nm + "_mcse"] = ratio_of_means(v, ex["full"])
    return out


__all__ = ["RULES", "ARMS", "AXES", "SIGNED_Q", "ABS_Q", "ABS_THR", "SPAN_EDGES", "STAT_NAMES", "JACK_KEEP",
           "counterfactual_latent", "measure_from_z", "view_rule_values", "simulate_view_rules",
           "jackknife_se", "error_summary", "conditional_summary", "paired_mean", "ratio_of_means",
           "decomposition"]
