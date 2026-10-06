"""Protocol amendment 3, package A3-1 (beat rules): post hoc sensitivity extensions.

Added 2026-10-05 in response to the senior author's review of manuscript v05
(comments C185, C196, C324). Nothing in the pre-specified library is changed;
this module adds variants that reduce exactly (bit-identical arrays for the
same generator state) to `model.draw_latent`, `model.measure` and
`rules.beat_rule` when every new option is at its default.

New options (all read with `p.get`, so a pre-specified parameter dict is valid):

  beat_ar1_rho      0.0        first-order autoregressive correlation of the residual
                               log-scale beat effect eta along the beat sequence of a
                               view; the marginal variance s_b^2 is preserved.
  resp_amp          0.0        amplitude A of a multiplicative sinusoidal (respiratory)
                               modulation of every beat span, as a fraction of the span.
  resp_period_beats [4, 6]     the period P (beats per respiratory cycle) is uniform on
                               this interval, one value per patient.
  resp_mode         "add"      "add": the cyclic component is added to the residual beat
                               variation (total beat CV rises); "preserve": the residual
                               CV is lowered so that the total marginal beat CV is unchanged.
  cal_noise_mode    "constant" "constant": per-beat caliper SD = sigma_cal_mm (library);
                               "proportional": SD = sigma_cal_mm * span / span_ref;
                               "mixed": SD = sigma_cal_mm * (f + (1 - f) * span / span_ref).
  cal_const_frac    0.5        f in the mixed form.
  cal_ref_span_mm   None       span_ref; None = closed-form expected true beat span of
                               the anchor view on the AP axis (`expected_beat_span_mm`),
                               so that the average caliper SD on that axis is sigma_cal_mm.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .model import Latent, log_sd_from_cv, n_beats_generated, per_view, zt_poisson
from .rules import BeatRuleResult, beat_rule

EXT_DEFAULTS = dict(beat_ar1_rho=0.0, resp_amp=0.0, resp_period_beats=(4.0, 6.0), resp_mode="add",
                    cal_noise_mode="constant", cal_const_frac=0.5, cal_ref_span_mm=None)


# ------------------------------------------------------------------ beat-noise scales

def residual_beat_log_sd(p: dict) -> float:
    """Log-scale SD s_b of the residual beat effect eta, after any `preserve` adjustment.

    Library value: s_b^2 = ln(1 + beat_cv^2) (+ ln(1 + af_extra_cv^2) in AF). With a
    respiratory component of amplitude A in `preserve` mode the residual CV is reduced
    so that (1 + cv_res^2) * (1 + A^2 / 2) = 1 + cv_lib^2, where cv_lib^2 = exp(s_b^2) - 1.
    """
    s_b = log_sd_from_cv(p["beat_cv"])
    if p["rhythm"] == "AF":
        s_b = float(np.sqrt(s_b ** 2 + log_sd_from_cv(p["af_extra_cv"]) ** 2))
    A = float(p.get("resp_amp", 0.0))
    mode = p.get("resp_mode", "add")
    if mode not in ("add", "preserve"):
        raise ValueError(mode)
    if A != 0.0 and mode == "preserve":
        ratio = np.exp(s_b ** 2) / (1.0 + A ** 2 / 2.0)
        if ratio < 1.0:
            raise ValueError("resp_amp too large to preserve the total beat variance")
        s_b = float(np.sqrt(np.log(ratio)))
    return s_b


def total_beat_cv(p: dict) -> float:
    """Marginal CV of a true beat span around the view's expected level, excluding the
    RR-interval term: sqrt((1 + cv_res^2) * (1 + A^2 / 2) - 1)."""
    s_b = residual_beat_log_sd(p)
    A = float(p.get("resp_amp", 0.0))
    return float(np.sqrt(np.exp(s_b ** 2) * (1.0 + A ** 2 / 2.0) - 1.0))


def expected_beat_span_mm(p: dict, axis: int = 0, view: int = 0) -> float:
    """Closed-form expectation of a true beat span (study settings, g applied).

    E[S_AP] * E[r] (SL only) * (1 - E[u_v] + E[o_v]) * E[g] * E[exp(beat terms)], with
    E[u_v] = u_scale_v * sqrt(2 / pi) (the cap u_max is ignored; it lies beyond 3.6
    half-normal scales for every configured view), E[o_v] = p_over_v * o_median *
    exp(o_log_sd^2 / 2), and E[exp(beat terms)] = exp(tot / 2) under median centring
    (1 under mean centring). The respiratory factor has expectation 1.
    """
    K = int(p["K"])
    e = float(p["S_median_mm"]) * np.exp(float(p["S_log_sd"]) ** 2 / 2.0)
    if axis == 1:
        lo, hi = float(p["r_lower"]), float(p["r_upper"])
        e *= min(max(float(p["r_mean"]), lo), hi) if hi > lo else hi
    on = bool(p.get("view_errors", True))
    eu = per_view(p["u_scale"], K)[view] * np.sqrt(2.0 / np.pi) if (on and p["view_under"]) else 0.0
    eo = (per_view(p["p_over"], K)[view] * p["o_median"] * np.exp(p["o_log_sd"] ** 2 / 2.0)
          if (on and p["view_over"]) else 0.0)
    e *= 1.0 - eu + eo
    e *= np.exp(float(p["g_log_sd"]) ** 2 / 2.0)
    if p["beat_noise_centering"] == "median":
        s_b = residual_beat_log_sd(p)
        s_rr = log_sd_from_cv(p["rr_cv_af"] if p["rhythm"] == "AF" else p["rr_cv_sinus"])
        b1, b2 = float(p["beta_rr"]), float(p["beta_drr"])
        e *= np.exp((s_b ** 2 + (b1 + b2) ** 2 * s_rr ** 2 + b2 ** 2 * s_rr ** 2) / 2.0)
    return float(e)


# ------------------------------------------------------------------ generative model variant

def ar1_from_innovations(z: np.ndarray, rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) along the last axis from standard normal innovations z.

    eta_0 = sd * z_0; eta_b = rho * eta_{b-1} + sqrt(1 - rho^2) * sd * z_b, so every
    eta_b has variance sd^2 and corr(eta_b, eta_{b+k}) = rho^k. rho = 0 returns
    sd * z exactly (the library expression).
    """
    rho = float(rho)
    if not -1.0 < rho < 1.0:
        raise ValueError("beat_ar1_rho must lie strictly between -1 and 1")
    if rho == 0.0:
        return sd * z
    eta = np.empty_like(z)
    eta[..., 0] = sd * z[..., 0]
    c = np.sqrt(1.0 - rho ** 2) * sd
    for b in range(1, z.shape[-1]):
        eta[..., b] = rho * eta[..., b - 1] + c * z[..., b]
    return eta


def resp_factor(period: np.ndarray, phase: np.ndarray, B: int, amp: float) -> np.ndarray:
    """Multiplicative respiratory modulation, shape (n, 1, K, B).

    factor[i, 0, k, b] = 1 + amp * sin(2 * pi * b / period[i] + phase[i, k]); the two
    axes of a view share the clip and therefore the factor.
    """
    b = np.arange(B)[None, None, :]
    f = 1.0 + float(amp) * np.sin(2.0 * np.pi * b / period[:, None, None] + phase[:, :, None])
    return f[:, None, :, :]


def draw_latent_ext(p: dict, n: int, rng: np.random.Generator) -> Latent:
    """Variant of `model.draw_latent` with serially correlated beat effects and an
    optional respiratory component.

    The sequence of random draws is that of the library; the two respiratory draws
    (period, phase) are made last and only when resp_amp > 0, so that with the
    defaults the returned arrays and the generator state equal the library's.
    """
    K = int(p["K"])
    D = 2
    # 3.1 case mix (as library)
    S_ap = np.exp(np.log(p["S_median_mm"]) + p["S_log_sd"] * rng.standard_normal(n))
    lo, hi = float(p["r_lower"]), float(p["r_upper"])
    m = (p["r_mean"] - lo) / (hi - lo) if hi > lo else 1.0
    if m >= 1.0:
        r = np.full(n, hi)
    elif m <= 0.0:
        r = np.full(n, lo)
    else:
        c = float(p["r_concentration"])
        r = lo + (hi - lo) * rng.beta(m * c, (1 - m) * c, n)
    S = np.stack([S_ap, S_ap * r], axis=1)

    # 3.2 view errors (as library)
    rho = float(p["rho"])

    def corr_normal():
        C = rng.standard_normal((n, D, 1))
        E = rng.standard_normal((n, D, K))
        return np.sqrt(rho) * C + np.sqrt(1.0 - rho) * E

    Zu = corr_normal()
    Zo = corr_normal()
    Mo = rng.standard_normal((n, D, K))
    on = bool(p.get("view_errors", True))
    if on and p["view_under"]:
        u = np.minimum(per_view(p["u_scale"], K) * np.abs(Zu), p["u_max"])
    else:
        u = np.zeros((n, D, K))
    if on and p["view_over"]:
        over = ndtr(Zo) < per_view(p["p_over"], K)
        o = np.where(over, p["o_median"] * np.exp(p["o_log_sd"] * Mo), 0.0)
    else:
        over = np.zeros((n, D, K), dtype=bool)
        o = np.zeros((n, D, K))
    mu = S[:, :, None] * (1.0 - u + o)

    # 3.5 instrument factor (as library)
    g = np.exp(p["g_log_sd"] * rng.standard_normal(n))

    # 3.3 beats
    B = n_beats_generated(p)
    af = p["rhythm"] == "AF"
    s_rr = log_sd_from_cv(p["rr_cv_af"] if af else p["rr_cv_sinus"])
    logrr = s_rr * rng.standard_normal((n, K, B + 1))
    if p["data_mode"] == "prospective":
        avail = np.full((n, K), B, dtype=np.int64)
    else:
        avail = zt_poisson(rng, p["avail_lambda"], (n, K), int(p["max_beats_retro"]))
    amp = float(p.get("resp_amp", 0.0))
    if amp < 0.0 or amp >= 1.0:
        raise ValueError("resp_amp must lie in [0, 1)")
    if amp == 0.0:
        # library expression, kept verbatim so that s_b is bit-identical
        s_b = log_sd_from_cv(p["beat_cv"])
        if af:
            s_b = float(np.sqrt(s_b ** 2 + log_sd_from_cv(p["af_extra_cv"]) ** 2))
    else:
        s_b = residual_beat_log_sd(p)
    z = rng.standard_normal((n, D, K, B))
    eta = ar1_from_innovations(z, float(p.get("beat_ar1_rho", 0.0)), s_b)   # A3-1 (d)
    b1, b2 = float(p["beta_rr"]), float(p["beta_drr"])
    rr2 = logrr[:, None, :, 1:]
    rr1 = logrr[:, None, :, :-1]
    log_span = np.log(mu)[..., None] + b1 * rr2 + b2 * (rr2 - rr1) + eta
    if p["beat_noise_centering"] == "mean":
        tot = s_b ** 2 + (b1 + b2) ** 2 * s_rr ** 2 + b2 ** 2 * s_rr ** 2
        log_span = log_span - tot / 2.0
    elif p["beat_noise_centering"] != "median":
        raise ValueError(p["beat_noise_centering"])
    spans = g[:, None, None, None] * np.exp(log_span)
    if amp > 0.0:                                                            # A3-1 (d)
        plo, phi = (float(v) for v in p.get("resp_period_beats", EXT_DEFAULTS["resp_period_beats"]))
        if not 2.0 <= plo <= phi:
            raise ValueError("resp_period_beats must satisfy 2 <= lower <= upper")
        period = rng.uniform(plo, phi, n)
        phase = rng.uniform(0.0, 2.0 * np.pi, (n, K))
        spans = spans * resp_factor(period, phase, B, amp)
    return Latent(S=S, r=r, u=u, o=o, over=over, mu=mu, g=g, logrr=logrr, spans=spans, avail=avail)


def caliper_sd(spans: np.ndarray, p: dict):
    """Per-beat caliper SD (mm): scalar for the constant form, array otherwise."""
    mode = p.get("cal_noise_mode", "constant")
    sigma = p["sigma_cal_mm"]
    if mode == "constant":
        return sigma
    if mode == "proportional":
        f = 0.0
    elif mode == "mixed":
        f = float(p.get("cal_const_frac", EXT_DEFAULTS["cal_const_frac"]))
        if not 0.0 <= f <= 1.0:
            raise ValueError("cal_const_frac must lie in [0, 1]")
    else:
        raise ValueError(mode)
    ref = p.get("cal_ref_span_mm")
    if ref is None:
        ref = expected_beat_span_mm(p, axis=0, view=0)
    return sigma * (f + (1.0 - f) * spans / float(ref))


def measure_ext(lat: Latent, p: dict, rng: np.random.Generator, reader_bias: np.ndarray) -> np.ndarray:
    """Variant of `model.measure` with span-dependent caliper noise.

    One standard normal per beat is drawn, as in the library, and multiplied by the
    per-beat SD of `caliper_sd`; in the constant form the expression is the library's.
    """
    cal = caliper_sd(lat.spans, p) * rng.standard_normal(lat.spans.shape)
    x = lat.spans + reader_bias[:, None, None, None] + cal
    floor = p.get("measure_floor_mm")
    if floor is not None:
        x = np.maximum(x, floor)
    return x


# ------------------------------------------------------------------ beat rules at a fixed budget

def plain_mean_first(x: np.ndarray, k) -> np.ndarray:
    """Mean of the first k beats along the last axis; k is a scalar or an integer array
    broadcastable to x.shape[:-1] (values clipped to 1..B)."""
    B = x.shape[-1]
    k = np.clip(np.broadcast_to(np.asarray(k, dtype=np.int64), x.shape[:-1]), 1, B)
    mask = np.arange(B) < k[..., None]
    return np.where(mask, x, 0.0).sum(-1) / k


def window_rule_budget(x: np.ndarray, avail, N: int, W: float | None, budget: int | None = None,
                       limited_value: str = "all_acquired") -> BeatRuleResult:
    """`rules.beat_rule` with a cap `budget` on the beats the reader may acquire in total.

    Only the first `budget` beats are visible to the rule; the search, stopping and
    fallback are the library's. budget = None or budget >= x.shape[-1] calls the
    library on the unchanged arrays.
    """
    B = x.shape[-1]
    if budget is None or int(budget) >= B:
        return beat_rule(x, avail, N, W, limited_value)
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")
    return beat_rule(x[..., :budget], np.minimum(avail, budget), N, W, limited_value)


def beat_rule_exhaustive(x: np.ndarray, avail, N: int, W: float,
                         limited_value: str = "all_acquired") -> BeatRuleResult:
    """Reference implementation that examines every N-subset of the first m beats
    (not only contiguous blocks of the sorted values). Same stopping rule (first m at
    which any subset qualifies), same criterion (every beat within +-W of the subset
    mean), same choice (smallest (max - min) / |mean|; first in lexicographic index
    order on exact ties) and same fallback as `rules.beat_rule`. For small B only.
    """
    shape = x.shape[:-1]
    B = x.shape[-1]
    X = x.reshape(-1, B)
    A = np.minimum(np.broadcast_to(avail, shape).reshape(-1), B).astype(np.int64)
    R = X.shape[0]
    N = int(N)
    W = float(W)
    value = np.empty(R)
    used = np.empty(R, dtype=np.int64)
    accepted = np.zeros(R, dtype=bool)
    met_first = np.zeros(R, dtype=bool)
    active = A >= N
    for m in range(N, B + 1):
        idx = np.nonzero(active & (A >= m))[0]
        if idx.size == 0:
            break
        # subsets that contain the newest beat suffice for m > N (others failed before)
        if m == N:
            combos = np.array([tuple(range(N))])
        else:
            combos = np.array([c + (m - 1,) for c in itertools.combinations(range(m - 1), N - 1)])
        best = np.full(idx.size, np.inf)
        bval = np.empty(idx.size)
        for lo in range(0, combos.shape[0], 512):
            cc = combos[lo:lo + 512]
            v = X[idx][:, cc]                               # (rows, combos, N)
            mn = v.mean(-1)
            tol = W * np.abs(mn)
            vmin, vmax = v.min(-1), v.max(-1)
            ok = (mn - vmin <= tol) & (vmax - mn <= tol)
            spread = np.where(ok, (vmax - vmin) / np.abs(mn), np.inf)
            j = np.argmin(spread, axis=1)
            s = spread[np.arange(idx.size), j]
            better = s < best
            best[better] = s[better]
            bval[better] = mn[np.arange(idx.size), j][better]
        hit = np.isfinite(best)
        sel = idx[hit]
        value[sel] = bval[hit]
        used[sel] = m
        accepted[sel] = True
        if m == N:
            met_first[sel] = True
        active[sel] = False
    lim = ~accepted
    if limited_value == "all_acquired":
        value[lim] = plain_mean_first(X[lim], A[lim]) if lim.any() else value[lim]
    elif limited_value == "first_n":
        value[lim] = plain_mean_first(X[lim], np.minimum(A[lim], N)) if lim.any() else value[lim]
    else:
        raise ValueError(limited_value)
    used[lim] = A[lim]
    return BeatRuleResult(value=value.reshape(shape), used=used.reshape(shape),
                          accepted=accepted.reshape(shape), met_first=met_first.reshape(shape),
                          limited=lim.reshape(shape))


# ------------------------------------------------------------------ error summaries with MCSE

def error_summary(e: np.ndarray, truth: np.ndarray) -> dict:
    """Bias, RMSE, MAE and relative bias of errors e = estimate - truth, each with its
    Monte Carlo standard error (RMSE by the delta method on the mean squared error)."""
    e = np.asarray(e, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    n = e.size
    e2 = e * e
    rmse = float(np.sqrt(e2.mean()))
    rel = e / t
    return dict(
        n=n,
        bias_mm=float(e.mean()), bias_mcse=float(e.std(ddof=1) / np.sqrt(n)),
        rmse_mm=rmse, rmse_mcse=float(e2.std(ddof=1) / np.sqrt(n) / (2.0 * rmse)) if rmse > 0 else 0.0,
        mae_mm=float(np.abs(e).mean()), mae_mcse=float(np.abs(e).std(ddof=1) / np.sqrt(n)),
        relbias_pct=float(100.0 * rel.mean()), relbias_mcse=float(100.0 * rel.std(ddof=1) / np.sqrt(n)),
    )


def paired_summary(ea: np.ndarray, eb: np.ndarray, truth: np.ndarray) -> dict:
    """Differences (a minus b) of bias, RMSE, MAE and relative bias for two estimators
    evaluated on the same simulated patients (common random numbers).

    Paired MCSE: SD of the per-patient difference / sqrt(n) for bias, MAE and relative
    bias. For RMSE the delta method on the paired squared errors:
        var(RMSE_a - RMSE_b) = [var(a2) / (4 MSE_a) + var(b2) / (4 MSE_b)
                                - cov(a2, b2) / (2 RMSE_a RMSE_b)] / n,
    with a2 = ea^2, b2 = eb^2. `*_mcse_indep` is the value the same data would give if
    the covariance were ignored (the formula for independently seeded arms).
    """
    ea = np.asarray(ea, dtype=float).ravel()
    eb = np.asarray(eb, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    n = ea.size
    rn = np.sqrt(n)
    a2, b2 = ea * ea, eb * eb
    ra, rb = np.sqrt(a2.mean()), np.sqrt(b2.mean())
    va, vb = a2.var(ddof=1), b2.var(ddof=1)
    cov = float(np.cov(a2, b2, ddof=1)[0, 1]) if n > 1 else 0.0
    if ra > 0 and rb > 0:
        var_p = (va / (4 * ra ** 2) + vb / (4 * rb ** 2) - cov / (2 * ra * rb)) / n
        var_i = (va / (4 * ra ** 2) + vb / (4 * rb ** 2)) / n
    else:
        var_p = var_i = 0.0
    aa, ab = np.abs(ea), np.abs(eb)
    rela, relb = 100.0 * ea / t, 100.0 * eb / t

    def ind(u, v):
        return float(np.sqrt(u.var(ddof=1) + v.var(ddof=1)) / rn)

    return dict(
        n=n,
        d_bias_mm=float((ea - eb).mean()), d_bias_mcse=float((ea - eb).std(ddof=1) / rn),
        d_bias_mcse_indep=ind(ea, eb),
        d_rmse_mm=float(ra - rb), d_rmse_mcse=float(np.sqrt(max(var_p, 0.0))),
        d_rmse_mcse_indep=float(np.sqrt(var_i)),
        d_mse_mm2=float((a2 - b2).mean()), d_mse_mcse=float((a2 - b2).std(ddof=1) / rn),
        d_mae_mm=float((aa - ab).mean()), d_mae_mcse=float((aa - ab).std(ddof=1) / rn),
        d_mae_mcse_indep=ind(aa, ab),
        d_relbias_pct=float((rela - relb).mean()), d_relbias_mcse=float((rela - relb).std(ddof=1) / rn),
        d_relbias_mcse_indep=ind(rela, relb),
        corr_sq_err=float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan"),
    )


def bootstrap_rmse_diff(ea: np.ndarray, eb: np.ndarray, n_boot: int, rng: np.random.Generator,
                        block: int = 50) -> dict:
    """Patient-level (paired) bootstrap of RMSE_a - RMSE_b: patients are resampled with
    replacement and both errors of a resampled patient are kept together."""
    a2 = np.asarray(ea, dtype=float).ravel() ** 2
    b2 = np.asarray(eb, dtype=float).ravel() ** 2
    n = a2.size
    out = np.empty(n_boot)
    for lo in range(0, n_boot, block):
        k = min(block, n_boot - lo)
        idx = rng.integers(0, n, size=(k, n))
        out[lo:lo + k] = np.sqrt(a2[idx].mean(1)) - np.sqrt(b2[idx].mean(1))
    q = np.percentile(out, [2.5, 97.5])
    return dict(n_boot=int(n_boot), boot_mean=float(out.mean()), boot_se=float(out.std(ddof=1)),
                boot_lo95=float(q[0]), boot_hi95=float(q[1]))
