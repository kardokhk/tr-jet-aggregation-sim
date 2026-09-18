"""Generative model (protocol section 3).

Array conventions: n patients, D = 2 axes (0 = AP, 1 = SL), K views (0 =
anchor), B beats. Views share their RR sequence and available-beat count
across axes (same clip); beat noise, view errors and caliper errors are drawn
independently per axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

AXES = ("AP", "SL")


def log_sd_from_cv(cv: float) -> float:
    """Log-scale SD of a lognormal with coefficient of variation `cv`."""
    return float(np.sqrt(np.log1p(cv ** 2)))


def per_view(values, K: int) -> np.ndarray:
    v = np.atleast_1d(np.asarray(values, dtype=float))
    if len(v) < K:
        v = np.concatenate([v, np.repeat(v[-1], K - len(v))])
    return v[:K]


def n_beats_generated(p: dict) -> int:
    N = int(p["N_beats"])
    if p["window"] is None:
        return N if p["data_mode"] == "prospective" else min(N, int(p["max_beats_retro"]))
    if p["data_mode"] == "prospective":
        return max(N, int(p["max_beats_prosp"]))
    return int(p["max_beats_retro"])


def zt_poisson(rng: np.random.Generator, lam: float, size, cap: int) -> np.ndarray:
    """Zero-truncated Poisson by rejection, capped at `cap`."""
    x = rng.poisson(lam, size)
    bad = x == 0
    while bad.any():
        x[bad] = rng.poisson(lam, int(bad.sum()))
        bad = x == 0
    return np.minimum(x, cap)


@dataclass
class Latent:
    S: np.ndarray        # (n, 2) true spans T1
    r: np.ndarray        # (n,) SL/AP ratio
    u: np.ndarray        # (n, 2, K) underestimation fraction
    o: np.ndarray        # (n, 2, K) overestimation fraction
    over: np.ndarray     # (n, 2, K) bool, overestimation present
    mu: np.ndarray       # (n, 2, K) view expected span (reference settings)
    g: np.ndarray        # (n,) instrument factor
    logrr: np.ndarray    # (n, K, B+1) centred log RR; beat b has RR1=logrr[b], RR2=logrr[b+1]
    spans: np.ndarray    # (n, 2, K, B) true beat spans under study settings (g applied)
    avail: np.ndarray    # (n, K) available beats (<= B for indexing via min)

    @property
    def n(self):
        return self.S.shape[0]


def draw_latent(p: dict, n: int, rng: np.random.Generator) -> Latent:
    K = int(p["K"])
    D = 2
    # 3.1 case mix
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

    # 3.2 view errors; latent Gaussian drivers with shared patient component
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

    # 3.5 instrument factor
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
    s_b = log_sd_from_cv(p["beat_cv"])
    if af:
        s_b = float(np.sqrt(s_b ** 2 + log_sd_from_cv(p["af_extra_cv"]) ** 2))
    eta = s_b * rng.standard_normal((n, D, K, B))
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
    return Latent(S=S, r=r, u=u, o=o, over=over, mu=mu, g=g, logrr=logrr, spans=spans, avail=avail)


def measure(lat: Latent, p: dict, rng: np.random.Generator, reader_bias: np.ndarray) -> np.ndarray:
    """One read of all beats: spans + reader bias + per-beat caliper error.

    `reader_bias` has shape (n,) (bias, including any drift, for this read).
    """
    cal = p["sigma_cal_mm"] * rng.standard_normal(lat.spans.shape)
    x = lat.spans + reader_bias[:, None, None, None] + cal
    floor = p.get("measure_floor_mm")
    if floor is not None:
        x = np.maximum(x, floor)
    return x
