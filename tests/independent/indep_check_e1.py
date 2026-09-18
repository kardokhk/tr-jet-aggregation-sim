"""Independent reimplementation of the DUO-Max E1 core (protocol v01, section 3-4).

Written 2026-09-18 without reading code/lib/duomaxsim/{model,rules,estimators}.py.
One axis at a time (bias/RMSE per axis are marginal, so cross-axis coupling is
irrelevant). Interpretation choices (from protocol + base.yaml notes):
  - S_AP lognormal(log 9, 0.49); SL = AP * r, r = 0.4 + 0.6*Beta(a, b), a+b = 8, mean 0.614.
  - latent per-view drivers z_v = sqrt(rho) z0 + sqrt(1-rho) e_v, separately for u and o;
    u_v = min(u_scale_v |z_v|, u_max); o present if Phi(zo_v) < p_over_v,
    magnitude lognormal(log 0.15, 0.5) (independent of the latent).
  - g ~ lognormal(0, 0.10), per study, all views.
  - beats: log span = log(g mu) + beta_rr * (log RR - log RR_med) + eta, eta ~ N(0, ln(1+cv^2)).
  - reading = span + N(0, 1) + b_r, b_r ~ N(0, 0.5^2) per patient (one reader).
  - window W: acquire N; if some N-subset (contiguous in sorted order) has all beats
    within +-W*|mean| of its mean, accept (the qualifying run with smallest max rel.
    deviation); else add beats one at a time up to 30 (prospective); if never met,
    report mean of all acquired ("limited").
  - A4: triggered (d >= t_warn) higher views judged; true overestimation rejected with
    prob s_det, genuine view rejected with prob f_rej; final = max of anchor and
    non-rejected verification means.
"""
import sys
import numpy as np
from scipy.stats import norm

rng_master = np.random.SeedSequence(987654321)

P = dict(S_median=9.0, S_log_sd=0.49, r_lo=0.4, r_hi=1.0, r_mean=0.614, r_conc=8.0,
         u_scale=[0.03, 0.10, 0.10, 0.15], u_max=0.9, p_over=[0.05, 0.15, 0.15, 0.20],
         o_median=0.15, o_log_sd=0.5, rr_cv=0.03, beta_rr=0.2, max_beats=30,
         window=0.15, sigma_cal=1.0, sigma_rb=0.5, g_log_sd=0.10, t_warn=3.0)


def window_rule(x, N, W, maxb):
    """x: (n, maxb) readings in acquisition order. Returns (value, limited)."""
    n = x.shape[0]
    if W is None:
        return x[:, :N].mean(1), np.zeros(n, bool)
    val = np.full(n, np.nan)
    done = np.zeros(n, bool)
    for k in range(N, maxb + 1):
        todo = ~done
        if not todo.any():
            break
        xs = np.sort(x[todo, :k], axis=1)
        best_dev = np.full(todo.sum(), np.inf)
        best_val = np.full(todo.sum(), np.nan)
        for s in range(0, k - N + 1):
            run = xs[:, s:s + N]
            m = run.mean(1)
            dev = np.abs(run - m[:, None]).max(1) / np.abs(m)
            ok = (dev <= W) & (dev < best_dev)
            best_dev = np.where(ok, dev, best_dev)
            best_val = np.where(ok, m, best_val)
        acc = np.isfinite(best_dev)
        idx = np.flatnonzero(todo)
        val[idx[acc]] = best_val[acc]
        done[idx[acc]] = True
    lim = ~done
    val[lim] = x[lim, :maxb].mean(1)
    return val, lim


def simulate(n, K, cv, over, rho, N, axis, seed, sdet_frej):
    rng = np.random.default_rng(seed)
    S = np.exp(np.log(P["S_median"]) + P["S_log_sd"] * rng.standard_normal(n))
    if axis == "SL":
        bm = (P["r_mean"] - P["r_lo"]) / (P["r_hi"] - P["r_lo"])
        a, b = bm * P["r_conc"], (1 - bm) * P["r_conc"]
        S = S * (P["r_lo"] + (P["r_hi"] - P["r_lo"]) * rng.beta(a, b, n))
    zu = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, K))
    zo = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, K))
    u = np.minimum(np.array(P["u_scale"][:K]) * np.abs(zu), P["u_max"])
    if over:
        ind = norm.cdf(zo) < np.array(P["p_over"][:K])
        mag = np.exp(np.log(P["o_median"]) + P["o_log_sd"] * rng.standard_normal((n, K)))
        o = ind * mag
    else:
        ind = np.zeros((n, K), bool)
        o = np.zeros((n, K))
    mu = S[:, None] * (1 - u + o)
    g = np.exp(P["g_log_sd"] * rng.standard_normal(n))
    sb = np.sqrt(np.log(1 + cv ** 2))
    srr = np.sqrt(np.log(1 + P["rr_cv"] ** 2))
    br = P["sigma_rb"] * rng.standard_normal(n)
    M = P["max_beats"]
    means = np.empty((n, K))
    for v in range(K):
        lrr = srr * rng.standard_normal((n, M))
        span = g[:, None] * mu[:, [v]] * np.exp(P["beta_rr"] * lrr + sb * rng.standard_normal((n, M)))
        x = span + P["sigma_cal"] * rng.standard_normal((n, M)) + br[:, None]
        means[:, v], _ = window_rule(x, N, P["window"], M)
    est = {"A1": means[:, 0], "A3": means.max(1)}
    for sd, fr in sdet_frej:
        final = means[:, 0].copy()
        for v in range(1, K):
            d = means[:, v] - means[:, 0]
            trig = d >= P["t_warn"]
            rej = np.where(ind[:, v], rng.random(n) < sd, rng.random(n) < fr) & trig
            final = np.where(rej, final, np.maximum(final, means[:, v]))
        est[f"A4_{sd}_{fr}"] = final
    T = {"T1": S, "T2": mu[:, 0]}
    out = []
    for e, x in est.items():
        for tn, t in T.items():
            err = x - t
            bias, se_b = err.mean(), err.std(ddof=1) / np.sqrt(n)
            mse = np.mean(err ** 2)
            rmse = np.sqrt(mse)
            se_r = np.std(err ** 2, ddof=1) / np.sqrt(n) / (2 * rmse)
            out.append((e, axis, tn, bias, se_b, rmse, se_r))
    return out


if __name__ == "__main__":
    import json
    settings = [
        (0, 1, 0.05, False, 0.0, 1), (409, 2, 0.15, True, 0.0, 3), (754, 3, 0.20, True, 0.3, 10),
        (842, 3, 0.30, True, 0.0, 5), (953, 4, 0.10, True, 0.6, 13), (1077, 4, 0.25, False, 0.9, 7)]
    sdfr = [(0.8, 0.1), (1.0, 0.0), (0.0, 0.3)]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    seeds = rng_master.spawn(len(settings) * 2)
    rows = []
    for i, (cell, K, cv, over, rho, N) in enumerate(settings):
        for j, axis in enumerate(["AP", "SL"]):
            for r in simulate(n, K, cv, over, rho, N, axis, seeds[2 * i + j], sdfr):
                rows.append(dict(cell=cell, estimator=r[0], axis=r[1], estimand=r[2],
                                 bias=r[3], se_bias=r[4], rmse=r[5], se_rmse=r[6]))
    print(json.dumps(rows))
