"""Independent check of amendment-2 (E5b shared-error AI, E6b Brown-Forsythe).

Written without reading the new experiments.py/metrics.py code. Uses only the
pre-existing generative model (model.draw_latent, model.measure) and rules
(beat_rule, a4_final). Own seeds (not the package's), so agreement is only
expected within Monte Carlo error.
"""
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, "code/lib")
from duomaxsim.config import load_config, cell_params  # noqa: E402
from duomaxsim.model import draw_latent, measure  # noqa: E402
from duomaxsim.rules import beat_rule, a4_final  # noqa: E402

CFG = load_config("code/configs/amend2_2026-09-18.yaml")
AX = 0  # AP


def view_means(x, lat, p):
    br = beat_rule(x, lat.avail[:, None, :], int(p["N_beats"]), p["window"], p["limited_value"])
    return br.value  # (n, 2, K)


def a4(vm, lat, p, U):
    f, _ = a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])
    return f


def read(lat, p, rng, bias, U):
    x = measure(lat, p, rng, bias)
    return a4(view_means(x, lat, p), lat, p, U)


# ---------------------------------------------------------------- (a) E5b
def e5(cell, n_studies=400, seed=1):
    p = cell_params(CFG, "E5", cell)
    lam = float(p["ai_shared_lambda"])
    rng = np.random.default_rng(seed)
    nv = int(p["n_val"])
    out = {"cell": cell, "lambda": lam, "beat_cv": p["beat_cv"], "view_over": p["view_over"]}
    mae = {k: [] for k in ("shared", "indep", "rimg_vs_T1")}
    lower = []
    for s in range(n_studies):
        lat = draw_latent(p, nv, rng)
        K = int(p["K"])
        U = rng.random((nv, 2, K - 1))
        b = p["sigma_rb_mm"] * rng.standard_normal()
        r1 = read(lat, p, rng, np.full(nv, b), U)[:, AX]
        # R_img: images only (g, u, o, beat noise), no caliper error, no reader bias
        rimg = a4(view_means(lat.spans, lat, p), lat, p, U)[:, AX]
        T1 = lat.S[:, AX]
        z = rng.standard_normal(nv)
        ai_ind = T1 + p["sigma_ai_mm"] * z
        ai_sh = T1 + lam * (rimg - T1) + p["sigma_ai_mm"] * z
        m_sh = np.abs(ai_sh - r1).mean()
        m_in = np.abs(ai_ind - r1).mean()
        mae["shared"].append(m_sh)
        mae["indep"].append(m_in)
        mae["rimg_vs_T1"].append(np.abs(rimg - T1).mean())
        lower.append(m_sh < m_in)
    for k, v in mae.items():
        v = np.asarray(v)
        out[f"mae_{k}"] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    d = np.asarray(mae["shared"]) - np.asarray(mae["indep"])
    out["paired_diff"] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    ph = np.mean(lower)
    out["p_lower"] = (ph, np.sqrt(ph * (1 - ph) / len(lower)))
    return out


# ---------------------------------------------------------------- (b) E6b
def bf_test(a, b):
    """Brown-Forsythe on rows of a (group 1) and b (group 2). Returns (p2, p1)."""
    za = np.abs(a - np.median(a, -1, keepdims=True))
    zb = np.abs(b - np.median(b, -1, keepdims=True))
    n1, n2 = za.shape[-1], zb.shape[-1]
    m1, m2 = za.mean(-1), zb.mean(-1)
    ss = ((za - m1[..., None]) ** 2).sum(-1) + ((zb - m2[..., None]) ** 2).sum(-1)
    sp2 = ss / (n1 + n2 - 2)
    t = (m1 - m2) / np.sqrt(sp2 * (1 / n1 + 1 / n2))
    p2 = 2 * stats.t.sf(np.abs(t), n1 + n2 - 2)  # t^2 = F(1, n-2)
    p1 = stats.t.sf(t, n1 + n2 - 2)
    return p2, p1


def e6(cell, n_studies=1000, seed=2, per=25):
    p = cell_params(CFG, "E6", cell)
    rng = np.random.default_rng(seed)
    prodn = int(p["production_n"])
    lvl = float(p["test_level"])
    K = int(p["K"])
    res = {}
    done = 0
    while done < n_studies:
        s_c = min(per, n_studies - done)
        lat = draw_latent(p, s_c * prodn, rng)
        U = rng.random((s_c * prodn, 2, K - 1))
        b = np.repeat(p["sigma_rb_mm"] * rng.standard_normal(s_c), prodn)
        man = read(lat, p, rng, b, U)[:, AX].reshape(s_c, prodn)
        T1 = lat.S[:, AX].reshape(s_c, prodn)
        ai = T1 + p["ai_draft_bias_mm"] + p["ai_draft_sigma_mm"] * rng.standard_normal((s_c, prodn))
        d = man - ai
        for ns in (10, 25, 50, 100):
            for al in (0.0, 0.2, 0.5):
                p2, p1 = bf_test(d[:, :ns], (1 - al) * d[:, ns:])
                res.setdefault((ns, al), [[], []])
                res[(ns, al)][0].extend(p2 < lvl)
                res[(ns, al)][1].extend(p1 < lvl)
        done += s_c
    out = {}
    for k, (r2, r1) in res.items():
        out[f"ns{k[0]}_a{k[1]}"] = (np.mean(r2), np.mean(r1))
    return out


def scipy_agreement(seed=3):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((50, 25)) * 1.3
    b = rng.standard_normal((50, 475))
    p2, _ = bf_test(a, b)
    ref = np.array([stats.levene(a[i], b[i], center="median").pvalue for i in range(50)])
    return float(np.max(np.abs(p2 - ref) / ref))


if __name__ == "__main__":
    res = {"scipy_rel_err": scipy_agreement()}
    for c in (9, 10, 11, 12, 13, 14):
        res[f"E5_{c}"] = e5(c)
    for c in (22, 0):
        res[f"E6_{c}"] = e6(c)
    print(json.dumps(res, indent=1, default=float))
