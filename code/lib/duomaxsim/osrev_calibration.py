"""Protocol amendment 3, package A3-6 (calibration): matched-estimand calibration of long-axis
underestimation against Singh et al. 2026 (doi:10.21037/acs-2025-1-72-tvd).

Nothing in the existing library is modified. Three parts:

1. Ellipse geometry. The vena contracta cross-section is an ellipse with major diameter Dmax and
   minor diameter Dmin. A measurement plane cuts it along a line at angle theta to the major
   axis and at perpendicular distance d from the centre; the measured width is the chord

       c(theta, delta) = c0(theta) * sqrt(1 - delta^2),
       c0(theta) = 1 / sqrt(cos^2(theta) / Dmax^2 + sin^2(theta) / Dmin^2),
       delta = |d| / h(theta),  h(theta) = 0.5 * sqrt(Dmax^2 sin^2(theta) + Dmin^2 cos^2(theta)),

   where c0 is the central chord (the longest chord in that direction) and h the half-extent of
   the ellipse perpendicular to the plane. Functions: `central_chord`, `half_extent`,
   `feret_width`, `chord`, `midpoint_offset`, `biplane_widths`.

2. Calibration. `fit_orifice` matches a joint distribution of (Dmax, Dmin) to the published
   means and derived SDs; `fit_view` tunes the two plane-geometry parameters of one view (mean
   plane angle and either an offset scale or a rotation-independent scale factor) to the
   published mean primary-plane and orthogonal-plane widths; `underestimation_table` reads off
   the fractional underestimation of the fixed-axis maximal span in the same simulated patients.

3. View-rule model. `draw_latent_axis` is a variant of `model.draw_latent` that accepts an
   axis-specific underestimation scale (`u_scale_axis`), a constant component (`u_shift_axis`)
   and a chord link (`u_link: chord`). With none of these keys in `p` it consumes the same
   random numbers and returns bit-identical arrays to `model.draw_latent`
   (tests/test_osrev_calibration.py). `simulate_rules` applies the published beat rule and the
   four view rules with the random-number order of `experiments.run_E1`.

4. Follow-up of 2026-10-06 (audit item 5.3): patient-level geometric draw. With the key `u_geom`
   in `p` (default: absent, nothing changes), the underestimation fraction of the long-axis
   views is computed patient by patient from the ellipse geometry (`geometric_underestimation`):
   an ellipticity ratio drawn from the calibrated orifice distribution and shared by the views of
   a patient, a plane angle drawn per view, and a relative offset. The additional normals come
   from a separate generator (`draw_geometry_normals`), so the main random-number stream, and
   with it true spans, anchor view, overestimation, beats and reading, is the same as without
   the option (common random numbers).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq, least_squares
from scipy.special import expit, ndtr

from .estimators import a1_anchor, a2_mean, a3_max
from .experiments import read_once
from .metrics import ErrAcc
from .model import AXES, Latent, draw_latent, log_sd_from_cv, measure, n_beats_generated, per_view, zt_poisson
from .rules import a4_final

RULES = ("A1", "A2", "A3", "A4")
VIEWS = ("4CH", "inflow", "mBC")
HALF_NORMAL_MEAN = float(np.sqrt(2.0 / np.pi))     # E|Z| = 0.7979

# Published values, Singh et al. 2026, Tables 2 and 3 (n = 30); mean, lower and upper 95% CI, mm.
# The upper limit of the 3D maximal diameter is printed as 19.96 (asymmetric about 16.82).
SINGH = {
    "n": 30,
    "d3_max": (16.82, 14.68, 19.96),
    "d3_min": (8.99, 7.69, 10.29),
    "d3_avg": (12.91, 11.38, 14.44),
    "d3_sphericity": (2.04, 1.73, 2.35),
    "views": {
        "4CH": {"primary": (8.70, 7.54, 9.86), "orthogonal": (10.92, 9.06, 12.79),
                "biplane": (9.81, 8.47, 11.15), "sphericity": (1.54, 1.35, 1.72)},
        "inflow": {"primary": (11.64, 9.37, 13.91), "orthogonal": (9.33, 7.59, 11.08),
                   "biplane": (10.49, 8.79, 12.19), "sphericity": (1.78, 1.43, 2.10)},
        "mBC": {"primary": (8.53, 7.25, 9.81), "orthogonal": (10.95, 9.06, 12.85),
                "biplane": (9.74, 8.41, 11.08), "sphericity": (1.61, 1.42, 1.80)},
    },
    # Table 3: mean difference (3D minus 2D), lower and upper 95% limits of agreement
    "ba_biplane_vs_avg": {"4CH": (3.09, -3.59, 9.77), "inflow": (2.42, -2.56, 7.40), "mBC": (3.17, -2.85, 9.18)},
    "ba_single_vs_max": {"4CH": (8.11, -1.52, 17.74), "inflow": (5.18, -3.72, 14.07), "mBC": (8.29, -1.23, 17.81)},
    "ba_single_vs_min": {"4CH": (0.29, -6.65, 7.24), "inflow": (-2.64, -12.07, 6.79), "mBC": (0.47, -5.70, 6.64)},
    "r_biplane_vs_avg": {"4CH": 0.613, "inflow": 0.832, "mBC": 0.687},
}


def _gen(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


# ------------------------------------------------------------------ published summaries

def sd_from_ci(mean: float, lo: float, hi: float, n: int, side: str = "both") -> float:
    """SD implied by a t-based 95% CI of a mean of n values.

    side 'both' uses (hi - lo) / 2, 'lower' uses mean - lo, 'upper' uses hi - mean.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    hw = {"both": (hi - lo) / 2.0, "lower": mean - lo, "upper": hi - mean}[side]
    return float(hw * np.sqrt(n) / stats.t.ppf(0.975, n - 1))


def se_from_ci(mean: float, lo: float, hi: float, n: int, side: str = "both") -> float:
    return sd_from_ci(mean, lo, hi, n, side) / float(np.sqrt(n))


def orifice_targets(s: dict = SINGH) -> dict:
    """Means and derived SDs of the 3D diameters. The SD of the maximal diameter uses the lower
    half-width because the printed upper limit is asymmetric (probable misprint)."""
    n = s["n"]
    return {
        "mean_max": s["d3_max"][0], "sd_max": sd_from_ci(*s["d3_max"], n, side="lower"),
        "mean_min": s["d3_min"][0], "sd_min": sd_from_ci(*s["d3_min"], n),
        "sd_avg": sd_from_ci(*s["d3_avg"], n),
    }


# ------------------------------------------------------------------ ellipse geometry

def central_chord(dmax, dmin, theta):
    """Chord through the centre at angle theta (radians) to the major axis."""
    dmax, dmin, theta = np.asarray(dmax, float), np.asarray(dmin, float), np.asarray(theta, float)
    return 1.0 / np.sqrt(np.cos(theta) ** 2 / dmax ** 2 + np.sin(theta) ** 2 / dmin ** 2)


def half_extent(dmax, dmin, theta):
    """Half-extent of the ellipse perpendicular to a line at angle theta to the major axis."""
    dmax, dmin, theta = np.asarray(dmax, float), np.asarray(dmin, float), np.asarray(theta, float)
    return 0.5 * np.sqrt(dmax ** 2 * np.sin(theta) ** 2 + dmin ** 2 * np.cos(theta) ** 2)


def feret_width(dmax, dmin, phi):
    """Projected extent (caliper width) of the ellipse along a direction at angle phi to the major axis."""
    return 2.0 * half_extent(dmax, dmin, np.asarray(phi, float) + np.pi / 2.0)


def chord(dmax, dmin, theta, delta):
    """Chord cut by a line at angle theta and relative offset delta = |d| / half_extent; 0 if the line misses."""
    delta = np.asarray(delta, float)
    return central_chord(dmax, dmin, theta) * np.sqrt(np.clip(1.0 - delta ** 2, 0.0, None))


def midpoint_offset(dmax, dmin, theta, delta):
    """Relative offset of the orthogonal plane when it passes through the midpoint of the
    primary-plane chord (primary plane at angle theta, relative offset delta).

    The midpoint lies at signed distance t = -d sin(theta) cos(theta) (1/b^2 - 1/a^2) / P along the
    primary plane, with a = Dmax/2, b = Dmin/2, d = delta * h(theta) and
    P = cos^2(theta)/a^2 + sin^2(theta)/b^2; the result is |t| / h(theta + 90 degrees), always < 1.
    """
    dmax, dmin, theta = np.asarray(dmax, float), np.asarray(dmin, float), np.asarray(theta, float)
    a, b = dmax / 2.0, dmin / 2.0
    sn, cs = np.sin(theta), np.cos(theta)
    P = cs ** 2 / a ** 2 + sn ** 2 / b ** 2
    d = np.asarray(delta, float) * half_extent(dmax, dmin, theta)
    t = d * sn * cs * (1.0 / b ** 2 - 1.0 / a ** 2) / P
    return np.abs(t) / half_extent(dmax, dmin, theta + np.pi / 2.0)


def biplane_widths(dmax, dmin, theta, delta1, delta2=None, kappa: float = 1.0):
    """Primary-plane width, orthogonal-plane width and their average (the 'biplane' width).

    delta2 None: the orthogonal plane passes through the midpoint of the primary chord.
    kappa: rotation-independent multiplicative factor on both widths (1 = none).
    """
    if delta2 is None:
        delta2 = midpoint_offset(dmax, dmin, theta, delta1)
    p = kappa * chord(dmax, dmin, theta, delta1)
    o = kappa * chord(dmax, dmin, np.asarray(theta, float) + np.pi / 2.0, delta2)
    return p, o, 0.5 * (p + o)


# ------------------------------------------------------------------ orifice distribution

def draw_orifice(par, z1, z2):
    """(Dmax, Dmin) from two standard normals.

    par = (m, s, mq, sq, rho): Dmax = exp(m + s z1); Dmin = Dmax * expit(mq + sq (rho z1 +
    sqrt(1 - rho^2) z2)), so that Dmin < Dmax for every patient.
    """
    m, s, mq, sq, rho = par
    z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
    dmax = np.exp(m + s * z1)
    q = expit(mq + sq * (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2))
    return dmax, q * dmax


def orifice_moments(par, n_nodes: int = 60) -> dict:
    """Moments of the orifice distribution by Gauss-Hermite quadrature (deterministic)."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    Z1, Z2 = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    D, d = draw_orifice(par, Z1, Z2)
    E = lambda v: float((W * v).sum())  # noqa: E731
    a = 0.5 * (D + d)
    sd = lambda v: float(np.sqrt(max(E(v * v) - E(v) ** 2, 0.0)))  # noqa: E731
    return {"mean_max": E(D), "sd_max": sd(D), "mean_min": E(d), "sd_min": sd(d), "mean_avg": E(a),
            "sd_avg": sd(a), "corr": (E(D * d) - E(D) * E(d)) / (sd(D) * sd(d)) if sd(D) * sd(d) > 0 else np.nan,
            "mean_sphericity": E(D / d), "mean_ratio_min_max": E(d / D)}


def fit_orifice(targets: dict, n_nodes: int = 60):
    """Solve the five orifice parameters for mean and SD of Dmax and Dmin and SD of their average.

    Returns (par, moments, max absolute residual)."""
    keys = ("mean_max", "sd_max", "mean_min", "sd_min", "sd_avg")
    tg = np.array([targets[k] for k in keys])

    def unpack(x):
        return (x[0], np.exp(x[1]), x[2], np.exp(x[3]), np.tanh(x[4]))

    def f(x):
        mo = orifice_moments(unpack(x), n_nodes)
        return np.array([mo[k] for k in keys]) - tg

    cv = targets["sd_max"] / targets["mean_max"]
    x0 = [np.log(targets["mean_max"]) - 0.5 * np.log1p(cv ** 2), np.log(np.sqrt(np.log1p(cv ** 2))),
          float(np.log(targets["mean_min"] / (targets["mean_max"] - targets["mean_min"]))), np.log(0.6), 0.0]
    r = least_squares(f, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    par = unpack(r.x)
    return par, orifice_moments(par, n_nodes), float(np.abs(r.fun).max())


# ------------------------------------------------------------------ one view

def simulate_view(dmax, dmin, zt, z1, z2, theta0: float, par: float, sigma_theta: float = 0.0,
                  mechanism: str = "offset", offset_rule: str = "midpoint", orientation: str = "fixed",
                  delta_max: float = 0.95) -> dict:
    """Widths of one biplane view in simulated patients.

    Plane angle: theta = theta0 + sigma_theta * zt (orientation 'fixed'), or uniform on
    [0, pi) from zt (orientation 'uniform', theta0 and sigma_theta ignored).
    mechanism 'offset': par is the scale s of the relative offset of the primary plane,
    delta1 = min(s |z1|, delta_max); the orthogonal plane passes through the midpoint of the
    primary chord (offset_rule 'midpoint') or has its own offset delta2 = min(s |z2|, delta_max)
    (offset_rule 'independent'). mechanism 'scale': both planes are central and par is a
    rotation-independent factor kappa on both widths.
    Returns theta, delta1, delta2, primary, orthogonal, biplane and the central chords c0_primary, c0_orthogonal.
    """
    zt = np.asarray(zt, float)
    if orientation == "uniform":
        theta = np.pi * ndtr(zt)
    elif orientation == "fixed":
        theta = theta0 + sigma_theta * zt
    else:
        raise ValueError(orientation)
    if mechanism == "scale":
        d1 = np.zeros_like(zt)
        d2 = np.zeros_like(zt)
        kappa = float(par)
    elif mechanism == "offset":
        d1 = np.minimum(par * np.abs(z1), delta_max)
        if offset_rule == "midpoint":
            d2 = midpoint_offset(dmax, dmin, theta, d1)
        elif offset_rule == "independent":
            d2 = np.minimum(par * np.abs(z2), delta_max)
        else:
            raise ValueError(offset_rule)
        kappa = 1.0
    else:
        raise ValueError(mechanism)
    p, o, bp = biplane_widths(dmax, dmin, theta, d1, d2, kappa)
    return {"theta": theta, "delta1": d1, "delta2": d2, "kappa": kappa, "primary": p, "orthogonal": o,
            "biplane": bp, "c0_primary": central_chord(dmax, dmin, theta),
            "c0_orthogonal": central_chord(dmax, dmin, theta + np.pi / 2.0)}


def fit_view(dmax, dmin, zt, z1, z2, target_primary: float, target_orthogonal: float, **kw) -> dict:
    """Tune (theta0, par) of one view to the published mean primary and orthogonal widths.

    theta0 is restricted to [0, 90] degrees from the major axis (the geometry is symmetric).
    With orientation 'uniform' only par is tuned, to the mean biplane width (one parameter
    cannot match two targets; the residuals are returned). Starts from three angles and keeps
    the best. Returns theta0, par, the two residuals (mm) and `converged` (both |residual| < 0.005 mm).
    """
    mech = kw.get("mechanism", "offset")
    lo_par, hi_par, x_par = (0.3, 1.5, 0.9) if mech == "scale" else (0.0, 3.0, 0.4)

    def resid(theta0, par):
        v = simulate_view(dmax, dmin, zt, z1, z2, theta0, par, **kw)
        return np.array([v["primary"].mean() - target_primary, v["orthogonal"].mean() - target_orthogonal])

    if kw.get("orientation", "fixed") == "uniform":
        r = least_squares(lambda x: [resid(0.0, x[0]).sum()], [x_par], bounds=([lo_par], [hi_par]))
        theta0, par = np.nan, float(r.x[0])
        res = resid(0.0, par)
    else:
        best = None
        for a0 in (20.0, 45.0, 70.0):
            r = least_squares(lambda x: resid(x[0], x[1]), [np.radians(a0), x_par],
                              bounds=([0.0, lo_par], [np.pi / 2.0, hi_par]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
        theta0, par = float(best.x[0]), float(best.x[1])
        res = resid(theta0, par)
    return {"theta0": theta0, "par": par, "resid_primary": float(res[0]), "resid_orthogonal": float(res[1]),
            "converged": bool(np.abs(res).max() < 0.005)}


def _mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def ratio_shortfall(meas, ref):
    """1 - mean(meas) / mean(ref) with delta-method MCSE for paired per-patient vectors."""
    meas, ref = np.asarray(meas, float), np.asarray(ref, float)
    mr = ref.mean()
    r = meas.mean() / mr
    se = (meas - r * ref).std(ddof=1) / np.sqrt(meas.size) / abs(mr) if meas.size > 1 else np.nan
    return float(1.0 - r), float(se)


QUANTILES = (2.5, 25.0, 50.0, 75.0, 97.5)


def underestimation_rows(meas, ref, n_groups: int = 50) -> dict:
    """Per-patient fractional underestimation u = 1 - meas / ref: mean (MCSE), SD and quantiles
    (delete-a-group jackknife MCSE), and the ratio-of-means version (delta-method MCSE)."""
    u = 1.0 - np.asarray(meas, float) / np.asarray(ref, float)
    n = u.size
    out = {"n": n}
    out["mean"], out["mean_mcse"] = _mean_se(u)
    rs, rse = ratio_shortfall(meas, ref)
    out["ratio_of_means"], out["ratio_of_means_mcse"] = rs, rse

    def sv(x):
        return np.concatenate([[x.std(ddof=1)], np.percentile(x, QUANTILES)])

    full = sv(u)
    G = int(min(n_groups, n))
    if G > 1:
        edges = np.linspace(0, n, G + 1).astype(np.int64)
        reps = np.array([sv(np.concatenate([u[:edges[g]], u[edges[g + 1]:]])) for g in range(G)])
        dv = reps - reps.mean(axis=0)
        jse = np.sqrt((G - 1) / G * (dv * dv).sum(axis=0))
    else:
        jse = np.full(full.size, np.nan)
    for k, nm in enumerate(["sd"] + [f"q{q:g}" for q in QUANTILES]):
        out[nm], out[nm + "_mcse"] = float(full[k]), float(jse[k])
    out["count_negative"] = int((u < 0).sum())
    return out


def underestimation_table(dmax, dmin, v: dict) -> list[dict]:
    """Fractional underestimation of each plane of one simulated view under the two readings.

    reading 'major_axis_is_target' (R1): the anatomical axis coincides with a principal axis of
    the ellipse, so the fixed-axis maximal span is Dmax (or Dmin) and plane rotation counts as error.
    reading 'plane_on_axis' (R2): the plane is parallel to the anatomical axis and the ellipse is
    rotated, so the fixed-axis maximal span is the central chord in the plane's direction and only
    the offset (or scale factor) counts. Components: rotation = 1 - c0 / Dmax and
    off_centre = 1 - width / c0, with (1 - total) = (1 - rotation)(1 - off_centre).
    """
    rows = []
    for plane in ("primary", "orthogonal"):
        w, c0 = v[plane], v["c0_" + plane]
        for ref_name, ref, reading, comp in (
                ("3D maximal diameter", dmax, "major_axis_is_target", "total"),
                ("3D minimal diameter", dmin, "minor_axis_is_target", "total"),
                ("central chord in plane direction", c0, "plane_on_axis", "off_centre")):
            rows.append({"plane": plane, "reference": ref_name, "reading": reading, "component": comp,
                         **underestimation_rows(w, ref)})
        rows.append({"plane": plane, "reference": "3D maximal diameter", "reading": "major_axis_is_target",
                     "component": "rotation", **underestimation_rows(c0, dmax)})
        rows.append({"plane": plane, "reference": "3D minimal diameter", "reading": "minor_axis_is_target",
                     "component": "rotation", **underestimation_rows(c0, dmin)})
    rows.append({"plane": "biplane", "reference": "3D average of maximal and minimal diameter",
                 "reading": "matched_published_estimand", "component": "total",
                 **underestimation_rows(v["biplane"], 0.5 * (dmax + dmin))})
    cp = np.maximum(v["primary"], v["orthogonal"])
    rows.append({"plane": "cross-plane maximum", "reference": "3D maximal diameter",
                 "reading": "major_axis_is_target", "component": "total", **underestimation_rows(cp, dmax)})
    return rows


# ------------------------------------------------------------------ library-model matched estimand

def library_biplane_shortfall(p: dict, L: float, n: int, ss: np.random.SeedSequence, view: int = 1) -> dict:
    """Published estimand evaluated in the library model for long-axis scale L.

    One beat, one reader (reader-bias SD and caliper SD from p), no window; u_scale =
    [anchor, L, L, 1.2 L]. Biplane width B = (x_AP + x_SL) / 2 in long-axis view `view`;
    reference R = (S_AP + S_SL) / 2. Returns 1 - mean(B)/mean(R), the AP-only and SL-only
    shortfalls, and the mean underestimation fraction u, each with MCSE. The same SeedSequence
    gives common random numbers across L (the draw order does not depend on u_scale).
    """
    q = dict(p, N_beats=1, window=None, data_mode="prospective",
             u_scale=[per_view(p["u_scale"], 1)[0], L, L, 1.2 * L])
    rng = _gen(ss)
    lat = draw_latent(q, n, rng)
    rb = q["sigma_rb_mm"] * rng.standard_normal(n)
    x = measure(lat, q, rng, rb)[..., 0]                   # (n, 2, K)
    B = x[:, :, view].mean(axis=1)
    R = lat.S.mean(axis=1)
    out = {"L": float(L), "n": n}
    out["biplane_shortfall"], out["biplane_shortfall_mcse"] = ratio_shortfall(B, R)
    out["ap_shortfall"], out["ap_shortfall_mcse"] = ratio_shortfall(x[:, 0, view], lat.S[:, 0])
    out["sl_shortfall"], out["sl_shortfall_mcse"] = ratio_shortfall(x[:, 1, view], lat.S[:, 1])
    out["mean_u"], out["mean_u_mcse"] = _mean_se(lat.u[:, :, view].mean(axis=1))
    return out


def solve_library_scale(p: dict, target: float, n: int, ss: np.random.SeedSequence,
                        lo: float = 0.0, hi: float = 0.9) -> dict:
    """Long-axis scale L at which the library-model biplane shortfall equals `target` (brentq on
    common random numbers). MCSE of L by the delta method with a central finite difference."""
    f = lambda L: library_biplane_shortfall(p, L, n, ss)["biplane_shortfall"] - target  # noqa: E731
    L = float(brentq(f, lo, hi, xtol=1e-6))
    r = library_biplane_shortfall(p, L, n, ss)
    h = 0.01
    slope = (f(L + h) - f(L - h)) / (2 * h)
    r["target"] = float(target)
    r["L_mcse"] = float(r["biplane_shortfall_mcse"] / abs(slope))
    return r


# ------------------------------------------------------------------ view-rule model variant

GEOM_READINGS = ("rotation_and_offset", "offset_only")


def draw_geometry_normals(rng: np.random.Generator, n: int, K: int) -> dict:
    """Standard normals of the patient-level geometric draw, in a fixed order and of fixed shapes
    whatever the options: z1, z2 (n) for the orifice, zt (n, 2, K) for the plane angles and
    zd (n, 2, K) for offsets that do not use the library's latent normals."""
    return {"z1": rng.standard_normal(n), "z2": rng.standard_normal(n),
            "zt": rng.standard_normal((n, 2, K)), "zd": rng.standard_normal((n, 2, K))}


def geometric_underestimation(Zu: np.ndarray, S_ap: np.ndarray, p: dict, gz: dict) -> np.ndarray:
    """Underestimation fraction u (n, 2, K) with the long-axis views (index 1 and above) drawn
    patient by patient from the ellipse geometry. The anchor view (index 0) keeps the library
    expression min(u_scale_0 |Zu|, u_max).

    p['u_geom'] is a dict:
      reading          'rotation_and_offset' (reading 1: reference Dmax on the AP axis and Dmin on
                       the SL axis; plane rotation counts as error) or 'offset_only' (reading 2:
                       reference is the central chord in the plane direction)
      offset_scale     [s_AP, s_SL], scale of the relative offset, delta = min(s |z|, delta_max)
      delta_max        cap on the relative offset (default 0.95)
      offset_source    'latent' (default): z is the library's latent normal Zu (between-view
                       correlation rho); 'independent': z is gz['zd']
      theta0_deg       [AP, SL] mean plane angle to the major axis (reading 1 only)
      sigma_theta_deg  SD of the plane angle between patients and views (reading 1 only)
      orifice          [mq, sq, rho]: ellipticity ratio q = Dmin / Dmax = expit(mq + sq (rho z1 +
                       sqrt(1 - rho^2) z2)), one value per patient, shared by axes and views
                       (reading 1 only)
      ellipticity      'independent' (default): z1 = gz['z1'], independent of the true span;
                       'span_linked': z1 is the standardized log true AP span, which carries the
                       calibrated dependence between ellipticity and the maximal diameter
    With the chord of a unit-major-axis ellipse c0(theta) = central_chord(1, q, theta):
      reading 1:  u_AP = 1 - c0(theta) sqrt(1 - delta^2),  u_SL = 1 - c0(theta) sqrt(1 - delta^2) / q
      reading 2:  u = 1 - sqrt(1 - delta^2) on both axes (the chord link of `underestimation_fraction`)
    followed by u = min(u, u_max). u_SL may be negative (a rotated plane is longer than Dmin).
    """
    g = p["u_geom"]
    K = Zu.shape[-1]
    reading = g["reading"]
    if reading not in GEOM_READINGS:
        raise ValueError(f"u_geom reading: {reading}")
    src = g.get("offset_source", "latent")
    if src not in ("latent", "independent"):
        raise ValueError(f"u_geom offset_source: {src}")
    ell = g.get("ellipticity", "independent")
    if ell not in ("independent", "span_linked"):
        raise ValueError(f"u_geom ellipticity: {ell}")
    s = np.asarray(g["offset_scale"], float)
    if s.shape != (2,) or (s < 0).any():
        raise ValueError("u_geom offset_scale must be two non-negative values (AP, SL)")
    u = np.minimum(per_view(p["u_scale"], K) * np.abs(Zu), p["u_max"])
    if K < 2:
        return u
    z_off = Zu[:, :, 1:] if src == "latent" else gz["zd"][:, :, 1:]
    w = s[None, :, None] * np.abs(z_off)
    wc = 1.0 - np.sqrt(1.0 - np.minimum(w, g.get("delta_max", 0.95)) ** 2)
    if reading == "offset_only":
        u[:, :, 1:] = np.minimum(wc, p["u_max"])
        return u
    mq, sq, rho = (float(x) for x in g["orifice"])
    if not -1.0 <= rho <= 1.0 or sq < 0:
        raise ValueError("u_geom orifice: need sq >= 0 and |rho| <= 1")
    if ell == "span_linked":
        if not p["S_log_sd"] > 0:
            raise ValueError("span_linked ellipticity needs S_log_sd > 0")
        z1 = (np.log(S_ap) - np.log(p["S_median_mm"])) / p["S_log_sd"]
    else:
        z1 = gz["z1"]
    q = expit(mq + sq * (rho * z1 + np.sqrt(1.0 - rho ** 2) * gz["z2"]))
    th0 = np.radians(np.asarray(g["theta0_deg"], float))
    if th0.shape != (2,):
        raise ValueError("u_geom theta0_deg must be two values (AP, SL)")
    theta = th0[None, :, None] + np.radians(float(g["sigma_theta_deg"])) * gz["zt"][:, :, 1:]
    c0 = central_chord(1.0, q[:, None, None], theta)
    ref = np.stack([np.ones_like(q), q], axis=1)[:, :, None]
    u[:, :, 1:] = np.minimum(1.0 - (c0 / ref) * (1.0 - wc), p["u_max"])
    return u


def underestimation_fraction(Zu: np.ndarray, p: dict) -> np.ndarray:
    """Underestimation fraction u (n, 2, K) from the latent normals Zu.

    Without the keys u_scale_axis, u_shift_axis and u_link this is the library expression
    min(u_scale_v |Zu|, u_max). Otherwise, with scale s (2, K) from u_scale_axis (default: u_scale
    for both axes) and shift c (2, K) from u_shift_axis (default 0):
      u_link 'halfnormal' (default): w = s |Zu|
      u_link 'chord':                w = 1 - sqrt(1 - min(s |Zu|, delta_max)^2), delta_max = u_delta_max (0.95)
      u = min(c + (1 - c) w, u_max).
    u_link may be one string or a list with one entry per view (anchor first).
    """
    K = Zu.shape[-1]
    ext = any(k in p for k in ("u_scale_axis", "u_shift_axis", "u_link"))
    if not ext:
        return np.minimum(per_view(p["u_scale"], K) * np.abs(Zu), p["u_max"])
    if "u_scale_axis" in p:
        s = np.stack([per_view(row, K) for row in p["u_scale_axis"]])
    else:
        s = np.stack([per_view(p["u_scale"], K)] * 2)
    if s.shape != (2, K):
        raise ValueError("u_scale_axis must have one row per axis")
    c = np.stack([per_view(row, K) for row in p["u_shift_axis"]]) if "u_shift_axis" in p else np.zeros((2, K))
    link = p.get("u_link", "halfnormal")
    links = [link] * K if isinstance(link, str) else list(link)[:K]
    if len(links) != K or any(lk not in ("halfnormal", "chord") for lk in links):
        raise ValueError(f"u_link: {link}")
    is_chord = np.array([lk == "chord" for lk in links])
    w = s * np.abs(Zu)
    if is_chord.any():
        wc = 1.0 - np.sqrt(1.0 - np.minimum(w, p.get("u_delta_max", 0.95)) ** 2)
        w = np.where(is_chord, wc, w)
    return np.minimum(c + (1.0 - c) * w, p["u_max"])


def draw_latent_axis(p: dict, n: int, rng: np.random.Generator, rng_geom: np.random.Generator | None = None) -> Latent:
    """Variant of `model.draw_latent`; the only change is that u comes from `underestimation_fraction`
    or, with the key `u_geom` in `p`, from `geometric_underestimation`.

    Statement order and random-number consumption from `rng` are those of `model.draw_latent`,
    with and without `u_geom`. The geometric draw takes its additional normals from `rng_geom`
    (required when `u_geom` is present, never touched otherwise).
    """
    if "u_geom" in p:
        if rng_geom is None:
            raise ValueError("u_geom needs rng_geom")
        if any(k in p for k in ("u_scale_axis", "u_shift_axis", "u_link")):
            raise ValueError("u_geom cannot be combined with u_scale_axis, u_shift_axis or u_link")
    K = int(p["K"])
    D = 2
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
        if "u_geom" in p:
            u = geometric_underestimation(Zu, S_ap, p, draw_geometry_normals(rng_geom, n, K))
        else:
            u = underestimation_fraction(Zu, p)
    else:
        u = np.zeros((n, D, K))
    if on and p["view_over"]:
        over = ndtr(Zo) < per_view(p["p_over"], K)
        o = np.where(over, p["o_median"] * np.exp(p["o_log_sd"] * Mo), 0.0)
    else:
        over = np.zeros((n, D, K), dtype=bool)
        o = np.zeros((n, D, K))
    mu = S[:, :, None] * (1.0 - u + o)

    g = np.exp(p["g_log_sd"] * rng.standard_normal(n))

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


def simulate_rules(p: dict, ss: np.random.SeedSequence, n_total: int, chunk: int,
                   ss_geom: np.random.SeedSequence | None = None) -> dict:
    """Per-patient values of the four view rules (n, 2) and true spans S (n, 2).

    Per chunk the random numbers are consumed as in `experiments.run_E1`: latent draw, reader
    bias, caliper error (inside `read_once`), A4 decision uniforms. Also returns `acc`, ErrAcc
    accumulators per (rule, axis) for estimand T1 filled chunk by chunk as in run_E1,
    `mean_u` (2, K), the realized mean underestimation fraction per axis and view, `sd_u` (2, K),
    its between-patient SD, and `corr_u_long` (2), the correlation of u between the first two
    long-axis views (nan if K < 3).
    `ss_geom` seeds the patient-level geometric draw (key `u_geom` in `p`); its children are
    spawned per chunk like those of `ss`, and it is ignored without `u_geom`.
    """
    K = int(p["K"])
    # SeedSequence.spawn is stateful: spawn from a fresh copy so that repeated calls with the same
    # object give the same children (common random numbers) and equal those of a new cell seed.
    ss = np.random.SeedSequence(entropy=ss.entropy, spawn_key=ss.spawn_key)
    kids = ss.spawn(-(-n_total // chunk))
    geom = "u_geom" in p
    if geom:
        if ss_geom is None:
            raise ValueError("u_geom needs ss_geom")
        gkids = np.random.SeedSequence(entropy=ss_geom.entropy, spawn_key=ss_geom.spawn_key).spawn(len(kids))
    S = []
    est = {r: [] for r in RULES}
    acc = {(r, d): ErrAcc() for r in RULES for d in range(2)}
    usum = np.zeros((2, K))
    u_all = []
    for i, ch in enumerate(kids):
        n = min(chunk, n_total - i * chunk)
        rng = _gen(ch)
        lat = draw_latent_axis(p, n, rng, _gen(gkids[i]) if geom else None)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        _, br = read_once(lat, p, rng, rb)
        vm = br.value
        U = rng.random((n, 2, K - 1))
        e = {"A1": a1_anchor(vm), "A2": a2_mean(vm), "A3": a3_max(vm),
             "A4": a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])[0]}
        S.append(lat.S)
        usum += lat.u.sum(axis=0)
        u_all.append(lat.u)
        for r in RULES:
            est[r].append(e[r])
            for d in range(2):
                acc[(r, d)].add(e[r][:, d], lat.S[:, d])
    u_all = np.concatenate(u_all)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([np.corrcoef(u_all[:, d, 1], u_all[:, d, 2])[0, 1] if K >= 3 else np.nan for d in range(2)])
    return {"S": np.concatenate(S), "est": {r: np.concatenate(v) for r, v in est.items()}, "acc": acc,
            "mean_u": usum / n_total, "sd_u": u_all.std(axis=0, ddof=1) if n_total > 1 else np.full((2, K), np.nan),
            "corr_u_long": corr}


def error_metrics(e: np.ndarray) -> dict:
    """bias, RMSE, MAE and SD of a per-patient error vector, each as (value, MCSE).

    MCSE: bias sd/sqrt(n); RMSE delta method sd(e^2)/sqrt(n)/(2 RMSE); MAE sd(|e|)/sqrt(n);
    SD normal-theory sd/sqrt(2(n-1)) (as in metrics.ErrAcc).
    """
    e = np.asarray(e, float).ravel()
    n = e.size
    if n < 2:
        raise ValueError("need at least two errors")
    sd = e.std(ddof=1)
    e2 = e * e
    rmse = float(np.sqrt(e2.mean()))
    a = np.abs(e)
    return {"bias_mm": (float(e.mean()), float(sd / np.sqrt(n))),
            "rmse_mm": (rmse, float(e2.std(ddof=0) / np.sqrt(n) / (2 * rmse)) if rmse > 0 else 0.0),
            "mae_mm": (float(a.mean()), float(a.std(ddof=1) / np.sqrt(n))),
            "sd_err_mm": (float(sd), float(sd / np.sqrt(2 * (n - 1))))}


def paired_contrast(e1: np.ndarray, e0: np.ndarray) -> dict:
    """Paired differences (arm 1 minus arm 0, same patients on common random numbers) of bias,
    RMSE and MAE, each as (value, MCSE). RMSE difference: delta method on the paired mean squared errors."""
    e1, e0 = np.asarray(e1, float).ravel(), np.asarray(e0, float).ravel()
    n = e1.size
    d = e1 - e0
    m1, m0 = (e1 * e1).mean(), (e0 * e0).mean()
    r1, r0 = np.sqrt(m1), np.sqrt(m0)
    infl = (e1 * e1 - m1) / (2 * r1) - (e0 * e0 - m0) / (2 * r0) if r1 > 0 and r0 > 0 else np.zeros(n)
    da = np.abs(e1) - np.abs(e0)
    return {"bias_mm": (float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))),
            "rmse_mm": (float(r1 - r0), float(infl.std(ddof=1) / np.sqrt(n))),
            "mae_mm": (float(da.mean()), float(da.std(ddof=1) / np.sqrt(n)))}


__all__ = ["RULES", "VIEWS", "AXES", "SINGH", "HALF_NORMAL_MEAN", "QUANTILES", "sd_from_ci", "se_from_ci",
           "orifice_targets", "central_chord", "half_extent", "feret_width", "chord", "midpoint_offset",
           "biplane_widths", "draw_orifice", "orifice_moments", "fit_orifice", "simulate_view", "fit_view",
           "ratio_shortfall", "underestimation_rows", "underestimation_table", "library_biplane_shortfall",
           "solve_library_scale", "underestimation_fraction", "GEOM_READINGS", "draw_geometry_normals",
           "geometric_underestimation", "draw_latent_axis", "simulate_rules",
           "error_metrics", "paired_contrast"]
