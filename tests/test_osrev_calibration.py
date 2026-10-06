"""Tests for code/lib/duomaxsim/osrev_calibration.py (protocol amendment 3, package A3-6).

Identity with the existing library when the new options are off, known answers and edge cases
for every new function. The last section covers the patient-level geometric draw added on
2026-10-06 (key u_geom, default off).
"""
import dataclasses

import numpy as np
import pytest

from duomaxsim import config as C
from duomaxsim import osrev_calibration as K
from duomaxsim.experiments import run_E1
from duomaxsim.model import draw_latent

from conftest import QUIET, ROOT

AMEND2 = ROOT / "code" / "configs" / "amend2_2026-09-18.yaml"


def _rng(seed=11):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _assert_latent_equal(a, b):
    for f in dataclasses.fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name), err_msg=f.name)


# ------------------------------------------------------------------ identity with the library

@pytest.mark.parametrize("over", [
    {},
    {"K": 4, "rhythm": "AF", "window": None},
    {"K": 2, "data_mode": "retrospective", "beat_noise_centering": "mean"},
    {"K": 1, "view_over": False},
    {"view_under": False},
    {"view_errors": False, "N_beats": 5},
    {"u_scale": [0.25, 0.5, 0.5, 0.6], "S_median_mm": 13.0},
])
def test_draw_latent_axis_bit_identical_when_options_off(base, over):
    p = dict(base, **over)
    r1, r2 = _rng(), _rng()
    a, b = draw_latent(p, 500, r1), K.draw_latent_axis(p, 500, r2)
    _assert_latent_equal(a, b)
    assert r1.random() == r2.random()            # same random-number consumption


def test_draw_latent_axis_equal_rows_reproduce_library(base):
    """Extension keys present but neutral: same values as the library."""
    K_ = int(base["K"])
    p = dict(base, u_scale_axis=[base["u_scale"], base["u_scale"]], u_shift_axis=[[0.0] * 4, [0.0] * 4],
             u_link="halfnormal")
    a, b = draw_latent(base, 400, _rng()), K.draw_latent_axis(p, 400, _rng())
    _assert_latent_equal(a, b)
    assert a.u.shape == (400, 2, K_)


def test_draw_latent_axis_changes_only_the_named_axis(base):
    p = dict(base, u_scale_axis=[base["u_scale"], [0.06, 0.02, 0.02, 0.02]])
    a, b = draw_latent(base, 400, _rng()), K.draw_latent_axis(p, 400, _rng())
    for nm in ("u", "mu", "spans"):
        np.testing.assert_array_equal(getattr(a, nm)[:, 0], getattr(b, nm)[:, 0], err_msg=nm)
    np.testing.assert_array_equal(a.u[:, 1, 0], b.u[:, 1, 0])          # anchor unchanged
    assert b.u[:, 1, 1].mean() < 0.2 * a.u[:, 1, 1].mean()
    np.testing.assert_array_equal(a.S, b.S)
    np.testing.assert_array_equal(a.g, b.g)


def test_simulate_rules_bit_identical_to_run_E1():
    cfg = C.load_config(AMEND2)
    grids = {k: v.get("grid", [v["value"]]) for k, v in cfg["parameters"].items()}
    for ci in (36, 757):                           # no window; window 0.15 with K = 2
        p = C.cell_params(cfg, "E1", ci)
        rows = run_E1(p, grids, C.cell_seed(cfg, "E1", ci), 3000, 1000)
        sim = K.simulate_rules(p, C.cell_seed(cfg, "E1", ci), 3000, 1000)
        n_checked = 0
        for r in rows:
            if r.get("estimand") != "T1" or r["estimator"] not in K.RULES:
                continue
            if r["estimator"] == "A4" and not (r["s_det"] == p["s_det"] and r["f_rej"] == p["f_rej"]):
                continue
            val, se = sim["acc"][(r["estimator"], K.AXES.index(r["axis"]))].summary()[r["metric"]]
            assert val == r["value"] and se == r["mcse"], (ci, r["estimator"], r["axis"], r["metric"])
            n_checked += 1
        assert n_checked == 4 * 2 * 4
        # per-patient values agree with the accumulators
        e = sim["est"]["A3"][:, 0] - sim["S"][:, 0]
        assert abs(e.mean() - sim["acc"][("A3", 0)].summary()["bias_mm"][0]) < 1e-12


def test_underestimation_fraction_off_is_library_expression(base):
    Zu = _rng().standard_normal((50, 2, 4))
    p = dict(base, K=4)
    lib = np.minimum(np.array(p["u_scale"]) * np.abs(Zu), p["u_max"])
    np.testing.assert_array_equal(K.underestimation_fraction(Zu, p), lib)


def test_underestimation_fraction_known_answers_and_edges(base):
    Zu = np.full((1, 2, 3), 2.0)
    p = dict(base, K=3, u_scale_axis=[[0.1, 0.3, 0.3], [0.1, 0.3, 0.3]], u_link=["halfnormal", "chord", "chord"])
    u = K.underestimation_fraction(Zu, p)
    np.testing.assert_allclose(u[0, 0], [0.2, 0.2, 0.2], atol=1e-15)   # 1 - sqrt(1 - 0.6^2) = 0.2
    p2 = dict(p, u_shift_axis=[[0.0, 0.25, 0.25], [0.0, -0.1, -0.1]])
    u2 = K.underestimation_fraction(Zu, p2)
    np.testing.assert_allclose(u2[0, 0], [0.2, 0.25 + 0.75 * 0.2, 0.4], atol=1e-15)
    np.testing.assert_allclose(u2[0, 1], [0.2, -0.1 + 1.1 * 0.2, 0.12], atol=1e-15)   # negative shift allowed
    big = K.underestimation_fraction(np.full((1, 2, 3), 50.0), dict(p, u_link="halfnormal"))
    assert (big == p["u_max"]).all()                                    # cap u_max
    capped = K.underestimation_fraction(np.full((1, 2, 3), 50.0), dict(p, u_link="chord", u_delta_max=0.6))
    np.testing.assert_allclose(capped, 0.2, atol=1e-15)                 # cap on the relative offset
    with pytest.raises(ValueError):
        K.underestimation_fraction(Zu, dict(p, u_link="linear"))
    with pytest.raises(ValueError):
        K.underestimation_fraction(Zu, dict(p, u_link=["chord"]))


# ------------------------------------------------------------------ published summaries

def test_sd_from_ci_known_answer_and_edge():
    from scipy import stats
    t = stats.t.ppf(0.975, 29)
    assert K.sd_from_ci(10.0, 9.0, 11.0, 30) == pytest.approx(np.sqrt(30) / t)
    assert K.sd_from_ci(16.82, 14.68, 19.96, 30, side="lower") == pytest.approx(2.14 * np.sqrt(30) / t)
    assert K.sd_from_ci(16.82, 14.68, 19.96, 30, side="upper") == pytest.approx(3.14 * np.sqrt(30) / t)
    assert K.se_from_ci(10.0, 9.0, 11.0, 30) == pytest.approx(1 / t)
    with pytest.raises(ValueError):
        K.sd_from_ci(1.0, 0.0, 2.0, 1)


def test_orifice_targets_values():
    tg = K.orifice_targets()
    assert tg["mean_max"] == 16.82 and tg["mean_min"] == 8.99
    assert tg["sd_max"] == pytest.approx(5.731, abs=1e-3)
    assert tg["sd_min"] == pytest.approx(3.481, abs=1e-3)
    assert tg["sd_avg"] == pytest.approx(4.097, abs=1e-3)


# ------------------------------------------------------------------ geometry

def _brute_chord(dmax, dmin, theta, d):
    """Independent calculation: intersect the line with the ellipse by solving the quadratic."""
    a, b = dmax / 2, dmin / 2
    e = np.array([np.cos(theta), np.sin(theta)])
    nrm = np.array([-np.sin(theta), np.cos(theta)])
    p0 = d * nrm
    A = e[0] ** 2 / a ** 2 + e[1] ** 2 / b ** 2
    B = 2 * (p0[0] * e[0] / a ** 2 + p0[1] * e[1] / b ** 2)
    Cc = p0[0] ** 2 / a ** 2 + p0[1] ** 2 / b ** 2 - 1
    disc = B * B - 4 * A * Cc
    if disc <= 0:
        return 0.0, np.nan
    t1, t2 = (-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)
    return t2 - t1, 0.5 * (t1 + t2)


def test_central_chord_half_extent_feret_known_answers():
    assert K.central_chord(16.0, 8.0, 0.0) == pytest.approx(16.0)
    assert K.central_chord(16.0, 8.0, np.pi / 2) == pytest.approx(8.0)
    assert K.central_chord(10.0, 10.0, 0.7) == pytest.approx(10.0)          # circle
    assert K.half_extent(16.0, 8.0, 0.0) == pytest.approx(4.0)
    assert K.half_extent(16.0, 8.0, np.pi / 2) == pytest.approx(8.0)
    assert K.feret_width(16.0, 8.0, 0.0) == pytest.approx(16.0)
    assert K.feret_width(16.0, 8.0, np.pi / 2) == pytest.approx(8.0)
    # 45 degrees: central chord 2ab sqrt(2/(a^2+b^2)); the Feret width is larger
    assert K.central_chord(16.0, 8.0, np.pi / 4) == pytest.approx(2 * 8 * 4 * np.sqrt(2 / 80))
    assert K.feret_width(16.0, 8.0, np.pi / 4) == pytest.approx(np.sqrt(2 * 80))
    th = np.linspace(0, np.pi, 13)
    inv = 1 / K.central_chord(16.82, 8.99, th) ** 2 + 1 / K.central_chord(16.82, 8.99, th + np.pi / 2) ** 2
    np.testing.assert_allclose(inv, 1 / 16.82 ** 2 + 1 / 8.99 ** 2, rtol=1e-12)   # orthogonal-chord invariant


def test_chord_matches_brute_force_and_edges():
    rng = _rng(3)
    for _ in range(200):
        dmax = rng.uniform(5, 30)
        dmin = dmax * rng.uniform(0.2, 1.0)
        th = rng.uniform(0, np.pi)
        delta = rng.uniform(0, 0.99)
        d = delta * K.half_extent(dmax, dmin, th)
        length, mid = _brute_chord(dmax, dmin, th, d)
        assert K.chord(dmax, dmin, th, delta) == pytest.approx(length, rel=1e-9)
        h2 = K.half_extent(dmax, dmin, th + np.pi / 2)
        assert K.midpoint_offset(dmax, dmin, th, delta) == pytest.approx(abs(mid) / h2, abs=1e-9)
        assert K.midpoint_offset(dmax, dmin, th, delta) < 1.0
    assert K.chord(16.0, 8.0, 0.3, 0.6) == pytest.approx(0.8 * K.central_chord(16.0, 8.0, 0.3))
    assert K.chord(16.0, 8.0, 0.3, 1.0) == 0.0            # tangent plane
    assert K.chord(16.0, 8.0, 0.3, 1.7) == 0.0            # plane misses the orifice
    assert K.midpoint_offset(16.0, 8.0, 0.0, 0.5) == pytest.approx(0.0)       # aligned with an axis
    assert K.midpoint_offset(10.0, 10.0, 0.6, 0.5) == pytest.approx(0.0)      # circle


def test_biplane_widths_known_answers():
    p, o, bp = K.biplane_widths(16.0, 8.0, 0.0, 0.0)
    assert (p, o, bp) == pytest.approx((16.0, 8.0, 12.0))
    p, o, bp = K.biplane_widths(16.0, 8.0, 0.0, 0.6, 0.8, kappa=0.5)
    assert (p, o, bp) == pytest.approx((0.5 * 16 * 0.8, 0.5 * 8 * 0.6, 0.5 * (6.4 + 2.4)))
    p, o, _ = K.biplane_widths(16.0, 8.0, np.pi / 2, 0.0)                     # primary on the minor axis
    assert (p, o) == pytest.approx((8.0, 16.0))


# ------------------------------------------------------------------ orifice distribution and fits

def test_fit_orifice_reproduces_targets_and_ordering():
    tg = K.orifice_targets()
    par, mo, res = K.fit_orifice(tg)
    assert res < 1e-6
    for k in tg:
        assert mo[k] == pytest.approx(tg[k], abs=1e-6)
    assert mo["mean_avg"] == pytest.approx(0.5 * (16.82 + 8.99), abs=1e-6)
    z = _rng(5).standard_normal((2, 20000))
    dmax, dmin = K.draw_orifice(par, z[0], z[1])
    assert (dmin < dmax).all() and (dmin > 0).all()
    assert dmax.mean() == pytest.approx(16.82, abs=0.15)
    # edge: zero spread gives a constant orifice
    d1, d2 = K.draw_orifice((np.log(10.0), 0.0, 0.0, 0.0, 0.0), z[0][:5], z[1][:5])
    np.testing.assert_allclose(d1, 10.0)
    np.testing.assert_allclose(d2, 5.0)


def test_simulate_view_known_answers_and_edges():
    n = 1000
    z = _rng(7).standard_normal((3, n))
    dmax, dmin = np.full(n, 16.0), np.full(n, 8.0)
    v = K.simulate_view(dmax, dmin, z[0], z[1], z[2], 0.0, 0.0)
    np.testing.assert_allclose(v["primary"], 16.0)
    np.testing.assert_allclose(v["orthogonal"], 8.0)
    np.testing.assert_allclose(v["biplane"], 12.0)
    v = K.simulate_view(dmax, dmin, z[0], z[1], z[2], np.pi / 2, 0.9, mechanism="scale")
    np.testing.assert_allclose(v["primary"], 0.9 * 8.0)
    np.testing.assert_allclose(v["orthogonal"], 0.9 * 16.0)
    v = K.simulate_view(dmax, dmin, z[0], z[1], z[2], 0.3, 5.0, offset_rule="independent", delta_max=0.6)
    assert (v["delta1"] <= 0.6).all() and (v["primary"] >= 0.8 * v["c0_primary"] - 1e-12).all()
    v = K.simulate_view(dmax, dmin, z[0], z[1], z[2], 0.0, 0.0, orientation="uniform")
    assert v["theta"].min() >= 0 and v["theta"].max() <= np.pi
    assert 8.0 - 1e-9 <= v["primary"].min() and v["primary"].max() <= 16.0 + 1e-9
    for bad in (dict(mechanism="x"), dict(offset_rule="x"), dict(orientation="x")):
        with pytest.raises(ValueError):
            K.simulate_view(dmax, dmin, z[0], z[1], z[2], 0.0, 0.1, **bad)


@pytest.mark.parametrize("kw,theta0,par", [
    (dict(mechanism="offset", offset_rule="midpoint", sigma_theta=0.2), np.radians(30.0), 0.45),
    (dict(mechanism="offset", offset_rule="independent", sigma_theta=0.0), np.radians(60.0), 0.35),
    (dict(mechanism="scale", sigma_theta=0.1), np.radians(40.0), 0.88),
])
def test_fit_view_recovers_known_parameters(kw, theta0, par):
    n = 20000
    z = _rng(9).standard_normal((5, n))
    opar, _, _ = K.fit_orifice(K.orifice_targets())
    dmax, dmin = K.draw_orifice(opar, z[0], z[1])
    truth = K.simulate_view(dmax, dmin, z[2], z[3], z[4], theta0, par, **kw)
    f = K.fit_view(dmax, dmin, z[2], z[3], z[4], truth["primary"].mean(), truth["orthogonal"].mean(), **kw)
    assert f["converged"]
    assert f["theta0"] == pytest.approx(theta0, abs=1e-3)
    assert f["par"] == pytest.approx(par, abs=1e-3)


def test_fit_view_reports_failure_for_unreachable_targets():
    """Edge: no plane through an ellipse can exceed the major diameter in both planes."""
    n = 5000
    z = _rng(9).standard_normal((5, n))
    dmax, dmin = np.full(n, 16.0), np.full(n, 8.0)
    f = K.fit_view(dmax, dmin, z[2], z[3], z[4], 20.0, 20.0)
    assert not f["converged"] and f["resid_primary"] < 0
    f = K.fit_view(dmax, dmin, z[2], z[3], z[4], 10.0, 9.0, orientation="uniform", offset_rule="independent")
    assert np.isnan(f["theta0"])
    assert abs(f["resid_primary"] + f["resid_orthogonal"]) < 1e-3     # only the sum can be matched


def test_underestimation_rows_and_table_known_answers():
    ref = np.linspace(5, 20, 400)
    r = K.underestimation_rows(0.8 * ref, ref)
    assert r["mean"] == pytest.approx(0.2) and r["ratio_of_means"] == pytest.approx(0.2)
    assert r["sd"] == pytest.approx(0.0, abs=1e-12) and r["q50"] == pytest.approx(0.2)
    assert r["mean_mcse"] == pytest.approx(0.0, abs=1e-12) and r["count_negative"] == 0 and r["n"] == 400
    assert K.underestimation_rows(1.1 * ref, ref)["count_negative"] == 400
    n = 400
    z = _rng(2).standard_normal((3, n))
    dmax, dmin = np.full(n, 16.0), np.full(n, 8.0)
    v = K.simulate_view(dmax, dmin, z[0], z[1], z[2], np.pi / 4, 0.0)          # central planes at 45 degrees
    tab = {(t["plane"], t["reference"], t["component"]): t for t in K.underestimation_table(dmax, dmin, v)}
    c45 = 2 * 8 * 4 * np.sqrt(2 / 80)
    assert tab[("primary", "3D maximal diameter", "total")]["mean"] == pytest.approx(1 - c45 / 16)
    assert tab[("primary", "3D maximal diameter", "rotation")]["mean"] == pytest.approx(1 - c45 / 16)
    assert tab[("primary", "central chord in plane direction", "off_centre")]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert tab[("primary", "3D minimal diameter", "total")]["mean"] == pytest.approx(1 - c45 / 8)   # negative
    assert tab[("biplane", "3D average of maximal and minimal diameter", "total")]["mean"] == pytest.approx(1 - c45 / 12)
    assert tab[("cross-plane maximum", "3D maximal diameter", "total")]["mean"] == pytest.approx(1 - c45 / 16)
    assert len(tab) == 12


def test_ratio_shortfall_known_answer_and_edge():
    v, se = K.ratio_shortfall(np.array([8.0, 16.0]), np.array([10.0, 20.0]))
    assert v == pytest.approx(0.2) and se == pytest.approx(0.0, abs=1e-12)
    v, se = K.ratio_shortfall(np.array([8.0]), np.array([10.0]))
    assert v == pytest.approx(0.2) and np.isnan(se)


# ------------------------------------------------------------------ library-model match

def test_library_biplane_shortfall_known_answers(base):
    p = dict(base, **QUIET, K=3)
    ss = np.random.SeedSequence(5)
    r0 = K.library_biplane_shortfall(p, 0.0, 2000, ss)
    assert r0["biplane_shortfall"] == pytest.approx(0.0, abs=1e-12)           # edge: no error at all
    assert r0["mean_u"] == 0.0
    p1 = dict(p, view_errors=True, view_over=False)
    r = K.library_biplane_shortfall(p1, 0.25, 200000, ss)
    hn = K.HALF_NORMAL_MEAN * 0.25                                             # cap u_max is not reached in practice
    assert r["mean_u"] == pytest.approx(hn, abs=5 * r["mean_u_mcse"])
    assert r["ap_shortfall"] == pytest.approx(hn, abs=5 * r["ap_shortfall_mcse"])
    assert r["biplane_shortfall"] == pytest.approx(hn, abs=5 * r["biplane_shortfall_mcse"])
    # common random numbers: identical call gives identical output
    assert K.library_biplane_shortfall(p1, 0.25, 2000, ss) == K.library_biplane_shortfall(p1, 0.25, 2000, ss)


def test_solve_library_scale_hits_target(base):
    p = dict(base, K=3, window=None)
    ss = np.random.SeedSequence(6)
    r = K.solve_library_scale(p, 0.2, 5000, ss)
    assert r["biplane_shortfall"] == pytest.approx(0.2, abs=1e-5)
    assert 0.2 < r["L"] < 0.4 and r["L_mcse"] > 0
    with pytest.raises(ValueError):                                             # edge: target outside the bracket
        K.solve_library_scale(p, 0.99, 2000, ss)


# ------------------------------------------------------------------ error summaries

def test_error_metrics_known_answer_and_edge():
    m = K.error_metrics(np.array([1.0, -1.0, 3.0, -3.0]))
    assert m["bias_mm"][0] == 0.0
    assert m["rmse_mm"][0] == pytest.approx(np.sqrt(5.0))
    assert m["mae_mm"][0] == pytest.approx(2.0)
    assert m["sd_err_mm"][0] == pytest.approx(np.sqrt(20.0 / 3.0))
    assert m["bias_mm"][1] == pytest.approx(np.sqrt(20.0 / 3.0) / 2.0)
    assert K.error_metrics(np.zeros(5))["rmse_mm"] == (0.0, 0.0)
    with pytest.raises(ValueError):
        K.error_metrics(np.array([1.0]))


def test_paired_contrast_known_answer_and_edge():
    e0 = np.array([1.0, -1.0, 3.0, -3.0])
    same = K.paired_contrast(e0, e0)
    for k in ("bias_mm", "rmse_mm", "mae_mm"):
        assert same[k] == (0.0, 0.0)
    c = K.paired_contrast(e0 + 1.0, e0)
    assert c["bias_mm"] == (1.0, 0.0)                                           # constant shift: no Monte Carlo error
    assert c["rmse_mm"][0] == pytest.approx(np.sqrt(6.0) - np.sqrt(5.0))
    assert c["mae_mm"][0] == pytest.approx(0.0)              # |e0 + 1| = 2, 0, 4, 2: mean 2, as for e0
    assert K.paired_contrast(e0 + 4.0, e0)["mae_mm"][0] == pytest.approx(2.0)   # 5, 3, 7, 1: mean 4


def test_simulate_rules_common_random_numbers(base):
    """Repeated calls with one SeedSequence object give identical draws (SeedSequence.spawn is
    stateful), and an extension that leaves the AP axis unchanged leaves the AP values unchanged."""
    p = dict(base, K=3, window=None)
    ss = np.random.SeedSequence(entropy=20261005, spawn_key=(76, 2, 0))
    a = K.simulate_rules(p, ss, 1500, 500)
    b = K.simulate_rules(p, ss, 1500, 500)
    for r in K.RULES:
        np.testing.assert_array_equal(a["est"][r], b["est"][r])
    c = K.simulate_rules(dict(p, u_scale_axis=[p["u_scale"], [0.06, 0.02, 0.02, 0.02]]), ss, 1500, 500)
    np.testing.assert_array_equal(a["S"], c["S"])
    for r in K.RULES:
        np.testing.assert_array_equal(a["est"][r][:, 0], c["est"][r][:, 0])
    assert c["mean_u"][1, 1] < 0.2 * a["mean_u"][1, 1]
    np.testing.assert_allclose(a["mean_u"][0], c["mean_u"][0])


# ------------------------------------------------------------------ follow-up 2026-10-06: patient-level geometric draw

# ind15 calibration (results/2026-10-05_osrev/calibration: geom_fit.csv, orifice_fit.csv)
GEOM_IND15 = dict(offset_scale=[0.3844257914, 0.5058149598], delta_max=0.95, theta0_deg=[31.75148, 58.72762],
                  sigma_theta_deg=15.0, orifice=[0.250354, 0.7867732, -0.3577115])


# regression fixture of test_simulate_rules_geometric_common_random_numbers_and_regression:
# AP bias of the largest view mean, of the mean across views (mm) and mean long-axis u, computed 2026-10-06
REGRESSION_GEOM = (0.13519244593382915, -1.90917668186951, 0.29291461313964057)


def _gz(n, K_, seed=5):
    return K.draw_geometry_normals(_rng(seed), n, K_)


@pytest.mark.parametrize("over", [{}, {"K": 4, "rhythm": "AF", "window": None}, {"view_under": False},
                                  {"u_scale_axis": [[0.06, 0.3, 0.3, 0.3], [0.06, 0.1, 0.1, 0.1]], "u_link": "chord"}])
def test_geometry_generator_untouched_when_option_off(base, over):
    """Default-off: with no u_geom key a supplied geometry generator is not consumed and every
    array equals the call without it (and the library, see the tests above)."""
    p = dict(base, **over)
    rg, rg0 = _rng(3), _rng(3)
    a, b = K.draw_latent_axis(p, 300, _rng()), K.draw_latent_axis(p, 300, _rng(), rg)
    _assert_latent_equal(a, b)
    assert rg.random() == rg0.random()


def test_simulate_rules_outputs_unchanged_by_geometry_seed_when_option_off(base):
    p = dict(base, K=3, window=None)
    ss = np.random.SeedSequence(entropy=20261005, spawn_key=(76, 2, 0))
    a = K.simulate_rules(p, ss, 1200, 500)
    b = K.simulate_rules(p, ss, 1200, 500, np.random.SeedSequence(entropy=1, spawn_key=(9,)))
    np.testing.assert_array_equal(a["S"], b["S"])
    np.testing.assert_array_equal(a["mean_u"], b["mean_u"])
    for r in K.RULES:
        np.testing.assert_array_equal(a["est"][r], b["est"][r])
        for d in range(2):
            assert a["acc"][(r, d)].summary() == b["acc"][(r, d)].summary()
    # new summary keys: SD and between-view correlation of u, consistent with a direct draw
    assert a["sd_u"].shape == (2, 3) and a["corr_u_long"].shape == (2,)
    assert 0.0 < a["corr_u_long"][0] < 0.3                 # latent correlation 0.3 of the normals, less for |Z|
    assert np.isnan(K.simulate_rules(dict(p, K=2), ss, 600, 300)["corr_u_long"]).all()


def test_geometric_draw_leaves_main_stream_unchanged(base):
    """u_geom on: the main generator is consumed exactly as without it; true spans, anchor view,
    overestimation, instrument factor and RR intervals are identical; long-axis u differs."""
    p = dict(base, K=3)
    pg = dict(p, u_geom=dict(GEOM_IND15, reading="rotation_and_offset"))
    r1, r2 = _rng(), _rng()
    a, b = K.draw_latent_axis(p, 500, r1), K.draw_latent_axis(pg, 500, r2, _rng(5))
    assert r1.random() == r2.random()
    for nm in ("S", "r", "o", "over", "g", "logrr", "avail"):
        np.testing.assert_array_equal(getattr(a, nm), getattr(b, nm), err_msg=nm)
    np.testing.assert_array_equal(a.u[:, :, 0], b.u[:, :, 0])
    np.testing.assert_array_equal(a.spans[:, :, 0], b.spans[:, :, 0])
    assert not np.array_equal(a.u[:, :, 1:], b.u[:, :, 1:])
    np.testing.assert_allclose(b.mu, b.S[:, :, None] * (1.0 - b.u + b.o), rtol=0, atol=0)
    # same geometry seed, same draw
    c = K.draw_latent_axis(pg, 500, _rng(), _rng(5))
    _assert_latent_equal(b, c)


def test_geometric_offset_only_is_the_chord_link(base):
    """Reading 2 with offsets from the latent normals is bit-identical to the stage B chord
    scenario with the same offset scales (no rotation, no ellipticity)."""
    p = dict(base, K=3)
    sa, ss_ = GEOM_IND15["offset_scale"]
    chord = dict(p, u_scale_axis=[[0.06, sa, sa, sa], [0.06, ss_, ss_, ss_]],
                 u_link=["halfnormal", "chord", "chord", "chord"], u_delta_max=0.95)
    geom = dict(p, u_scale=[0.06, 0.0, 0.0, 0.0], u_geom=dict(GEOM_IND15, reading="offset_only"))
    a, b = K.draw_latent_axis(chord, 800, _rng()), K.draw_latent_axis(geom, 800, _rng(), _rng(5))
    _assert_latent_equal(a, b)


def test_geometric_underestimation_known_answers(base):
    n, K_ = 4, 3
    p = dict(base, K=K_, u_scale=[0.1, 0.0, 0.0, 0.0])
    gz = {"z1": np.zeros(n), "z2": np.zeros(n), "zt": np.zeros((n, 2, K_)), "zd": np.full((n, 2, K_), 2.0)}
    S_ap = np.full(n, 10.0)
    g0 = dict(reading="rotation_and_offset", offset_scale=[0.3, 0.3], orifice=[0.0, 0.0, 0.0], sigma_theta_deg=0.0)
    Zu = np.zeros((n, 2, K_))
    # q = expit(0) = 0.5; plane on the major axis, central: AP exact, SL chord is twice Dmin
    u = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(g0, theta0_deg=[0.0, 0.0])), gz)
    np.testing.assert_allclose(u[0, 0], [0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(u[0, 1], [0.0, -1.0, -1.0], atol=1e-15)
    # plane on the minor axis: AP reads Dmin (u = 1 - q = 0.5), SL exact
    u = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(g0, theta0_deg=[90.0, 90.0])), gz)
    np.testing.assert_allclose(u[0, 0, 1:], 0.5, atol=1e-12)
    np.testing.assert_allclose(u[0, 1, 1:], 0.0, atol=1e-12)
    # relative offset 0.3 * 2 = 0.6 shortens every chord by the factor 0.8; anchor = 0.1 * |Zu|
    Z2 = np.full((n, 2, K_), 2.0)
    u = K.geometric_underestimation(Z2, S_ap, dict(p, u_geom=dict(g0, theta0_deg=[0.0, 90.0])), gz)
    np.testing.assert_allclose(u[0, 0], [0.2, 0.2, 0.2], atol=1e-12)
    np.testing.assert_allclose(u[0, 1], [0.2, 0.2, 0.2], atol=1e-12)
    # independent offsets come from zd, not from Zu
    u = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(g0, theta0_deg=[0.0, 90.0], offset_source="independent")), gz)
    np.testing.assert_allclose(u[0, :, 1:], 0.2, atol=1e-12)
    np.testing.assert_allclose(u[0, :, 0], 0.0, atol=1e-15)
    # offset_only ignores angle and ellipticity
    u = K.geometric_underestimation(Z2, S_ap, dict(p, u_geom=dict(g0, reading="offset_only", theta0_deg=[45.0, 45.0])), gz)
    np.testing.assert_allclose(u[0, :, 1:], 0.2, atol=1e-15)


def test_geometric_underestimation_matches_chord_function(base):
    """Reading 1 equals 1 - chord / Dmax (AP) and 1 - chord / Dmin (SL) from the tested geometry functions."""
    n, K_ = 2000, 3
    p = dict(base, K=K_, u_scale=[0.06, 0.0, 0.0, 0.0])
    Zu, gz = _rng(1).standard_normal((n, 2, K_)), _gz(n, K_)
    S_ap = np.exp(np.log(p["S_median_mm"]) + p["S_log_sd"] * _rng(2).standard_normal(n))
    g = dict(GEOM_IND15, reading="rotation_and_offset")
    u = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=g), gz)
    mq, sq, rho = g["orifice"]
    dmax = 17.0 * np.ones(n)
    dmin = dmax / (1.0 + np.exp(-(mq + sq * (rho * gz["z1"] + np.sqrt(1 - rho ** 2) * gz["z2"]))))
    for d, ref in ((0, dmax), (1, dmin)):
        for k in (1, 2):
            th = np.radians(g["theta0_deg"][d]) + np.radians(15.0) * gz["zt"][:, d, k]
            c = K.chord(dmax, dmin, th, np.minimum(g["offset_scale"][d] * np.abs(Zu[:, d, k]), 0.95))
            np.testing.assert_allclose(u[:, d, k], np.minimum(1.0 - c / ref, p["u_max"]), atol=1e-12)
    assert (u[:, 1, 1:] < 0).mean() > 0.3                      # rotated planes are longer than Dmin
    # span-linked ellipticity: z1 is the standardized log span
    us = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(g, ellipticity="span_linked")), gz)
    z1 = (np.log(S_ap) - np.log(p["S_median_mm"])) / p["S_log_sd"]
    dmin_s = dmax / (1.0 + np.exp(-(mq + sq * (rho * z1 + np.sqrt(1 - rho ** 2) * gz["z2"]))))
    th = np.radians(g["theta0_deg"][0]) + np.radians(15.0) * gz["zt"][:, 0, 1]
    c = K.chord(dmax, dmin_s, th, np.minimum(g["offset_scale"][0] * np.abs(Zu[:, 0, 1]), 0.95))
    np.testing.assert_allclose(us[:, 0, 1], 1.0 - c / dmax, atol=1e-12)


def test_geometric_draw_reproduces_calibrated_distribution(base):
    """Between-patient mean and SD of the AP long-axis fraction equal the calibrated values of
    calibration_summary.csv (ind15: mean 0.2951, SD 0.1904; off-centre 0.0836, SD 0.1267), which
    the constant-rotation scenario did not (SD 0.098)."""
    n, K_ = 200_000, 3
    p = dict(base, K=K_, u_scale=[0.06, 0.0, 0.0, 0.0])
    Zu, gz = _rng(1).standard_normal((n, 2, K_)), _gz(n, K_)
    S_ap = np.full(n, 10.0)
    for src in ("latent", "independent"):
        u = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(GEOM_IND15, reading="rotation_and_offset", offset_source=src)), gz)
        assert u[:, 0, 1].mean() == pytest.approx(0.2951, abs=0.003)
        assert u[:, 0, 1].std() == pytest.approx(0.1904, abs=0.003)
        assert u[:, 1, 1].mean() == pytest.approx(0.0240, abs=0.004)
        assert u[:, 1, 1].std() == pytest.approx(0.2547, abs=0.004)
    u2 = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(GEOM_IND15, reading="offset_only")), gz)
    assert u2[:, 0, 1].mean() == pytest.approx(0.0836, abs=0.002)
    assert u2[:, 0, 1].std() == pytest.approx(0.1267, abs=0.002)


def test_geometric_underestimation_edges(base):
    n = 3
    p = dict(base, K=3, u_scale=[0.06, 0.0, 0.0, 0.0])
    gz = _gz(n, 3)
    Zu, S_ap = np.full((n, 2, 3), 50.0), np.full(n, 10.0)
    g = dict(GEOM_IND15, reading="rotation_and_offset")
    u = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=g), gz)
    assert (u[:, 0] <= p["u_max"]).all() and (u[:, :, 0] == p["u_max"]).all()      # caps
    capped = K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=dict(g, reading="offset_only", delta_max=0.6)), gz)
    np.testing.assert_allclose(capped[:, :, 1:], 0.2, atol=1e-15)
    one = K.geometric_underestimation(np.ones((n, 2, 1)), S_ap, dict(p, K=1, u_geom=g), _gz(n, 1))
    np.testing.assert_array_equal(one, np.full((n, 2, 1), 0.06))                  # anchor only
    for bad in (dict(g, reading="both"), dict(g, offset_source="x"), dict(g, ellipticity="x"),
                dict(g, offset_scale=[0.3]), dict(g, offset_scale=[-0.1, 0.3]), dict(g, theta0_deg=[10.0]),
                dict(g, orifice=[0.0, 1.0, 1.5])):
        with pytest.raises(ValueError):
            K.geometric_underestimation(Zu, S_ap, dict(p, u_geom=bad), gz)
    with pytest.raises(ValueError):
        K.geometric_underestimation(Zu, S_ap, dict(p, S_log_sd=0.0, u_geom=dict(g, ellipticity="span_linked")), gz)
    with pytest.raises(ValueError):                             # generator required
        K.draw_latent_axis(dict(p, u_geom=g), 10, _rng())
    with pytest.raises(ValueError):                             # no mixing with the stage B keys
        K.draw_latent_axis(dict(p, u_geom=g, u_link="chord"), 10, _rng(), _rng(5))
    with pytest.raises(ValueError):
        K.simulate_rules(dict(p, window=None, u_geom=g), np.random.SeedSequence(1), 100, 50)


def test_simulate_rules_geometric_common_random_numbers_and_regression(base):
    p = dict(base, K=3, window=None, u_scale=[0.06, 0.25, 0.25, 0.3])
    pg = dict(p, u_scale=[0.06, 0.0, 0.0, 0.0], u_geom=dict(GEOM_IND15, reading="rotation_and_offset"))
    ss = np.random.SeedSequence(entropy=20261005, spawn_key=(76, 2, 0))
    sg = np.random.SeedSequence(entropy=20261005, spawn_key=(76, 5, 0))
    a = K.simulate_rules(p, ss, 1500, 500)
    b = K.simulate_rules(pg, ss, 1500, 500, sg)
    c = K.simulate_rules(pg, ss, 1500, 500, sg)
    np.testing.assert_array_equal(a["S"], b["S"])
    np.testing.assert_array_equal(a["est"]["A1"], b["est"]["A1"])          # anchor-view mean unaffected
    assert not np.array_equal(a["est"]["A3"], b["est"]["A3"])
    for r in K.RULES:
        np.testing.assert_array_equal(b["est"][r], c["est"][r])            # repeatable
    d = K.simulate_rules(pg, ss, 1500, 500, np.random.SeedSequence(entropy=20261005, spawn_key=(76, 5, 2)))
    assert not np.array_equal(b["est"]["A3"], d["est"]["A3"])              # the geometry seed matters
    np.testing.assert_array_equal(b["est"]["A1"], d["est"]["A1"])
    # regression fixture (values of 2026-10-06, 1,500 patients)
    e3 = b["est"]["A3"][:, 0] - b["S"][:, 0]
    e2 = b["est"]["A2"][:, 0] - b["S"][:, 0]
    assert (e3.mean(), e2.mean(), b["mean_u"][0, 1]) == pytest.approx(REGRESSION_GEOM, rel=1e-9)
