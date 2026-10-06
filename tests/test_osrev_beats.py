"""Tests for the amendment-3 beat-rule extensions (code/lib/duomaxsim/osrev_beats.py).

Three kinds: (1) with every new option off, each variant reproduces the library output
bit for bit, including the generator state afterwards; (2) known-answer cases;
(3) edge cases.
"""
import numpy as np
import pytest

from conftest import QUIET
from duomaxsim import osrev_beats as OB
from duomaxsim.model import draw_latent, measure
from duomaxsim.rules import _mean_first, beat_rule


def rng_of(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


VARIANTS = [
    {},
    {"K": 1},
    {"rhythm": "AF", "rr_cv_af": 0.2},
    {"data_mode": "retrospective"},
    {"window": None, "N_beats": 5},
    {"beat_noise_centering": "mean", "K": 4},
    {"view_errors": False, "beta_drr": 0.1},
]


# ---------------------------------------------------------------- (1) bit-identical when off

@pytest.mark.parametrize("ov", VARIANTS)
def test_draw_latent_ext_identical_when_off(base, ov):
    p = dict(base, **ov)
    r1, r2 = rng_of(11), rng_of(11)
    a = draw_latent(p, 500, r1)
    b = OB.draw_latent_ext(dict(p, **OB.EXT_DEFAULTS), 500, r2)
    c = OB.draw_latent_ext(p, 500, rng_of(11))          # options absent
    for f in ("S", "r", "u", "o", "over", "mu", "g", "logrr", "spans", "avail"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
        assert np.array_equal(getattr(a, f), getattr(c, f)), f
    assert np.array_equal(r1.standard_normal(7), r2.standard_normal(7))      # same generator state


@pytest.mark.parametrize("ov", [{}, {"measure_floor_mm": 0.5}, {"rhythm": "AF"}])
def test_measure_ext_identical_when_off(base, ov):
    p = dict(base, **ov)
    lat = draw_latent(p, 400, rng_of(3))
    rb = 0.75 * rng_of(4).standard_normal(400)
    r1, r2 = rng_of(5), rng_of(5)
    a = measure(lat, p, r1, rb)
    b = OB.measure_ext(lat, dict(p, cal_noise_mode="constant"), r2, rb)
    assert np.array_equal(a, b)
    assert np.array_equal(r1.random(5), r2.random(5))


@pytest.mark.parametrize("N,W,lv", [(3, 0.15, "all_acquired"), (5, 0.10, "first_n"), (3, None, "all_acquired")])
def test_window_rule_budget_identical_when_unlimited(base, N, W, lv):
    lat = draw_latent(base, 600, rng_of(8))
    x = measure(lat, base, rng_of(9), np.zeros(600))
    av = lat.avail[:, None, :]
    ref = beat_rule(x, av, N, W, lv)
    for bud in (None, x.shape[-1], x.shape[-1] + 5):
        got = OB.window_rule_budget(x, av, N, W, bud, lv)
        for f in ("value", "used", "accepted", "met_first", "limited"):
            assert np.array_equal(getattr(ref, f), getattr(got, f)), (f, bud)


def test_plain_mean_first_identical_to_library(base):
    lat = draw_latent(dict(base, K=1), 300, rng_of(2))
    x = measure(lat, base, rng_of(6), np.zeros(300))[:, :, 0, :]
    for k in (1, 3, 5, 30):
        ref = beat_rule(x, 30, k, None).value
        assert np.array_equal(ref, OB.plain_mean_first(x, k))
    k = rng_of(1).integers(1, 31, size=x.shape[:-1])
    ref = _mean_first(x.reshape(-1, 30), k.reshape(-1)).reshape(k.shape)
    assert np.array_equal(ref, OB.plain_mean_first(x, k))


def test_ar1_rho_zero_is_library_expression():
    z = rng_of(1).standard_normal((50, 2, 3, 30))
    assert np.array_equal(OB.ar1_from_innovations(z, 0.0, 0.149), 0.149 * z)


# ---------------------------------------------------------------- (2) known answers

def test_ar1_known_answer_and_moments():
    out = OB.ar1_from_innovations(np.array([[1.0, 1.0, -1.0]]), 0.6, 2.0)
    np.testing.assert_allclose(out, [[2.0, 0.6 * 2.0 + 0.8 * 2.0, 0.6 * 2.8 - 0.8 * 2.0]])
    eta = OB.ar1_from_innovations(rng_of(5).standard_normal((200000, 12)), 0.6, 0.15)
    np.testing.assert_allclose(eta.std(axis=0), 0.15, rtol=0.01)          # marginal SD preserved at every beat
    assert abs(np.corrcoef(eta[:, 3], eta[:, 4])[0, 1] - 0.6) < 0.01
    assert abs(np.corrcoef(eta[:, 3], eta[:, 6])[0, 1] - 0.6 ** 3) < 0.01


def test_resp_factor_known_answer():
    f = OB.resp_factor(np.array([4.0]), np.array([[0.0]]), 4, 0.10)
    assert f.shape == (1, 1, 1, 4)
    np.testing.assert_allclose(f[0, 0, 0], [1.0, 1.1, 1.0, 0.9], atol=1e-12)


def test_respiratory_component_in_model(base):
    p = dict(dict(base, **QUIET), K=2, window=0.15, resp_amp=0.10, resp_period_beats=[4.0, 6.0])
    lat = OB.draw_latent_ext(p, 40000, rng_of(21))
    ratio = lat.spans / lat.mu[..., None]
    assert ratio.min() >= 0.9 - 1e-12 and ratio.max() <= 1.1 + 1e-12
    np.testing.assert_allclose(ratio.var(), 0.10 ** 2 / 2, rtol=0.02)     # A^2 / 2 with a random phase
    np.testing.assert_allclose(ratio.mean(), 1.0, atol=1e-3)
    assert np.allclose(ratio[:, 0], ratio[:, 1], rtol=1e-12, atol=0)                     # axes of a view share the clip
    assert not np.allclose(ratio[:, 0, 0], ratio[:, 0, 1])                # views have their own phase


def test_total_beat_cv_known_answers(base):
    assert OB.total_beat_cv(base) == pytest.approx(0.15, abs=1e-12)
    add = dict(base, resp_amp=0.10, resp_mode="add")
    assert OB.total_beat_cv(add) == pytest.approx(np.sqrt(1.0225 * 1.005 - 1.0), abs=1e-12)
    pres = dict(base, resp_amp=0.10, resp_mode="preserve")
    assert OB.total_beat_cv(pres) == pytest.approx(0.15, abs=1e-12)
    assert OB.residual_beat_log_sd(pres) < OB.residual_beat_log_sd(base)
    # simulated marginal CV of beats around g * mu matches (no RR effect)
    q = dict(pres, beta_rr=0.0, K=1)
    lat = OB.draw_latent_ext(q, 60000, rng_of(4))
    ratio = lat.spans[:, 0, 0, :] / (lat.g[:, None] * lat.mu[:, 0, 0, None])
    np.testing.assert_allclose(ratio.std() / ratio.mean(), 0.15, rtol=0.01)


def test_expected_beat_span_known_answer(base):
    q = dict(base, **QUIET)
    assert OB.expected_beat_span_mm(q) == pytest.approx(10.0 * np.exp(0.4 ** 2 / 2), rel=1e-12)
    assert OB.expected_beat_span_mm(q, axis=1) == pytest.approx(0.64 * 10.0 * np.exp(0.08), rel=1e-12)
    for ov in ({"K": 1}, {"K": 1, "rhythm": "AF", "rr_cv_af": 0.2}, {"K": 1, "beat_noise_centering": "mean"}):
        p = dict(base, **ov)
        lat = draw_latent(p, 300000, rng_of(17))
        for ax in (0, 1):
            np.testing.assert_allclose(lat.spans[:, ax, 0, :].mean(), OB.expected_beat_span_mm(p, ax),
                                       rtol=0.004)


def test_caliper_sd_known_answers(base):
    spans = np.array([5.0, 10.0, 20.0])
    assert OB.caliper_sd(spans, base) == 1.0
    p = dict(base, cal_noise_mode="proportional", cal_ref_span_mm=10.0)
    np.testing.assert_allclose(OB.caliper_sd(spans, p), [0.5, 1.0, 2.0])
    m = dict(base, cal_noise_mode="mixed", cal_const_frac=0.5, cal_ref_span_mm=10.0)
    np.testing.assert_allclose(OB.caliper_sd(spans, m), [0.75, 1.0, 1.5])
    # default reference: average SD on the AP axis of the anchor view equals sigma_cal_mm
    q = dict(base, K=1, cal_noise_mode="proportional")
    lat = draw_latent(q, 200000, rng_of(23))
    np.testing.assert_allclose(OB.caliper_sd(lat.spans[:, 0], q).mean(), 1.0, rtol=0.005)
    x = OB.measure_ext(lat, q, rng_of(24), np.zeros(200000))
    resid = (x - lat.spans)[:, 0]
    np.testing.assert_allclose(np.abs(resid).mean(), np.sqrt(2 / np.pi) * 1.0, rtol=0.01)


def test_window_rule_budget_known_answer():
    x = np.array([[10.0, 13.0, 10.2, 9.9, 10.1]])
    full = OB.window_rule_budget(x, 5, 3, 0.15, None)
    assert full.used[0] == 4 and full.accepted[0] and not full.met_first[0]
    assert full.value[0] == pytest.approx((9.9 + 10.0 + 10.2) / 3)
    b3 = OB.window_rule_budget(x, 5, 3, 0.15, 3)                # budget = N: cannot search, falls back
    assert b3.limited[0] and b3.used[0] == 3 and b3.value[0] == pytest.approx((10.0 + 13.0 + 10.2) / 3)
    b3f = OB.window_rule_budget(x, 5, 3, 0.15, 3, "first_n")
    assert b3f.value[0] == pytest.approx(b3.value[0])
    b4 = OB.window_rule_budget(x, 5, 3, 0.15, 4)
    assert b4.value[0] == pytest.approx(full.value[0]) and b4.used[0] == 4


def test_window_rule_budget_equals_slice(base):
    lat = draw_latent(dict(base, K=1), 800, rng_of(31))
    x = measure(lat, base, rng_of(32), np.zeros(800))[:, :, 0, :]
    for bud in (3, 4, 6, 10):
        a = OB.window_rule_budget(x, 30, 3, 0.15, bud)
        b = beat_rule(x[..., :bud].copy(), bud, 3, 0.15)
        assert np.array_equal(a.value, b.value) and np.array_equal(a.used, b.used)
        assert a.used.max() <= bud
    b3 = OB.window_rule_budget(x, 30, 3, 0.15, 3)
    np.testing.assert_allclose(b3.value, OB.plain_mean_first(x, 3), rtol=1e-13)   # budget = N is the plain mean
    assert np.array_equal(b3.accepted, b3.met_first)


def test_exhaustive_known_answer_and_agreement(base):
    x = np.array([[10.0, 13.0, 10.2, 9.9, 10.1], [10.0, 20.0, 30.0, 45.0, 70.0]])
    r = OB.beat_rule_exhaustive(x, 5, 3, 0.15)
    assert r.used.tolist() == [4, 5] and r.accepted.tolist() == [True, False]
    assert r.value[0] == pytest.approx((9.9 + 10.0 + 10.2) / 3)
    assert r.value[1] == pytest.approx(35.0)                    # limited: mean of all five
    lat = draw_latent(dict(base, K=1), 3000, rng_of(41))
    xm = measure(lat, base, rng_of(42), np.zeros(3000))[:, :, 0, :8]
    a = beat_rule(xm, 8, 3, 0.15)
    b = OB.beat_rule_exhaustive(xm, 8, 3, 0.15)
    assert np.array_equal(a.met_first, b.met_first)             # identical when m = N
    assert (a.accepted != b.accepted).mean() < 0.005
    same = a.used == b.used
    assert same.mean() > 0.995
    np.testing.assert_allclose(a.value[same & a.accepted], b.value[same & a.accepted], rtol=1e-12)


def test_error_summary_known_answer():
    e = np.array([1.0, -1.0, 2.0, -2.0])
    s = OB.error_summary(e, np.full(4, 10.0))
    assert s["n"] == 4 and s["bias_mm"] == 0.0
    assert s["rmse_mm"] == pytest.approx(np.sqrt(2.5))
    assert s["mae_mm"] == pytest.approx(1.5)
    assert s["bias_mcse"] == pytest.approx(np.std(e, ddof=1) / 2.0)
    assert s["mae_mcse"] == pytest.approx(np.std([1.0, 1.0, 2.0, 2.0], ddof=1) / 2.0)
    assert s["rmse_mcse"] == pytest.approx(np.std([1.0, 1.0, 4.0, 4.0], ddof=1) / 2.0 / (2 * np.sqrt(2.5)))
    assert s["relbias_pct"] == pytest.approx(0.0)


def test_paired_summary_known_answer_and_delta_method():
    r = rng_of(77)
    n = 40000
    shared = r.standard_normal(n)
    ea = 1.0 * shared + 0.5 * r.standard_normal(n) + 0.1
    eb = 0.9 * shared + 0.5 * r.standard_normal(n)
    t = np.full(n, 10.0)
    ps = OB.paired_summary(ea, eb, t)
    assert ps["d_bias_mm"] == pytest.approx(ea.mean() - eb.mean())
    assert ps["d_bias_mcse"] == pytest.approx((ea - eb).std(ddof=1) / np.sqrt(n))
    assert ps["d_rmse_mm"] == pytest.approx(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean()))
    assert ps["d_mae_mm"] == pytest.approx(np.abs(ea).mean() - np.abs(eb).mean())
    assert ps["d_relbias_pct"] == pytest.approx(10.0 * (ea.mean() - eb.mean()))
    assert ps["d_rmse_mcse"] < ps["d_rmse_mcse_indep"]          # positive correlation narrows the interval
    bs = OB.bootstrap_rmse_diff(ea, eb, 400, rng_of(78))
    assert bs["boot_se"] == pytest.approx(ps["d_rmse_mcse"], rel=0.15)   # delta method agrees with bootstrap
    assert bs["boot_lo95"] < ps["d_rmse_mm"] < bs["boot_hi95"]
    # independent arms: the paired formula reduces to the independent one (up to sampling covariance)
    ec = r.standard_normal(n)
    ed = r.standard_normal(n)
    pi = OB.paired_summary(ec, ed, t)
    assert pi["d_rmse_mcse"] == pytest.approx(pi["d_rmse_mcse_indep"], rel=0.02)


# ---------------------------------------------------------------- (3) edge cases

def test_edge_identical_arms_and_zero_error():
    e = np.array([0.5, -1.5, 2.0, 0.0, 1.0])
    t = np.full(5, 8.0)
    ps = OB.paired_summary(e, e, t)
    for k in ("d_bias_mm", "d_rmse_mm", "d_mae_mm", "d_relbias_pct", "d_bias_mcse", "d_mae_mcse"):
        assert ps[k] == 0.0
    assert ps["d_rmse_mcse"] == pytest.approx(0.0, abs=1e-7)
    z = OB.error_summary(np.zeros(5), t)
    assert z["rmse_mm"] == 0.0 and z["rmse_mcse"] == 0.0 and z["mae_mm"] == 0.0
    b = OB.bootstrap_rmse_diff(e, e, 20, rng_of(1))
    assert b["boot_lo95"] == 0.0 and b["boot_hi95"] == 0.0 and b["boot_se"] == 0.0
    b1 = OB.bootstrap_rmse_diff(e, 2 * e, 30, rng_of(9))
    b2 = OB.bootstrap_rmse_diff(e, 2 * e, 30, rng_of(9))
    assert b1 == b2                                              # reproducible for a given seed


def test_edge_plain_mean_first():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert OB.plain_mean_first(x, 2)[0] == 1.5
    assert OB.plain_mean_first(x, 99)[0] == 2.5                 # clipped to the beats that exist
    assert OB.plain_mean_first(x, 0)[0] == 1.0                  # at least one beat
    assert OB.plain_mean_first(x, np.array([3]))[0] == 2.0


def test_edge_invalid_options(base):
    with pytest.raises(ValueError):
        OB.ar1_from_innovations(np.zeros((2, 3)), 1.0, 1.0)
    with pytest.raises(ValueError):
        OB.caliper_sd(np.ones(3), dict(base, cal_noise_mode="quadratic"))
    with pytest.raises(ValueError):
        OB.caliper_sd(np.ones(3), dict(base, cal_noise_mode="mixed", cal_const_frac=1.5))
    with pytest.raises(ValueError):
        OB.window_rule_budget(np.ones((2, 5)), 5, 3, 0.15, 0)
    with pytest.raises(ValueError):
        OB.residual_beat_log_sd(dict(base, resp_amp=0.5, resp_mode="preserve"))   # A^2/2 exceeds CV^2
    with pytest.raises(ValueError):
        OB.residual_beat_log_sd(dict(base, resp_amp=0.1, resp_mode="other"))
    with pytest.raises(ValueError):
        OB.draw_latent_ext(dict(base, resp_amp=0.1, resp_period_beats=[1.0, 6.0]), 10, rng_of(1))
    with pytest.raises(ValueError):
        OB.draw_latent_ext(dict(base, resp_amp=1.2), 10, rng_of(1))
    with pytest.raises(ValueError):
        OB.beat_rule_exhaustive(np.ones((2, 4)), 4, 3, 0.15, "other")


def test_edge_single_beat_and_short_views(base):
    # one generated beat: AR(1) and the respiratory factor are defined and finite
    p = dict(base, K=1, window=None, N_beats=1, beat_ar1_rho=0.9, resp_amp=0.1)
    lat = OB.draw_latent_ext(p, 200, rng_of(3))
    assert lat.spans.shape == (200, 2, 1, 1) and np.isfinite(lat.spans).all() and (lat.spans > 0).all()
    # proportional caliper noise with zero base SD leaves beats unchanged
    q = dict(base, K=1, sigma_cal_mm=0.0, cal_noise_mode="proportional")
    lat = OB.draw_latent_ext(q, 50, rng_of(4))
    assert np.array_equal(OB.measure_ext(lat, q, rng_of(5), np.zeros(50)), lat.spans)
    # exhaustive rule with fewer available beats than N: limited, mean of what exists
    x = np.array([[4.0, 6.0, 100.0, 100.0]])
    r = OB.beat_rule_exhaustive(x, 2, 3, 0.15)
    assert r.limited[0] and r.used[0] == 2 and r.value[0] == 5.0
    lib = beat_rule(x, 2, 3, 0.15)
    assert lib.limited[0] and lib.used[0] == 2 and lib.value[0] == 5.0
    # AR(1) and respiration keep the sensitivity model on the library's first-beat marginal
    a = OB.draw_latent_ext(dict(base, K=1), 100, rng_of(6))
    b = OB.draw_latent_ext(dict(base, K=1, beat_ar1_rho=0.6), 100, rng_of(6))
    assert np.array_equal(a.spans[..., 0], b.spans[..., 0]) and not np.array_equal(a.spans[..., 1], b.spans[..., 1])
