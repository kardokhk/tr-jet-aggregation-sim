"""Edge cases, rule behaviour, seeding and agreement statistics (task item 5 and extras)."""
import numpy as np
import pytest

from conftest import QUIET
from duomaxsim import estimators as E
from duomaxsim.experiments import read_once, run_cells
from duomaxsim.metrics import bland_altman, icc_a1
from duomaxsim.model import draw_latent
from duomaxsim.rules import a4_final, beat_rule, cross_view_flags


def _sim(base, n=20000, seed=1, **ov):
    p = dict(base, **ov)
    rng = np.random.default_rng(seed)
    lat = draw_latent(p, n, rng)
    rb = p["sigma_rb_mm"] * rng.standard_normal(n)
    meas, br = read_once(lat, p, rng, rb)
    U = rng.random((n, 2, p["K"] - 1))
    return p, lat, meas, br, U


def test_K1_all_view_combiners_equal_anchor(base):
    p, lat, meas, br, U = _sim(base, K=1)
    vm = br.value
    est = E.standard_estimators(vm, lat, meas, br.used)
    a4 = a4_final(vm, lat.over, U, 3, 5, 0.8, 0.1)[0]
    for k in ("A2", "A3", "A5"):
        np.testing.assert_array_equal(est[k], est["A1"])
    np.testing.assert_array_equal(a4, est["A1"])


@pytest.mark.parametrize("W", [None, 0.1])
def test_N1_uses_first_beat(base, W):
    p, lat, meas, br, U = _sim(base, N_beats=1, window=W)
    np.testing.assert_allclose(br.value, meas[..., 0])
    assert (br.used == 1).all() and br.met_first.all() and not br.limited.any()


@pytest.mark.parametrize("rhythm", ["sinus", "AF"])
@pytest.mark.parametrize("W", [None, 0.15])
def test_zero_noise_gives_truth(base, rhythm, W):
    p, lat, meas, br, U = _sim(base, **dict(QUIET, K=4, N_beats=5, rhythm=rhythm, window=W))
    est = E.standard_estimators(br.value, lat, meas, br.used)
    est["A4"] = a4_final(br.value, lat.over, U, 3, 5, 0.8, 0.1)[0]
    for k, v in est.items():
        np.testing.assert_allclose(v, lat.S, rtol=1e-12, err_msg=k)


def test_view_errors_only_A7_recovers_truth(base):
    p, lat, meas, br, U = _sim(base, **dict(QUIET, K=4, N_beats=3, view_errors=True))
    est = E.standard_estimators(br.value, lat, meas, br.used)
    np.testing.assert_allclose(est["A7"], lat.S, rtol=1e-12)
    assert lat.over.any() and (lat.u > 0).any()


def test_retrospective_short_clips_are_limited():
    x = np.arange(1, 11, dtype=float)[None, :].repeat(3, 0) * 0 + np.array([[5, 6, 7, 20, 1, 1, 1, 1, 1, 1]])
    avail = np.array([2, 4, 10])
    r = beat_rule(x, avail, N=3, W=0.1)
    assert r.limited[0] and r.value[0] == pytest.approx(5.5) and r.used[0] == 2
    # 5,6,7 exceeds +-10% of 6; with 4 beats no 3-subset qualifies -> limited, mean of 4
    assert r.limited[1] and r.value[1] == pytest.approx((5 + 6 + 7 + 20) / 4)
    # (1,1,1) first available at the 7th beat
    assert r.accepted[2] and r.value[2] == pytest.approx(1.0) and r.used[2] == 7


def test_window_rule_accepts_consistent_first_beats():
    x = np.array([[10.0, 10.5, 9.8, 30.0]])
    r = beat_rule(x, np.array([4]), N=3, W=0.1)
    assert r.met_first[0] and r.used[0] == 3 and r.value[0] == pytest.approx((10 + 10.5 + 9.8) / 3)


def test_a4_reduces_to_a3_without_rejection(base):
    p, lat, meas, br, U = _sim(base, K=4, beat_cv=0.25)
    a4 = a4_final(br.value, lat.over, U, 2, 4, 0.0, 0.0)[0]
    np.testing.assert_array_equal(a4, E.a3_max(br.value))


def test_a4_full_rejection_keeps_only_untriggered(base):
    p, lat, meas, br, U = _sim(base, K=4, beat_cv=0.25)
    vm = br.value
    a4 = a4_final(vm, lat.over, U, 2, 4, 1.0, 1.0)[0]
    diff, warn, adj = cross_view_flags(vm, 2, 4)
    keep = np.where(warn | adj, -np.inf, vm[..., 1:])
    np.testing.assert_array_equal(a4, np.maximum(vm[..., 0], keep.max(-1)))


@pytest.mark.parametrize("s_det,f_rej", [(0.5, 0.1), (0.8, 0.3)])
def test_a4_detection_rates(base, s_det, f_rej):
    p, lat, meas, br, U = _sim(base, n=100000, K=4, beat_cv=0.2)
    _, info = a4_final(br.value, lat.over, U, 2, 4, s_det, f_rej)
    scr = info["warn"] | info["adj"]
    ov = lat.over[..., 1:]
    for mask, target in ((scr & ov, s_det), (scr & ~ov, f_rej)):
        k = mask.sum()
        rate = info["reject"][mask].mean()
        assert abs(rate - target) < 4 * np.sqrt(target * (1 - target) / k) + 1e-9


def test_seed_independent_of_cell_chunking(cfg):
    a = run_cells(cfg, "E1", [2, 3], n_rep=3000)
    b = run_cells(cfg, "E1", [3], n_rep=3000)
    cols = ["estimator", "s_det", "f_rej", "axis", "estimand", "metric", "value", "mcse"]
    a3 = a[a.cell == 3][cols].reset_index(drop=True)
    np.testing.assert_array_equal(a3.value.to_numpy(), b[cols].value.to_numpy())
    assert (a.seed_spawn_key[a.cell == 3] == "1,3").all()


def test_icc_ci_coverage_two_way_random():
    rng = np.random.default_rng(7)
    S, n = 2000, 50
    subj = 2.0 * rng.standard_normal((S, n, 1))
    rater = 0.5 * rng.standard_normal((S, 1, 2))
    Y = subj + rater + rng.standard_normal((S, n, 2))
    true = 4.0 / (4.0 + 0.25 + 1.0)
    icc, (lo, hi) = icc_a1(Y)
    cover = np.mean((lo <= true) & (true <= hi))
    assert abs(np.median(icc) - true) < 0.02
    assert 0.92 <= cover <= 0.985, cover


def test_icc_perfect_agreement_and_ba():
    Y = np.stack([np.arange(10.0), np.arange(10.0)], -1)
    assert icc_a1(Y, ci=False) == pytest.approx(1.0)
    bias, sd, lo, hi, hw = bland_altman(np.array([1.0, 1.0, 1.0, 1.0]))
    assert bias == 1.0 and sd == 0.0 and lo == hi == 1.0
