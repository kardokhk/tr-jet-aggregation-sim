"""Tests for duomaxsim.osrev_triggers (protocol amendment 3, package A3-4).

Three kinds of test per function: exact reduction to the pre-specified library
when the new options are off, a known-answer case, and an edge case.
"""
import copy

import numpy as np
import pytest
from scipy import stats

from duomaxsim import config as C
from duomaxsim import osrev_triggers as T
from duomaxsim.experiments import run_cells
from duomaxsim.metrics import ErrAcc
from duomaxsim.model import Latent, draw_latent
from duomaxsim.rules import a4_final, cross_view_flags

E4_BASE, E1_BASE, E3_BASE = 193, 703, 29


def _val(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    assert len(q) >= 1, kw
    assert q["value"].nunique() == 1, kw
    return float(q["value"].iloc[0])


def _latent(S, g, u, o):
    """Hand-built latent state for one axis pair: S (n, 2), g (n,), u and o (n, 2, K)."""
    S = np.asarray(S, float)
    u = np.asarray(u, float)
    o = np.asarray(o, float)
    g = np.asarray(g, float)
    n, _, K = u.shape
    mu = S[:, :, None] * (1 - u + o)
    return Latent(S=S, r=S[:, 1] / S[:, 0], u=u, o=o, over=o > 0, mu=mu, g=g,
                  logrr=np.zeros((n, K, 2)), spans=g[:, None, None, None] * mu[..., None], avail=np.ones((n, K), int))


def _one(u_a, u_v, o_a=0.0, o_v=0.0, g=1.0, S=10.0):
    """One patient, K = 2, the same errors on both axes."""
    return _latent([[S, S]], [g], [[[u_a, u_v]] * 2], [[[o_a, o_v]] * 2])


# ------------------------------------------------------------------ cohort_chunks vs library runners

@pytest.mark.parametrize("chunk", [20000, 1000])
def test_cohort_reproduces_run_E4_exactly(cfg, chunk):
    cfg = copy.deepcopy(cfg)
    cfg["meta"]["chunk_size"] = chunk
    n = 3000
    df = run_cells(cfg, "E4", [E4_BASE], n_rep=n)
    p = C.cell_params(cfg, "E4", E4_BASE)
    thr = p["error_threshold_mm"]
    k = {m: [0, 0] for m in ("any5", "ev", "fa5", "miss3")}
    for _, rc in T.cohort_chunks(p, C.cell_seed(cfg, "E4", E4_BASE), n, chunk):
        lat, vm = rc["lat"], rc["vm"]
        ev = T.pair_event_definitions(lat, thr)["D0_published"][:, 0]
        f5, f3 = T.trigger_fired(vm, 5.0)[:, 0], T.trigger_fired(vm, 3.0)[:, 0]
        for m, (a, b) in {"any5": (f5.any(1).sum(), lat.n), "ev": (ev.sum(), ev.size),
                          "fa5": (f5[~ev].sum(), (~ev).sum()), "miss3": ((~f3)[ev].sum(), ev.sum())}.items():
            k[m][0] += int(a)
            k[m][1] += int(b)
    ap = df[df.axis == "AP"]
    assert k["any5"][0] / k["any5"][1] == _val(ap, metric="p_any_adj", t_adj=5.0)
    assert k["ev"][0] / k["ev"][1] == _val(ap, metric="p_error_event_pair")
    assert k["fa5"][0] / k["fa5"][1] == _val(ap, metric="false_alarm_adj_pair", t_adj=5.0)
    assert k["miss3"][0] / k["miss3"][1] == _val(ap, metric="miss_trigger_pair", t_warn=3.0)


def test_cohort_reproduces_run_E1_and_E3_exactly(cfg):
    n = 3000
    df = run_cells(cfg, "E1", [E1_BASE], n_rep=n)
    p = C.cell_params(cfg, "E1", E1_BASE)
    (_, rc), = T.cohort_chunks(p, C.cell_seed(cfg, "E1", E1_BASE), n, cfg["meta"]["chunk_size"])
    grid = [(0.8, 0.1), (0.0, 0.0), (1.0, 0.3)]
    res = T.reviewed_rule_grid(rc["vm"], rc["lat"].over, rc["U"], p["t_warn"], p["t_adj"], grid, p["a4_scrutiny"])
    for s, f in grid:
        acc = ErrAcc()
        acc.add(res[(s, f)]["value"][:, 0], rc["lat"].S[:, 0])
        lib = df[(df.estimator == "A4") & (df.s_det == s) & (df.f_rej == f) & (df.axis == "AP") & (df.estimand == "T1")]
        for metric in ("bias_mm", "rmse_mm"):
            assert acc.summary()[metric][0] == _val(lib, metric=metric)
            # the module's own metric function agrees to rounding error (different summation order)
        em = T.error_metrics(res[(s, f)]["value"][:, 0], rc["lat"].S[:, 0])
        assert em["bias_mm"][0] == pytest.approx(_val(lib, metric="bias_mm"), abs=1e-12)
        assert em["rmse_mm"][0] == pytest.approx(_val(lib, metric="rmse_mm"), abs=1e-12)
        assert em["rmse_mm"][1] == pytest.approx(float(lib[lib.metric == "rmse_mm"].mcse.iloc[0]), rel=1e-3)
    acc = ErrAcc()
    acc.add(rc["vm"].max(-1)[:, 0], rc["lat"].S[:, 0])
    assert acc.summary()["bias_mm"][0] == _val(df[(df.estimator == "A3") & (df.axis == "AP") & (df.estimand == "T1")],
                                               metric="bias_mm")
    # E3: sensitivity and specificity at 13 mm
    d3 = run_cells(cfg, "E3", [E3_BASE], n_rep=n)
    p3 = C.cell_params(cfg, "E3", E3_BASE)
    (_, r3), = T.cohort_chunks(p3, C.cell_seed(cfg, "E3", E3_BASE), n, cfg["meta"]["chunk_size"])
    a4 = T.reviewed_rule_grid(r3["vm"], r3["lat"].over, r3["U"], p3["t_warn"], p3["t_adj"],
                              [(p3["s_det"], p3["f_rej"])], p3["a4_scrutiny"])[(p3["s_det"], p3["f_rej"])]["value"]
    em = T.error_metrics(a4[:, 0], r3["lat"].S[:, 0], [13.0])
    lib = d3[(d3.estimator == "A4") & (d3.axis == "AP") & (d3.cutoff_mm == 13.0)]
    assert em["sensitivity_13mm"][0] == _val(lib, metric="sensitivity")
    assert em["specificity_13mm"][0] == _val(lib, metric="specificity")


def test_cohort_only_selects_identical_chunk(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    ss = C.cell_seed(cfg, "E4", E4_BASE)
    full = dict(T.cohort_chunks(p, ss, 2500, 1000))
    assert sorted(full) == [0, 1, 2] and full[2]["lat"].n == 500
    for i in (0, 2):
        (j, rc), = T.cohort_chunks(p, C.cell_seed(cfg, "E4", E4_BASE), 2500, 1000, only=i)
        assert j == i
        for k in ("vm", "U", "meas"):
            assert np.array_equal(rc[k], full[i][k])
        assert np.array_equal(rc["lat"].spans, full[i]["lat"].spans)


def test_cohort_edge_single_view(cfg):
    p = C.cell_params(cfg, "E1", E1_BASE, {"K": 1})
    (_, rc), = T.cohort_chunks(p, np.random.SeedSequence(1), 50, 100)
    assert rc["vm"].shape == (50, 2, 1) and rc["U"].shape == (50, 2, 0)
    assert T.trigger_fired(rc["vm"], 3.0).shape == (50, 2, 0)


# ------------------------------------------------------------------ error definitions

def test_published_definition_equals_run_E4_expression(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    lat = draw_latent(p, 4000, np.random.default_rng(5))
    thr = 2.0
    gS = lat.g[:, None] * lat.S
    event = (gS[..., None] * lat.o[..., 1:] > thr) | (gS * lat.u[..., 0] > thr)[..., None]   # run_E4, verbatim
    d = T.pair_event_definitions(lat, thr, 5.0)
    assert np.array_equal(d["D0_published"], event)
    assert np.array_equal(d["D0_published"], d["D0a_other_over"] | d["D0b_anchor_under"])
    for k in ("D1_plus_other_under", "D2_net_either_view"):
        assert d["D0_published"].sum() <= d["D1_plus_other_under"].sum()
        assert d[k].shape == event.shape
    assert not (d["H_shared_total_agree"] & ~d["D5_shared_total"]).any()
    e = T.view_error_mm(lat)
    assert np.allclose(e["total"], e["net"] + ((lat.g - 1)[:, None] * lat.S)[..., None], atol=1e-12)
    assert np.allclose(e["net"], e["over"] - e["under"], atol=1e-12)


def test_definitions_known_answers():
    # anchor 3 mm low, other view 2.5 mm high: published event through both components
    d = T.pair_event_definitions(_one(0.3, 0.0, 0.0, 0.25), 2.0, 5.0)
    get = lambda k: bool(d[k][0, 0, 0])
    assert get("D0_published") and get("D0a_other_over") and get("D0b_anchor_under")
    assert get("D4_expected_excess") and get("X_expected_excess_ge_tau")       # 5.5 mm expected excess
    assert not get("D5_shared_total") and not get("H_shared_total_agree")      # opposite directions
    # both views 3 mm low: an offset shared by the pair, invisible between views
    d = T.pair_event_definitions(_one(0.3, 0.3), 2.0, 3.0)
    assert get("D0_published") and get("D5_shared_total") and get("D5n_shared_net")
    assert get("H_shared_total_agree") and get("Hn_shared_net_agree") and not get("D4_expected_excess")
    # only the other view low: not a published event, counted by D1 and D2
    d = T.pair_event_definitions(_one(0.0, 0.3), 2.0, 3.0)
    assert not get("D0_published") and get("D1_plus_other_under") and get("D2_net_either_view")
    assert not get("D5_shared_total")
    # no plane error, instrument factor 1.3: both views 3 mm high in total, no net error
    d = T.pair_event_definitions(_one(0.0, 0.0, g=1.3), 2.0, 3.0)
    assert not get("D0_published") and not get("D2_net_either_view") and get("D3_total_either_view")
    assert get("D5_shared_total") and not get("D5n_shared_net") and get("H_shared_total_agree")


def test_definitions_edge_threshold_is_strict_and_no_tau():
    d = T.pair_event_definitions(_one(0.2, 0.0), 2.0)          # exactly 2.0 mm: not an error
    assert not d["D0_published"].any() and "H_shared_total_agree" not in d
    assert (T.pair_partition(_one(0.2, 0.2), 2.0) == 0).all()


def test_partition_known_answers_and_exhaustive(cfg):
    cases = {(0.0, 0.0): 0, (0.3, 0.0): 1, (0.0, 0.3): 3, (0.3, 0.3): 5}
    for (ua, uv), code in cases.items():
        assert (T.pair_partition(_one(ua, uv), 2.0, "net") == code).all()
    assert (T.pair_partition(_one(0.0, 0.0, 0.3, 0.0), 2.0) == 2).all()
    assert (T.pair_partition(_one(0.0, 0.0, 0.0, 0.3), 2.0) == 4).all()
    assert (T.pair_partition(_one(0.0, 0.0, 0.3, 0.3), 2.0) == 6).all()
    assert (T.pair_partition(_one(0.3, 0.0, 0.0, 0.3), 2.0) == 7).all()
    assert (T.pair_partition(_one(0.0, 0.3, 0.3, 0.0), 2.0) == 8).all()
    assert (T.pair_partition(_one(0.0, 0.0, g=1.3), 2.0, "total") == 6).all()
    assert (T.pair_partition(_one(0.0, 0.0, g=1.3), 2.0, "net") == 0).all()
    p = C.cell_params(cfg, "E4", E4_BASE)
    lat = draw_latent(p, 3000, np.random.default_rng(2))
    code = T.pair_partition(lat, 2.0, "total")
    assert code.min() >= 0 and code.max() <= 8
    d = T.pair_event_definitions(lat, 2.0)
    assert np.array_equal(np.isin(code, (5, 6)), d["D5_shared_total"])
    assert np.array_equal(code != 0, d["D3_total_either_view"])


def test_trigger_fired_matches_library_flags():
    vm = np.array([[10.0, 12.9, 13.0, 15.0, 9.0]])
    _, warn, adj = cross_view_flags(vm, 3.0, 5.0)
    assert np.array_equal(T.trigger_fired(vm, 3.0), warn | adj)
    assert np.array_equal(T.trigger_fired(vm, 5.0), adj)
    assert T.trigger_fired(vm, 3.0).tolist() == [[False, True, True, False]]   # one-sided; >= inclusive


# ------------------------------------------------------------------ 2 by 2 tables

def test_two_by_two_known_answer_and_edge():
    fired = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    event = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0], bool)
    c = T.two_by_two(fired, event)
    assert c == dict(tp=2, fp=1, fn=3, tn=4)
    r = T.rates_2x2(**c)
    assert r["prevalence"][0] == 0.5 and r["sensitivity"][0] == 0.4 and r["false_positive_rate"][0] == 0.2
    assert r["ppv"][0] == pytest.approx(2 / 3) and r["npv"][0] == pytest.approx(4 / 7)
    assert r["ppv"][1] == pytest.approx(np.sqrt((2 / 3) * (1 / 3) / 3)) and r["ppv"][2:] == (2, 3)
    # Bayes consistency: PPV from prevalence, sensitivity and false-positive rate
    pv, se, fp = r["prevalence"][0], r["sensitivity"][0], r["false_positive_rate"][0]
    assert r["ppv"][0] == pytest.approx(pv * se / (pv * se + (1 - pv) * fp))
    e = T.rates_2x2(0, 0, 0, 5)                       # nothing fired, no event
    assert np.isnan(e["ppv"][0]) and np.isnan(e["sensitivity"][0]) and e["specificity"][0] == 1.0
    with pytest.raises(ValueError):
        T.two_by_two(fired, event[:3])


def test_pattern_hist_known_answer_and_consistency():
    fired = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]], bool)
    event = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]], bool)
    H = T.pattern_hist(fired, event)
    assert H.shape == (4, 4, 4) and H.sum() == 4
    assert H[1, 1, 0] == 1 and H[0, 0, 2] == 1 and H[0, 1, 0] == 1 and H[0, 0, 0] == 1
    r = T.rates_from_patterns(H)
    c = T.two_by_two(fired, event)
    assert {k: r["counts"][k] for k in c} == c
    flat = T.rates_2x2(**c)
    for m in ("prevalence", "sensitivity", "false_positive_rate", "ppv", "npv", "specificity", "fired"):
        assert r[m]["value"] == pytest.approx(flat[m][0])
        assert r[m]["se_iid"] == pytest.approx(flat[m][1])
        assert (r[m]["num"], r[m]["den"]) == flat[m][2:]
        assert r[m]["se_sqrt_m"] == pytest.approx(flat[m][1] * np.sqrt(3))
    with pytest.raises(ValueError):
        T.pattern_hist(fired, event[:, :2])


def test_cluster_se_limits():
    rng = np.random.default_rng(3)
    n = 4000
    # one pair per patient: the cluster SE is the binomial SE (times sqrt(n/(n-1)))
    f1, e1 = rng.random((n, 1)) < 0.3, rng.random((n, 1)) < 0.4
    r = T.rates_from_patterns(T.pattern_hist(f1, e1))
    for m in ("sensitivity", "ppv", "false_positive_rate", "prevalence"):
        assert r[m]["se_cluster"] == pytest.approx(r[m]["se_iid"] * np.sqrt(n / (n - 1)), rel=1e-12)
    # three perfectly dependent pairs: prevalence and fired reach the sqrt(m) bound
    f3, e3 = np.repeat(f1, 3, axis=1), np.repeat(e1, 3, axis=1)
    r3 = T.rates_from_patterns(T.pattern_hist(f3, e3))
    for m in ("prevalence", "fired", "ppv", "sensitivity"):
        assert r3[m]["value"] == pytest.approx(r[m]["value"])
        assert r3[m]["se_cluster"] == pytest.approx(r3[m]["se_sqrt_m"] * np.sqrt(n / (n - 1)), rel=1e-12)
        assert r3[m]["deff"] == pytest.approx(3 * n / (n - 1))
    # three independent pairs: design effect close to 1
    r_ind = T.rates_from_patterns(T.pattern_hist(rng.random((n, 3)) < 0.3, rng.random((n, 3)) < 0.4))
    assert 0.9 < r_ind["ppv"]["deff"] < 1.1


def test_rates_from_patterns_edge_nothing_fired():
    H = T.pattern_hist(np.zeros((10, 3), bool), np.eye(10, 3, dtype=bool))
    r = T.rates_from_patterns(H)
    assert np.isnan(r["ppv"]["value"]) and r["ppv"]["den"] == 0 and r["sensitivity"]["value"] == 0.0
    b = T.bootstrap_from_patterns(H, 200, np.random.default_rng(0))
    assert all(np.isnan(x) for x in b["ppv"])
    bad = np.zeros((4, 4, 4), int)
    bad[3, 3, 3] = 1                                       # nine pairs in a patient with three
    with pytest.raises(ValueError):
        T.rates_from_patterns(bad)


def test_bootstrap_agrees_with_linearization():
    rng = np.random.default_rng(11)
    n = 20000
    shared = rng.random((n, 1)) < 0.1                       # patient-level component induces clustering
    fired = (rng.random((n, 3)) < 0.05) | (shared & (rng.random((n, 3)) < 0.5))
    event = (rng.random((n, 3)) < 0.05) | (shared & (rng.random((n, 3)) < 0.6))
    H = T.pattern_hist(fired, event)
    r = T.rates_from_patterns(H)
    b = T.bootstrap_from_patterns(H, 4000, np.random.default_rng(1))
    for m in ("ppv", "sensitivity", "false_positive_rate", "prevalence"):
        assert b[m][0] == pytest.approx(r[m]["se_cluster"], rel=0.06)
        assert b[m][1] < r[m]["value"] < b[m][2]
    assert r["prevalence"]["deff"] > 1.3
    # direct resampling of patients gives the same sampling distribution
    idx = np.random.default_rng(2).integers(0, n, (300, n))
    ppv = [(fired[i] & event[i]).sum() / fired[i].sum() for i in idx]
    assert np.std(ppv, ddof=1) == pytest.approx(r["ppv"]["se_cluster"], rel=0.15)


# ------------------------------------------------------------------ reviewed rule

def test_review_grid_equals_library_and_no_review_is_largest_view_mean(cfg):
    p = C.cell_params(cfg, "E1", E1_BASE)
    (_, rc), = T.cohort_chunks(p, np.random.SeedSequence(7), 5000, 20000)
    vm, lat, U = rc["vm"], rc["lat"], rc["U"]
    grid = [(s, f) for s in (0.0, 0.2, 0.5, 0.8, 1.0) for f in (0.0, 0.1, 0.3, 0.5)]
    for scr in ("triggered", "adj_only", "all_higher"):
        res = T.reviewed_rule_grid(vm, lat.over, U, 3.0, 5.0, grid, scr)
        for s, f in grid:
            ref, info = a4_final(vm, lat.over, U, 3.0, 5.0, s, f, scr)
            assert np.array_equal(res[(s, f)]["value"], ref)
            assert np.array_equal(res[(s, f)]["reject"], info["reject"])
            assert np.array_equal(res[(s, f)]["reject"], res[(s, f)]["reject_over"] | res[(s, f)]["reject_valid"])
            assert not (res[(s, f)]["reject"] & ~res[(s, f)]["reviewed"]).any()
        assert np.array_equal(res[(0.0, 0.0)]["value"], vm.max(-1))
        assert not res[(0.0, 0.0)]["reject"].any()
    # a perfect reviewer never excludes a valid view; exclusions are nested in s_det on common uniforms
    res = T.reviewed_rule_grid(vm, lat.over, U, 3.0, 5.0, grid)
    assert not res[(1.0, 0.0)]["reject_valid"].any()
    assert (res[(0.5, 0.1)]["reject"] <= res[(0.8, 0.1)]["reject"]).all()
    assert (res[(0.8, 0.1)]["value"] <= vm.max(-1)).all() and (res[(0.8, 0.1)]["value"] >= vm[..., 0]).all()


def test_review_known_answers():
    vm = np.array([[10.0, 14.0, 12.0]])
    over = np.array([[False, True, False]])
    U = np.array([[0.5, 0.5]])
    val = lambda s, f, o=over: float(T.reviewed_rule_grid(vm, o, U, 3.0, 5.0, [(s, f)])[(s, f)]["value"][0])
    assert val(1.0, 0.0) == 12.0            # the overestimated view is reviewed (+4 mm) and excluded
    assert val(0.0, 1.0) == 14.0            # reviewer never detects true overestimation
    assert val(0.4, 0.0) == 14.0 and val(0.6, 0.0) == 12.0      # U = 0.5 against s_det
    none = np.zeros_like(over)
    assert val(1.0, 0.0, none) == 14.0      # nothing overestimated, nothing excluded
    assert val(0.0, 1.0, none) == 12.0      # valid view wrongly excluded; +2 mm view is not reviewed
    r = T.reviewed_rule_grid(vm, over, U, 3.0, 5.0, [(1.0, 1.0)])[(1.0, 1.0)]
    assert r["reviewed"].tolist() == [[True, False]] and r["reject_over"].tolist() == [[True, False]]


def test_review_edge_single_view_and_all_excluded():
    vm = np.array([[10.0]])
    r = T.reviewed_rule_grid(vm, np.array([[False]]), np.zeros((1, 0)), 3.0, 5.0, [(0.8, 0.1)])[(0.8, 0.1)]
    assert r["value"].tolist() == [10.0] and r["reject"].shape == (1, 0)
    vm = np.array([[10.0, 20.0, 30.0]])
    r = T.reviewed_rule_grid(vm, np.ones((1, 3), bool), np.zeros((1, 2)), 3.0, 5.0, [(1.0, 1.0)])[(1.0, 1.0)]
    assert r["value"].tolist() == [10.0]     # every other view excluded: the anchor is always retained


def test_error_metrics_known_answer_and_edge():
    m = T.error_metrics(np.array([1.0, 3.0, 14.0, 12.0]), np.array([2.0, 2.0, 13.0, 13.0]), [13.0])
    assert m["bias_mm"][0] == 0.0 and m["rmse_mm"][0] == 1.0 and m["mae_mm"][0] == 1.0
    assert m["sensitivity_13mm"] == (0.5, pytest.approx(np.sqrt(0.25 / 2)), 2, 1)
    assert m["specificity_13mm"][0] == 1.0 and m["specificity_13mm"][2:] == (2, 2)
    assert m["bias_mm"][1] == pytest.approx(np.std([-1, 1, 1, -1], ddof=1) / 2)
    e = T.error_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), [13.0])      # no events, exact estimates
    assert e["rmse_mm"][:2] == (0.0, 0.0) and np.isnan(e["sensitivity_13mm"][0])


def test_paired_contrast_known_answer_and_edge():
    rng = np.random.default_rng(4)
    truth = 5 + 10 * rng.random(20000)
    a = truth + rng.standard_normal(20000) + 0.5
    b = a - 0.5
    d = T.paired_contrast(a, b, truth, [10.0])
    assert d["bias_mm"][0] == pytest.approx(0.5, abs=1e-12) and d["bias_mm"][1] < 1e-12
    ma, mb = T.error_metrics(a, truth, [10.0]), T.error_metrics(b, truth, [10.0])
    for k in ("rmse_mm", "mae_mm", "sensitivity_10mm", "specificity_10mm"):
        assert d[k][0] == pytest.approx(ma[k][0] - mb[k][0], abs=1e-12)
    # paired MCSE is far below the independent-sample MCSE
    assert d["rmse_mm"][1] < 0.5 * np.hypot(ma["rmse_mm"][1], mb["rmse_mm"][1])
    z = T.paired_contrast(a, a, truth, [10.0, 99.0])         # identical rules; cut-off with no events
    assert z["rmse_mm"][:2] == (0.0, 0.0) and z["sensitivity_10mm"][0] == 0.0
    assert np.isnan(z["sensitivity_99mm"][0]) and z["specificity_99mm"][0] == 0.0


def test_paired_rmse_mcse_matches_replication():
    rng = np.random.default_rng(8)
    reps, n = 400, 2000
    diffs, ses = [], []
    for _ in range(reps):
        truth = np.full(n, 10.0)
        c = rng.standard_normal(n)
        a = truth + 1.5 * c + 0.5 * rng.standard_normal(n)
        b = truth + 1.5 * c + 0.2 * rng.standard_normal(n) + 0.3
        d = T.paired_contrast(a, b, truth)["rmse_mm"]
        diffs.append(d[0])
        ses.append(d[1])
    assert np.mean(ses) == pytest.approx(np.std(diffs, ddof=1), rel=0.12)


# ------------------------------------------------------------------ AP/SL joint distribution

def test_ratio_map_off_returns_same_objects(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    lat = draw_latent(p, 500, np.random.default_rng(0))
    assert T.ratio_map(lat.r, p, None) is lat.r
    assert T.ratio_map(lat.r, p, T.ratio_spec_from_params(p)) is lat.r
    assert T.rescale_sl(lat, T.ratio_map(lat.r, p, None)) is lat


def test_ratio_map_known_distributions(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    lat = draw_latent(p, 200000, np.random.default_rng(1))
    spec = dict(family="beta", r_mean=0.8, r_lower=0.4, r_upper=1.0, r_concentration=2.0)
    r = T.ratio_map(lat.r, p, spec)
    a, b = T.beta_shapes(0.8, 0.4, 1.0, 2.0)
    assert stats.kstest((r - 0.4) / 0.6, stats.beta(a, b).cdf).pvalue > 1e-3
    assert r.mean() == pytest.approx(0.8, abs=4 * r.std() / np.sqrt(r.size))
    ln = T.ratio_map(lat.r, p, dict(family="lognormal", mu_log=-0.1, sd_log=0.2))
    assert stats.kstest(np.log(ln), stats.norm(-0.1, 0.2).cdf).pvalue > 1e-3
    # monotone in the coded ratio: common random numbers across ratio models
    o = np.argsort(lat.r)
    assert (np.diff(r[o]) >= 0).all() and (np.diff(ln[o]) >= 0).all()
    assert (T.ratio_map(lat.r, p, dict(family="constant", value=1.0)) == 1.0).all()
    with pytest.raises(ValueError):
        T.ratio_map(lat.r, p, dict(family="gamma"))
    with pytest.raises(ValueError):
        T.beta_shapes(1.0, 0.4, 1.0, 8.0)


def test_rescale_sl_equals_direct_draw(cfg):
    # with r_lower = r_upper the library draws no ratio variate, so two ratios share every random number
    base = C.cell_params(cfg, "E4", E4_BASE)
    p5 = dict(base, r_lower=0.5, r_upper=0.5)
    p8 = dict(base, r_lower=0.8, r_upper=0.8)
    l5 = draw_latent(p5, 2000, np.random.default_rng(9))
    l8 = draw_latent(p8, 2000, np.random.default_rng(9))
    out = T.rescale_sl(l5, np.full(2000, 0.8))
    for k in ("S", "mu", "spans"):
        np.testing.assert_allclose(getattr(out, k), getattr(l8, k), rtol=1e-13, atol=0)
        assert np.array_equal(getattr(out, k)[:, 0], getattr(l5, k)[:, 0])      # AP axis bit-identical
    for k in ("u", "o", "over", "g", "logrr", "avail"):
        assert getattr(out, k) is getattr(l5, k)
    assert np.array_equal(l5.S[:, 1], 0.5 * l5.S[:, 0])                         # input not modified in place
    ev5 = T.pair_event_definitions(l5, 2.0)["D0_published"]
    ev8 = T.pair_event_definitions(out, 2.0)["D0_published"]
    assert np.array_equal(ev5[:, 0], ev8[:, 0]) and ev8[:, 1].sum() > ev5[:, 1].sum()


def test_rescale_sl_edges(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    lat = draw_latent(p, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        T.rescale_sl(lat, np.zeros(10))
    with pytest.raises(ValueError):
        T.rescale_sl(lat, np.ones(9))
    one = T.rescale_sl(lat, np.ones(10))                   # circular orifice
    assert np.array_equal(one.S[:, 0], one.S[:, 1])


def test_axis_prob_known_answers(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    # values of the independent audit of 2026-09-18 (closed-form quadrature, notes/scratch/2026-09-18-findings-E4.md)
    assert T.axis_prob_ge(p, None, 3.0) == pytest.approx(0.6241, abs=5e-5)
    assert T.axis_prob_ge(p, None, 5.0) == pytest.approx(0.2337, abs=5e-5)
    assert T.axis_prob_ge(p, dict(T.ratio_spec_from_params(p), r_mean=0.53), 3.0) == pytest.approx(0.8352, abs=5e-5)
    assert T.axis_prob_ge(p, dict(T.ratio_spec_from_params(p), r_mean=0.90), 3.0) == pytest.approx(0.0499, abs=5e-5)
    # constant ratio: a lognormal tail in closed form
    sap = stats.lognorm(s=p["S_log_sd"], scale=p["S_median_mm"])
    assert T.axis_prob_ge(p, dict(family="constant", value=0.5), 3.0) == pytest.approx(sap.sf(6.0), rel=1e-12)
    assert T.axis_prob_ge(p, dict(family="constant", value=1.0), 3.0) == 0.0     # circular orifice
    # lognormal family against simulation
    spec = T.spec_for_mean_difference(p, "lognormal", 3.9, log_corr=0.9)
    rng = np.random.default_rng(6)
    n = 400000
    S = p["S_median_mm"] * np.exp(p["S_log_sd"] * rng.standard_normal(n))
    r = np.exp(spec["mu_log"] + spec["sd_log"] * rng.standard_normal(n))
    for tau in (3.0, 5.0):
        emp = np.mean(np.abs(S * (1 - r)) >= tau)
        assert T.axis_prob_ge(p, spec, tau) == pytest.approx(emp, abs=4 * np.sqrt(emp * (1 - emp) / n))
    assert np.corrcoef(np.log(S), np.log(S * r))[0, 1] == pytest.approx(0.9, abs=0.005)
    assert np.mean(S - S * r) == pytest.approx(3.9, abs=0.05)


def test_spec_for_mean_difference(cfg):
    p = C.cell_params(cfg, "E4", E4_BASE)
    assert T.mean_ap_mm(p) == pytest.approx(10.0 * np.exp(0.08))
    b = T.spec_for_mean_difference(p, "beta", 3.9)
    assert b["r_mean"] == pytest.approx(0.64, abs=1e-4) and b["r_concentration"] == 8.0   # the coded calibration
    ln = T.spec_for_mean_difference(p, "lognormal", 2.0, log_corr=0.98)
    assert np.exp(ln["mu_log"] + ln["sd_log"] ** 2 / 2) == pytest.approx(1 - 2.0 / T.mean_ap_mm(p))
    assert 0.4 / np.hypot(0.4, ln["sd_log"]) == pytest.approx(0.98)
    with pytest.raises(ValueError):
        T.spec_for_mean_difference(p, "gamma", 2.0)
