"""Tests for protocol amendment 3, package A3-3 (code/lib/duomaxsim/osrev_thresholds.py)."""
import copy

import numpy as np
import pytest
from scipy import integrate, stats

from duomaxsim import config as C
from duomaxsim import osrev_thresholds as T
from duomaxsim.experiments import _gen, run_E3
from duomaxsim.model import draw_latent


def _grids(cfg):
    return {k: v.get("grid", [v["value"]]) for k, v in cfg["parameters"].items()}


# ---------------------------------------------------------------- identity with the library

@pytest.mark.parametrize("cell,overrides", [
    (29, {}),                                                   # pre-specified base cell
    (77, {}),                                                   # 8% long-axis setting
    (29, {"window": None, "K": 2}),                             # no window, two views
    (29, {"data_mode": "retrospective", "rhythm": "AF", "K": 4, "S_median_mm": 13.0}),
])
def test_rows_identical_to_run_E3(cfg, cell, overrides):
    """Same seed, same chunking: the rebuilt summary rows equal run_E3 exactly."""
    p = C.cell_params(cfg, "E3", cell, overrides)
    n, chunk = 5000, 2000                                       # three chunks, the last one short
    ref = run_E3(p, _grids(cfg), C.cell_seed(cfg, "E3", cell), n, chunk)
    pat = T.simulate_e3_patients(p, C.cell_seed(cfg, "E3", cell), n, chunk)
    new = T.e3_rows(pat["S"], pat["est"], p["cutoffs_mm"])
    assert len(new) == len(ref) == 2 * 3 * 7 * 7
    for a, b in zip(ref, new):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], float):
                # bit-identical, NaN-safe
                assert np.array_equal(np.float64(a[k]), np.float64(b[k]), equal_nan=True), (k, a, b)
            else:
                assert a[k] == b[k], (k, a, b)


def test_true_spans_are_the_library_draws(cfg):
    """S returned per patient is bit-identical to draw_latent on the same chunk generators."""
    p = C.cell_params(cfg, "E3", 29)
    ss = C.cell_seed(cfg, "E3", 29)
    pat = T.simulate_e3_patients(p, ss, 3000, 1500)
    kids = C.cell_seed(cfg, "E3", 29).spawn(2)
    S = np.concatenate([draw_latent(p, 1500, _gen(k)).S for k in kids])
    assert np.array_equal(pat["S"], S)
    assert pat["S"].shape == (3000, 2)
    assert set(pat["est"]) == set(T.E3_NAMES)
    # orderings that hold by construction
    assert (pat["est"]["A3"] >= pat["est"]["A4"]).all() and (pat["est"]["A4"] >= pat["est"]["A1"]).all()
    assert (pat["est"]["A3"] >= pat["est"]["A2"]).all()


def test_simulate_single_chunk_edge(cfg):
    p = C.cell_params(cfg, "E3", 29)
    pat = T.simulate_e3_patients(p, C.cell_seed(cfg, "E3", 29), 7, 20000)   # n smaller than one chunk
    assert pat["S"].shape == (7, 2) and pat["est"]["A1"].shape == (7, 2)


# ---------------------------------------------------------------- counts

TRUTH = np.array([5.0, 9.0, 10.0, 10.0, 12.0, 15.0, 9.99, 11.0])
EST = np.array([10.0, 8.0, 9.99, 10.0, 13.0, 9.0, 10.5, 10.0])
REF = np.array([9.0, 11.0, 10.2, 9.0, 13.0, 16.0, 9.0, 9.5])


def test_confusion_counts_known_answer():
    g = T.confusion_counts(TRUTH, EST, 10.0, ref=REF)
    # events (truth >= 10): idx 2,3,4,5,7; positives (est >= 10): idx 0,3,4,6,7
    assert (g["n"], g["tp"], g["fn"], g["fp"], g["tn"]) == (8, 3, 2, 2, 1)
    # ref positives: idx 1,2,4,5. events: up = {3,7}, dn = {2,5}; non-events: up = {0,6}, dn = {1}
    assert (g["ev_up"], g["ev_dn"], g["ne_up"], g["ne_dn"]) == (2, 2, 2, 1)


def test_confusion_counts_edges():
    # a value exactly at the cut-off is an event and a positive (>=, as in run_E3)
    g = T.confusion_counts(np.array([10.0]), np.array([10.0]), 10.0)
    assert (g["tp"], g["fp"], g["fn"], g["tn"]) == (1, 0, 0, 0)
    # empty mask
    g = T.confusion_counts(TRUTH, EST, 10.0, ref=REF, mask=np.zeros(8, bool))
    assert all(v == 0 for v in g.values())
    # comparator equal to the rule: nothing is reclassified
    g = T.confusion_counts(TRUTH, EST, 10.0, ref=EST)
    assert (g["ev_up"], g["ev_dn"], g["ne_up"], g["ne_dn"]) == (0, 0, 0, 0)


def test_classification_metrics_known_answer():
    m = T.classification_metrics(80, 30, 20, 870)
    assert m["sensitivity"] == pytest.approx(0.8)
    assert m["specificity"] == pytest.approx(870 / 900)
    assert m["ppv"] == pytest.approx(80 / 110)
    assert m["npv"] == pytest.approx(870 / 890)
    assert m["prevalence"] == pytest.approx(0.1)
    assert m["accuracy"] == pytest.approx(0.95)
    assert m["sensitivity_mcse"] == pytest.approx(np.sqrt(0.8 * 0.2 / 100))
    assert m["sensitivity_lo95"] == pytest.approx(0.8 - 1.959964 * 0.04)
    assert m["sensitivity_hi95"] == pytest.approx(0.8 + 1.959964 * 0.04)
    assert m["npv_n"] == 890 and m["ppv_n"] == 110


def test_classification_metrics_edges():
    m = T.classification_metrics(0, 0, 0, 50)                   # no events, no positives
    assert np.isnan(m["sensitivity"]) and np.isnan(m["ppv"]) and np.isnan(m["sensitivity_mcse"])
    assert m["specificity"] == 1.0 and m["specificity_mcse"] == 0.0
    assert m["specificity_lo95"] == 1.0 and m["specificity_hi95"] == 1.0
    # vectorised and clipped to [0, 1]
    m = T.classification_metrics(np.array([1, 99]), np.array([0, 5]), np.array([99, 1]), np.array([5, 0]))
    assert m["sensitivity"].shape == (2,)
    assert (m["sensitivity_lo95"] >= 0).all() and (m["sensitivity_hi95"] <= 1).all()


def test_reclassification_known_answer_and_identity():
    g = T.confusion_counts(TRUTH, EST, 10.0, ref=REF)
    r = T.reclassification_metrics(g["ev_up"], g["ev_dn"], g["ne_up"], g["ne_dn"], 5, 3)
    assert r["nri_events"] == pytest.approx(0.0)
    assert r["nri_nonevents"] == pytest.approx(-1 / 3)
    assert r["nri"] == pytest.approx(-1 / 3)
    assert r["net_correct"] == pytest.approx(((2 - 2) + (1 - 2)) / 8)
    assert r["nri_events_mcse"] == pytest.approx(np.sqrt((0.4 + 0.4 - 0.0) / 5))
    # identities: components are differences in sensitivity, specificity and accuracy
    a = T.classification_metrics(g["tp"], g["fp"], g["fn"], g["tn"])
    gr = T.confusion_counts(TRUTH, REF, 10.0)
    b = T.classification_metrics(gr["tp"], gr["fp"], gr["fn"], gr["tn"])
    assert r["nri_events"] == pytest.approx(a["sensitivity"] - b["sensitivity"])
    assert r["nri_nonevents"] == pytest.approx(a["specificity"] - b["specificity"])
    assert r["net_correct"] == pytest.approx(a["accuracy"] - b["accuracy"])


def test_reclassification_edges():
    r = T.reclassification_metrics(0, 0, 0, 0, 10, 20)          # rule = comparator
    assert all(r[k] == 0.0 for k in r)
    r = T.reclassification_metrics(0, 0, 0, 0, 0, 20)           # no events
    assert np.isnan(r["nri_events"]) and r["nri_nonevents"] == 0.0


def test_near_band_known_answer_and_edge():
    g = T.near_band_counts(TRUTH, EST, 10.0, 1.0)
    # |truth - 10| <= 1: idx 1 (9), 2, 3 (10), 6 (9.99), 7 (11)
    assert (g["n"], g["tp"], g["fn"], g["fp"], g["tn"]) == (5, 2, 1, 1, 1)
    g0 = T.near_band_counts(TRUTH, EST, 30.0, 1.0)              # nobody near the cut-off
    assert all(v == 0 for v in g0.values())
    # closed band: a truth exactly half_width away is included on both sides
    g1 = T.near_band_counts(np.array([9.0, 11.0]), np.array([9.0, 11.0]), 10.0, 1.0)
    assert (g1["n"], g1["tp"], g1["tn"]) == (2, 1, 1)


def test_distance_bins_known_answer():
    b = T.distance_bin_counts(TRUTH, EST, 10.0, bin_width=0.5, max_dist=1.0)
    assert np.allclose(b["lo"], [-1.0, -0.5, 0.0, 0.5]) and np.allclose(b["hi"], [-0.5, 0.0, 0.5, 1.0])
    # d = truth - 10: -5 (out), -1, 0, 0, 2 (out), 5 (out), -0.01, 1 (out: upper edge is open)
    assert b["n"].tolist() == [1, 1, 2, 0]
    assert b["n_pos"].tolist() == [0, 1, 1, 0]
    assert b["n_mis"].tolist() == [0, 1, 1, 0]   # FP below the cut-off, FN at or above it


def test_distance_bins_edges_and_totals(cfg):
    e = T.distance_bin_edges(0.5, 5.0)
    assert len(e) == 21 and e[0] == -5.0 and e[-1] == 5.0 and 0.0 in e
    # no patient in range
    b = T.distance_bin_counts(np.array([100.0]), np.array([1.0]), 10.0)
    assert b["n"].sum() == 0 and b["n_mis"].sum() == 0
    # wide range: bins partition all patients, and misclassified = FP + FN
    p = C.cell_params(cfg, "E3", 29)
    pat = T.simulate_e3_patients(p, C.cell_seed(cfg, "E3", 29), 4000, 4000)
    S, est = pat["S"][:, 0], pat["est"]["A3"][:, 0]
    b = T.distance_bin_counts(S, est, 10.0, bin_width=0.5, max_dist=500.0)
    g = T.confusion_counts(S, est, 10.0)
    assert b["n"].sum() == 4000
    assert b["n_mis"][b["lo"] < 0].sum() == g["fp"] and b["n_mis"][b["lo"] >= 0].sum() == g["fn"]
    # the +-1 mm band is the four central 0.5 mm bins up to boundary ties (none in continuous data)
    nb = T.near_band_counts(S, est, 10.0, 1.0)
    cen = (b["lo"] >= -1.0) & (b["hi"] <= 1.0)
    assert b["n"][cen].sum() == nb["n"] and b["n_mis"][cen].sum() == nb["fp"] + nb["fn"]


def test_count_tables_consistency(cfg):
    p = C.cell_params(cfg, "E3", 29)
    pat = T.simulate_e3_patients(p, C.cell_seed(cfg, "E3", 29), 3000, 3000)
    tabs = T.count_tables(pat["S"], pat["est"], [7.0, 10.0, 13.0])
    assert len(tabs["overall"]) == 2 * 3 * 4 and len(tabs["band"]) == 2 * 3 * 4 * 2
    assert len(tabs["bins"]) == 2 * 3 * 4 * 20
    for r in tabs["overall"]:
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == r["n"] == 3000
        if r["rule"] == "A1":
            assert r["ev_up"] == r["ev_dn"] == r["ne_up"] == r["ne_dn"] == 0
        if r["rule"] in ("A3", "A4"):                           # never below the anchor-view mean
            assert r["ev_dn"] == 0 and r["ne_dn"] == 0
    for ax in ("AP", "SL"):
        assert sum(r["n"] for r in tabs["hist"] if r["axis"] == ax) == 3000
    # the overall table agrees with the library summary rows
    rows = {(r["estimator"], r["axis"], r["cutoff_mm"], r["metric"]): r["value"]
            for r in T.e3_rows(pat["S"], pat["est"], [7.0, 10.0, 13.0])}
    for r in tabs["overall"]:
        m = T.classification_metrics(r["tp"], r["fp"], r["fn"], r["tn"])
        assert float(m["sensitivity"]) == rows[(r["rule"], r["axis"], r["cutoff_mm"], "sensitivity")]
        assert float(m["specificity"]) == rows[(r["rule"], r["axis"], r["cutoff_mm"], "specificity")]


# ---------------------------------------------------------------- true-span distribution

def test_span_distribution_ap_known_answer(base):
    ln = stats.lognorm(s=0.40, scale=10.0)
    x = np.array([0.5, 7.0, 10.0, 13.0, 30.0])
    assert np.allclose(T.span_cdf(base, "AP", x), ln.cdf(x), rtol=0, atol=1e-14)
    assert np.allclose(T.span_pdf(base, "AP", x), ln.pdf(x), rtol=1e-13)
    q = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    assert np.allclose(T.span_quantile(base, "AP", q), ln.ppf(q), rtol=1e-12)
    d = T.span_distribution(base, "AP")
    assert d["median_mm"] == pytest.approx(10.0) and d["prop_ge_10mm"] == pytest.approx(0.5)
    assert d["p25_mm"] == pytest.approx(10 * np.exp(-0.40 * 0.6744897501960817))
    assert d["mean_mm"] == pytest.approx(10 * np.exp(0.08)) and d["sd_mm"] == pytest.approx(ln.std())


def test_span_distribution_sl_against_independent_quadrature(base):
    lo, hi = base["r_lower"], base["r_upper"]
    m = (base["r_mean"] - lo) / (hi - lo)
    a, b = m * base["r_concentration"], (1 - m) * base["r_concentration"]
    assert (a, b) == pytest.approx((3.2, 4.8))

    def cdf(x):
        f = lambda t: stats.norm.cdf((np.log(x / (lo + (hi - lo) * t)) - np.log(10.0)) / 0.40) * stats.beta.pdf(t, a, b)
        return integrate.quad(f, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]

    for x in (3.0, 7.0, 10.0, 13.0, 20.0):
        assert T.span_cdf(base, "SL", x)[0] == pytest.approx(cdf(x), abs=1e-9)
    # density integrates to the CDF, quantiles invert it, mean = E[AP] E[r]
    xs = np.linspace(1e-6, 13.0, 20001)
    assert integrate.trapezoid(T.span_pdf(base, "SL", xs), xs) == pytest.approx(cdf(13.0), abs=1e-6)
    q = T.span_quantile(base, "SL", [0.25, 0.5, 0.75])
    assert np.allclose(T.span_cdf(base, "SL", q), [0.25, 0.5, 0.75], atol=1e-9)
    mean, sd = T.span_mean_sd(base, "SL")
    assert mean == pytest.approx(10 * np.exp(0.08) * 0.64)
    # agreement with the generator (Monte Carlo, n = 200,000; tolerance about 5 MCSE)
    p = copy.deepcopy(base)
    S = draw_latent(p, 200000, np.random.default_rng(12345)).S
    assert S[:, 1].mean() == pytest.approx(mean, abs=5 * sd / np.sqrt(200000))
    for c in (7.0, 10.0, 13.0):
        pr = 1 - T.span_cdf(base, "SL", c)[0]
        assert (S[:, 1] >= c).mean() == pytest.approx(pr, abs=5 * np.sqrt(pr * (1 - pr) / 200000))


def test_span_distribution_edges(base):
    # degenerate ratio (r_mean at the upper bound): SL equals AP
    p = copy.deepcopy(base)
    p["r_mean"] = p["r_upper"]
    x = np.array([5.0, 10.0, 20.0])
    assert np.allclose(T.span_cdf(p, "SL", x), T.span_cdf(p, "AP", x), atol=1e-15)
    assert np.allclose(T.span_pdf(p, "SL", x), T.span_pdf(p, "AP", x), rtol=1e-13)
    assert T.span_mean_sd(p, "SL") == pytest.approx(T.span_mean_sd(p, "AP"))
    # non-positive spans have zero density and zero probability; scaling the median scales quantiles
    assert T.span_pdf(base, "AP", [0.0, -1.0]).tolist() == [0.0, 0.0]
    assert T.span_cdf(base, "SL", [0.0])[0] == 0.0
    p13 = copy.deepcopy(base)
    p13["S_median_mm"] = 13.0
    assert np.allclose(T.span_quantile(p13, "SL", [0.1, 0.9]), 1.3 * T.span_quantile(base, "SL", [0.1, 0.9]),
                       rtol=1e-9)
    with pytest.raises(ValueError):
        T.span_cdf(base, "XX", 1.0)
