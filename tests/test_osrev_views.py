"""Protocol amendment 3, package A3-2 (view rules): identity, known-answer and edge-case tests
for code/lib/duomaxsim/osrev_views.py."""
import numpy as np
import pandas as pd
import pytest

import make_fixture as MF
from conftest import QUIET, ROOT
from duomaxsim import config as C
from duomaxsim import osrev_views as V
from duomaxsim.experiments import run_cells
from duomaxsim.metrics import ErrAcc, expected_max_std_normals
from duomaxsim.model import draw_latent, measure

AMEND2 = ROOT / "code" / "configs" / "amend2_2026-09-18.yaml"


def _gen(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


# ---------------------------------------------------------------- identity with the library

@pytest.mark.parametrize("cfg_path,cells,n", [
    (ROOT / "tests" / "fixtures" / "config_v01.yaml", [0, 700], 3000),
    (AMEND2, [5, 437, 863], 2500),          # includes window cells, K 2/3/4, chunked RNG (n < chunk)
])
def test_full_arm_reproduces_run_E1_bit_for_bit(cfg_path, cells, n):
    """Options off: bias, RMSE, SD, relative bias (and their MCSE), limited-sampling proportion and
    selection inflation equal the rows written by experiments.run_E1 exactly."""
    cfg = C.load_config(cfg_path)
    ref = run_cells(cfg, "E1", cells, n_rep=n)
    chunk = int(cfg["meta"]["chunk_size"])
    for ci in cells:
        p = C.cell_params(cfg, "E1", ci)
        sim = V.simulate_view_rules(p, C.cell_seed(cfg, "E1", ci), n, chunk)
        r = ref[(ref.cell == ci) & (ref.estimand.isin(["T1"]) | ref.estimand.isna())]
        for rule in V.RULES:
            for d, ax in enumerate(V.AXES):
                rr = r[(r.estimator == rule) & (r.axis == ax)]
                if rule == "A4":
                    rr = rr[(rr.s_det == p["s_det"]) & (rr.f_rej == p["f_rej"])]
                summ = sim["acc"][(rule, d)].summary()
                assert len(rr) == 4
                for _, row in rr.iterrows():
                    v, se = summ[row.metric]
                    assert v == row.value and se == row.mcse, (ci, rule, ax, row.metric)
                # the per-patient arrays carry the same information as the accumulator
                e = sim["est"]["full"][rule][:, d] - sim["S"][:, d]
                np.testing.assert_allclose(e.mean(), summ["bias_mm"][0], rtol=0, atol=1e-12)
        for d, ax in enumerate(V.AXES):
            lim = r[(r.metric == "p_limited_anchor") & (r.axis == ax)].iloc[0]
            assert sim["lim_anchor"][d].summary()[0] == lim.value
            inf = r[(r.metric == "selection_inflation_mm") & (r.axis == ax)]
            if len(inf):
                assert sim["infl"][d].summary()[0] == inf.iloc[0].value


def test_regression_fixture_rows_reproduced():
    """Same check against the frozen regression fixture (csv, 12 significant digits)."""
    cfg = C.load_config(ROOT / "tests" / "fixtures" / "config_v01.yaml")
    ref = pd.read_csv(MF.FIXTURE)
    ref = ref[(ref.experiment == "E1") & (ref.cell == 700) & (ref.estimand == "T1")]
    p = C.cell_params(cfg, "E1", 700)
    sim = V.simulate_view_rules(p, C.cell_seed(cfg, "E1", 700), 3000, int(cfg["meta"]["chunk_size"]))
    for rule in ("A1", "A2", "A3"):
        for d, ax in enumerate(V.AXES):
            rr = ref[(ref.estimator == rule) & (ref.axis == ax)].set_index("metric")
            s = sim["acc"][(rule, d)].summary()
            for m in ("bias_mm", "rmse_mm", "sd_err_mm"):
                np.testing.assert_allclose(s[m][0], rr.at[m, "value"], rtol=1e-9, atol=1e-12)


def test_measure_from_z_identical_to_measure(base):
    p = dict(base, measure_floor_mm=0.5)
    lat = draw_latent(p, 500, _gen(3))
    rb = 0.75 * _gen(4).standard_normal(500)
    a = measure(lat, p, _gen(5), rb)
    b = V.measure_from_z(lat, p, _gen(5).standard_normal(lat.spans.shape), rb)
    np.testing.assert_array_equal(a, b)
    # edge: z None means no caliper error
    np.testing.assert_array_equal(V.measure_from_z(lat, dict(p, measure_floor_mm=None), None, np.zeros(500)),
                                  lat.spans)


def test_counterfactual_latent_options_off_is_identity(base):
    lat = draw_latent(base, 200, _gen(1))
    assert V.counterfactual_latent(lat) is lat


def test_counterfactual_latent_known_answers(base):
    p = dict(base, K=4)
    lat = draw_latent(p, 400, _gen(2))
    zn = V.counterfactual_latent(lat, zero_noise=True)
    np.testing.assert_array_equal(zn.spans, np.broadcast_to((lat.g[:, None, None] * lat.mu)[..., None],
                                                            lat.spans.shape))
    np.testing.assert_array_equal(zn.mu, lat.mu)
    ev = V.counterfactual_latent(lat, equal_views=True)
    for k in range(4):
        np.testing.assert_array_equal(ev.mu[..., k], lat.mu[..., 0])
        np.testing.assert_array_equal(ev.over[..., k], lat.over[..., 0])
    np.testing.assert_array_equal(ev.spans[:, :, 0, :], lat.spans[:, :, 0, :])     # anchor untouched
    # equal_views equals the published model evaluated with the anchor's offsets: the beat-level
    # multiplicative noise spans / (g mu) of each view is preserved
    np.testing.assert_allclose(ev.spans / (lat.g[:, None, None, None] * ev.mu[..., None]),
                               lat.spans / (lat.g[:, None, None, None] * lat.mu[..., None]), rtol=1e-12)
    both = V.counterfactual_latent(lat, True, True)
    np.testing.assert_array_equal(both.spans, np.broadcast_to((lat.g[:, None, None] * lat.mu[..., :1])[..., None],
                                                              lat.spans.shape))
    # the original is not modified
    assert not np.shares_memory(ev.mu, lat.mu) and (lat.mu[..., 1] != lat.mu[..., 0]).any()


# ---------------------------------------------------------------- decomposition known answers

def test_decomposition_known_answers(base):
    p = dict(base, window=None, K=3)
    sim = V.simulate_view_rules(p, np.random.SeedSequence(11), 6000, 2000, arms=tuple(V.ARMS))
    for d in range(2):
        dec = V.decomposition(sim, d)
        # both off: every view mean identical, so the excess is exactly 0
        ex = sim["est"]["zero_noise_equal_views"]["A3"][:, d] - sim["est"]["zero_noise_equal_views"]["A1"][:, d]
        assert (ex == 0).all() and dec["excess_zero_noise_equal_views"] == 0.0
        # identity of the 2 x 2 factorial
        np.testing.assert_allclose(dec["excess_full"], dec["excess_zero_noise"] + dec["excess_equal_views"]
                                   + dec["interaction"], atol=1e-12)
        np.testing.assert_allclose(dec["share_offset_first"] + dec["share_noise_last"], 1.0, atol=1e-12)
        np.testing.assert_allclose(dec["share_offset_last"] + dec["share_noise_first"], 1.0, atol=1e-12)
        # A1 is unchanged by equal_views (same anchor draws): common random numbers
        np.testing.assert_array_equal(sim["est"]["equal_views"]["A1"][:, d], sim["est"]["full"]["A1"][:, d])
    # zero noise: excess = g * S * (max_v f_v - f_0), f_v = 1 - u_v + o_v, recomputed from the latent draws
    kids = np.random.SeedSequence(11).spawn(3)   # spawn is stateful: use a fresh SeedSequence
    exp = []
    for ch in kids:
        lat = draw_latent(p, 2000, np.random.Generator(np.random.PCG64(ch)))
        f = 1.0 - lat.u + lat.o
        exp.append(lat.g[:, None] * lat.S * (f.max(-1) - f[..., 0]))
    exp = np.concatenate(exp)
    got = sim["est"]["zero_noise"]["A3"] - sim["est"]["zero_noise"]["A1"]
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-9)
    assert (got >= 0).all()


def test_noise_only_excess_matches_order_statistics(base):
    """No view errors, no beat variation: view means are iid N(S, sigma_cal^2 / N), so
    E[A3 - A1] = sigma_cal / sqrt(N) * E[max of K standard normals]."""
    for K in (2, 4):
        p = dict(base, **QUIET, K=K, N_beats=3)
        p["sigma_cal_mm"] = 1.0
        sim = V.simulate_view_rules(p, np.random.SeedSequence(7 + K), 40000, 20000, arms=("full", "equal_views"))
        x = sim["est"]["full"]["A3"] - sim["est"]["full"]["A1"]
        m, se = V.paired_mean(x)
        assert abs(m - expected_max_std_normals(K) / np.sqrt(3)) < 4 * se
        # with no view errors the equal_views arm changes nothing
        np.testing.assert_allclose(sim["est"]["equal_views"]["A3"], sim["est"]["full"]["A3"], rtol=1e-12)


def test_single_view_edge_case(base):
    p = dict(base, K=1, window=None)
    sim = V.simulate_view_rules(p, np.random.SeedSequence(5), 300, 200, arms=tuple(V.ARMS))
    for arm in V.ARMS:
        for r in ("A2", "A3", "A4"):
            np.testing.assert_array_equal(sim["est"][arm][r], sim["est"][arm]["A1"])
        assert (sim["sel"][arm] == 0).all()
    assert sim["S"].shape == (300, 2)
    with pytest.raises(ValueError):
        V.simulate_view_rules(p, np.random.SeedSequence(5), 10, 10, arms=("nonsense",))


# ---------------------------------------------------------------- summaries

def test_error_summary_known_answer():
    e = np.arange(-50, 51, dtype=float)             # 101 values, symmetric
    rows, reps = V.error_summary(e, n_groups=101)   # leave-one-out jackknife
    m = {r["metric"]: r for r in rows}
    assert m["bias_mm"]["value"] == 0.0
    np.testing.assert_allclose(m["rmse_mm"]["value"], np.sqrt(850.0))        # (n^2 - 1) / 12 with n = 101
    np.testing.assert_allclose(m["mae_mm"]["value"], 2 * 1275 / 101)
    assert m["q50_err_mm"]["value"] == 0.0 and m["q2.5_err_mm"]["value"] == -47.5
    assert m["q97.5_err_mm"]["value"] == 47.5 and m["q95_abs_mm"]["value"] == 48.0
    np.testing.assert_allclose(m["ratio_q95abs_rmse"]["value"], 48.0 / np.sqrt(850.0))
    assert m["p_abs_gt3mm"]["count"] == 94 and m["p_abs_gt3mm"]["n"] == 101
    np.testing.assert_allclose(m["bias_mm"]["mcse"], e.std(ddof=1) / np.sqrt(101))
    # RMSE value and MCSE agree with the library accumulator
    acc = ErrAcc()
    acc.add(e + 100.0, np.full(101, 100.0))
    np.testing.assert_allclose([m["rmse_mm"]["value"], m["rmse_mm"]["mcse"]], acc.summary()["rmse_mm"])
    # leave-one-out jackknife of the mean equals sd / sqrt(n)
    k = V.STAT_NAMES.index("bias_mm")
    np.testing.assert_allclose(V.jackknife_se(reps)[k], e.std(ddof=1) / np.sqrt(101))


def test_error_summary_normal_and_edges():
    e = _gen(9).standard_normal(100000)
    rows, _ = V.error_summary(e)
    m = {r["metric"]: r for r in rows}
    assert abs(m["ratio_q95abs_rmse"]["value"] - 1.959964) < 4 * m["ratio_q95abs_rmse"]["mcse"]
    # jackknife SE of the median close to the asymptotic value sqrt(pi / 2) / sqrt(n)
    assert 0.6 < m["q50_err_mm"]["mcse"] / (np.sqrt(np.pi / 2) / np.sqrt(1e5)) < 1.6
    assert abs(m["p_abs_gt2mm"]["value"] - 0.0455) < 4 * m["p_abs_gt2mm"]["mcse"]
    # edges: constant error and a single observation
    rows, _ = V.error_summary(np.full(10, 2.0), n_groups=5)
    m = {r["metric"]: r for r in rows}
    assert m["rmse_mm"]["value"] == 2.0 and m["sd_err_mm"]["value"] == 0.0 and m["q95_abs_mm"]["mcse"] == 0.0
    rows, _ = V.error_summary(np.array([1.5]))
    assert {r["metric"]: r for r in rows}["mae_mm"]["value"] == 1.5
    with pytest.raises(ValueError):
        V.error_summary(np.array([]))


def test_conditional_summary_known_answer_and_empty_bin():
    s = np.repeat([6.0, 8.0, 20.0], 40)                    # bins 5 to 7, 7 to 10, >= 16; others empty
    e = np.concatenate([np.linspace(-1, 1, 40), np.linspace(-2, 2, 40), np.linspace(0, 4, 40)])
    perm = _gen(1).permutation(120)
    rows = pd.DataFrame(V.conditional_summary(e[perm], s[perm], n_groups=10))
    g = rows.pivot(index="span_bin", columns="metric", values="value")
    n = rows.drop_duplicates("span_bin").set_index("span_bin").n
    assert n["5 to 7"] == 40 and n["7 to 10"] == 40 and n[">= 16"] == 40 and n["all"] == 120
    assert n["0 to 5"] == 0 and n["10 to 13"] == 0 and np.isnan(g.at["13 to 16", "rmse_mm"])
    np.testing.assert_allclose(g.at["7 to 10", "q50_err_mm"], 0.0, atol=1e-12)
    np.testing.assert_allclose(g.at[">= 16", "q50_err_mm"], 2.0)
    np.testing.assert_allclose(g.at[">= 16", "q97.5_err_mm"], np.percentile(np.linspace(0, 4, 40), 97.5))
    np.testing.assert_allclose(g.at["5 to 7", "bias_mm"], 0.0, atol=1e-12)
    # boundary: a span exactly on an edge belongs to the upper bin (left-closed)
    r2 = pd.DataFrame(V.conditional_summary(np.array([1.0, 2.0]), np.array([7.0, 7.0]), n_groups=2))
    assert r2[r2.span_bin == "7 to 10"].n.iloc[0] == 2 and r2[r2.span_bin == "5 to 7"].n.iloc[0] == 0


def test_ratio_and_paired_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert V.paired_mean(x) == (2.5, pytest.approx(x.std(ddof=1) / 2))
    r, se = V.ratio_of_means(2 * x, x)
    assert r == 2.0 and se == pytest.approx(0.0)
    assert np.isnan(V.ratio_of_means(x, np.zeros(4))[0])
