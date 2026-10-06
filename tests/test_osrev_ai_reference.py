"""Tests for code/lib/duomaxsim/osrev_ai_reference.py (protocol amendment 3, A3-5).

1. With the new options off, the draw functions reproduce the arrays and the stored summaries of
   experiments.run_E5 and run_E6 exactly (same seed, bit-identical).
2. Known-answer and edge-case tests for every new function.
"""
import itertools

import numpy as np
import pytest
from scipy import stats

from duomaxsim import config as C
from duomaxsim import experiments as EX
from duomaxsim import osrev_ai_reference as OA
from duomaxsim.experiments import run_cells

from conftest import QUIET, ROOT

AMEND2 = ROOT / "code" / "configs" / "amend2_2026-09-18.yaml"


@pytest.fixture(scope="module")
def cfg2():
    cfg = C.load_config(AMEND2)
    cfg["parameters"]["calib_n"]["value"] = 2000
    return cfg


def _capture(monkeypatch):
    """Record every array returned by experiments.protocol_read while a runner executes."""
    got = []
    orig = EX.protocol_read

    def wrapped(*a, **k):
        out = orig(*a, **k)
        got.append(out[0].copy() if isinstance(out, tuple) else out.copy())
        return out

    monkeypatch.setattr(EX, "protocol_read", wrapped)
    return got


# ---------------------------------------------------------------- 1. exact reproduction, options off

@pytest.mark.parametrize("cell", [9, 11])
def test_e5_draws_reproduce_run_E5(cfg2, monkeypatch, cell):
    n_studies = 150            # two chunks of 100 studies: checks the chunk seeding as well
    got = _capture(monkeypatch)
    df = run_cells(cfg2, "E5", [cell], n_rep=n_studies)
    monkeypatch.undo()
    p = C.cell_params(cfg2, "E5", cell)
    ss = C.cell_seed(cfg2, "E5", cell)
    chunk = int(cfg2["meta"]["chunk_size"])
    d = OA.e5_draws(p, ss, n_studies, chunk)
    # arrays: calls 1..3 and 4..6 are the three reads of chunks 1 and 2 (call 0 = calibration)
    assert len(got) == 7
    for j in range(3):
        lib = np.concatenate([got[1 + j][:, 0].reshape(100, 200), got[4 + j][:, 0].reshape(50, 200)])
        assert np.array_equal(lib, d[f"r{j + 1}"])
    # stored summaries
    grids = {k: v.get("grid", [v["value"]]) for k, v in cfg2["parameters"].items()}
    lam = float(p["ai_shared_lambda"])
    m = OA.e5_library_metrics(d, p, grids["sigma_ai_mm"], grids["adj_tol_mm"], lam=lam)
    n_checked = 0
    for (ai, sig, ref, met), (val, se) in ((k, v) for k, v in m.items() if len(k) == 4 and k[0] not in ("paired",)):
        row = df[(df.ai == ai) & (df.sigma_ai_mm == sig) & (df.ref == ref) & (df.metric == met)]
        assert len(row) == 1
        assert row.value.iloc[0] == val and row.mcse.iloc[0] == se, (ai, sig, ref, met)
        n_checked += 1
    assert n_checked == 3 * 3 * 5 * 15
    for (_, ref, met), (val, se) in ((k, v) for k, v in m.items() if k[0] == "ref"):
        row = df[df.ai.isna() & (df.ref == ref) & (df.metric == met)]
        assert row.value.iloc[0] == val and row.mcse.iloc[0] == se
    for (_, ai, sig, ref), val in ((k, v) for k, v in m.items() if k[0] == "paired"):
        row = df[(df.ai == ai) & (df.sigma_ai_mm == sig) & (df.ref == ref)
                 & (df.metric == "p_lower_mae_than_independent")]
        assert row.value.iloc[0] == val
    cal = df[df.estimator == "calibration"].set_index("metric").value
    assert cal["inherited_intercept_mm"] == d["a_int"] and cal["inherited_slope"] == d["b_slope"]


def test_e5_extra_noise_leaves_other_arrays_unchanged(cfg2):
    p = C.cell_params(cfg2, "E5", 9)
    # SeedSequence.spawn is stateful: a fresh SeedSequence is needed for each call
    a = OA.e5_draws(p, C.cell_seed(cfg2, "E5", 9), 120, 20000)
    b = OA.e5_draws(p, C.cell_seed(cfg2, "E5", 9), 120, 20000, extra_noise=True)
    for k in ("T1", "R_img", "r1", "r2", "r3", "b1", "b2", "b3", "z"):
        assert np.array_equal(a[k], b[k]), k
    assert "z2" not in a and b["z2"].shape == a["z"].shape
    assert abs(np.corrcoef(b["z"].ravel(), b["z2"].ravel())[0, 1]) < 0.03


def test_e5_components_add_up(cfg2):
    """Quiet model (no noise, no view error, no reader offsets): every read and R_img equal T1."""
    p = C.cell_params(cfg2, "E5", 9)
    p.update(QUIET)
    d = OA.e5_draws(p, np.random.SeedSequence(5), 5, 20000)
    for k in ("r1", "r2", "r3", "R_img"):
        np.testing.assert_allclose(d[k], d["T1"], rtol=0, atol=1e-12)
    assert np.all(d["b1"] == 0)
    assert d["a_int"] == pytest.approx(0.0, abs=1e-9) and d["b_slope"] == pytest.approx(1.0, abs=1e-9)


def test_e5_reader_offsets_are_three_distinct_pool_members(cfg2):
    p = C.cell_params(cfg2, "E5", 9)
    d = OA.e5_draws(p, np.random.SeedSequence(6), 4, 20000)
    b = np.stack([d["b1"], d["b2"], d["b3"]], -1)               # (S, n, 3)
    assert np.all(b[..., 0] != b[..., 1]) and np.all(b[..., 0] != b[..., 2]) and np.all(b[..., 1] != b[..., 2])
    for s in range(4):
        assert len(np.unique(b[s])) == 4                        # pool of four readers per study


@pytest.mark.parametrize("cell", [22])
def test_e6_draws_reproduce_run_E6(cfg2, monkeypatch, cell):
    n_studies = 30             # 20000 // 750 = 26 studies per chunk: two chunks
    got = _capture(monkeypatch)
    df = run_cells(cfg2, "E6", [cell], n_rep=n_studies)
    monkeypatch.undo()
    p = C.cell_params(cfg2, "E6", cell)
    ss = C.cell_seed(cfg2, "E6", cell)
    d = OA.e6_draws(p, ss, n_studies, int(cfg2["meta"]["chunk_size"]))
    assert len(got) == 6       # per chunk: r1, r2, manual
    for j, k, n in ((0, "r1", 250), (1, "r2", 250), (2, "man", 500)):
        lib = np.concatenate([got[j][:, 0].reshape(26, n), got[3 + j][:, 0].reshape(4, n)])
        assert np.array_equal(lib, d[k]), k
    m = OA.e6_library_metrics(d, p)
    for (kind, *rest), val in m.items():
        if kind == "double":
            nn, met = rest
            row = df[(df.n_double == nn) & (df.metric == met)]
            assert len(row) >= 1 and np.all(row.value.to_numpy() == val), (nn, met)
        else:
            ns, al, name = rest
            row = df[(df.sentinel_n == ns) & (df.anchor_alpha == al) & (df.metric == "power_" + name)]
            assert len(row) == 1 and row.value.iloc[0] == val, (ns, al, name)


def test_e6_skip_double_changes_stream_but_not_model(cfg2):
    p = C.cell_params(cfg2, "E6", 22)
    d = OA.e6_draws(p, np.random.SeedSequence(8), 30, 20000, skip_double=True)
    assert "r1" not in d and d["man"].shape == (30, 500)
    # the AI draft is T1 + bias + sigma x noise
    e = d["ai"] - d["T1"]
    assert e.mean() == pytest.approx(p["ai_draft_bias_mm"], abs=0.08)
    assert e.std() == pytest.approx(p["ai_draft_sigma_mm"], abs=0.08)


def test_reader_panel_reads_reproduce_run_E6_double_reads(cfg2, monkeypatch):
    got = _capture(monkeypatch)
    run_cells(cfg2, "E6", [22], n_rep=20)       # one chunk of 20 studies, 250 double-read cases
    monkeypatch.undo()
    p = C.cell_params(cfg2, "E6", 22)
    ss = C.cell_seed(cfg2, "E6", 22)
    rng = EX._gen(ss.spawn(1)[0])
    reads, b = OA.reader_panel_reads(p, rng, n_panels=20, n_rep=1, n_cases=250, n_readers=2)
    assert np.array_equal(reads[:, 0, :, 0], got[0][:, 0].reshape(20, 250))
    assert np.array_equal(reads[:, 0, :, 1], got[1][:, 0].reshape(20, 250))
    assert b.shape == (20, 2)


def test_reference_designs_match_runner_arithmetic():
    rng = np.random.default_rng(0)
    r1, r2, r3 = rng.normal(10, 2, (3, 4, 50))
    refs = OA.reference_designs(r1, r2, r3, [1.0, 2.0])
    assert refs["single"] is r1
    np.testing.assert_array_equal(refs["mean2"], 0.5 * (r1 + r2))
    far = np.abs(r1 - r2) > 2.0
    np.testing.assert_array_equal(refs["adj_tol2"][far], r3[far])
    np.testing.assert_array_equal(refs["adj_tol2"][~far], (0.5 * (r1 + r2))[~far])
    # edge: tolerance larger than every disagreement gives the mean of two; negative gives the adjudicator
    np.testing.assert_array_equal(OA.reference_designs(r1, r2, r3, [1e9])["adj_tol1e+09"], 0.5 * (r1 + r2))
    np.testing.assert_array_equal(OA.reference_designs(r1, r2, r3, [-1.0])["adj_tol-1"], r3)
    cp = OA.reference_designs(np.array([1.0]), np.array([5.0]), np.array([4.0]), [1.0], "closest_pair_mean")
    assert cp["adj_tol1"][0] == 4.5
    with pytest.raises(ValueError):
        OA.reference_designs(r1, r2, r3, [1.0], "other")


# ---------------------------------------------------------------- 2. analytic helpers

def test_folded_normal_mean_known_values():
    assert OA.folded_normal_mean(0.0, 2.0) == pytest.approx(2.0 * np.sqrt(2 / np.pi), rel=1e-14)
    assert OA.folded_normal_mean(3.0, 0.0) == 3.0                # edge: no spread
    assert OA.folded_normal_mean(-3.0, 0.0) == 3.0
    for mu, sd in ((0.4, 2.8), (-1.0, 0.5)):
        assert OA.folded_normal_mean(mu, sd) == pytest.approx(stats.foldnorm.mean(abs(mu) / sd, scale=sd), rel=1e-12)
    assert OA.folded_normal_mean(-0.4, 2.8) == pytest.approx(OA.folded_normal_mean(0.4, 2.8), rel=1e-14)


def test_predicted_apparent_known_values():
    q = OA.predicted_apparent(4.0, 3.0, 0.0)
    assert q["var"] == 7.0 and q["mse"] == 7.0
    assert q["loa_width"] == pytest.approx(2 * 1.96 * np.sqrt(7.0))
    assert q["mae_normal"] == pytest.approx(np.sqrt(2 / np.pi) * np.sqrt(7.0))
    # correlated errors: rho = 0.5, sd 2 and 3 -> cov 3
    q = OA.predicted_apparent(4.0, 9.0, 0.5 * 2 * 3, bias_ai=0.2, bias_ref=0.5)
    assert q["var"] == pytest.approx(7.0) and q["bias"] == pytest.approx(-0.3)
    assert q["mse"] == pytest.approx(7.09)
    # edge: model identical to the reference
    q = OA.predicted_apparent(4.0, 4.0, 4.0)
    assert q["var"] == 0.0 and q["mae_normal"] == 0.0
    # Monte Carlo confirmation of the variance identity
    rng = np.random.default_rng(1)
    em, er = rng.multivariate_normal([0, 0], [[4, 1.5], [1.5, 2.25]], 400000).T
    q = OA.predicted_apparent(4.0, 2.25, 1.5)
    assert np.var(em - er) == pytest.approx(q["var"], rel=0.01)
    assert np.abs(em - er).mean() == pytest.approx(q["mae_normal"], rel=0.01)


def test_match_sigma_for_mae():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(400000)
    target = 2.0 * np.sqrt(2 / np.pi)
    assert OA.match_sigma_for_mae(np.zeros_like(z), z, float(np.abs(2.0 * z).mean())) == pytest.approx(2.0, abs=1e-8)
    e = rng.normal(0, 1.2, z.size)                 # normal systematic part: s^2 = 4 - 1.44
    s = OA.match_sigma_for_mae(e, z, target)
    assert s == pytest.approx(np.sqrt(4 - 1.44), abs=0.01)
    assert np.abs(e + s * z).mean() == pytest.approx(target, abs=1e-8)
    with pytest.raises(ValueError):                # edge: systematic error alone above the target
        OA.match_sigma_for_mae(np.full(10, 5.0), z[:10], 1.0)
    assert OA.match_sigma_for_mae(np.full(10, 1.0), z[:10], 1.0) == 0.0


def test_study_agreement_known_answer():
    ai = np.array([[1.0, 2.0, 3.0, 6.0]])
    ref = np.array([[0.0, 3.0, 3.0, 4.0]])
    q = OA.study_agreement(ai, ref)                # d = 1, -1, 0, 2
    assert q["mae"][0] == 1.0 and q["bias"][0] == 0.5
    assert q["sd"][0] == pytest.approx(np.std([1, -1, 0, 2], ddof=1))
    assert q["loa_width"][0] == pytest.approx(3.92 * q["sd"][0])
    assert q["mse"][0] == 1.5
    q0 = OA.study_agreement(ai, ai)                # edge: identical
    assert q0["mae"][0] == 0 and q0["loa_width"][0] == 0


# ---------------------------------------------------------------- 3. permutation test

def _brute_perm_p(d, ns, w, perms):
    x = np.concatenate([d[:ns], (1 - w) * d[ns:]])
    t_obs = np.var(x[:ns], ddof=1) / np.var(x[ns:], ddof=1)
    ge = 0
    for pm in perms:
        xp = x[pm]
        ge += np.var(xp[:ns], ddof=1) / np.var(xp[ns:], ddof=1) >= t_obs * (1 - 1e-12)
    return (1 + ge) / (len(perms) + 1)


def test_perm_test_matches_brute_force():
    rng = np.random.default_rng(3)
    S, N, B = 6, 40, 60
    d = rng.standard_t(4, (S, N)) + 0.7
    perms = np.stack([[rng.permutation(N) for _ in range(B)] for _ in range(S)])
    pv = OA.perm_variance_ratio_pvalues(d, [5, 12], [0.0, 0.3], B, perms=perms, study_chunk=4)
    for ns in (5, 12):
        for w in (0.0, 0.3):
            for s in range(S):
                assert pv[(ns, w)][s] == pytest.approx(_brute_perm_p(d[s], ns, w, perms[s]), abs=1e-12)


def test_perm_test_identity_permutations_give_p_one():
    rng = np.random.default_rng(4)
    d = rng.normal(size=(3, 30))
    perms = np.broadcast_to(np.arange(30), (3, 7, 30))
    pv = OA.perm_variance_ratio_pvalues(d, [10], [0.0], 7, perms=perms)
    np.testing.assert_allclose(pv[(10, 0.0)], 1.0)


def test_perm_test_size_under_heavy_tails_and_power():
    """Exchangeable heavy-tailed null: size at the nominal level where the F test is not."""
    rng = np.random.default_rng(5)
    S, N = 1500, 120
    d = rng.standard_t(5, (S, N)) * 1.5 + 0.3
    pv = OA.perm_variance_ratio_pvalues(d, [20], [0.0, 0.5], 199, rng=np.random.default_rng(6))
    size = float((pv[(20, 0.0)] <= 0.05).mean())
    assert abs(size - 0.05) < 3.5 * np.sqrt(0.05 * 0.95 / S)
    f = OA.sentinel_tests(d, 20, 0.0)["f"]
    assert (f < 0.05).mean() > 0.07                   # the F test is liberal here
    assert (pv[(20, 0.5)] <= 0.05).mean() > 0.8       # halved SD in the assisted sample is detected
    # p-values lie on the grid k / (n_perm + 1), k >= 1
    k = pv[(20, 0.0)] * 200
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)
    assert k.min() >= 1


def test_perm_test_edge_cases():
    d = np.random.default_rng(7).normal(size=(2, 10))
    with pytest.raises(ValueError):
        OA.perm_variance_ratio_pvalues(d, [1], [0.0], 9, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        OA.perm_variance_ratio_pvalues(d, [9], [0.0], 9, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        OA.perm_variance_ratio_pvalues(d, [3], [1.0], 9, rng=np.random.default_rng(0))
    pv = OA.perm_variance_ratio_pvalues(d, [2, 8], [0.0], 9, rng=np.random.default_rng(0), study_chunk=1)
    assert pv[(2, 0.0)].shape == (2,) and np.all((pv[(8, 0.0)] >= 0.1) & (pv[(8, 0.0)] <= 1.0))


def test_sentinel_tests_match_scipy():
    rng = np.random.default_rng(8)
    d = rng.normal(1.0, 2.0, (3, 60))
    pv = OA.sentinel_tests(d, 15, 0.2)
    for s in range(3):
        x1, x2 = d[s, :15], 0.8 * d[s, 15:]
        assert pv["welch"][s] == pytest.approx(stats.ttest_ind(x1, x2, equal_var=False).pvalue, rel=1e-10)
        assert pv["f"][s] == pytest.approx(stats.f.sf(np.var(x1, ddof=1) / np.var(x2, ddof=1), 14, 44), rel=1e-12)
        assert pv["bf_two"][s] == pytest.approx(stats.levene(x1, x2, center="median").pvalue, rel=1e-10)
    # edge: w = 0 with identical halves gives F p-value 0.5 for equal variances
    e = np.tile(rng.normal(size=20), 2)[None, :]
    assert OA.sentinel_tests(e, 20, 0.0)["f"][0] == pytest.approx(0.5)


# ---------------------------------------------------------------- 4. reader panels and multi-reader estimators

def test_reader_panel_reads_quiet_known_answer(cfg2):
    """No measurement noise: every read is the true span plus the reader's offset."""
    p = C.cell_params(cfg2, "E6", 22)
    p.update(QUIET)
    p["sigma_rb_mm"] = 0.75
    p["reference_estimator"] = "A1"
    reads, b = OA.reader_panel_reads(p, np.random.default_rng(9), n_panels=3, n_rep=4, n_cases=5, n_readers=4)
    assert reads.shape == (3, 4, 5, 4) and b.shape == (3, 4)
    d01 = reads[..., 0] - reads[..., 1]
    np.testing.assert_allclose(d01, np.broadcast_to((b[:, 0] - b[:, 1])[:, None, None], d01.shape), atol=1e-12)
    # offsets are fixed across a panel's case samples and differ between panels
    assert len(np.unique(np.round(d01, 9))) == 3
    # edge: one panel, one case sample, one case
    r1, b1 = OA.reader_panel_reads(p, np.random.default_rng(9), 1, 1, 1, 2)
    assert r1.shape == (1, 1, 1, 2)


def test_allocations():
    m = OA.crossed_allocation(4, 400)
    assert m.shape == (100, 4) and m.all()
    assert OA.crossed_allocation(3, 400).shape == (133, 3)
    assert OA.crossed_allocation(6, 400).sum() == 396
    for R in (2, 3, 4, 6):
        m = OA.rotating_allocation(R, 400)
        assert m.shape == (200, R) and np.all(m.sum(1) == 2) and m.sum() == 400
        per_reader = m.sum(0)
        assert per_reader.max() - per_reader.min() <= R        # reads per reader nearly balanced
        pairs = list(itertools.combinations(range(R), 2))
        counts = [int((m[:, j] & m[:, k]).sum()) for j, k in pairs]
        assert max(counts) - min(counts) <= 1 and sum(counts) == 200
        assert tuple(np.nonzero(m[len(pairs)])[0]) == pairs[0]  # rotation restarts after the last pair
    np.testing.assert_array_equal(OA.rotating_allocation(2, 400), OA.crossed_allocation(2, 400))
    assert OA.rotating_allocation(3, 0).shape == (0, 3)        # edge: no reads
    with pytest.raises(ValueError):
        OA.rotating_allocation(1, 10)
    with pytest.raises(ValueError):
        OA.crossed_allocation(1, 10)


def test_between_reader_mom_known_answers():
    rng = np.random.default_rng(10)
    case = rng.normal(10, 3, (20, 1))
    b = np.array([0.5, -0.25, 1.0])
    reads = case + b                                   # no noise: differences are exact offsets
    q = OA.between_reader_mom(reads, np.ones((20, 3), bool))
    want = np.mean([(b[j] - b[k]) ** 2 for j, k in itertools.combinations(range(3), 2)])
    assert q["tau2"] == pytest.approx(want) and q["sigw2"] == pytest.approx(0.0, abs=1e-20)
    assert q["sd_b"] == pytest.approx(np.sqrt(want / 2)) and q["n_pairs"] == 3
    assert q["loa_marginal"] == pytest.approx(1.96 * np.sqrt(want))
    # two readers: tau2 = dbar^2 - s^2 / n
    r2 = rng.normal(size=(50, 2))
    d = r2[:, 0] - r2[:, 1]
    q = OA.between_reader_mom(r2, np.ones((50, 2), bool))
    assert q["tau2"] == pytest.approx(d.mean() ** 2 - d.var(ddof=1) / 50)
    assert q["sigw2"] == pytest.approx(d.var(ddof=1))
    # unbiasedness in a large simulation: sigma_b 0.75, noise SD 1 per read, crossed and rotating
    S, n, R = 4000, 60, 4
    bb = rng.normal(0, 0.75, (S, 1, R))
    reads = rng.normal(10, 3, (S, n, 1)) + bb + rng.normal(0, 1.0, (S, n, R))
    for mask in (np.ones((n, R), bool), OA.rotating_allocation(R, 2 * n)):
        q = OA.between_reader_mom(reads, mask)
        se = q["tau2"].std(ddof=1) / np.sqrt(S)
        assert abs(q["tau2"].mean() - 2 * 0.75 ** 2) < 4 * se
        assert q["sigw2"].mean() == pytest.approx(2.0, rel=0.02)
    # edge: truncation at zero and a design with no shared cases
    q = OA.between_reader_mom(np.array([[1.0, 1.0], [2.0, 4.0], [3.0, 1.0]]), np.ones((3, 2), bool))
    assert q["tau2"] < 0 and q["sd_b"] == 0.0
    with pytest.raises(ValueError):
        OA.between_reader_mom(np.zeros((4, 2)), np.eye(2, dtype=bool)[[0, 1, 0, 1]])


def test_oneway_components_known_answers():
    x = np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])      # within var 2; group means 2, 6, 10
    q = OA.oneway_components(x)
    assert q["within_var"] == 2.0 and q["msb"] == 32.0
    assert q["between_var"] == pytest.approx(15.0) and q["total_var"] == pytest.approx(17.0)
    rng = np.random.default_rng(11)
    y = rng.normal(0, 1.06, (3000, 1)) + rng.normal(0, 0.2, (3000, 30))
    q = OA.oneway_components(y)
    assert q["between_var"] == pytest.approx(1.06 ** 2, rel=0.08) and q["within_var"] == pytest.approx(0.04, rel=0.03)
    q0 = OA.oneway_components(np.tile(np.arange(4.0)[:, None], (1, 3)))   # edge: no within-group variation
    assert q0["within_var"] == 0.0 and q0["between_var"] == pytest.approx(np.var(np.arange(4.0), ddof=1))
    with pytest.raises(ValueError):
        OA.oneway_components(np.zeros((5, 1)))
