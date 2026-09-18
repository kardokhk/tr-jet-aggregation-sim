"""Protocol amendment 2 (E1b, E3b, E5b, E6b): identity, known-answer and property tests."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import make_fixture as MF
from conftest import ROOT
from duomaxsim import config as C
from duomaxsim.experiments import ai_shared, image_composite, protocol_read, run_cells
from duomaxsim.metrics import brown_forsythe
from duomaxsim.model import draw_latent

AMEND2 = ROOT / "code" / "configs" / "amend2_2026-09-18.yaml"
DROP = ["cpu_s", "wall_s"]


@pytest.fixture(scope="module")
def acfg():
    return C.load_config(AMEND2)


# ---------------------------------------------------------------- (a) lambda = 0 identity

def test_lambda0_reproduces_previous_E5_outputs():
    """With ai_shared_lambda = 0 every pre-existing E5 row is bit-identical to the
    frozen regression fixture, and the shared AI reproduces the independent AI."""
    cfg = C.load_config(ROOT / "tests" / "fixtures" / "config_v01.yaml")
    cfg["parameters"]["calib_n"]["value"] = 2000
    e, cells, n = [s for s in MF.SPEC if s[0] == "E5"][0]
    new = run_cells(cfg, e, cells, n_rep=n, overrides={"ai_shared_lambda": 0.0}).drop(columns=DROP)
    ref = pd.read_csv(MF.FIXTURE)
    ref = ref[ref.experiment == "E5"].reset_index(drop=True)
    added = (new.ai == "shared") | new.comparator.notna() | (new.ref == "R_img") | (new.estimator == "shared_ai")
    new_old = new[~added].reset_index(drop=True)
    assert len(new_old) == len(ref)
    for c in ("metric", "ai", "ref"):
        assert (new_old[c].astype(object).fillna("NA").astype(str).to_numpy()
                == ref[c].astype(object).fillna("NA").astype(str).to_numpy()).all(), c
    for c in ("value", "mcse"):
        np.testing.assert_allclose(new_old[c].to_numpy(float), ref[c].to_numpy(float), rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=c)
    # shared AI at lambda 0 == independent AI, exactly (same z)
    k = ["sigma_ai_mm", "ref", "metric"]
    sh = new[(new.ai == "shared") & new.comparator.isna()].set_index(k).sort_index()
    ind = new[new.ai == "independent"].set_index(k).sort_index()
    assert len(sh) == len(ind) > 0
    np.testing.assert_array_equal(sh.value.to_numpy(), ind.value.loc[sh.index].to_numpy())
    pl = new[(new.ai == "shared") & (new.metric == "paired_mae_diff_mean")]
    assert (pl.value == 0.0).all()


def test_ai_shared_lambda0_exact():
    rng = np.random.default_rng(1)
    T1, R, z = rng.random(50) * 10, rng.random(50) * 10, rng.standard_normal(50)
    np.testing.assert_array_equal(ai_shared(T1, R, 0.0, 2.0, z), T1 + 2.0 * z)


# ---------------------------------------------------------------- (b) lambda = 1, sigma_AI -> 0

def test_lambda1_sigma0_equals_R_img(base):
    p = dict(base, K=4, beat_cv=0.2)
    rng = np.random.default_rng(4)
    lat = draw_latent(p, 5000, rng)
    _, U = protocol_read(lat, p, rng, p["sigma_rb_mm"] * rng.standard_normal(lat.n), return_u=True)
    R = image_composite(lat, p, U)[:, 0]
    T1 = lat.S[:, 0]
    z = rng.standard_normal(lat.n)
    np.testing.assert_allclose(ai_shared(T1, R, 1.0, 0.0, z), R, rtol=0, atol=1e-12)
    for sig in (1e-3, 1e-6):
        assert np.max(np.abs(ai_shared(T1, R, 1.0, sig, z) - R)) < 10 * sig
    # R_img carries the image-level error: it differs from T1
    assert np.mean(np.abs(R - T1)) > 0.3


def test_R_img_equals_error_free_read_in_E5(acfg):
    """End to end: with no caliper error and no reader bias, reader 1's read is the
    image composite, so the lambda = 1, sigma_AI = 0 AI agrees exactly with the
    single-read reference (apparent MAE 0)."""
    cfg = copy.deepcopy(acfg)
    cfg["parameters"]["sigma_ai_mm"]["grid"] = [0.0]
    cfg["parameters"]["adj_tol_mm"]["grid"] = [2.0]
    cfg["parameters"]["calib_n"]["value"] = 2000
    ov = {"sigma_cal_mm": 0.0, "sigma_rb_mm": 0.0, "ai_shared_lambda": 1.0}
    df = run_cells(cfg, "E5", [4], n_rep=3, overrides=ov)
    r = df[(df.ai == "shared") & (df.ref == "single") & (df.metric == "apparent_mae")]
    assert len(r) == 1 and r.value.iloc[0] < 1e-12
    # and the stored per-study R_img error vs T1 equals the shared AI's true MAE
    ri = df[(df.ref == "R_img") & (df.metric == "ref_mae_vs_T1")].value.iloc[0]
    tm = df[(df.ai == "shared") & (df.ref == "single") & (df.metric == "true_mae")].value.iloc[0]
    assert ri == pytest.approx(tm, rel=1e-12)


# ---------------------------------------------------------------- (c) Brown-Forsythe

def test_bf_matches_scipy_levene_median():
    rng = np.random.default_rng(9)
    x1 = rng.standard_normal((20, 25)) * 1.3
    x2 = rng.standard_normal((20, 175))
    p2, p1 = brown_forsythe(x1, x2)
    for i in range(20):
        ref = stats.levene(x1[i], x2[i], center="median").pvalue
        assert p2[i] == pytest.approx(ref, rel=1e-9)
        # one-sided p is the matching tail of the same t statistic
        assert min(p1[i], 1 - p1[i]) == pytest.approx(ref / 2, rel=1e-9)


@pytest.mark.parametrize("n1,n2", [(10, 490), (50, 450)])
def test_bf_size_normal_null(n1, n2):
    rng = np.random.default_rng(2026)
    S = 200
    x1 = 2.0 + rng.standard_normal((S, n1))
    x2 = -1.0 + rng.standard_normal((S, n2))
    p2, p1 = brown_forsythe(x1, x2)
    se = np.sqrt(0.05 * 0.95 / S)
    for pv in (p2, p1):
        assert abs(np.mean(pv < 0.05) - 0.05) < 4 * se, np.mean(pv < 0.05)


def test_bf_one_sided_direction():
    rng = np.random.default_rng(3)
    x1 = 2.0 * rng.standard_normal((200, 50))
    x2 = rng.standard_normal((200, 450))
    p2, p1 = brown_forsythe(x1, x2)
    assert np.mean(p1 < 0.05) > 0.9
    p2r, p1r = brown_forsythe(x2, x1)
    assert np.mean(p1r < 0.05) < 0.01
    np.testing.assert_allclose(p2, p2r)


def test_E6b_reports_bf_and_keeps_existing_rows(acfg, cfg):
    a = run_cells(acfg, "E6", [0], n_rep=2).drop(columns=DROP)
    ms = set(a.metric)
    assert {"power_reject_bf", "power_reject_bf_onesided", "power_reject_f_variance",
            "power_reject_welch_mean"} <= ms
    # base config (no BF flag): no BF rows
    b = run_cells(cfg, "E6", [0], n_rep=2)
    assert not b.metric.str.contains("bf").any()


# ---------------------------------------------------------------- (d) cell enumeration

@pytest.mark.parametrize("exp,count", [("E1", 864), ("E3", 12), ("E5", 18), ("E6", 54)])
def test_amend2_cell_counts(acfg, exp, count):
    assert len(C.experiment_cells(acfg, exp)) == count


@pytest.mark.parametrize("exp", ["E1", "E3"])
def test_amend2_u_combo_levels(acfg, exp):
    cells = C.experiment_cells(acfg, exp)
    combos = {(c["u_anchor"], c["u_long"]) for c in cells}
    assert combos == {(a, L) for a in (0.03, 0.06, 0.125, 0.25) for L in (0.10, 0.25, 0.50)}
    for c in cells:
        a, L = c["u_anchor"], c["u_long"]
        np.testing.assert_allclose(c["u_scale"], [a, L, L, 1.2 * L], rtol=1e-12)


def test_amend2_E1_grid_and_fixed(acfg):
    cells = C.experiment_cells(acfg, "E1")
    df = pd.DataFrame(cells).drop(columns=["u_scale"])
    assert df.drop_duplicates().shape[0] == 864
    assert sorted(df.K.unique()) == [2, 3, 4]
    assert sorted(df.beat_cv.unique()) == [0.05, 0.15, 0.30]
    assert {c["window"] for c in cells} == {None, 0.15}
    assert sorted(df.S_median_mm.unique()) == [10.0, 13.0]
    p = C.cell_params(acfg, "E1", 0)
    assert p["rho"] == 0.3 and p["N_beats"] == 3 and p["rhythm"] == "sinus" and p["data_mode"] == "prospective"
    E5 = pd.DataFrame(C.experiment_cells(acfg, "E5"))
    assert sorted(E5.ai_shared_lambda.unique()) == [0.0, 0.5, 1.0]


def test_amend2_E1_output_columns_and_inflation(acfg):
    df = run_cells(acfg, "E1", [0, 863], n_rep=2000)
    assert {"p_u_anchor", "p_u_long", "p_u_scale"} <= set(df.columns)
    last = df[df.cell == 863].iloc[0]
    assert last.p_u_anchor == 0.25 and last.p_u_long == 0.5
    inf = df[(df.metric == "selection_inflation_mm")]
    assert len(inf) == 4
    # E[A3 - A1] equals bias(A3) - bias(A1) against the same estimand
    for (c, ax), g in inf.groupby(["cell", "axis"]):
        sub = df[(df.cell == c) & (df.axis == ax) & (df.estimand == "T1") & (df.metric == "bias_mm")]
        d = sub[sub.estimator == "A3"].value.iloc[0] - sub[sub.estimator == "A1"].value.iloc[0]
        assert g.value.iloc[0] == pytest.approx(d, abs=1e-9)
