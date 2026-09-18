"""Known-answer tests (task items 1 to 4)."""
import numpy as np
import pytest
from scipy import stats

from conftest import QUIET
from duomaxsim import estimators as E
from duomaxsim.experiments import read_once, run_cells
from duomaxsim.metrics import clark_max2, expected_max_std_normals, mvn_expected_max
from duomaxsim.model import draw_latent, log_sd_from_cv

Z = 4.0  # tolerance in Monte Carlo SEs


# ---------------------------------------------------------------- (1)

@pytest.mark.parametrize("K,closed", [(2, 1 / np.sqrt(np.pi)), (3, 3 / (2 * np.sqrt(np.pi))), (4, 1.0293753730)])
def test_expected_max_closed_forms(K, closed):
    assert expected_max_std_normals(K) == pytest.approx(closed, abs=1e-8)


@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("N", [1, 5])
def test_bias_of_max_of_view_means(cfg, K, N):
    sigma = 2.0
    ov = dict(QUIET, K=K, N_beats=N, sigma_cal_mm=sigma)
    df = run_cells(cfg, "E1", [0], n_rep=200000, overrides=ov)
    r = df[(df.estimator == "A3") & (df.axis == "AP") & (df.estimand == "T1") & (df.metric == "bias_mm")].iloc[0]
    expected = expected_max_std_normals(K) * sigma / np.sqrt(N)
    assert abs(r.value - expected) < Z * r.mcse, (r.value, expected, r.mcse)


# ---------------------------------------------------------------- (2)

@pytest.mark.parametrize("mu1,mu2,s1,s2,rho", [(1.0, 0.5, 1.0, 2.0, 0.4), (0.0, 0.0, 1.0, 1.0, 0.0),
                                                 (3.0, 1.0, 0.5, 1.5, -0.3)])
def test_clark_max2_against_mc_through_a3(mu1, mu2, s1, s2, rho):
    rng = np.random.default_rng(11)
    cov = [[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]]
    x = rng.multivariate_normal([mu1, mu2], cov, size=400000)
    m = E.a3_max(x)
    cm, cv = clark_max2(mu1, mu2, s1, s2, rho)
    se_m = m.std() / np.sqrt(m.size)
    se_v = np.sqrt(np.var((m - m.mean()) ** 2) / m.size)
    assert abs(m.mean() - cm) < Z * se_m
    assert abs(m.var() - cv) < Z * se_v


def test_clark_iid_standard_equals_closed_form():
    m, v = clark_max2(0.0, 0.0, 1.0, 1.0, 0.0)
    assert m == pytest.approx(1 / np.sqrt(np.pi), abs=1e-12)
    assert v == pytest.approx(1 - 1 / np.pi, abs=1e-12)


def test_numerical_mvn_max_matches_clark_k2():
    s1, s2, rho = 1.0, 2.0, 0.4
    cov = [[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]]
    assert mvn_expected_max([1.0, 0.5], cov) == pytest.approx(clark_max2(1.0, 0.5, s1, s2, rho)[0], abs=2e-3)


@pytest.mark.parametrize("K", [3, 4])
def test_numerical_mvn_max_k3_k4(K):
    # equicorrelated standard normals: E[max] = sqrt(1-rho) * E[max of K iid]
    rho = 0.5
    cov = np.full((K, K), rho) + (1 - rho) * np.eye(K)
    assert mvn_expected_max(np.zeros(K), cov) == pytest.approx(np.sqrt(1 - rho) * expected_max_std_normals(K), abs=3e-3)
    # unequal means and variances vs Monte Carlo through A3
    mu = np.array([0.0, 0.7, -0.4, 1.1])[:K]
    sd = np.array([1.0, 1.5, 0.7, 2.0])[:K]
    R = np.full((K, K), 0.3) + 0.7 * np.eye(K)
    cov = R * np.outer(sd, sd)
    x = np.random.default_rng(5).multivariate_normal(mu, cov, size=400000)
    m = E.a3_max(x)
    assert abs(mvn_expected_max(mu, cov) - m.mean()) < Z * m.std() / np.sqrt(m.size) + 3e-3


# ---------------------------------------------------------------- (3)

def test_mean_estimators_unbiased_with_mean_centred_noise(cfg):
    ov = dict(QUIET, K=3, N_beats=5, beat_cv=0.2, beat_noise_centering="mean", sigma_cal_mm=1.0, sigma_rb_mm=0.5)
    df = run_cells(cfg, "E1", [0], n_rep=200000, overrides=ov)
    sel = df[(df.estimand == "T1") & (df.metric == "bias_mm") & df.estimator.isin(["A1", "A2", "A7"])]
    assert len(sel) == 6
    for _, r in sel.iterrows():
        assert abs(r.value) < Z * r.mcse, (r.estimator, r.axis, r.value, r.mcse)


def test_median_centred_beat_mean_has_lognormal_bias(cfg, base):
    # protocol-literal eta ~ N(0, s^2): E[A1] = S * exp(s^2/2)
    p = dict(base, **dict(QUIET, K=1, N_beats=4, beat_cv=0.3))
    rng = np.random.default_rng(3)
    lat = draw_latent(p, 200000, rng)
    _, br = read_once(lat, p, rng, np.zeros(lat.n))
    s = log_sd_from_cv(0.3)
    ratio = br.value[:, 0, 0] / lat.S[:, 0]
    assert abs(ratio.mean() - np.exp(s ** 2 / 2)) < Z * ratio.std() / np.sqrt(ratio.size)


# ---------------------------------------------------------------- (4)

def _p_window_lognormal_n3(cv, W, n=2001, lim=8.0):
    """P(3 iid lognormal beats all within +-W of their mean): 2D grid integral over
    (d2, d3) = (eta2-eta1, eta3-eta1), which are bivariate normal (var 2s^2, cov s^2)."""
    s = log_sd_from_cv(cv)
    t = np.linspace(-lim * s, lim * s, n)
    d2, d3 = np.meshgrid(t, t, indexing="ij")
    x = np.stack([np.ones_like(d2), np.exp(d2), np.exp(d3)])
    m = x.mean(0)
    ok = ((x >= (1 - W) * m) & (x <= (1 + W) * m)).all(0)
    dens = stats.multivariate_normal([0, 0], [[2 * s * s, s * s], [s * s, 2 * s * s]]).pdf(np.dstack([d2, d3]))
    h = t[1] - t[0]
    return float((dens * ok).sum() * h * h)


def _p_window_additive_n3(mu, sigma, W):
    """Same probability for x_i = mu + e_i, e_i ~ N(0, sigma^2): outer quadrature over
    the mean error, inner 2D grid over the residual plane (independent of the mean)."""
    u1 = np.array([1, -1, 0]) / np.sqrt(2)
    u2 = np.array([1, 1, -2]) / np.sqrt(6)
    t = np.linspace(-8, 8, 1601)
    a, b = np.meshgrid(t, t, indexing="ij")
    mx = np.max(np.abs(a[..., None] * u1 + b[..., None] * u2), axis=-1).ravel()
    w = (stats.norm.pdf(a) * stats.norm.pdf(b)).ravel() * (t[1] - t[0]) ** 2
    order = np.argsort(mx)
    mx, cw = mx[order], np.cumsum(w[order])
    G = lambda c: np.interp(c, mx, cw, left=0.0, right=cw[-1])  # P(max|r_i|/sigma <= c)
    x, wq = np.polynomial.hermite_e.hermegauss(80)
    ebar = x * sigma / np.sqrt(3)
    return float(np.sum(wq * G(W * (mu + ebar) / sigma)) / np.sqrt(2 * np.pi))


@pytest.mark.parametrize("cv,W", [(0.15, 0.15), (0.25, 0.10), (0.10, 0.20)])
def test_window_probability_lognormal_beats(cfg, cv, W):
    ov = dict(QUIET, K=1, N_beats=3, window=W, beat_cv=cv)
    df = run_cells(cfg, "E2", [0], n_rep=200000, overrides=ov)
    r = df[(df.metric == "p_window_met_first_N") & (df.axis == "AP")].iloc[0]
    exact = _p_window_lognormal_n3(cv, W)
    assert abs(r.value - exact) < Z * r.mcse + 1e-4, (r.value, exact, r.mcse)


@pytest.mark.parametrize("sigma,W", [(1.0, 0.15), (2.0, 0.20)])
def test_window_probability_additive_gaussian_beats(cfg, sigma, W):
    mu = 9.0
    ov = dict(QUIET, K=1, N_beats=3, window=W, sigma_cal_mm=sigma, S_median_mm=mu, S_log_sd=0.0)
    df = run_cells(cfg, "E2", [0], n_rep=200000, overrides=ov)
    r = df[(df.metric == "p_window_met_first_N") & (df.axis == "AP")].iloc[0]
    exact = _p_window_additive_n3(mu, sigma, W)
    assert abs(r.value - exact) < Z * r.mcse + 1e-4, (r.value, exact, r.mcse)
