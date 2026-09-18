"""Performance measures, Monte Carlo SEs, agreement statistics and closed forms."""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats
from scipy.special import ndtr

# ---------------------------------------------------------------- accumulators


class ErrAcc:
    """Streaming moments of an error e = est - truth and relative error e/truth."""

    def __init__(self):
        self.n = 0
        self.s1 = self.s2 = self.s4 = self.r1 = self.r2 = 0.0

    def add(self, est, truth):
        e = np.asarray(est - truth, dtype=float).ravel()
        rel = (e / np.asarray(truth, dtype=float).ravel())
        self.n += e.size
        self.s1 += e.sum()
        e2 = e * e
        self.s2 += e2.sum()
        self.s4 += (e2 * e2).sum()
        self.r1 += rel.sum()
        self.r2 += (rel * rel).sum()

    def summary(self) -> dict:
        """metric -> (value, mcse)."""
        n = self.n
        m1 = self.s1 / n
        m2 = self.s2 / n
        var = max(m2 - m1 * m1, 0.0) * n / (n - 1)
        sd = np.sqrt(var)
        rmse = np.sqrt(m2)
        var_e2 = max(self.s4 / n - m2 * m2, 0.0)
        se_mse = np.sqrt(var_e2 / n)
        rm = self.r1 / n
        rvar = max(self.r2 / n - rm * rm, 0.0) * n / (n - 1)
        return {
            "bias_mm": (m1, sd / np.sqrt(n)),
            "rmse_mm": (rmse, se_mse / (2 * rmse) if rmse > 0 else 0.0),
            "sd_err_mm": (sd, sd / np.sqrt(2 * (n - 1))),  # normal-theory MCSE
            "relbias_pct": (100 * rm, 100 * np.sqrt(rvar / n)),
        }


class PropAcc:
    def __init__(self):
        self.k = 0
        self.n = 0

    def add(self, flags, where=None):
        f = np.asarray(flags, dtype=bool)
        if where is not None:
            f = f[np.asarray(where, dtype=bool)]
        self.k += int(f.sum())
        self.n += int(f.size)

    def summary(self):
        if self.n == 0:
            return (np.nan, np.nan, 0)
        p = self.k / self.n
        return (p, np.sqrt(p * (1 - p) / self.n), self.n)


class MeanAcc:
    def __init__(self):
        self.n = 0
        self.s1 = self.s2 = 0.0

    def add(self, x, where=None):
        x = np.asarray(x, dtype=float)
        if where is not None:
            x = x[np.asarray(where, dtype=bool)]
        x = x.ravel()
        self.n += x.size
        self.s1 += x.sum()
        self.s2 += (x * x).sum()

    def summary(self):
        if self.n < 2:
            return (np.nan, np.nan, self.n)
        m = self.s1 / self.n
        v = max(self.s2 / self.n - m * m, 0.0) * self.n / (self.n - 1)
        return (m, np.sqrt(v / self.n), self.n)


def mean_mcse(x, axis=0):
    """Mean over replicates and its Monte Carlo SE (for per-study metrics)."""
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=axis)
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1)
    return m, s / np.sqrt(n)


# ---------------------------------------------------------------- agreement


def icc_a1(Y: np.ndarray, alpha: float = 0.05, ci: bool = True):
    """ICC(A,1), two-way random effects, absolute agreement, single rater.

    Y: (..., n, k). Returns icc, (lo, hi) using the McGraw and Wong (1996)
    F-based approximation with Satterthwaite df (as implemented in common
    software). Vectorised over leading axes.
    """
    Y = np.asarray(Y, dtype=float)
    n, k = Y.shape[-2], Y.shape[-1]
    gm = Y.mean(axis=(-2, -1), keepdims=True)
    rm = Y.mean(axis=-1, keepdims=True)
    cm = Y.mean(axis=-2, keepdims=True)
    ssr = k * ((rm - gm) ** 2).sum(axis=(-2, -1))
    ssc = n * ((cm - gm) ** 2).sum(axis=(-2, -1))
    sst = ((Y - gm) ** 2).sum(axis=(-2, -1))
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    if not ci:
        return icc
    with np.errstate(divide="ignore", invalid="ignore"):
        a = k * icc / (n * (1 - icc))
        b = 1 + k * icc * (n - 1) / (n * (1 - icc))
        v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
        fu = stats.f.ppf(1 - alpha / 2, n - 1, v)
        fl = stats.f.ppf(1 - alpha / 2, v, n - 1)
        lo = n * (msr - fu * mse) / (fu * (k * msc + (k * n - k - n) * mse) + n * msr)
        hi = n * (fl * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fl * msr)
    return icc, (lo, hi)


def bland_altman(d: np.ndarray, alpha: float = 0.05):
    """Bias, SD, LoA and approximate CI half-width of each LoA (Bland and Altman 1986).

    d: (..., n) differences. Half-width = t_{n-1} * sqrt(3 s^2 / n).
    """
    d = np.asarray(d, dtype=float)
    n = d.shape[-1]
    bias = d.mean(axis=-1)
    sd = d.std(axis=-1, ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    hw = stats.t.ppf(1 - alpha / 2, n - 1) * np.sqrt(3 * sd ** 2 / n)
    return bias, sd, lo, hi, hw


def brown_forsythe(x1: np.ndarray, x2: np.ndarray):
    """Brown-Forsythe (median-centred Levene) test for equal variances of two groups.

    x1: (..., n1), x2: (..., n2); vectorised over leading axes. Returns
    (p_two_sided, p_one_sided). The two-sided p equals
    scipy.stats.levene(x1, x2, center='median'): a one-way ANOVA on absolute
    deviations from each group's median, which for two groups is F = t^2 of a
    pooled-variance t test on those deviations. The one-sided p is that t test
    with H1: mean |x1 - median(x1)| > mean |x2 - median(x2)|, i.e. group 1 more
    dispersed than group 2 (amendment 2, E6b: sentinel variance > assisted).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = x1.shape[-1], x2.shape[-1]
    z1 = np.abs(x1 - np.median(x1, axis=-1, keepdims=True))
    z2 = np.abs(x2 - np.median(x2, axis=-1, keepdims=True))
    m1, m2 = z1.mean(-1), z2.mean(-1)
    df = n1 + n2 - 2
    sp2 = (((z1 - m1[..., None]) ** 2).sum(-1) + ((z2 - m2[..., None]) ** 2).sum(-1)) / df
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m2) / np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    p_two = stats.f.sf(t * t, 1, df)
    p_one = stats.t.sf(t, df)
    return p_two, p_one


# ---------------------------------------------------------------- closed forms


def expected_max_std_normals(K: int) -> float:
    """E[max of K iid N(0,1)] by quadrature."""
    if K == 1:
        return 0.0
    f = lambda x: x * K * stats.norm.pdf(x) * stats.norm.cdf(x) ** (K - 1)
    return integrate.quad(f, -12, 12, limit=200)[0]


def clark_max2(mu1, mu2, s1, s2, rho):
    """Clark (1961) exact mean and variance of max(X1, X2), bivariate normal."""
    a = np.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    al = (mu1 - mu2) / a
    phi = np.exp(-0.5 * al ** 2) / np.sqrt(2 * np.pi)
    m1 = mu1 * ndtr(al) + mu2 * ndtr(-al) + a * phi
    m2 = (mu1 ** 2 + s1 ** 2) * ndtr(al) + (mu2 ** 2 + s2 ** 2) * ndtr(-al) + (mu1 + mu2) * a * phi
    return m1, m2 - m1 ** 2


def mvn_expected_max(mu, cov, n_grid: int = 801) -> float:
    """E[max X], X ~ MVN(mu, cov), by integrating the CDF of the max numerically.

    E[M] = int_0^inf (1 - F(t)) dt - int_-inf^0 F(t) dt, F(t) = P(all X_i <= t).
    """
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    sd = np.sqrt(np.diag(cov))
    lo = float((mu - 9 * sd).min())
    hi = float((mu + 9 * sd).max())
    t = np.linspace(lo, hi, n_grid)
    mvn = stats.multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    F = np.array([mvn.cdf(np.full(len(mu), ti)) for ti in t])
    # E[M] = lo + int_lo^hi (1 - F) dt, with F(lo) ~ 0 and F(hi) ~ 1
    return lo + integrate.trapezoid(1.0 - F, t)
