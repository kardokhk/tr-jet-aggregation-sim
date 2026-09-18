"""Independent re-implementation of the E2 anchor-view beat rule (does not import duomaxsim).

Checks, at base case (sinus, AP axis, beat CV 15%, N = 3, prospective):
  p(window met by first N), mean beats acquired, RMSE and bias of the anchor mean vs Tv,
  with and without a +-15% window. Window search uses all N-subsets (itertools),
  not the library's contiguous-sorted-block search.
Mechanism check: the same with sigma_cal = 0 and sigma_rb = 0 (pure multiplicative beat noise).
Seed 7 (numpy default_rng). n = 20,000 patients per scenario.
"""
import itertools
import sys

import numpy as np

rng = np.random.default_rng(7)
n = 20000
N = 3
B = 30


def draw(sig_cal, sig_rb, cv=0.15):
    S = 10.0 * np.exp(0.40 * rng.standard_normal(n))
    u = np.minimum(np.abs(0.06 * rng.standard_normal(n)), 0.9)
    over = rng.random(n) < 0.05
    o = np.where(over, 0.15 * np.exp(0.5 * rng.standard_normal(n)), 0.0)
    mu = S * (1 - u + o)
    g = np.exp(0.10 * rng.standard_normal(n))
    s = np.sqrt(np.log(1 + cv ** 2))
    rr = np.log(1 + 0.03 ** 2) ** 0.5 * rng.standard_normal((n, B))
    spans = g[:, None] * mu[:, None] * np.exp(0.2 * rr + s * rng.standard_normal((n, B)))
    x = spans + sig_rb * rng.standard_normal(n)[:, None] + sig_cal * rng.standard_normal((n, B))
    return x, g * mu


def window_rule(x, W):
    val = np.empty(n)
    used = np.empty(n, int)
    met1 = np.zeros(n, bool)
    for i in range(n):
        xi = x[i]
        done = False
        for m in range(N, B + 1):
            best = None
            for c in itertools.combinations(range(m), N):
                v = xi[list(c)]
                mn = v.mean()
                if np.all(np.abs(v - mn) <= W * abs(mn)):
                    sp = (v.max() - v.min()) / abs(mn)
                    if best is None or sp < best[0]:
                        best = (sp, mn)
            if best is not None:
                val[i], used[i], done = best[1], m, True
                met1[i] = m == N
                break
        if not done:
            val[i], used[i] = xi.mean(), B
    return val, used, met1


for label, sc, sr in [("base (sigma_cal 1.0, sigma_rb 0.75)", 1.0, 0.75), ("no additive error", 0.0, 0.0)]:
    x, tv = draw(sc, sr)
    v0 = x[:, :N].mean(1)
    v1, used, met1 = window_rule(x, 0.15)
    e0, e1 = v0 - tv, v1 - tv
    r0, r1 = 100 * e0 / tv, 100 * e1 / tv
    print(label)
    print(f"  p_met_first_N {met1.mean():.4f} (MCSE {np.sqrt(met1.mean()*(1-met1.mean())/n):.4f})")
    print(f"  beats acquired {used.mean():.3f} (MCSE {used.std()/np.sqrt(n):.3f})")
    print(f"  RMSE vs Tv no window {np.sqrt((e0**2).mean()):.4f}, window {np.sqrt((e1**2).mean()):.4f}")
    d = r1 - r0
    print(f"  relbias% vs Tv no window {r0.mean():.3f}, window {r1.mean():.3f}; "
          f"paired diff {d.mean():.3f} (MCSE {d.std()/np.sqrt(n):.3f})")
    sys.stdout.flush()
