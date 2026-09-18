"""Beat acceptance rule and cross-view triggers / rule A4 (protocol section 4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BeatRuleResult:
    value: np.ndarray       # view mean reported by the rule
    used: np.ndarray        # beats acquired (examined) by the reader
    accepted: np.ndarray    # window met (always True where W is None and avail >= N)
    met_first: np.ndarray   # window met by the first N beats
    limited: np.ndarray     # "limited sampling" flag


def _mean_first(X: np.ndarray, k: np.ndarray) -> np.ndarray:
    B = X.shape[1]
    mask = np.arange(B)[None, :] < k[:, None]
    return np.where(mask, X, 0.0).sum(1) / np.maximum(k, 1)


def beat_rule(x: np.ndarray, avail: np.ndarray, N: int, W: float | None,
              limited_value: str = "all_acquired") -> BeatRuleResult:
    """Apply the N-beat rule with optional +-W acceptance window.

    x: (..., B) measured beats in acquisition order; avail: (...) beats available.
    With W: the reader first measures N beats; if all lie within +-W of their
    mean the rule is met. Otherwise beats are added one at a time and the
    reader keeps the N most mutually consistent beats (searched over contiguous
    blocks of the sorted values, smallest relative range first). If the window
    is never met once all available beats are used, the view is flagged
    "limited sampling" and the mean of all acquired beats is reported
    (`limited_value='first_n'`: mean of the first min(N, avail) beats).
    """
    shape = x.shape[:-1]
    B = x.shape[-1]
    X = x.reshape(-1, B)
    A = np.minimum(np.broadcast_to(avail, shape).reshape(-1), B).astype(np.int64)
    R = X.shape[0]
    N = int(N)
    short = A < N
    value = np.empty(R)
    used = np.empty(R, dtype=np.int64)
    accepted = np.zeros(R, dtype=bool)
    met_first = np.zeros(R, dtype=bool)
    if W is None:
        k = np.minimum(A, N)
        value[:] = _mean_first(X, k)
        used[:] = k
        accepted[:] = ~short
        met_first[:] = ~short
    else:
        W = float(W)
        active = ~short
        for m in range(N, B + 1):
            idx = np.nonzero(active & (A >= m))[0]
            if idx.size == 0:
                break
            xs = np.sort(X[idx, :m], axis=1)
            cs = np.concatenate([np.zeros((idx.size, 1)), np.cumsum(xs, axis=1)], axis=1)
            wmean = (cs[:, N:] - cs[:, :-N]) / N
            lo = xs[:, : m - N + 1]
            hi = xs[:, N - 1:]
            tol = W * np.abs(wmean)          # sign-robust: a single beat always passes
            ok = (wmean - lo <= tol) & (hi - wmean <= tol)
            any_ok = ok.any(axis=1)
            if any_ok.any():
                spread = np.where(ok, (hi - lo) / np.abs(wmean), np.inf)
                j = np.argmin(spread, axis=1)
                sel = idx[any_ok]
                value[sel] = wmean[any_ok, j[any_ok]]
                used[sel] = m
                accepted[sel] = True
                if m == N:
                    met_first[sel] = True
                active[sel] = False
        lim = ~accepted
        if limited_value == "all_acquired":
            value[lim] = _mean_first(X[lim], A[lim])
            used[lim] = A[lim]
        elif limited_value == "first_n":
            k = np.minimum(A[lim], N)
            value[lim] = _mean_first(X[lim], k)
            used[lim] = A[lim]
        else:
            raise ValueError(limited_value)
    limited = ~accepted
    return BeatRuleResult(value=value.reshape(shape), used=used.reshape(shape),
                          accepted=accepted.reshape(shape), met_first=met_first.reshape(shape),
                          limited=limited.reshape(shape))


def cross_view_flags(vm: np.ndarray, t_warn: float, t_adj: float):
    """Cross-view triggers within an axis: verification mean minus anchor mean.

    vm: (..., K). Returns diff, warn, adj with shape (..., K-1).
    """
    diff = vm[..., 1:] - vm[..., :1]
    warn = (diff >= t_warn) & (diff < t_adj)
    adj = diff >= t_adj
    return diff, warn, adj


def a4_final(vm: np.ndarray, over: np.ndarray, U: np.ndarray, t_warn: float, t_adj: float,
             s_det: float, f_rej: float, scrutiny: str = "triggered"):
    """Composite protocol rule A4.

    The anchor mean is always retained. A verification view is scrutinised
    according to `scrutiny` (default: it raised a warning or adjudication
    trigger). A scrutinised view with true overestimation (over=True) is
    rejected with probability s_det; a scrutinised genuine view is rejected
    with probability f_rej. Final value = highest mean not rejected.
    U: uniforms (..., K-1), shared across variants for common random numbers.
    """
    if vm.shape[-1] == 1:
        z = np.zeros(vm.shape[:-1] + (0,), dtype=bool)
        return vm[..., 0].copy(), {"warn": z, "adj": z, "reject": z}
    diff, warn, adj = cross_view_flags(vm, t_warn, t_adj)
    if scrutiny == "triggered":
        scr = warn | adj
    elif scrutiny == "adj_only":
        scr = adj
    elif scrutiny == "all_higher":
        scr = diff > 0
    else:
        raise ValueError(scrutiny)
    p_rej = np.where(over[..., 1:], s_det, f_rej)
    reject = scr & (U < p_rej)
    ver = np.where(reject, -np.inf, vm[..., 1:])
    final = np.maximum(vm[..., 0], ver.max(axis=-1))
    return final, {"warn": warn, "adj": adj, "reject": reject}
