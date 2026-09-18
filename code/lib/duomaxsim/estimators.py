"""Estimators A1 to A7 (protocol section 4). A4 lives in rules.a4_final."""
from __future__ import annotations

import numpy as np

from .model import Latent


def a1_anchor(vm):
    return vm[..., 0]


def a2_mean(vm):
    return vm.mean(axis=-1)


def a3_max(vm):
    return vm.max(axis=-1)


def a5_median(vm):
    return np.median(vm, axis=-1)


def a6_index_beat(meas_anchor: np.ndarray, logrr_anchor: np.ndarray, used_anchor: np.ndarray):
    """Index-beat estimator on the anchor view.

    meas_anchor: (n, D, B) measured beats; logrr_anchor: (n, B+1);
    used_anchor: (n, D) beats acquired. Picks, among acquired beats, the beat
    whose RR1/RR2 is closest to 1 (|log RR1 - log RR2| minimal).
    """
    B = meas_anchor.shape[-1]
    score = np.abs(logrr_anchor[:, :-1] - logrr_anchor[:, 1:])          # (n, B)
    score = np.broadcast_to(score[:, None, :], meas_anchor.shape)
    mask = np.arange(B)[None, None, :] < np.maximum(used_anchor, 1)[..., None]
    score = np.where(mask, score, np.inf)
    j = np.argmin(score, axis=-1)
    return np.take_along_axis(meas_anchor, j[..., None], axis=-1)[..., 0]


def a7_oracle(vm, lat: Latent):
    """Mean of view means after dividing out each view's true multiplicative offset."""
    return (vm / (1.0 - lat.u + lat.o)).mean(axis=-1)


def standard_estimators(vm, lat: Latent, meas, used):
    """A1, A2, A3, A5, A6, A7 as a dict of (n, D) arrays."""
    return {
        "A1": a1_anchor(vm),
        "A2": a2_mean(vm),
        "A3": a3_max(vm),
        "A5": a5_median(vm),
        "A6": a6_index_beat(meas[:, :, 0, :], lat.logrr[:, 0, :], used[:, :, 0]),
        "A7": a7_oracle(vm, lat),
    }
