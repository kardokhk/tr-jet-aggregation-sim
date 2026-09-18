import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))

from duomaxsim import config as C  # noqa: E402

CONFIG = ROOT / "code" / "configs" / "base.yaml"


@pytest.fixture(scope="session")
def cfg():
    return C.load_config(CONFIG)


@pytest.fixture
def base(cfg):
    return C.base_params(cfg)


# All noise and systematic error switched off; tests switch on what they need.
QUIET = dict(view_errors=False, beat_cv=0.0, g_log_sd=0.0, sigma_rb_mm=0.0, sigma_cal_mm=0.0,
             rhythm="sinus", beta_rr=0.0, beta_drr=0.0, af_extra_cv=0.0, window=None,
             data_mode="prospective")
