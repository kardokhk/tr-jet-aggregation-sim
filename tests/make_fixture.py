"""Regenerate the regression fixture. Run only when a model change is intended,
and record the reason in notes/lab-notebook.md.

  python tests/make_fixture.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))

import pandas as pd  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim.experiments import run_cells  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "regression_v01.csv"
SPEC = [("E1", [0, 700], 3000), ("E2", [5, 1000], 3000), ("E3", [30], 3000),
        ("E4", [100], 3000), ("E5", [3], 3), ("E6", [10], 2)]
DROP = ["cpu_s", "wall_s"]


def build():
    cfg = C.load_config(ROOT / "tests" / "fixtures" / "config_v01.yaml")  # frozen: detects code changes, not parameter changes
    cfg["parameters"]["calib_n"]["value"] = 2000
    frames = [run_cells(cfg, e, cells, n_rep=n) for e, cells, n in SPEC]
    df = pd.concat(frames, ignore_index=True).drop(columns=DROP)
    return df


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(FIXTURE, index=False, float_format="%.12g")
    print(FIXTURE, df.shape)
