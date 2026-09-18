"""Regression test against a small fixed-seed fixture (task item 6)."""
import numpy as np
import pandas as pd

import make_fixture as MF


def test_regression_fixture_unchanged():
    ref = pd.read_csv(MF.FIXTURE)
    new = MF.build()
    assert list(new.columns) == list(ref.columns)
    assert len(new) == len(ref)
    key = ["experiment", "cell", "metric"]
    for c in key:
        assert (new[c].astype(str).to_numpy() == ref[c].astype(str).to_numpy()).all(), c
    for c in ("value", "mcse", "n"):
        np.testing.assert_allclose(new[c].to_numpy(float), ref[c].to_numpy(float), rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=c)
