"""Smoke tests for CreditModel — monotonic GBM scorecard."""
from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import make_synthetic; from src.model import fit_and_evaluate
from src.core import monotonicity_check, lorenz_curve


def test_data():
    d = make_synthetic(500); assert d["n_samples"] == 500 and 0.1 < d["positive_rate"] < 0.5


def test_monotonicity():
    assert monotonicity_check([1, -2, 3], [1, -1, 1]) == 1.0


def test_gini():
    g = lorenz_curve([0,0,1,1],[0.1,0.2,0.8,0.9]); assert g > 0


def test_fit():
    d = make_synthetic(500); m, met = fit_and_evaluate(d)
    assert "gini" in met and met["gini"] > 0


if __name__ == "__main__":
    test_data(); test_monotonicity(); test_gini(); test_fit(); print("OK")
