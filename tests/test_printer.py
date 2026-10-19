"""Printed coefficients keep their significant figures."""
import numpy as np

from dragon.utils.symbolic import dag_to_formula
from dragon.utils.symbolic.dag_to_formula import coef, format_nested


def test_small_coefficient_keeps_its_figures():
    # reverse-ablation Kepler run 0: the fitted w = 0.000817 printed as 0.0008
    w = 0.000817
    assert coef(w) == "+0.000817"
    assert abs(float(coef(w)) ** 3 / w**3 - 1) < 1e-6
    assert coef(np.float32(w)) == "+0.000817"
    assert coef(6.674e-11, ".4f") == "6.674e-11"


def test_never_fewer_decimals_than_before():
    for x in (1234.5678, -20.0181, 1.4745322, 0.5, 171.0, 1e15 + 0.25):
        assert abs(float(coef(x)) - x) <= abs(float(format(x, "+.4f")) - x)
    assert coef(1234.5678) == "+1234.5678"
    assert coef(1.4745322) == "+1.47453"


def test_zero_signs_and_scale():
    assert coef(0.0) == "+0" and coef(0.0, ".4f") == "0"
    assert coef(-2.5, ".4f") == "-2.5"
    assert coef(np.exp(30.0), ".4e") == f"{np.exp(30.0):.6e}"
    assert coef(float("inf")) == "+inf"


def test_legacy_format_prints_as_the_paper_code(monkeypatch):
    monkeypatch.setattr(dag_to_formula, "LEGACY_FORMAT", True)
    assert coef(0.000817) == "+0.0008"
    assert coef(np.exp(1.0), ".4e") == "2.7183e+00"


def test_nested_formula_carries_the_fitted_weight():
    nested = {"link": "log", "unaries": ["log"], "w": [0.000817], "b": 0.0, "valid": [True]}
    s = format_nested(nested, ["x"])
    assert "0.000817" in s, s
