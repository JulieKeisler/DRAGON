"""End-to-end: DragonSR recovers Nguyen-4 on a small budget, and predict()
evaluates the reported expression."""
import numpy as np
import pandas as pd

from dragon.search_algorithm.dragonsr import DragonSR, select_stream


def test_fit_predict_n4():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 2000)
    f = lambda v: v**6 + v**5 + v**4 + v**3 + v**2 + v
    m = DragonSR(n_iterations=240, families=["rational"], seed=1).fit(
        pd.DataFrame({"x": x}), f(x))
    # A level can end a few evaluations past the stream budget (the paper's
    # runs performed 19,922-20,040 evaluations for 20,000).
    assert abs(m.n_evaluations_ - 240) <= 12
    assert {s["stream"] for s in m.streams_} == {"rational|channel", "rational|full"}
    xt = rng.uniform(-1, 1, 500)
    assert m.score(pd.DataFrame({"x": xt}), f(xt)) > 0.9999, m.formula_


def test_tie_break_prefers_simpler_class():
    rs = [{"loss": 1e-14, "winner_type": "rational", "formula": "x/(1+x)"},
          {"loss": 2e-12, "winner_type": "channel", "formula": "2*x + 1"},
          {"loss": 1e-14, "winner_type": "channel", "formula": "3.0"},
          {"loss": 0.5, "winner_type": "channel", "formula": "x"}]
    assert select_stream(rs)["formula"] == "2*x + 1"


def test_numpy_input_and_names():
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 2, (800, 2))
    m = DragonSR(n_iterations=120, families=["rational"], objectives=["full"],
                 var_aug=False, seed=0).fit(X, X[:, 0] * X[:, 1])
    assert m.raw_feature_names_ == ["x0", "x1"]
    assert np.all(np.isfinite(m.predict(X)))


def test_budget_beyond_curriculum_warns():
    import pytest
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 200)
    m = DragonSR(n_iterations=400, families=["rational"], objectives=["full"],
                 k_init=5, t_per_level=5, max_complexity=2, var_aug=False, seed=0)
    with pytest.warns(UserWarning, match="t_per_level >= 198"):
        m.fit(pd.DataFrame({"x": x}), 2 * x)
    # the curriculum caps the stream: about k_init + max_complexity * t_per_level
    assert m.n_evaluations_ < 30
