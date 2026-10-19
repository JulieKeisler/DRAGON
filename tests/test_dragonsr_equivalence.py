"""The library DragonSR against the frozen experiment code (leaderboard/).

Each test feeds the same inputs to a library component and to the frozen one
it was ported from, and requires identical outputs.  The frozen harness reads
its settings from ``leaderboard/Config.py``; the paper's runs overrode the
composition penalty (0.01), which is set here the same way.
"""
import copy
import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import Config as frozen_config
from dataprocessing.Features import (VarAugmentor as FrozenVarAugmentor,
                                     XGBoostSelector as FrozenSelector,
                                     CombinationBuilder as FrozenCombos)
from pipeline.DragonOrchestrator import (SearchLoss as FrozenSearchLoss,
                                         AlignmentLoss as FrozenAlignmentLoss)
from runner.Dragon import DragonSearcher as FrozenSearcher, RegressionDataset as FrozenDataset
from runner.Ols import OLSPostProcessor as FrozenOLS
from searchspace.SearchSpace import SearchSpaceBuilder as FrozenSpace

from dragon.search_space.symbolic import (CombinationBuilder, SymbolicSearchSpace,
                                          OPERATOR_FAMILIES)
from dragon.utils.symbolic import dag_to_formula
from dragon.utils.symbolic.combination import OLSPostProcessor
from dragon.utils.symbolic.evaluation import DragonSearcher, RegressionDataset
from dragon.utils.symbolic.losses import SearchLoss, AlignmentLoss
from dragon.utils.symbolic.preprocessing import VarAugmentor, XGBoostSelector

frozen_config.Loss.COMPOSITION_PENALTY = 0.01   # paper configuration
frozen_config.Dragon.PER_TARGET_OPS = False


@pytest.fixture(autouse=True)
def _frozen_number_format(monkeypatch):
    """The frozen code prints coefficients with fixed decimals; the library
    prints them to six significant figures unless told to print as it did."""
    monkeypatch.setattr(dag_to_formula, "LEGACY_FORMAT", True)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(1, 2, n), "b": rng.uniform(1, 2, n),
                      "c": rng.uniform(-1, 1, n)})
    y = pd.Series(X.a ** X.b + 0.5 * np.sin(X.c), name="y")
    return X, y


@pytest.mark.parametrize("name", ["corr", "mse", "raw_mse", "mae", "huber"])
def test_search_losses(name):
    rng = np.random.default_rng(1)
    y, p = rng.normal(size=500), rng.normal(size=500)
    assert SearchLoss.score(name, p, y) == FrozenSearchLoss.score(name, p, y)


@pytest.mark.parametrize("kind", ["mse", "huber"])
def test_alignment_losses(kind):
    rng = np.random.default_rng(2)
    y, p = rng.normal(size=500), rng.normal(size=500)
    v = float(np.var(y))
    assert (AlignmentLoss.compute(kind, p, y, v, 0.2)
            == FrozenAlignmentLoss.compute(kind, p, y, v, 0.2))


def test_preprocessing():
    X, y = _data()
    Xl, Xf = VarAugmentor().transform(X, y), FrozenVarAugmentor().transform(X, y)
    assert list(Xl.columns) == list(Xf.columns)
    np.testing.assert_array_equal(Xl.values, Xf.values)
    names_l, scores_l = XGBoostSelector(n_top=14).transform(Xl, y)
    names_f, scores_f = FrozenSelector(n_top=14, seed=42).transform(Xf, y)
    assert names_l == names_f and scores_l == scores_f
    assert (CombinationBuilder().build(names_l, scores_l)
            == FrozenCombos().build(names_f, scores_f))


@pytest.mark.parametrize("mode", ["channel", "full"])
def test_model_classes(mode):
    rng = np.random.default_rng(3)
    a, b = rng.uniform(1, 2, 400), rng.uniform(1, 2, 400)
    P = torch.tensor(np.column_stack([a, b, a * b, 1 / (a + b), rng.normal(size=400)]),
                     dtype=torch.float32)
    y = torch.tensor((a - b) / (a + b), dtype=torch.float32).reshape(-1, 1)
    out_l = OLSPostProcessor(search_loss="corr").evaluate(P, y, mode)
    out_f = FrozenOLS(search_loss="corr").evaluate(P, y, mode)
    # (loss, selected channel, weights, alignment loss, ...): compare every field
    for l_, f_ in zip(out_l[:4], out_f[:4]):
        if isinstance(l_, np.ndarray):
            np.testing.assert_array_equal(l_, f_)
        else:
            assert l_ == f_
    assert (out_l[5] is None) == (out_f[5] is None) and (out_l[6] is None) == (out_f[6] is None)


def _spaces(family):
    X, y = _data()
    X = VarAugmentor().transform(X, y)
    names, scores = XGBoostSelector(n_top=14).transform(X, y)
    combos = CombinationBuilder().build(names, scores)
    ops = OPERATOR_FAMILIES[family]
    lib_space, lib_dag = SymbolicSearchSpace(names, scores, ops).build(combos)
    frz_space, frz_dag = FrozenSpace(names, scores, ops, target=None).build(combos)
    return X[names], y, names, lib_space, lib_dag, frz_space, frz_dag


@pytest.mark.parametrize("family", list(OPERATOR_FAMILIES))
def test_search_space_draws(family):
    *_, lib_space, lib_dag, frz_space, frz_dag = _spaces(family)
    for dag in (lib_dag, frz_dag):
        dag.complexity = 3
    for seed in range(5):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        g_l = lib_space.random()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        g_f = frz_space.random()
        assert str(g_l) == str(g_f)


@pytest.mark.parametrize("family,objective", [("general_power", "full"),
                                              ("trig", "channel"), ("rational", "full")])
def test_candidate_evaluation(family, objective, tmp_path):
    X, y, names, lib_space, lib_dag, _, _ = _spaces(family)
    lib_dag.complexity = 4
    loader_l = DataLoader(RegressionDataset(X, y), batch_size=32, shuffle=False)
    loader_f = DataLoader(FrozenDataset(X, y), batch_size=32, shuffle=False)
    kw = dict(loss_mode=objective, search_loss="corr", optimize_constants=True,
              constants_optimizer="dichotomy")
    lib = DragonSearcher(lib_space, loader_l, torch.device("cpu"), X.shape[1], names,
                         log_path=None, **kw)
    frz = FrozenSearcher(lib_space, loader_f, torch.device("cpu"), X.shape[1], names,
                         log_path=str(tmp_path / "frozen.log"), **kw)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    graphs = [lib_space.random() for _ in range(8)]
    def _seeded(searcher, g, idx):
        # Building a graph draws the initial values of its constants from the
        # torch generator, and the line search starts from them: reseed.
        random.seed(idx); np.random.seed(idx); torch.manual_seed(idx)
        return searcher(copy.deepcopy(g), idx)[0]
    for idx, g in enumerate(graphs):
        loss_l, loss_f = _seeded(lib, g, idx), _seeded(frz, g, idx)
        assert loss_l == loss_f or (np.isinf(loss_l) and np.isinf(loss_f))
    torch.manual_seed(99); lib.finalize_constants()
    torch.manual_seed(99); frz.finalize_constants()
    assert lib.state["best_loss"] == frz.state["best_loss"]
    assert lib.state["best_formula"] == frz.state["best_formula"]
    assert lib.state["winner_type"] == frz.state["winner_type"]
