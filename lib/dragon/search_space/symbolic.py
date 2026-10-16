"""Search space of DragonSR: symbolic operator families on a DRAGON DAG.

Ported from the frozen experiment code (``leaderboard/searchspace/
SearchSpace.py``, ``leaderboard/target_categories.py`` and the
``CombinationBuilder`` of ``leaderboard/dataprocessing/Features.py``,
CODE_VERSION AGGREGATE 2846c144).  The search space is unchanged; the one
difference is that the operator set never depends on the target: the frozen
code could replace every stream's operators by the target's class
(``PER_TARGET_OPS``), which the paper's runs switched off, and the library has
no target to look up.

A DragonSR run searches the six operator families of ``OPERATOR_FAMILIES``,
each in its own stream; every family contains the algebraic core
(``select``, ``unary``, ``power``, ``sum``, ``boost``, ``const``).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch.nn as nn

from dragon.search_space.bricks_variables import operations_var
from dragon.search_space.base_variables import CatVar, Constant, ArrayVar
from dragon.search_space.dag_encoding import SymbolicNode
from dragon.search_space.dag_variables import HpVar, EvoDagVariable
from dragon.search_operators.base_neighborhoods import (
    CatInterval, ConstantInterval, ArrayInterval,
)
from dragon.search_operators.dag_neighborhoods import EvoDagInterval, HpInterval
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import (
    SelectFeatures, Inverse, Negate, Power, SumFeatures, ConstantBrick,
    Ln, Sin, Cos, Exp, ExpAffine, ChannelBoost,
)

# Maximum number of nodes of a graph (paper: 15; frozen Config.Dragon.MAX_NODES).
MAX_NODES = 15

# The six operator families (frozen target_categories.OPERATOR_GROUPS_BY_CATEGORY).
# Keys of each list are keys of SymbolicSearchSpace._ops_map.
OPERATOR_FAMILIES: dict[str, list[str]] = {
    "rational":       ["select", "unary", "power", "sum", "boost", "const"],
    "trig":           ["select", "unary", "power", "sum", "sin", "cos", "boost", "const"],
    "log":            ["select", "unary", "power", "sum", "ln", "boost", "const"],
    "sqrt":           ["select", "unary", "power", "sum", "dilation", "boost", "const"],
    "exp":            ["select", "unary", "power", "sum", "exp", "expa", "boost", "const"],
    # x**y is only expressible as exp(y * ln(x)), so this family needs both ln and exp.
    "general_power":  ["select", "unary", "power", "sum", "ln", "exp", "expa", "boost", "const"],
}


def operator_streams(families=None) -> dict[str, list[str]]:
    """Operator lists of the streams of one run, keyed by family name.

    ``families`` selects a subset of ``OPERATOR_FAMILIES`` (default: all six, as
    in the paper).  The frozen code built the same six lists for every target,
    labelling the target's own class ``base``; only the labels differ.
    """
    names = list(OPERATOR_FAMILIES) if families is None else list(families)
    unknown = [n for n in names if n not in OPERATOR_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown operator families {unknown}; available: {list(OPERATOR_FAMILIES)}")
    return {n: list(OPERATOR_FAMILIES[n]) for n in names}


class CombinationBuilder:
    """Builds weighted feature-index combinations for Dragon's SelectFeatures node.

    Strategy
    --------
    For sizes 1 and 2 every possible combination is generated.
    For sizes 3+, only the top-M features (by importance) are considered,
    then combinations are ranked by sum of log-probabilities and the top
    `budget` are kept.

    Parameters
    ----------
    max_combo       : maximum subset size (clipped to n features)
    budget_per_size : optional dict overriding the per-size sample budget
    """

    _TOP_M = {3: 30, 4: 20, 5: 14, 6: 12}   # candidate pool size for sizes >= 3

    def __init__(self, max_combo: int = 6, budget_per_size: dict = None):
        self.max_combo       = max_combo
        self.budget_per_size = budget_per_size or {}

    def build(self, feature_names: list[str], feature_scores: dict[str, float]) -> list[list[int]]:
        """Return list of index lists, each representing a candidate feature subset."""
        n = len(feature_names)
        if n == 1:
            return [[0]]

        probs    = np.array([feature_scores.get(c, 1.0 / n) for c in feature_names])
        probs    = probs / probs.sum()
        log_prob = np.log(probs + 1e-300)

        max_combo = min(self.max_combo, n)
        budget    = {1: n, 2: n * (n - 1) // 2}
        for s in range(3, max_combo + 1):
            budget[s] = max(50, 200 // (s - 1))
        budget.update(self.budget_per_size)

        all_combos = []
        for size in range(1, max_combo + 1):
            cap = budget[size]
            if size <= 2:
                pool = list(combinations(range(n), size))
            else:
                M       = min(self._TOP_M.get(size, 12), n)
                top_idx = np.argsort(probs)[-M:]
                pool    = list(combinations(sorted(top_idx.tolist()), size))

            scored = [(sum(log_prob[i] for i in combo), list(combo)) for combo in pool]
            scored.sort(key=lambda x: -x[0])
            all_combos.extend(combo for _, combo in scored[:cap])

        return all_combos


class SymbolicSearchSpace:
    """Builds the DRAGON EvoDag search space of DragonSR for a set of operators.

    Parameters
    ----------
    feature_names  : ordered list of feature name strings
    feature_scores : dict mapping name -> importance weight
    operator_keys  : list of operator tags, e.g. ['select', 'unary', 'power', ...]
    max_nodes      : maximum number of nodes of a graph (paper: 15)
    """

    def __init__(
        self,
        feature_names:  list[str],
        feature_scores: dict[str, float],
        operator_keys:  list[str],
        max_nodes:      int = MAX_NODES,
    ):
        self.feature_names  = feature_names
        self.feature_scores = feature_scores
        self.operator_keys  = operator_keys
        self.max_nodes      = max_nodes
        self._combo_builder = CombinationBuilder()

    def build(self, all_combos: list = None) -> tuple:
        """Return (ArrayVar search_space, EvoDagVariable dag)."""
        n = len(self.feature_names)
        if all_combos is None:
            all_combos = self._combo_builder.build(self.feature_names, self.feature_scores)

        combo_weights = SelectFeatures.combination_weights(
            [self.feature_scores.get(c, 1.0 / n) for c in self.feature_names],
            all_combos,
        )
        candidates = [v for k, v in self._ops_map(all_combos, combo_weights).items()
                      if k in self.operator_keys]

        cand_ops = operations_var(
            "CandidateOperations",
            size=self.max_nodes,
            candidates=candidates,
            combiner_features=['add', 'mul'],
            activations=Constant("id", value=nn.Identity(), neighbor=ConstantInterval()),
            node_type=SymbolicNode,
        )
        dag = EvoDagVariable(
            label="Dag",
            operations=cand_ops,
            init_complexity=4,
            neighbor=EvoDagInterval(nb_mutations=2),
        )
        return ArrayVar(dag, label="Search Space", neighbor=ArrayInterval()), dag

    def _ops_map(self, all_combos, combo_weights) -> dict:
        def _hpv(label, brick, hps=None):
            return HpVar(label, brick, hyperparameters=hps or {}, neighbor=HpInterval())
        def _const(label, cls):
            return Constant(label, cls, neighbor=ConstantInterval())
        def _cat(label, features):
            return CatVar(label, features=features, neighbor=CatInterval())

        return {
            "select": HpVar(
                "SelectFeatures", _const("SelectFeaturesOp", SelectFeatures),
                hyperparameters={"feature_indices": CatVar(
                    "feature_indices", features=all_combos, weights=combo_weights,
                    neighbor=CatInterval())},
                neighbor=HpInterval()),
            "unary":    _hpv("UnaryOp",    _cat("UnaryOpType", [Identity, Inverse, Negate])),
            "power":    _hpv("Power",      _const("PowerOp", Power),
                             {"exponent": _cat("exponent", [-3, -2, -1, 1, 2, 3])}),
            "sum":      _hpv("Sum",        _const("SumOp",  SumFeatures)),
            "ln":       _hpv("Ln",         _const("LnOp",   Ln)),
            "sin":      _hpv("Sin",        _const("SinOp",  Sin)),
            "cos":      _hpv("Cos",        _const("CosOp",  Cos)),
            "exp":      _hpv("Exp",        _const("ExpOp",  Exp)),
            "expa":     _hpv("ExpAffine",  _const("ExpAffineOp", ExpAffine)),
            "boost":    _hpv("ChannelBoost", _const("ChannelBoostOp", ChannelBoost),
                             {"mode": _cat("mode", ["add", "sub", "mul", "div"])}),
            "const":    _hpv("ConstantBrick", _const("ConstOp", ConstantBrick)),
            "dilation": _hpv("Dilation",   _const("DilOp", Power),
                             {"exponent": _cat("exponent", [0.5, 1, 2, 3])}),
        }