from dataprocessing.Features import CombinationBuilder
import sys
from pathlib import Path
import os
from Config import Dragon as _CfgDragon
import torch.nn as nn

# local helper path
sys.path.insert(0, str(Path(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))))

# leaderboard path
leaderboard_root = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, str(leaderboard_root))

# DRAGON root path (for lib.dragon)
dragon_root = Path(os.path.abspath(os.path.join(leaderboard_root, "..")))
sys.path.insert(0, str(dragon_root))
import lib.dragon
sys.modules['dragon'] = sys.modules['lib.dragon']

# ── Dragon library imports ────────────────────────────────────────────────────
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

class SearchSpaceBuilder:
    """Builds the Dragon EvoDag search space for a set of operators.

    Parameters
    ----------
    feature_names  : ordered list of feature name strings
    feature_scores : dict mapping name → importance weight
    operator_keys  : list of operator tags, e.g. ['select', 'unary', 'power', ...]
    """

    def __init__(
        self,
        feature_names:  list[str],
        feature_scores: dict[str, float],
        operator_keys:  list[str],
    ):
        self.feature_names  = feature_names
        self.feature_scores = feature_scores
        self.operator_keys  = operator_keys
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
            size=_CfgDragon.MAX_NODES,
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