from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import traceback
from Config import Experiment as _CfgExp, Dragon as _CfgDragon, Paths as _CfgPaths
from Features import CombinationBuilder
from Preprocessing import PreprocessingPipeline
from ols import OLSPostProcessor, correl
from Data import DatasetLoader
from stats import DAGInspector
from denoise import apply_denoise, MCDropoutWeighter

_dataset_loader = DatasetLoader()


# dragon path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import lib.dragon
sys.modules['dragon'] = sys.modules['lib.dragon']

# ── Dragon library imports ────────────────────────────────────────────────────
from dragon.search_space.bricks_variables import operations_var
from dragon.search_space.base_variables import CatVar, Constant, ArrayVar
from dragon.search_space.dag_encoding import AdjMatrix, SymbolicNode
from dragon.search_space.dag_variables import HpVar, EvoDagVariable
from dragon.search_operators.base_neighborhoods import (
    CatInterval, ConstantInterval, ArrayInterval,
)
from dragon.search_operators.dag_neighborhoods import EvoDagInterval, HpInterval
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import (
    SelectFeatures, Inverse, Negate, Power, SumFeatures, ConstantBrick,
    Ln, Sin, Cos, Exp,
)
from dragon.utils.plot_functions import graph_to_all_formulas
import threading as _threading
import dragon.search_algorithm.search_algorithm as _dragon_sa
from dragon.search_algorithm.mutant_ucb import Mutant_UCB
from itertools import combinations

# Thread-safe SIGALRM patch -----------------------------------------------------
_original_timed_evaluation = _dragon_sa.timed_evaluation

def _thread_safe_timed_evaluation(x, idx, max_duration, evaluation):
    if _threading.current_thread() is not _threading.main_thread():
        return evaluation(x, idx)
    return _original_timed_evaluation(x, idx, max_duration, evaluation)

_dragon_sa.timed_evaluation = _thread_safe_timed_evaluation


# ══════════════════════════════════════════════════════════════════════════════
#  INIT STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def _build_random_seed_dags(feature_names, operator_keys, seed=0, n_seeds=200):
    """Build a population of seed DAGs for the 'random' init strategy."""
    import random as _rnd
    rng = _rnd.Random(seed)
    if not feature_names:
        return None

    op_pool = [
        ("Identity", Identity, {}, nn.Identity()),
        ("Inverse", Inverse, {}, nn.Identity()),
        ("Negate", Negate, {}, nn.Identity()),
        ("SumFeatures", SumFeatures, {}, nn.Identity()),
        *[(f"Power_{exp}", Power, {"exponent": exp}, nn.Identity())
          for exp in (-3, -2, -1, 1, 2, 3)],
        *[(f"Sel_{i}", SelectFeatures, {"indices": [i]}, nn.Identity())
          for i in range(len(feature_names))],
        *[(f"Sel_{i}_{j}", SelectFeatures, {"indices": [i, j]}, nn.Identity())
          for i, j in combinations(range(len(feature_names)), 2)],
    ]
    if "const" in operator_keys:
        op_pool.append(("Const", ConstantBrick, {}, nn.Identity()))

    combiner_patterns = (
        lambda k: "add",
        lambda k: "mul",
        lambda k: "add" if k % 2 == 0 else "mul",
        lambda k: "mul" if k % 2 == 0 else "add",
    )
    topologies = ("chain", "fan", "skip", "rand")

    seed_dags = []
    for s in range(n_seeds):
        size = rng.randint(3, 7)
        topo = topologies[s % len(topologies)]
        comb_fn = combiner_patterns[s % len(combiner_patterns)]
        nodes = [
            SymbolicNode(
                combiner=comb_fn(k),
                operation=op_cls,
                hp=dict(hp),
                activation=act,
            )
            for k in range(size)
            for _, op_cls, hp, act in [op_pool[rng.randrange(len(op_pool))]]
        ]

        M = np.zeros((size, size), dtype=int)
        if topo in ("chain", "skip", "rand"):
            M[np.arange(size - 1), np.arange(1, size)] = 1
        if topo == "fan":
            M[0, 1:] = 1
        if topo == "skip":
            M[np.arange(size - 2), np.arange(2, size)] = 1
        if topo == "rand":
            for i in range(size):
                for j in range(i + 1, size):
                    if rng.random() < 0.3:
                        M[i, j] = 1
        try:
            seed_dags.append(AdjMatrix(operations=nodes, matrix=M))
        except Exception:
            continue

    return [[m] for m in seed_dags] if seed_dags else None


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class MetaArchi(nn.Module):
    def __init__(self, args: dict, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape
        assert isinstance(args['Dag'], AdjMatrix)
        self.dag = args['Dag']
        self.dag.set(input_shape)

    def forward(self, X):
        return self.dag(X)

    def set_prediction_to_save(self, name: str, df: pd.DataFrame):
        if hasattr(self, "prediction"):
            self.prediction[name] = df
        else:
            self.prediction = {name: df}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "best_model.pth"))
        if hasattr(self, "prediction"):
            for k, v in self.prediction.items():
                v.to_csv(os.path.join(path, f"best_model_{k}_outputs.csv"))


# ── Dataset ──────────────────────────────────────────────────────────────────

class RegressionDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.DoubleTensor(X.values)
        self.y = torch.DoubleTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE
# ══════════════════════════════════════════════════════════════════════════════

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
            "const":    _hpv("ConstantBrick", _const("ConstOp", ConstantBrick)),
            "dilation": _hpv("Dilation",   _const("DilOp", Power),
                             {"exponent": _cat("exponent", [0.5, 1, 2, 3])}),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS & WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════

class DragonEvaluator:
    """Manage OLS evaluation, loss mode, and optional sample-weighted subsampling."""

    def __init__(
        self,
        loss_mode: str = "full",
        subsample_ratio: float = 1.0,
        X_np: np.ndarray = None,
        y_np: np.ndarray = None,
        sample_weights: np.ndarray = None,
    ):
        self.loss_mode = loss_mode
        self.subsample_ratio = subsample_ratio
        self.X_np = X_np
        self.y_np = y_np
        self.sample_weights = sample_weights
        self._ols = OLSPostProcessor()

    def forward(self, model, train_loader, idx, device=None):
        if device is None:
            device = self.device if hasattr(self, "device") else torch.device("cpu")
        if self.subsample_ratio < 1.0 and self.X_np is not None:
            n_total = len(self.y_np)
            n_sub = max(int(n_total * self.subsample_ratio), min(n_total, 20))
            prob = None
            if self.sample_weights is not None and len(self.sample_weights) == n_total:
                w = np.maximum(self.sample_weights, 0.0)
                ws = w.sum()
                prob = (w / ws) if ws > 0 else None
            rng = np.random.default_rng(int(idx) % (2 ** 31))
            sub_idx = rng.choice(n_total, n_sub, replace=False, p=prob)
            Xb = torch.tensor(self.X_np[sub_idx], dtype=torch.float32).to(device)
            yb = torch.tensor(self.y_np[sub_idx], dtype=torch.float32).reshape(-1, 1).to(device)
            with torch.no_grad():
                return model(Xb).detach().cpu(), yb.detach().cpu()

        all_pred, all_true = [], []
        with torch.no_grad():
            for Xb, yb in train_loader:
                all_pred.append(model(Xb.to(device)).detach().cpu())
                all_true.append(yb.to(device).detach().cpu())
        return torch.cat(all_pred), torch.cat(all_true)

    def evaluate(self, pred_all, true_all):
        return self._ols.evaluate(pred_all, true_all, self.loss_mode)

    def resolve_prediction(self, pred_all, true_all, selected_c,
                           ols_weights, nested, rational):
        y_np = true_all.squeeze().numpy()
        if rational is not None:
            return y_np, rational["pred"], "rational", rational.get("max_degree")
        if nested is not None:
            return y_np, nested["pred"], "nested", None
        if ols_weights is not None and np.any(ols_weights != 0):
            P = pred_all.numpy(); kept = ols_weights != 0
            yp = P[:, kept] @ ols_weights[kept]
            return y_np, yp + float(np.mean(y_np - yp)), "ols", None
        ch_raw = (pred_all[:, selected_c].numpy()
                  if pred_all.ndim > 1 and pred_all.shape[-1] > 1
                  else pred_all.squeeze().numpy())
        A = np.column_stack([ch_raw, np.ones_like(ch_raw)])
        c, *_ = np.linalg.lstsq(A, y_np, rcond=None)
        return y_np, float(c[0]) * ch_raw + float(c[1]), "channel", None

    def apply_weighted_loss(self, mse, y_np, y_pred):
        if self.sample_weights is None or self.subsample_ratio < 1.0 or y_pred is None:
            return mse
        try:
            w = np.maximum(np.asarray(self.sample_weights, dtype=np.float64), 0.0)
            ws = w.sum()
            if ws <= 0 or len(w) != len(y_np):
                return mse
            w /= ws
            ym = float(np.dot(w, y_np))
            vw = float(np.dot(w, (y_np - ym) ** 2))
            if vw <= 1e-30:
                return mse
            r = y_np - np.asarray(y_pred).ravel()
            if self._ols.loss_kind == "huber":
                d = self._ols.huber_delta_frac * float(np.sqrt(vw))
                a = np.abs(r); q = np.minimum(a, d)
                return float(np.dot(w, 0.5 * q ** 2 + d * (a - q)) / (0.5 * vw))
            return float(np.dot(w, r ** 2) / vw)
        except Exception:
            return mse


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH LOOP  (callable loss function + live state)
# ══════════════════════════════════════════════════════════════════════════════

class DragonSearcher:
    """Callable loss function for Dragon's Mutant_UCB, with live result tracking.

    The instance is passed directly as the loss function to Mutant_UCB.
    After the search, all results are in `searcher.state`.

    Usage
    -----
    searcher = DragonSearcher(search_space, train_loader, ...)
    algo     = Mutant_UCB(..., loss_func=searcher)
    algo.run()
    print(searcher.state["best_formula"])

    Parameters
    ----------
    search_space       : ArrayVar returned by SearchSpaceBuilder.build()
    train_loader       : DataLoader for full-dataset evaluation
    device             : torch device
    num_features       : number of input features
    feature_names      : list of feature name strings
    log_path           : path to append OLS analysis log
    loss_mode          : 'full', 'ols', or 'channel'
    optimize_constants : if True, run 100-epoch Adam on ConstantBrick
    subsample_ratio    : fraction of dataset to use per evaluation (< 1 = stochastic)
    X_np, y_np         : numpy arrays for stochastic subsampling path
    sample_weights     : per-sample MC-Dropout confidence weights
    """

    def __init__(
        self,
        search_space,
        train_loader,
        device,
        num_features:       int,
        feature_names:      list[str],
        log_path:           str,
        loss_mode:          str        = "full",
        optimize_constants: bool       = False,
        subsample_ratio:    float      = 1.0,
        X_np:               np.ndarray = None,
        y_np:               np.ndarray = None,
        sample_weights:     np.ndarray = None,
    ):
        self.search_space       = search_space
        self.train_loader       = train_loader
        self.device             = device
        self.num_features       = num_features
        self.feature_names      = feature_names
        self.log_path           = log_path
        self.loss_mode          = loss_mode
        self.optimize_constants = optimize_constants
        self.subsample_ratio    = subsample_ratio
        self.X_np               = X_np
        self.y_np               = y_np
        self.sample_weights     = sample_weights

        self._evaluator = DragonEvaluator(
            loss_mode=loss_mode,
            subsample_ratio=subsample_ratio,
            X_np=X_np,
            y_np=y_np,
            sample_weights=sample_weights,
        )
        self._ols = self._evaluator._ols
        self.state = {
            "best_loss":       np.inf,
            "winner_type":     "channel",
            "corr_value":      0.0,
            "alignment_loss":  1.0,
            "rat_degree":      None,
            "best_formula":    "N/A",
            "formula_channel": None,
            "formula_ols":     None,
            "formula_nested":  None,
            "formula_polyrat": {},
            "loss_channel":    None,
            "loss_ols":        None,
            "loss_nested":     None,
            "loss_polyrat":    {},
        }

    def __call__(self, args, idx, *kwargs):
        labels    = [e.label for e in self.search_space]
        args_dict = ({labels[0]: args} if isinstance(args, AdjMatrix)
                     else dict(zip(labels, args)))
        model     = MetaArchi(args_dict, input_shape=(self.num_features,)).to(self.device)

        self._maybe_optimize_constants(model)
        model.eval()

        pred_all, true_all = self._evaluator.forward(model, self.train_loader, idx, self.device)
        (mse, selected_c, ols_weights, alignment_loss,
         _lr, nested, rational, valid_idx_global, analysis) = self._evaluator.evaluate(
            pred_all, true_all)

        y_np, y_pred, winner_type, rat_degree = self._evaluator.resolve_prediction(
            pred_all, true_all, selected_c, ols_weights, nested, rational)

        mse      = self._evaluator.apply_weighted_loss(mse, y_np, y_pred)
        corr_val = float(correl(
            torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred,
            true_all))

        if mse < self.state["best_loss"]:
            self._update_state(mse, winner_type, corr_val, alignment_loss, rat_degree,
                               y_pred, y_np, selected_c, ols_weights, nested, rational,
                               valid_idx_global, analysis, pred_all, model, idx)

        print(f"Idx={idx}, Loss = {mse:.10f}")
        model.set_prediction_to_save("prediction", pd.DataFrame({"pred": y_pred, "true": y_np}))
        return float(mse), model

    def _maybe_optimize_constants(self, model):
        if not self.optimize_constants:
            return
        if sum(p.numel() for p in model.parameters()) != 1:
            return
        model  = model.float()
        opt    = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_fn = nn.MSELoss()
        model.train()
        for epoch in range(100):
            epoch_loss = 0.0; n_seen = 0
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(self.device).float(), yb.to(self.device).float()
                opt.zero_grad()
                loss = mse_fn(model(Xb), yb)
                loss.backward(); opt.step()
                epoch_loss += loss.item() * Xb.size(0); n_seen += Xb.size(0)
            epoch_loss /= max(n_seen, 1)
            if epoch == 0 and (epoch_loss > 1.0 or np.isnan(epoch_loss)):
                break
            if epoch_loss < 1e-12:
                break

    # ── State update ─────────────────────────────────────────────────────────

    def _update_state(self, mse, winner_type, corr_val, alignment_loss, rat_degree,
                      y_pred, y_np, selected_c, ols_weights, nested, rational,
                      valid_idx_global, analysis, pred_all, model, idx):
        s = self.state
        s["best_loss"]      = mse
        s["winner_type"]    = winner_type
        s["corr_value"]     = corr_val
        s["alignment_loss"] = float(alignment_loss)
        s["rat_degree"]     = rat_degree
        try:
            yp_np            = (y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor)
                                else np.asarray(y_pred, dtype=np.float64))
            s["best_pred_np"] = yp_np.ravel()
            s["best_y_np"]    = np.asarray(y_np, dtype=np.float64).ravel()
        except Exception:
            pass
        try:
            adj      = model.dag.matrix
            nodes    = model.dag.operations
            formulas = graph_to_all_formulas(adj, self.feature_names, nodes)
            s["best_formula"] = str(formulas[selected_c]) if formulas else "N/A"

            ops_str, const_str, dag_size, dag_text, dag_svg, dag_data = DAGInspector.summarize(adj, nodes)
            s.update(ops_used=ops_str, has_const=const_str, dag_size=dag_size,
                     dag_text=dag_text, dag_svg=dag_svg, dag_data=dag_data)

            s["formula_channel"]      = self._channel_formula(formulas, selected_c, pred_all, y_np)
            s["all_channel_formulas"] = self._all_channel_formulas(formulas, analysis, selected_c)
            s["formula_ols"]          = self._ols_formula(formulas, analysis)
            s["formula_nested"]       = (self._ols.format_nested(analysis["nested"], formulas)
                                         if analysis.get("nested") and formulas else None)
            s["formula_polyrat"]      = self._polyrat_formulas(formulas, valid_idx_global, analysis)

            self._update_losses(s, analysis, selected_c)
            self._set_best_formula(s, winner_type, rat_degree)

            with open(self.log_path, "a") as lf:
                lf.write(f"=== NEW BEST  Idx={idx}  Loss={mse:.10f}  "
                         f"Winner={winner_type.upper()}  alignment_loss={alignment_loss:.10f}\n")
                self._ols.print_analysis(analysis, formulas, valid_idx_global, selected_c, f=lf)
                lf.write("\n")
            print(f"\n  >> NEW BEST loss={mse:.10f} [{winner_type.upper()}]  Idx={idx}")
            self._ols.print_analysis(analysis, formulas, valid_idx_global, selected_c)
        except Exception as e:
            print(f"  >> Could not extract formula: {e}")

    # ── Formula helpers ───────────────────────────────────────────────────────

    def _channel_formula(self, formulas, selected_c, pred_all, y_np) -> str | None:
        raw = str(formulas[selected_c]) if formulas and selected_c < len(formulas) else None
        if raw is None:
            return None
        ch = (pred_all[:, selected_c].numpy().ravel()
              if pred_all.ndim > 1 and pred_all.shape[-1] > 1
              else pred_all.squeeze().numpy().ravel())
        A  = np.column_stack([ch, np.ones_like(ch)])
        c, *_ = np.linalg.lstsq(A, y_np.ravel(), rcond=None)
        a1, a0 = float(c[0]), float(c[1])
        if abs(a1 - 1.0) > 1e-4 or abs(a0) > 1e-4:
            return f"{a1:.6e} * ({raw}) + {a0:.6e}"
        return raw

    def _all_channel_formulas(self, formulas, analysis, selected_c) -> list:
        ch_map = {int(c): (float(v) if v is not None and np.isfinite(v) else None)
                  for c, v in analysis.get("channel_results", [])}
        return [{"idx": ci, "formula": str(f), "loss": ch_map.get(ci),
                 "selected": ci == selected_c}
                for ci, f in enumerate(formulas or [])]

    def _ols_formula(self, formulas, analysis) -> str | None:
        wts  = analysis.get("ols_weights")
        bias = analysis.get("ols_bias", 0.0)
        if wts is None or not formulas:
            return None
        terms = [f"{w:+.4f}*({formulas[i]})"
                 for i, w in enumerate(wts) if abs(w) > 1e-6 and i < len(formulas)]
        if abs(bias) > 1e-6:
            terms.append(f"{bias:+.4f}")
        return " ".join(terms) if terms else None

    def _polyrat_formulas(self, formulas, valid_idx_global, analysis) -> dict:
        out = {}
        for rc in analysis.get("rational_candidates", []):
            dk = str(rc.get("max_degree", "?"))
            if valid_idx_global is not None and formulas:
                out[dk] = self._ols.format_rational(rc, formulas, valid_idx_global)
        return out

    def _update_losses(self, s, analysis, selected_c):
        var_y  = analysis.get("var_y", 1.0) or 1.0
        ch_map = {c: v for c, v in analysis.get("channel_results", [])}
        s["loss_channel"] = (float(ch_map[selected_c])
                             if selected_c in ch_map and ch_map[selected_c] is not None else None)
        s["mse_channel"]  = (s["loss_channel"] * var_y if s["loss_channel"] is not None else None)
        ols_nm = analysis.get("ols_mse_norm")
        s["loss_ols"] = (float(ols_nm) if ols_nm is not None and np.isfinite(ols_nm) else None)
        s["mse_ols"]  = (float(ols_nm) * var_y if s["loss_ols"] is not None else None)
        nest = analysis.get("nested")
        s["loss_nested"] = (float(nest["mse"] / var_y) if nest and np.isfinite(nest["mse"]) else None)
        s["mse_nested"]  = (float(nest["mse"])          if nest and np.isfinite(nest["mse"]) else None)
        lp = {}; mp = {}
        for rc in analysis.get("rational_candidates", []):
            dk = str(rc.get("max_degree", "?")); nm = rc["mse"] / var_y
            if np.isfinite(nm):
                lp[dk] = float(nm); mp[dk] = float(rc["mse"])
        s["loss_polyrat"] = lp; s["mse_polyrat"] = mp

    def _set_best_formula(self, s, winner_type, rat_degree):
        if winner_type == "rational":
            key = str(rat_degree) if rat_degree is not None else next(iter(s["formula_polyrat"]), None)
            s["best_formula"] = s["formula_polyrat"].get(key) or s["best_formula"]
        elif winner_type == "nested":
            s["best_formula"] = s["formula_nested"] or s["best_formula"]
        elif winner_type == "ols":
            s["best_formula"] = s["formula_ols"] or s["best_formula"]
        else:
            s["best_formula"] = s.get("formula_channel") or s["best_formula"]



# ══════════════════════════════════════════════════════════════════════════════
#  LOG HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _extract_best_formula_from_log(log_path: str) -> str:
    """Return the formula with the smallest logged loss."""
    if not os.path.exists(log_path):
        return "N/A"
    best_loss = np.inf; best_formula = "N/A"
    current_formula = None; current_loss = None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- Idx="):
                    try:
                        current_loss = float(line.split("Loss=")[1].split()[0].rstrip("---").strip())
                    except Exception:
                        current_loss = None
                    current_formula = None
                elif "<<< SELECTED" in line and current_loss is not None:
                    try:
                        formula_part = line.split("Formula[")[1].split("]: ", 1)[1]
                        formula_part = formula_part.replace(" <<< SELECTED", "").strip()
                        current_formula = formula_part
                    except Exception:
                        pass
                    if current_loss is not None and current_loss < best_loss:
                        best_loss = current_loss; best_formula = current_formula or "N/A"
    except Exception:
        pass
    return best_formula


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER  — entry point for each parallel process
# ══════════════════════════════════════════════════════════════════════════════

# ── Helper: smart-parallel dispatcher ────────────────────────────────────────

def _run_smart_parallel(method_cfg, target, run_id, *,
                        _y_predenoised, _denoise_info, _max_iters,
                        _X_preloaded, _y_preloaded) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    method_id      = method_cfg["id"]
    _n_streams     = len(_CfgDragon.SPAR_OP_GROUPS)
    _iter_cap      = _max_iters if _max_iters is not None else _CfgDragon.N_ITERATIONS
    _stream_budget = max(1, _iter_cap // _n_streams)

    # Preprocess once for all smart-parallel streams.
    shared_X_sel, shared_y, shared_feature_names, shared_feat_scores, shared_seed = _prepare_data(
        method_cfg, target, run_id, _CfgExp.INIT_STRATEGIES[run_id],
        _y_predenoised=_y_predenoised, _denoise_info=_denoise_info,
        _X_preloaded=_X_preloaded, _y_preloaded=_y_preloaded)

    stream_results = []
    with ThreadPoolExecutor(max_workers=_n_streams) as ex:
        futs = {}
        for stream_id, ops in _CfgDragon.SPAR_OP_GROUPS.items():
            sub_cfg = {**method_cfg, "operators": list(ops),
                       "_in_smart_stream": True, "_stream_id": stream_id}
            futs[ex.submit(dragon_worker, sub_cfg, target, run_id,
                           _y_predenoised=_y_predenoised, _denoise_info=_denoise_info,
                           _max_iters=_stream_budget,
                           _X_preprocessed=shared_X_sel, _y_preprocessed=shared_y,
                           _feature_names_preloaded=shared_feature_names,
                           _feat_scores_preloaded=shared_feat_scores)] = stream_id
        for f in as_completed(futs):
            try:
                stream_results.append(f.result())
            except Exception:
                print(f"[spar/{target}/run{run_id}] stream {futs[f]} ERROR:")
                traceback.print_exc()

    if not stream_results:
        return {
            "target": target, "method": method_id, "run_id": run_id,
            "strategy": _CfgExp.INIT_STRATEGIES[run_id],
            "loss": float(np.inf), "r2": 0.0,
            "formula": "ERROR: all smart-parallel streams failed",
            "time_s": 0.0, "log_path": "", "description": method_cfg["description"],
            "search_space_ops": list(method_cfg.get("operators", [])),
        }

    best = min(stream_results, key=lambda r: r.get("loss", float("inf")))
    best.update(method=method_id, description=method_cfg["description"],
                search_space_ops=list(method_cfg.get("operators", [])))
    per_stream_T = {r.get("_stream_id", "?"): int(r["actualT"])
                    for r in stream_results if r.get("actualT") is not None}
    best["actualT_per_stream"] = per_stream_T
    best["actualT_winner"]     = best.get("actualT")
    if per_stream_T:
        best["actualT"] = int(sum(per_stream_T.values()))
    best["smart_parallel_streams"] = [
        {"stream": r.get("_stream_id", "?"), "ops": r.get("search_space_ops", []),
         "loss": r.get("loss"), "formula": r.get("formula"),
         "time_s": r.get("time_s"), "actualT": r.get("actualT")}
        for r in stream_results
    ]

    if method_cfg.get("boosted"):
        try:
            preds, sids, y_ref, ref_order = [], [], None, None
            for r in stream_results:
                p  = r.get("_best_pred_np"); yy = r.get("_best_y_np")
                if p is None or yy is None: continue
                p  = np.asarray(p,  dtype=np.float64).ravel()
                yy = np.asarray(yy, dtype=np.float64).ravel()
                if p.shape != yy.shape or p.size < 4: continue
                if not np.all(np.isfinite(p)): continue
                if y_ref is None:
                    y_ref = yy.copy(); ref_order = np.argsort(y_ref, kind="stable")
                else:
                    if yy.size != y_ref.size: continue
                    if not np.allclose(np.sort(yy), np.sort(y_ref), rtol=1e-8, atol=1e-12): continue
                p_aligned = np.empty_like(p)
                p_aligned[ref_order] = p[np.argsort(yy, kind="stable")]
                preds.append(p_aligned); sids.append(r.get("_stream_id", "?"))
            if len(preds) >= 2:
                P = np.column_stack(preds).astype(np.float64)
                (mse_b, sel_c_b, w_b, ch_loss_b,
                 _lr_b, nested_b, rational_b, _vidx_b, ana_b) = OLSPostProcessor().evaluate(
                    torch.tensor(P), torch.tensor(y_ref).reshape(-1, 1), "full")
                if np.isfinite(mse_b) and mse_b < best.get("loss", float("inf")):
                    if rational_b is not None:
                        wtype = "boosted-polyrat"
                        f_str = (f"poly-rational(deg≤{rational_b.get('max_degree','?')}) "
                                 f"over {len(sids)} streams")
                    elif nested_b is not None:
                        wtype = "boosted-nested"; f_str = f"nested-OLS over {len(sids)} streams"
                    elif w_b is not None and np.any(np.asarray(w_b) != 0):
                        wtype = "boosted-ols"
                        wts  = np.asarray(w_b).ravel()
                        bias = float(ana_b.get("ols_bias", 0.0)) if ana_b else 0.0
                        terms = [f"{float(w):+.4g}·ŷ_{sid}"
                                 for sid, w in zip(sids, wts) if abs(float(w)) > 1e-8]
                        if abs(bias) > 1e-8: terms.append(f"{bias:+.4g}")
                        f_str = " ".join(terms) if terms else "(boosted-OLS)"
                    else:
                        wtype = "boosted-channel"; f_str = f"best ŷ_{sids[sel_c_b]} (boosted)"
                    best.update(loss=float(mse_b),
                                r2=float(1.0 - mse_b) if mse_b <= 1.0 else 0.0,
                                formula="boosted: " + f_str, winner_type=wtype,
                                alignment_loss=float(ch_loss_b) if ch_loss_b is not None else float(mse_b),
                                boosted_streams=list(sids))
        except Exception as _e:
            print(f"[{method_id}/{target}/run{run_id}] boost FAILED: {_e}")

    for r in stream_results:
        r.pop("_best_pred_np", None); r.pop("_best_y_np", None)
    best.pop("_best_pred_np", None); best.pop("_best_y_np", None)
    return best


# ── Helper: data preparation ──────────────────────────────────────────────────

def _prepare_data(method_cfg, target, run_id, strategy, *,
                  _y_predenoised, _denoise_info,
                  _X_preprocessed=None, _y_preprocessed=None,
                  _feature_names_preloaded=None, _feat_scores_preloaded=None,
                  _X_preloaded=None, _y_preloaded=None):
    method_id = method_cfg["id"]
    seed = _CfgExp.RANDOM_SEED + run_id
    np.random.seed(seed); torch.manual_seed(seed)

    if (_X_preprocessed is not None and _y_preprocessed is not None
            and _feature_names_preloaded is not None
            and _feat_scores_preloaded is not None):
        return (_X_preprocessed,
                _y_preprocessed.copy(),
                _feature_names_preloaded,
                _feat_scores_preloaded,
                seed)

    if _X_preloaded is not None and _y_preloaded is not None:
        X_df = _X_preloaded; y = _y_preloaded.copy()
    else:
        X_df, y = _dataset_loader.load(target, run_id=run_id, strategy=strategy)

    if _y_predenoised is not None:
        y = _y_predenoised

    pre_dm = method_cfg.get("pre_denoise_method")
    pre_denoise = bool(pre_dm and _y_predenoised is None)
    pipeline = PreprocessingPipeline(
        add_noise=method_cfg.get("add_noise", False),
        pre_denoise=pre_denoise,
        var_aug=method_cfg.get("var_aug", True),
        run_id=run_id,
    )
    X_sel, y, info = pipeline.transform(X_df, y)
    if pre_denoise and "gpr_denoise" in info:
        print(f"[{method_id}/{target}/run{run_id}] pre_denoise({pre_dm}): {info['gpr_denoise']}")

    if method_cfg.get("denoise_method") == "stoch_sub":
        pass  # handled in _setup_searcher

    feature_names = info["feature_names"]
    feat_scores = info["feat_scores"]
    return X_sel, y, feature_names, feat_scores, seed


# ── Helper: searcher setup ────────────────────────────────────────────────────

def _setup_searcher(method_cfg, X_sel, y, feature_names, feat_scores, log_path, seed, device):
    from torch.utils.data import DataLoader as _DL
    method_id = method_cfg["id"]

    loader = _DL(RegressionDataset(X_sel, y.to_frame()), batch_size=32, shuffle=True)
    all_combos = CombinationBuilder().build(feature_names, feat_scores)
    search_space, dag = SearchSpaceBuilder(
        feature_names, feat_scores, method_cfg["operators"]).build(all_combos)

    subsample_ratio = 1.0; X_np = None; y_np = None
    if method_cfg.get("denoise_method") == "stoch_sub":
        subsample_ratio = float(method_cfg.get("subsample_ratio", 0.5))
        X_np = X_sel.values.astype(np.float32)
        y_np = y.values.astype(np.float32).ravel()

    sample_weights = None
    if method_cfg.get("mc_dropout", False):
        try:
            kw = dict(n_forward=method_cfg.get("mc_dropout_n_forward", 50),
                      dropout_p=method_cfg.get("mc_dropout_p", 0.15),
                      n_epochs=method_cfg.get("mc_dropout_epochs", 300),
                      hidden=method_cfg.get("mc_dropout_hidden", 64),
                      random_seed=seed)
            sample_weights = MCDropoutWeighter(**kw).compute_weights(X_sel, y)
            print(f"[{method_id}] MC-Dropout: min={sample_weights.min():.3f} max={sample_weights.max():.3f}")
        except Exception as e:
            print(f"[{method_id}] MC-Dropout FAILED ({e})")

    searcher = DragonSearcher(
        search_space, loader, device, X_sel.shape[1], feature_names, log_path,
        loss_mode=method_cfg.get("loss_mode", "full"),
        optimize_constants=method_cfg.get("optimize_constants", False),
        subsample_ratio=subsample_ratio, X_np=X_np, y_np=y_np, sample_weights=sample_weights,
    )

    seed_models = None
    strategy = _CfgExp.INIT_STRATEGIES[method_cfg.get("_run_id", 0)] if "_run_id" in method_cfg else None
    return searcher, dag, search_space, seed_models


# ── Helper: search execution ──────────────────────────────────────────────────

def _run_search(method_cfg, search_space, dag, searcher, save_dir, seed_models, _max_iters):
    parallel_N = method_cfg.get("parallel_N", 1)
    os.makedirs(save_dir, exist_ok=True)

    def _make_sa(T, clean, extra=None):
        kw = dict(search_space=search_space, evaluation=searcher,
                  T=T, K=_CfgDragon.K_INIT, N=parallel_N, E=1000,
                  save_dir=save_dir, clean_all=clean, verbose=True,
                  loss_threshold=_CfgDragon.LOSS_THRESHOLD, **(extra or {}))
        if clean and seed_models is not None:
            kw["models"] = seed_models
        return Mutant_UCB(**kw)

    _iter_cap = _max_iters if _max_iters is not None else _CfgDragon.N_ITERATIONS

    if not method_cfg.get("curriculum", False):
        sa = _make_sa(_iter_cap, clean=True)
        sa.run()
        return sa.min_loss

    global_best = np.inf; total_iters = 0
    budget_mode = method_cfg.get("budget_mode", "default")
    for complexity in range(1, _CfgDragon.MAX_COMPLEXITY + 1):
        remaining = _iter_cap - total_iters
        if remaining <= 0: break
        dag.complexity = complexity
        clean = (complexity == 1)
        _csv  = os.path.join(save_dir, "computation_file.csv")
        pop   = len(pd.read_csv(_csv)) if not clean and os.path.exists(_csv) else 0
        extra = {} if clean or budget_mode == "complexity" else {"pop_path": save_dir}
        T     = min(pop + complexity * _CfgDragon.T_PER_LEVEL, remaining)
        sa    = _make_sa(T, clean, extra)
        sa.run()
        try:
            total_iters = len(pd.read_csv(_csv))
        except Exception:
            total_iters += T
        global_best = min(global_best, sa.min_loss)
        if global_best <= _CfgDragon.LOSS_THRESHOLD: break
    return global_best


# ── Helper: collect landscape ─────────────────────────────────────────────────

def _collect_landscape(save_dir):
    actual_T = None; loss_history = []; landscape_svg = None
    comp_csv = os.path.join(save_dir, "computation_file.csv")
    if not os.path.exists(comp_csv):
        return actual_T, loss_history, landscape_svg
    try:
        df = pd.read_csv(comp_csv); actual_T = len(df)
        if "Loss" in df.columns:
            losses  = df["Loss"].replace([np.inf, -np.inf], np.nan).dropna().values
            idx_pts = np.linspace(0, len(losses) - 1, min(len(losses), 80), dtype=int)
            loss_history = [[int(i), float(losses[i])] for i in idx_pts]
    except Exception:
        pass
    try:
        from stats import _make_dragon_landscape_svg
        landscape_svg = _make_dragon_landscape_svg(comp_csv)
    except Exception:
        pass
    return actual_T, loss_history, landscape_svg


# ── Helper: build result dict ─────────────────────────────────────────────────

def _build_result(target, method_cfg, run_id, strategy, best_loss, best_formula,
                  log_path, loss_state, actual_T, loss_history, landscape_svg, elapsed) -> dict:
    result = {
        "target": target, "method": method_cfg["id"], "run_id": run_id,
        "strategy": strategy, "_stream_id": method_cfg.get("_stream_id"),
        "loss": float(best_loss), "r2": float(1.0 - best_loss) if best_loss <= 1.0 else 0.0,
        "formula": str(best_formula), "time_s": elapsed, "log_path": log_path,
        "description": method_cfg["description"],
        "search_space_ops": list(method_cfg.get("operators", [])),
        "loss_mode":      method_cfg.get("loss_mode", "full"),
        "winner_type":    loss_state.get("winner_type", "channel"),
        "corr_value":     float(loss_state.get("corr_value", 0.0)),
        "alignment_loss": float(loss_state.get("alignment_loss", 1.0)),
        "rat_degree":     loss_state.get("rat_degree"),
        "loss_history":   loss_history,
        "actualT":        int(actual_T) if actual_T is not None else None,
        "all_channel_formulas": loss_state.get("all_channel_formulas", []),
        "ops_used":         loss_state.get("ops_used"),
        "has_const":        loss_state.get("has_const"),
        "dag_size":         loss_state.get("dag_size"),
        "dag_text":         loss_state.get("dag_text"),
        "dag_svg":          loss_state.get("dag_svg"),
        "dag_data":         loss_state.get("dag_data", []),
        "formula_channel":  loss_state.get("formula_channel"),
        "formula_ols":      loss_state.get("formula_ols"),
        "formula_nested":   loss_state.get("formula_nested"),
        "formula_polyrat":  loss_state.get("formula_polyrat", {}),
        "loss_channel":     loss_state.get("loss_channel"),
        "loss_ols":         loss_state.get("loss_ols"),
        "loss_nested":      loss_state.get("loss_nested"),
        "loss_polyrat":     loss_state.get("loss_polyrat", {}),
        "mse_channel":      loss_state.get("mse_channel"),
        "mse_ols":          loss_state.get("mse_ols"),
        "mse_nested":       loss_state.get("mse_nested"),
        "mse_polyrat":      loss_state.get("mse_polyrat", {}),
        "landscape_svg":    landscape_svg,
    }
    if method_cfg.get("_in_smart_stream"):
        result["_best_pred_np"] = loss_state.get("best_pred_np")
        result["_best_y_np"]    = loss_state.get("best_y_np")
    return result


# ── Main worker ───────────────────────────────────────────────────────────────

def dragon_worker(method_cfg: dict, target: str, run_id: int,
                  *, _y_predenoised=None, _denoise_info=None,
                  _max_iters: int = None,
                  _X_preprocessed=None, _y_preprocessed=None,
                  _feature_names_preloaded=None, _feat_scores_preloaded=None,
                  _X_preloaded=None, _y_preloaded=None) -> dict:
    """Entry point for each parallel DRAGON process.

    method_cfg : one element of DRAGON_METHODS
    run_id     : 0..N_RUNS-1, also indexes Experiment.INIT_STRATEGIES
    """
    method_id = method_cfg["id"]

    if method_cfg.get("smart_parallel") and not method_cfg.get("_in_smart_stream"):
        return _run_smart_parallel(
            method_cfg, target, run_id,
            _y_predenoised=_y_predenoised, _denoise_info=_denoise_info,
            _max_iters=_max_iters, _X_preloaded=_X_preloaded, _y_preloaded=_y_preloaded)

    strategy = _CfgExp.INIT_STRATEGIES[run_id]
    run_dir  = (os.path.join(_CfgPaths.OUTPUT_DIR, target, method_id,
                             f"run_{run_id}", f"stream_{method_cfg['_stream_id']}")
                if method_cfg.get("_in_smart_stream")
                else os.path.join(_CfgPaths.OUTPUT_DIR, target, method_id, f"run_{run_id}"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"{target}_{method_id}{_CfgPaths.LOG_SUFFIX}")
    save_dir = os.path.join(run_dir, "save")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.time(); best_loss = np.inf; best_formula = "N/A"; loss_state = {}

    try:
        X_sel, y, feature_names, feat_scores, seed = _prepare_data(
            method_cfg, target, run_id, strategy,
            _y_predenoised=_y_predenoised, _denoise_info=_denoise_info,
            _X_preprocessed=_X_preprocessed, _y_preprocessed=_y_preprocessed,
            _feature_names_preloaded=_feature_names_preloaded,
            _feat_scores_preloaded=_feat_scores_preloaded,
            _X_preloaded=_X_preloaded, _y_preloaded=_y_preloaded)

        searcher, dag, search_space, _ = _setup_searcher(
            method_cfg, X_sel, y, feature_names, feat_scores, log_path, seed, device)
        loss_state = searcher.state

        seed_models = None
        if strategy == "diverse":
            seed_models = _build_random_seed_dags(
                feature_names, method_cfg["operators"], seed=run_id)

        best_loss = _run_search(method_cfg, search_space, dag, searcher,
                                save_dir, seed_models, _max_iters)

        best_formula = loss_state.get("best_formula", "N/A")
        if best_formula == "N/A":
            best_formula = _extract_best_formula_from_log(log_path)

    except Exception:
        tb = traceback.format_exc()
        best_formula = f"ERROR: {tb[:200]}"
        print(f"[{method_id}/{target}/run{run_id}] ERROR:\n{tb}")

    actual_T, loss_history, landscape_svg = _collect_landscape(save_dir)
    return _build_result(target, method_cfg, run_id, strategy, best_loss, best_formula,
                         log_path, loss_state, actual_T, loss_history, landscape_svg,
                         time.time() - t_start)
