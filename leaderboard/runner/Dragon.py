from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from runner.Worker import _run_smart_parallel, dragon_worker
from dataprocessing.Preprocessing import PreprocessingPipeline
from runner.Ols import OLSPostProcessor
from pipeline.DragonOrchestrator import SearchLoss
from helpers.stats import DAGInspector
from pathlib import Path
from Config import Loss as _LossCfg



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
from dragon.search_space.dag_encoding import AdjMatrix
from dragon.utils.plot_functions import graph_to_all_formulas
import threading as _threading
import dragon.search_algorithm.search_algorithm as _dragon_sa
from itertools import combinations

# Thread-safe SIGALRM patch -----------------------------------------------------
_original_timed_evaluation = _dragon_sa.timed_evaluation

def _thread_safe_timed_evaluation(x, idx, max_duration, evaluation):
    if _threading.current_thread() is not _threading.main_thread():
        return evaluation(x, idx)
    return _original_timed_evaluation(x, idx, max_duration, evaluation)

_dragon_sa.timed_evaluation = _thread_safe_timed_evaluation






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
        search_loss:        str        = "channel",
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
        self.search_loss        = search_loss
        self.optimize_constants = optimize_constants
        self.subsample_ratio    = subsample_ratio
        self.X_np               = X_np
        self.y_np               = y_np
        self.sample_weights     = sample_weights

        self._ols = OLSPostProcessor(search_loss=self.search_loss)

        self._comp_penalty = float(getattr(_LossCfg, "COMPOSITION_PENALTY", 0.0) or 0.0)
        self._comp_base    = float(getattr(_LossCfg, "COMPOSITION_BASE", 2.0) or 2.0)
        self._comp_groups  = dict(getattr(_LossCfg, "COMPOSITION_GROUPS", {}) or {})

        self.state = {
            "best_loss":       np.inf,
            "winner_type":     "channel",
            "corr_value":      0.0,
            "alignment_loss":  1.0,
            "search_loss":     self.search_loss,
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

        pred_all, true_all = self.forward(model, self.train_loader, idx, self.device)
        (baseline_loss, selected_c, ols_weights, alignment_loss,
         _lr, nested, rational, valid_idx_global, analysis) = self.evaluate(
            pred_all, true_all)

        # Defensive guard: if no channel produced a finite score, reject candidate.
        ch_res = analysis.get("channel_results", []) if isinstance(analysis, dict) else []
        has_valid_channel = any(v is not None and np.isfinite(v) for _, v in ch_res)
        if not has_valid_channel:
            selected_c = None

        y_np, y_pred, winner_type, rat_degree = self.resolve_prediction(
            pred_all, true_all, selected_c, ols_weights, nested, rational)

        search_loss = SearchLoss.score(self.search_loss, y_pred, y_np)
        search_loss = self.apply_weighted_loss(search_loss, y_np, y_pred)
        comp_pen = self._composition_penalty(model)
        if comp_pen and np.isfinite(search_loss):
            search_loss = float(search_loss) + comp_pen
        corr_val = float(SearchLoss.correl(
            torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred,
            true_all))

        if search_loss < self.state["best_loss"]:
            self._update_state(search_loss, winner_type, corr_val, alignment_loss, rat_degree,
                               y_pred, y_np, selected_c, ols_weights, nested, rational,
                               valid_idx_global, analysis, pred_all, model, idx)

        print(f"Idx={idx}, Loss = {search_loss:.10f}")
        model.set_prediction_to_save("prediction", pd.DataFrame({"pred": y_pred, "true": y_np}))
        return float(search_loss), model

    # ── Evaluation core (ex-DragonEvaluator) ─────────────────────────────────

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
        if selected_c is None:
            return y_np, None, "channel", None
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
        if self.search_loss == "corr":
            A = np.column_stack([ch_raw, np.ones_like(ch_raw)])
            c, *_ = np.linalg.lstsq(A, y_np, rcond=None)
            return y_np, float(c[0]) * ch_raw + float(c[1]), "channel", None
        return y_np, ch_raw, "channel", None

    def apply_weighted_loss(self, mse, y_np, y_pred):
        if mse is None or not np.isfinite(mse):
            return mse
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

    def _composition_penalty(self, model):
        """Exponential penalty for nesting functions of the same family.

        Walks the candidate DAG and, for every function node (sin/cos/exp/ln),
        computes the length of the longest chain of same-family function nodes
        that ends at it (its nesting depth). Each nested level adds an
        exponentially growing cost ``base**(depth-1) - 1`` so that deeper
        same-type compositions (e.g. ``sin(cos(sin(x)))``) are penalized far
        more than shallow ones. Returns ``0.0`` when disabled.
        """
        if self._comp_penalty <= 0.0 or not self._comp_groups:
            return 0.0
        try:
            adj   = model.dag.matrix
            nodes = model.dag.operations
        except Exception:
            return 0.0
        n = len(nodes)
        if n == 0:
            return 0.0

        # Family label per node (None when the node is not a tracked function).
        fam = []
        for nd in nodes:
            cls = getattr(nd, "name", None)
            cname = getattr(cls, "__name__", None)
            if cname is None:
                cname = str(cls).split(".")[-1].strip("'>\" ")
            fam.append(self._comp_groups.get(cname))

        # depth[v] = longest chain of same-family function nodes ending at v.
        # Fixed-point relaxation makes this robust to node ordering (small DAG).
        depth = [1 if fam[i] else 0 for i in range(n)]
        for _ in range(n):
            changed = False
            for j in range(n):
                if not fam[j]:
                    continue
                best = 0
                for i in range(n):
                    if fam[i] == fam[j] and adj[i, j] and depth[i] > best:
                        best = depth[i]
                if best + 1 > depth[j]:
                    depth[j] = best + 1
                    changed = True
            if not changed:
                break

        base = self._comp_base
        raw = 0.0
        for j in range(n):
            if fam[j] and depth[j] >= 2:
                raw += base ** (depth[j] - 1) - 1.0
        return self._comp_penalty * raw

    def _maybe_optimize_constants(self, model):
        if not self.optimize_constants:
            return

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return

        model  = model.float()
        opt    = torch.optim.Adam(params, lr=0.001)
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
            s["best_pred_all_np"] = (pred_all.detach().cpu().numpy()
                                      if isinstance(pred_all, torch.Tensor)
                                      else np.asarray(pred_all, dtype=np.float64))
        except Exception:
            pass
        try:
            adj      = model.dag.matrix
            nodes    = model.dag.operations
            formulas = graph_to_all_formulas(adj, self.feature_names, nodes)
            s["best_formula"] = str(formulas[selected_c]) if formulas else "N/A"
            s["best_formulas"] = [str(f) for f in formulas] if formulas else []

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
                         f"Winner={winner_type.upper()}  search_loss={self.search_loss}\n")
                self._ols.print_analysis(analysis, formulas, valid_idx_global, selected_c, f=lf)
                lf.write("\n")
            print(f"\n  >> NEW BEST loss={mse:.10f} [{winner_type.upper()}]  Idx={idx}")
            self._ols.print_analysis(analysis, formulas, valid_idx_global, selected_c)
        except Exception as e:
            print(f"  >> Could not extract formula: {e}")

    # ── Formula helpers ─────────────────────────────────────────────────────── #todo: do we keep them in the searcher or move them to a separate helper class ?

    def _channel_formula(self, formulas, selected_c, pred_all, y_np) -> str | None:
        raw = str(formulas[selected_c]) if formulas and selected_c < len(formulas) else None
        if raw is None:
            return None
        if self.search_loss != "corr":
            return raw
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

    def finalize_ols_postprocessing(self):
        if self.loss_mode != "channel" or self.state.get("best_pred_all_np") is None:
            return
        try:
            P = np.asarray(self.state["best_pred_all_np"], dtype=np.float32)
            y = np.asarray(self.state["best_y_np"], dtype=np.float32).reshape(-1, 1)
            P_t = torch.tensor(P, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            (_, selected_c, _, _, _lr, nested, rational, valid_idx_global, analysis) = \
                self._ols.evaluate(P_t, y_t, "full")
            formulas = self.state.get("best_formulas", [])
            self.state["formula_ols"] = self._ols_formula(formulas, analysis)
            self.state["formula_nested"] = (
                self._ols.format_nested(analysis["nested"], formulas)
                if analysis.get("nested") and formulas else None)
            self.state["formula_polyrat"] = self._polyrat_formulas(formulas, valid_idx_global, analysis)
            self._update_losses(self.state, analysis, selected_c)
        except Exception:
            pass



def run_dragon_method(method_cfg, target, run_id, *, _max_iters=None):

    X_sel, y, feature_names, feat_scores = PreprocessingPipeline.prepare_data(
        method_cfg, target, run_id
    )

    if method_cfg.get("parallel_mode") == "smart" or method_cfg.get("smart_parallel"):
        return _run_smart_parallel(
            method_cfg, target, run_id,
            _max_iters=_max_iters,
            _X_preprocessed=X_sel, _y_preprocessed=y,
            _feature_names_preloaded=feature_names,
            _feat_scores_preloaded=feat_scores)

    return dragon_worker(
        method_cfg, target, run_id,
        _X_preprocessed=X_sel, _y_preprocessed=y,
        _feature_names_preloaded=feature_names,
        _feat_scores_preloaded=feat_scores)