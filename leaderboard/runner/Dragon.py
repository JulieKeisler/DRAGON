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
        constants_optimizer: str       = "dichotomy",
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
        self.constants_optimizer = (constants_optimizer or "dichotomy").lower()
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

        self._optimize_constants_search(model)
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

        if search_loss < self.state["best_loss"]:
            self._last_comp_pen = float(comp_pen or 0.0)
            self._update_state(search_loss, winner_type, alignment_loss, rat_degree,
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

    def _const_fit_batch(self, max_points=512):
        """Cache a small (X, y) tensor batch used as the cheap dichotomy objective."""
        if getattr(self, "_cfb", None) is not None:
            return self._cfb
        if self.X_np is not None and self.y_np is not None:
            n = len(self.y_np)
            m = min(n, max_points)
            rng = np.random.default_rng(0)
            sub = rng.choice(n, m, replace=False) if n > m else np.arange(n)
            Xb = torch.tensor(self.X_np[sub], dtype=torch.float32, device=self.device)
            yb = torch.tensor(self.y_np[sub], dtype=torch.float32,
                              device=self.device).reshape(-1, 1)
        else:
            xs, ys, got = [], [], 0
            for Xb_, yb_ in self.train_loader:
                xs.append(Xb_); ys.append(yb_); got += len(Xb_)
                if got >= max_points:
                    break
            Xb = torch.cat(xs).to(self.device).float()
            yb = torch.cat(ys).to(self.device).float().reshape(-1, 1)
        self._cfb = (Xb, yb)
        return self._cfb

    def _probe_loss(self, model, Xb, yb):
        """Scalar objective for the dichotomy, aligned with ``self.search_loss``.

        Mirrors the channel-selection metric of the real pipeline: for every
        output channel the per-channel prediction is scored with
        ``SearchLoss.score(self.search_loss, ...)`` (with an affine alignment per
        channel when the search loss is scale/shift invariant, i.e. ``corr``).
        The per-channel scores are averaged so that every constant gets a signal,
        even when it only affects a non-winning channel. Constant / non-finite
        channels are penalised with the worst normalised value (1.0).
        """
        with torch.no_grad():
            pred = model(Xb)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        P = pred.detach().cpu().numpy()
        y = yb.detach().cpu().numpy().ravel()
        n_ch = P.shape[1]
        affine = (self.search_loss == "corr")
        total = 0.0
        for c in range(n_ch):
            raw = P[:, c]
            if not np.all(np.isfinite(raw)) or np.std(raw) < 1e-12:
                total += 1.0
                continue
            if affine:
                A = np.column_stack([raw, np.ones_like(raw)])
                try:
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                    yp = coef[0] * raw + coef[1]
                except Exception:
                    yp = raw
            else:
                yp = raw
            s = SearchLoss.score(self.search_loss, yp, y)
            total += s if np.isfinite(s) else 1e6
        return total / max(n_ch, 1)

    def _scalefree_line_search(self, f, current, max_mag=1e12, iters=40, flat_rtol=1e-6):
        """Scale-free 1-D minimizer for a scalar constant.

        A coarse logarithmic grid (both signs + zero + the current value,
        ``1e-3 … max_mag``) locates the correct order of magnitude, then a
        golden-section refinement runs on the bracket formed by the two grid
        points neighbouring the grid minimum. Handles constants from ~0 up to
        ~1e12. When the objective is (numerically) invariant across the grid
        (e.g. a pure-scale constant under a scale-invariant search loss), the
        current value is kept unchanged to avoid injecting meaningless constants.
        """
        cands = [0.0, float(current)]
        m = 1e-3
        while m <= max_mag * (1.0 + 1e-9):
            cands.append(m); cands.append(-m)
            m *= 10.0
        cands = sorted(set(cands))
        vals = [f(x) for x in cands]
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin <= flat_rtol * (abs(vmin) + 1e-12):
            return float(current)  # objective invariant → keep current (clean)
        i = int(np.argmin(vals))
        lo = cands[max(0, i - 1)]
        hi = cands[min(len(cands) - 1, i + 1)]
        best_x, best_f = cands[i], vals[i]
        if hi > lo:
            gr = (5.0 ** 0.5 - 1.0) / 2.0
            c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
            fc = f(c); fd = f(d)
            for _ in range(iters):
                if fc < fd:
                    hi, d, fd = d, c, fc
                    c = hi - gr * (hi - lo); fc = f(c)
                else:
                    lo, c, fc = c, d, fd
                    d = lo + gr * (hi - lo); fd = f(d)
            mid = (lo + hi) / 2.0
            fm = f(mid)
            if fm <= best_f:
                best_x, best_f = mid, fm
        return best_x

    def _dichotomy_constants(self, model, iters=40, max_mag=1e12, sweeps=1):
        """Cheap approximation of scalar learnable constants via a scale-free search.

        For every ExpAffine ``a`` and every ConstantBrick ``value`` we minimize a
        objective aligned with ``self.search_loss`` (see ``_probe_loss``) using a
        scale-free line search (log grid + local golden-section). This is a
        coordinate-descent pass (optionally repeated ``sweeps`` times) that stays
        cheap — a handful of forward passes per constant on a subsample. The
        precise, multi-sweep refinement on the best model is done in
        ``finalize_constants``.
        """
        if not self.optimize_constants:
            return
        targets = []  # (module, attribute)
        for m in model.modules():
            cls = m.__class__.__name__
            if cls == "ExpAffine" and hasattr(m, "a"):
                targets.append((m, "a"))
            elif cls == "ConstantBrick" and hasattr(m, "value"):
                targets.append((m, "value"))
        if not targets:
            return
        Xb, yb = self._const_fit_batch()

        for _ in range(max(1, sweeps)):
            for mod, attr in targets:
                param = getattr(mod, attr)
                cur = float(param.detach().reshape(-1)[0])

                def obj(val, _p=param):
                    old = _p.detach().clone()
                    with torch.no_grad():
                        _p.fill_(float(val))
                        v = self._probe_loss(model, Xb, yb)
                        _p.copy_(old)
                    return v

                best = self._scalefree_line_search(obj, cur, max_mag=max_mag, iters=iters)
                with torch.no_grad():
                    param.fill_(float(best))

    def _optimize_constants_search(self, model):
        """Per-candidate constant optimization during search (optimizer-dependent)."""
        if not self.optimize_constants:
            return
        if self.constants_optimizer == "dichotomy":
            self._dichotomy_constants(model)
        else:
            self._gradient_optimize_constants(model, epochs=100)

    def _make_constant_optimizer(self, params):
        """Build the gradient optimizer selected via ``constants_optimizer``."""
        name = self.constants_optimizer
        lr = 1e-3
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr)
        return torch.optim.Adam(params, lr=lr)

    def _gradient_optimize_constants(self, model, epochs=300):
        """Gradient fit of every learnable constant with the selected optimizer.

        Robust to divergence: the best finite-loss parameter state is snapshotted
        and restored at the end, so NaN/Inf constants can never leak out.
        """
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return
        model  = model.float()
        opt    = self._make_constant_optimizer(params)
        mse_fn = nn.MSELoss()
        best_loss  = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0; n_seen = 0
            diverged = False
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(self.device).float(), yb.to(self.device).float()
                opt.zero_grad()
                pred = model(Xb)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                target = yb.reshape(-1, 1).expand_as(pred)
                loss = mse_fn(pred, target)
                if not torch.isfinite(loss):
                    diverged = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                epoch_loss += loss.item() * Xb.size(0); n_seen += Xb.size(0)
            if diverged:
                break
            epoch_loss /= max(n_seen, 1)
            if not np.isfinite(epoch_loss):
                break
            if epoch_loss < best_loss:
                best_loss  = epoch_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch_loss < 1e-12:
                break
        model.load_state_dict(best_state)
        model.eval()

    def _full_forward(self, model):
        """Forward pass over the full training set (no subsampling)."""
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for Xb, yb in self.train_loader:
                all_pred.append(model(Xb.to(self.device)).detach().cpu())
                all_true.append(yb.to(self.device).detach().cpu())
        return torch.cat(all_pred), torch.cat(all_true)

    def finalize_constants(self):
        """Final precise optimization of the best model's constants, then re-score.

        Runs a proper gradient fit (Adam) on the learnable constants of the best
        model found during search, then recomputes predictions / loss / formula
        fields so the reported results reflect the refined constants.
        """
        if not self.optimize_constants:
            return
        model = self.state.get("best_model")
        if model is None:
            return
        try:
            # Final refinement must be consistent with the chosen optimizer:
            #  - dichotomy: same search-loss-aligned line search, finer (more
            #    golden iterations) and multi-sweep on the best model only;
            #  - adam/adamw: a longer gradient fit of all constants.
            if self.constants_optimizer == "dichotomy":
                self._dichotomy_constants(model, iters=64, sweeps=4)
            else:
                self._gradient_optimize_constants(model, epochs=300)
        except Exception as e:
            print(f"  >> final constant optimization failed: {e}")
            return
        try:
            pred_all, true_all = self._full_forward(model)
            (_baseline, selected_c, ols_weights, _align, _lr,
             nested, rational, valid_idx_global, analysis) = self.evaluate(pred_all, true_all)
            ch_res = analysis.get("channel_results", []) if isinstance(analysis, dict) else []
            if not any(v is not None and np.isfinite(v) for _, v in ch_res):
                selected_c = None
            y_np, y_pred, winner_type, rat_degree = self.resolve_prediction(
                pred_all, true_all, selected_c, ols_weights, nested, rational)
            search_loss = SearchLoss.score(self.search_loss, y_pred, y_np)
            comp_pen = self._composition_penalty(model)
            if comp_pen and np.isfinite(search_loss):
                search_loss = float(search_loss) + comp_pen

            if not np.isfinite(search_loss):
                print("  >> final re-score is non-finite, keeping pre-optimization result")
                return

            s = self.state
            s["best_loss"]        = float(search_loss)
            s["winner_type"]      = winner_type
            s["rat_degree"]       = rat_degree
            s["best_pred_all_np"] = pred_all.detach().cpu().numpy()
            s["best_y_np"]        = np.asarray(y_np, dtype=np.float64).ravel()
            if y_pred is not None:
                s["best_pred_np"] = np.asarray(y_pred, dtype=np.float64).ravel()
            self._compute_formulas_into_state(
                s, model.dag.matrix, model.dag.operations, selected_c, analysis,
                valid_idx_global, pred_all, y_np, winner_type, rat_degree)
            print(f"  >> FINAL constant optimization done, refined loss={search_loss:.10f}")
        except Exception as e:
            print(f"  >> final re-score failed: {e}")

    # ── State update ─────────────────────────────────────────────────────────

    def _update_state(self, mse, winner_type, alignment_loss, rat_degree,
                      y_pred, y_np, selected_c, ols_weights, nested, rational,
                      valid_idx_global, analysis, pred_all, model, idx):
        s = self.state
        s["best_loss"]      = mse
        s["winner_type"]    = winner_type
        s["alignment_loss"] = float(alignment_loss)
        s["rat_degree"]     = rat_degree
        s["best_model"]     = model
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

            ops_str, const_str, dag_size, dag_text, dag_svg, dag_data = DAGInspector.summarize(adj, nodes)
            s.update(ops_used=ops_str, has_const=const_str, dag_size=dag_size,
                     dag_text=dag_text, dag_svg=dag_svg, dag_data=dag_data)
            self._update_losses(s, analysis, selected_c)
            formulas = self._compute_formulas_into_state(
                s, adj, nodes, selected_c, analysis, valid_idx_global,
                pred_all, y_np, winner_type, rat_degree)

            with open(self.log_path, "a") as lf:
                lf.write(f"=== NEW BEST  Idx={idx}  Loss={mse:.10f}  "
                         f"CompPen={getattr(self, '_last_comp_pen', 0.0):.10f}  "
                         f"Winner={winner_type.upper()}  search_loss={self.search_loss}\n")
                self._ols.print_analysis(analysis, formulas, valid_idx_global, selected_c, f=lf)
                lf.write("\n")
            print(f"\n  >> NEW BEST loss={mse:.10f} [{winner_type.upper()}]  "
                  f"CompPen={getattr(self, '_last_comp_pen', 0.0):.10f}  Idx={idx}")
            self._ols.print_analysis(analysis, formulas, valid_idx_global, selected_c)
        except Exception as e:
            print(f"  >> Could not extract formula: {e}")

    def _compute_formulas_into_state(self, s, adj, nodes, selected_c, analysis,
                                     valid_idx_global, pred_all, y_np,
                                     winner_type, rat_degree):
        """Build readable formula strings from the graph (no SymPy) and store them."""
        formulas = graph_to_all_formulas(adj, self.feature_names, nodes, parse_sympy=False)
        _sel = selected_c if (formulas and selected_c is not None and selected_c < len(formulas)) else 0
        s["best_formula"]  = str(formulas[_sel]) if formulas else "N/A"
        s["best_formulas"] = [str(f) for f in formulas] if formulas else []
        s["formula_channel"]      = self._channel_formula(formulas, selected_c, pred_all, y_np)
        s["all_channel_formulas"] = self._all_channel_formulas(formulas, analysis, selected_c)
        s["formula_ols"]          = self._ols_formula(formulas, analysis)
        s["formula_nested"]       = (self._ols.format_nested(analysis["nested"], formulas)
                                     if analysis.get("nested") and formulas else None)
        s["formula_polyrat"]      = self._polyrat_formulas(formulas, valid_idx_global, analysis)
        self._set_best_formula(s, winner_type, rat_degree)
        return formulas

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