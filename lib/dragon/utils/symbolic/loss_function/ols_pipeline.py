"""End-to-end OLS evaluation pipeline for DRAGON symbolic regression.

The ``evaluate`` function takes raw multi-channel predictions and ground truth,
then runs channel selection, sparse OLS, nested link-function OLS,
polynomial-rational OLS, and parsimony-aware model selection to produce
the best scalar loss and corresponding formula metadata.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge

from dragon.utils.symbolic.loss_function.search_loss import SearchLoss, AlignmentLoss
from dragon.utils.symbolic.loss_function.solvers import safe_lstsq, sparse_lstsq
from dragon.utils.symbolic.loss_function.fitting import (
    nested_ols as _nested_ols,
    poly_rational_ols as _poly_rational_ols,
    _LINKS, _UNARIES,
)
from dragon.utils.symbolic.loss_function.selection import select_model


def evaluate(pred_all, true_all, search_loss="corr", loss_kind="mse",
             huber_delta_frac=0.20, loss_mode="full",
             rat_max_degree=2, max_basis_channels=3, max_features=4,
             parsimony_rel_tol=0.02, complexity_max=None):
    """Evaluate a multi-channel prediction through the OLS pipeline.

    Parameters
    ----------
    pred_all : torch.Tensor
        Prediction tensor of shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    true_all : torch.Tensor
        Ground truth tensor.
    search_loss : str
        Loss name (``"corr"``, ``"mse"``, ``"raw_mse"``, ``"mae"``, ``"huber"``).
    loss_kind : str
        ``"mse"`` or ``"huber"`` for alignment scoring.
    huber_delta_frac : float
        Huber delta as fraction of ``sqrt(var_y)``.
    loss_mode : str
        ``"full"``, ``"ols"``, or ``"channel"``.
    rat_max_degree : int
        Max polynomial degree for rational OLS.
    max_basis_channels : int
        Number of top channels for rational basis.
    max_features : int
        Max monomial features in rational model.
    parsimony_rel_tol : float
        Relative tolerance for parsimony-aware selection.
    complexity_max : int or None
        Hard cap on effective parameters.

    Returns
    -------
    tuple
        ``(mse_norm, selected_c, ols_weights, best_channel_loss,
          lr_stub, nested, rational, valid_idx_global, analysis)``
    """
    y = true_all.squeeze().numpy()
    var_y = float(np.var(y))
    if var_y < 1e-30:
        return 1.0, 0, None, 1.0, None, None, None, None, {}
    if pred_all.ndim == 1 or pred_all.shape[-1] == 1:
        return _eval_single(pred_all, y, var_y, search_loss, loss_kind,
                            huber_delta_frac, loss_mode, rat_max_degree,
                            max_basis_channels, max_features,
                            parsimony_rel_tol, complexity_max)
    return _eval_multi(pred_all, y, var_y, search_loss, loss_kind,
                       huber_delta_frac, loss_mode, rat_max_degree,
                       max_basis_channels, max_features,
                       parsimony_rel_tol, complexity_max)


def _eval_single(pred_all, y, var_y, search_loss, loss_kind,
                 huber_delta_frac, loss_mode, rat_max_degree,
                 max_basis_channels, max_features,
                 parsimony_rel_tol, complexity_max):
    ch = pred_all.squeeze().numpy()
    if not np.all(np.isfinite(ch)) or np.std(ch) < 1e-12:
        ana = {"var_y": var_y, "channel_results": [(0, None)], "n_ch": 1,
               "ols_mse_norm": None, "ols_kept_global": None, "ols_bias": None,
               "ols_weights": None, "nested": None, "rational_candidates": []}
        return np.inf, None, None, np.inf, None, None, None, np.array([0]), ana
    if loss_mode == "channel":
        l = SearchLoss.score(search_loss, ch, y)
        return l, 0, None, l, None, None, None, np.array([0]), {}

    w, b, pred = safe_lstsq(ch.reshape(-1, 1), y)
    if w is None:
        l = SearchLoss.score(search_loss, ch, y)
        return l, 0, None, l, None, None, None, np.array([0]), {}

    if search_loss == "corr":
        mse_lin = AlignmentLoss.channel_score(
            search_loss, loss_kind, pred, y, var_y, huber_delta_frac)
    else:
        mse_lin = SearchLoss.score(search_loss, ch, y)

    if loss_mode == "ols":
        ana = {"var_y": var_y, "channel_results": [(0, mse_lin)], "n_ch": 1,
               "ols_mse_norm": mse_lin, "ols_kept_global": None,
               "ols_bias": float(b), "ols_weights": None,
               "nested": None, "rational_candidates": []}
        return mse_lin, 0, None, mse_lin, None, None, None, np.array([0]), ana

    ch_mat = ch.reshape(-1, 1)
    nested = _nested_ols(ch_mat, y, mse_floor=mse_lin * var_y,
                         search_loss=search_loss, loss_kind=loss_kind,
                         huber_delta_frac=huber_delta_frac)
    floor = min(mse_lin, (nested["mse"] / var_y) if nested else mse_lin) * var_y
    rat_cands = []
    rational = None
    best_rat_mse = np.inf
    for deg in range(1, rat_max_degree + 1):
        rc = _poly_rational_ols(ch_mat, y, floor, max_degree=deg,
                                search_loss=search_loss,
                                max_basis_channels=max_basis_channels,
                                max_features=max_features)
        if rc is not None:
            rat_cands.append(rc)
            if rational is None or rc["mse"] < best_rat_mse:
                rational = rc
                best_rat_mse = rc["mse"]

    cands = [{"kind": "linear", "mse_norm": mse_lin}]
    if nested:
        cands.append({"kind": "nested", "mse_norm": nested["mse"] / var_y, "nested": nested})
    if rational:
        cands.append({"kind": "rational", "mse_norm": rational["mse"] / var_y, "rational": rational})
    winner = select_model(cands, parsimony_rel_tol, complexity_max)

    ana = {"var_y": var_y, "channel_results": [(0, mse_lin)], "n_ch": 1,
           "ols_mse_norm": mse_lin, "ols_kept_global": None, "ols_bias": float(b),
           "ols_weights": None, "nested": nested, "rational_candidates": rat_cands}
    if winner["kind"] == "rational":
        return rational["mse"] / var_y, 0, None, mse_lin, None, None, rational, np.array([0]), ana
    if winner["kind"] == "nested":
        return nested["mse"] / var_y, 0, None, mse_lin, None, nested, None, np.array([0]), ana
    return mse_lin, 0, None, mse_lin, None, None, None, np.array([0]), ana


def _eval_multi(pred_all, y, var_y, search_loss, loss_kind,
                huber_delta_frac, loss_mode, rat_max_degree,
                max_basis_channels, max_features,
                parsimony_rel_tol, complexity_max):
    P = pred_all.numpy()
    n_ch = P.shape[1]

    if n_ch == 0:
        ana = {"var_y": var_y, "channel_results": [], "n_ch": 0,
               "ols_mse_norm": None, "ols_kept_global": None, "ols_bias": None,
               "ols_weights": None, "nested": None, "rational_candidates": []}
        return np.inf, 0, None, np.inf, None, None, None, None, ana

    best_ch_loss = np.inf
    selected_c = None
    ch_res = []
    for c in range(n_ch):
        if not np.all(np.isfinite(P[:, c])) or np.std(P[:, c]) < 1e-12:
            ch_res.append((c, None))
            continue
        if search_loss == "corr":
            w, b, pred = safe_lstsq(P[:, c:c+1], y)
            if w is None:
                ch_res.append((c, None))
                continue
            loss = AlignmentLoss.channel_score(
                search_loss, loss_kind, pred, y, var_y, huber_delta_frac)
        else:
            loss = SearchLoss.score(search_loss, P[:, c], y)
        ch_res.append((c, loss))
        if loss < best_ch_loss:
            best_ch_loss = loss
            selected_c = c

    if selected_c is None:
        ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
               "ols_mse_norm": None, "ols_kept_global": None, "ols_bias": None,
               "ols_weights": None, "nested": None, "rational_candidates": []}
        return np.inf, None, None, np.inf, None, None, None, None, ana

    if loss_mode == "channel":
        ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
               "ols_mse_norm": None, "ols_kept_global": None, "ols_bias": None,
               "ols_weights": None, "nested": None, "rational_candidates": []}
        return best_ch_loss, selected_c, None, best_ch_loss, None, None, None, None, ana

    mask = np.array([np.all(np.isfinite(P[:, c])) and np.std(P[:, c]) > 1e-15
                     for c in range(n_ch)])
    valid_idx_global = np.where(mask)[0]
    ols_loss = np.inf
    ols_weights = None
    ols_bias = 0.0
    ols_kept = None
    lr_obj = None
    if mask.sum() > 0:
        try:
            lr = Ridge(alpha=1e-8).fit(P[:, mask], y)
            r2 = lr.score(P[:, mask], y)
            if np.isfinite(r2) and r2 > 0:
                ols_loss = max(0.0, float(1.0 - r2))
                ols_kept = np.zeros(n_ch, dtype=bool)
                ols_kept[valid_idx_global] = True
                ols_weights = np.zeros(n_ch)
                ols_weights[mask] = lr.coef_
                ols_bias = float(lr.intercept_)
                class _LR: pass
                lr_obj = _LR()
                lr_obj.coef_ = ols_weights
                lr_obj.intercept_ = lr.intercept_
                lr_obj._kept = ols_kept
        except Exception:
            pass

    if loss_mode == "ols":
        ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
               "ols_mse_norm": ols_loss, "ols_kept_global": ols_kept, "ols_bias": ols_bias,
               "ols_weights": ols_weights, "nested": None, "rational_candidates": []}
        if ols_weights is not None and ols_loss <= best_ch_loss:
            return ols_loss, selected_c, ols_weights, best_ch_loss, lr_obj, None, None, valid_idx_global, ana
        return best_ch_loss, selected_c, None, best_ch_loss, None, None, None, valid_idx_global, ana

    nested = None
    if mask.sum() > 0:
        if search_loss == "corr":
            P_v = P[:, mask]
            y_c = y - y.mean()
            sy = np.sqrt((y_c ** 2).sum()) + 1e-30
            corrs = np.array([
                abs(((P_v[:, j] - P_v[:, j].mean()) * y_c).sum() /
                    ((((P_v[:, j] - P_v[:, j].mean()) ** 2).sum() ** 0.5 + 1e-30) * sy))
                if np.all(np.isfinite(P_v[:, j])) and np.std(P_v[:, j]) > 1e-15 else 0.0
                for j in range(P_v.shape[1])])
            top_local = np.argsort(-corrs)[:max_basis_channels]
            top_global = valid_idx_global[top_local]
        else:
            local_indices = np.where(mask)[0]
            losses = np.array([
                SearchLoss.score(search_loss, P[:, c], y)
                for c in local_indices])
            top_local = np.argsort(losses)[:max_basis_channels]
            top_global = local_indices[top_local]
        ns = _nested_ols(P[:, top_global], y,
                         mse_floor=min(ols_loss, best_ch_loss) * var_y,
                         search_loss=search_loss, loss_kind=loss_kind,
                         huber_delta_frac=huber_delta_frac)
        if ns is not None:
            w_full = np.zeros(n_ch)
            w_full[top_global] = ns["w"]
            u_full = ["id"] * n_ch
            for ki, j in enumerate(top_global):
                u_full[j] = ns["unaries"][ki]
            vf = np.zeros(n_ch, dtype=bool)
            vf[top_global] = ns["valid"]
            nested = dict(mse=ns["mse"], link=ns["link"], unaries=u_full,
                          w=w_full, b=ns["b"], valid=vf, pred=ns["pred"])

    rat_cands = []
    rational = None
    best_rat_mse = np.inf
    if mask.sum() > 0:
        floor = min(ols_loss, best_ch_loss,
                    (nested["mse"] / var_y) if nested else np.inf) * var_y
        for deg in range(1, rat_max_degree + 1):
            rc = _poly_rational_ols(P[:, mask], y, floor, max_degree=deg,
                                    search_loss=search_loss,
                                    max_basis_channels=max_basis_channels,
                                    max_features=max_features)
            if rc is not None:
                rat_cands.append(rc)
                if rational is None or rc["mse"] < best_rat_mse:
                    rational = rc
                    best_rat_mse = rc["mse"]

    ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
           "ols_mse_norm": ols_loss, "ols_kept_global": ols_kept, "ols_bias": ols_bias,
           "ols_weights": ols_weights, "nested": nested, "rational_candidates": rat_cands}

    cands = [{"kind": "channel", "mse_norm": best_ch_loss}]
    if ols_weights is not None:
        cands.append({"kind": "ols", "mse_norm": ols_loss, "ols_weights": ols_weights})
    if nested:
        cands.append({"kind": "nested", "mse_norm": nested["mse"] / var_y, "nested": nested})
    if rational:
        cands.append({"kind": "rational", "mse_norm": rational["mse"] / var_y, "rational": rational})
    winner = select_model(cands, parsimony_rel_tol, complexity_max)

    if winner["kind"] == "rational":
        return rational["mse"] / var_y, selected_c, ols_weights, best_ch_loss, lr_obj, None, rational, valid_idx_global, ana
    if winner["kind"] == "nested":
        return nested["mse"] / var_y, selected_c, ols_weights, best_ch_loss, lr_obj, nested, None, valid_idx_global, ana
    if winner["kind"] == "ols":
        return ols_loss, selected_c, ols_weights, best_ch_loss, lr_obj, None, None, valid_idx_global, ana
    return best_ch_loss, selected_c, None, best_ch_loss, None, None, None, valid_idx_global, ana
