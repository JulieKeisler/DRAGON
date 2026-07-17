"""Nested and polynomial-rational OLS fitting routines for DRAGON.

Provides ``nested_ols`` (link-function composition with sparse OLS) and
``poly_rational_ols`` (polynomial-basis rational model fitting) used by
the OLS pipeline to produce compact symbolic formulas.
"""

from __future__ import annotations

import numpy as np

from dragon.utils.symbolic.loss_function.search_loss import SearchLoss
from dragon.utils.symbolic.loss_function.solvers import sparse_lstsq


_LINKS = {
    'id'  : (lambda y: y,                                lambda p: p),
    'log' : (lambda y: np.log(np.abs(y) + 1e-8),        lambda p: np.exp(np.clip(p, -50, 50))),
    'sqrt': (lambda y: np.sqrt(np.abs(y)),               lambda p: p ** 2),
    'sq'  : (lambda y: y ** 2,                           lambda p: np.sqrt(np.abs(p))),
    'inv' : (lambda y: 1.0 / (np.abs(y) + 1e-8),        lambda p: 1.0 / (np.abs(p) + 1e-8)),
    'cbrt': (lambda y: np.sign(y) * np.abs(y) ** (1/3), lambda p: p ** 3),
}

_UNARIES = {
    'id'  : lambda x: x,
    'sq'  : lambda x: x ** 2,
    'sqrt': lambda x: np.sqrt(np.abs(x)),
    'inv' : lambda x: 1.0 / (np.abs(x) + 1e-8),
    'neg' : lambda x: -x,
    'log' : lambda x: np.log(np.abs(x) + 1e-8),
}


def build_poly_features(channels, max_degree: int):
    """Build polynomial features (degree 1 and 2) from channel columns.

    Returns
    -------
    feats : np.ndarray
        Feature matrix of shape ``(n, n_features)``.
    names : list[tuple[int, ...]]
        Monomial index tuples for each column.
    """
    n, k = channels.shape
    feats, names = [], []
    for j in range(k):
        feats.append(channels[:, j])
        names.append((j,))
    if max_degree >= 2:
        for j in range(k):
            feats.append(channels[:, j] ** 2)
            names.append((j, j))
        for i in range(k):
            for j in range(i + 1, k):
                feats.append(channels[:, i] * channels[:, j])
                names.append((i, j))
    if not feats:
        return np.empty((n, 0)), []
    return np.column_stack(feats), names


def nested_ols(channels, y, mse_floor, search_loss="corr", loss_kind="mse",
               huber_delta_frac=0.20):
    """Fit a nested OLS model with link functions and per-channel unaries.

    Tries every link function x unary combination, fits a sparse linear model
    on the transformed channels, and returns the best result.

    Returns
    -------
    dict or None
        Keys: ``mse``, ``link``, ``unaries``, ``w``, ``b``, ``valid``, ``pred``.
    """
    n_ch = channels.shape[1]
    best = None
    for link_name, (fwd, inv) in _LINKS.items():
        with np.errstate(all='ignore'):
            y_t = fwd(y)
        if not np.all(np.isfinite(y_t)) or np.std(y_t) < 1e-12:
            continue
        unaries_pick = []
        chans_t = np.empty_like(channels)
        valid_mask = np.zeros(n_ch, dtype=bool)
        for j in range(n_ch):
            best_score, best_u, best_vals = np.inf, 'id', channels[:, j]
            for u_name, u_fn in _UNARIES.items():
                with np.errstate(all='ignore'):
                    v = u_fn(channels[:, j])
                if not np.all(np.isfinite(v)) or np.std(v) < 1e-12:
                    continue
                score = SearchLoss.score(search_loss, v, y_t)
                if np.isfinite(score) and score < best_score:
                    best_score, best_u, best_vals = score, u_name, v
            unaries_pick.append(best_u)
            if np.isfinite(best_score):
                chans_t[:, j] = best_vals
                valid_mask[j] = True
        if valid_mask.sum() == 0:
            continue
        w_kept, b, pred_t, kept_local = sparse_lstsq(chans_t[:, valid_mask], y_t)
        if w_kept is None:
            continue
        valid_idx_sub = np.where(valid_mask)[0]
        new_valid = np.zeros(n_ch, dtype=bool)
        new_valid[valid_idx_sub[kept_local]] = True
        with np.errstate(all='ignore'):
            pred = inv(pred_t)
        if not np.all(np.isfinite(pred)):
            continue
        mse = float(np.mean((y - pred) ** 2))
        if not np.isfinite(mse) or mse >= mse_floor:
            continue
        w_full = np.zeros(n_ch)
        w_full[new_valid] = w_kept
        best = dict(mse=mse, link=link_name, unaries=unaries_pick,
                    w=w_full, b=b, valid=new_valid, pred=pred)
        mse_floor = mse
    return best


def poly_rational_ols(channels, y, mse_floor, max_degree,
                      search_loss="corr", max_basis_channels=3,
                      max_features=4, rel_thresh=1e-3, abs_thresh=1e-8,
                      max_iter=10):
    """Fit a rational polynomial ``P(x)/Q(x)`` where P and Q are sparse.

    Returns
    -------
    dict or None
        Keys: ``mse``, ``a``, ``a0``, ``b``, ``b0``, ``kept_a``, ``kept_b``,
        ``monomials``, ``pred``, ``max_degree``.
    """
    n, k = channels.shape
    if k == 0 or n < 4:
        return None
    y_c = y - y.mean()
    sy = (y_c ** 2).sum() ** 0.5 + 1e-30
    scores = np.array([
        SearchLoss.score(search_loss, channels[:, j], y)
        for j in range(k)])
    sel = np.argsort(scores)[:min(k, max_basis_channels)]
    feats, mono_local = build_poly_features(channels[:, sel], max_degree)
    if feats.shape[1] == 0:
        return None
    ok = np.array([np.all(np.isfinite(feats[:, j])) and np.std(feats[:, j]) > 1e-15
                   for j in range(feats.shape[1])])
    if not ok.any():
        return None
    feats = feats[:, ok]
    mono_local = [m for m, ko in zip(mono_local, ok) if ko]
    if feats.shape[1] > max_features:
        sc = np.array([
            SearchLoss.score(search_loss, feats[:, j], y)
            for j in range(feats.shape[1])])
        keep = np.argsort(sc)[:max_features]
        feats = feats[:, keep]
        mono_local = [mono_local[i] for i in keep]
    m = feats.shape[1]
    if n < 2 * m + 2:
        sc = np.array([
            SearchLoss.score(search_loss, feats[:, j], y)
            if np.std(feats[:, j]) > 1e-15 else np.inf
            for j in range(m)])
        keep = np.argsort(sc)[:max(1, (n - 2) // 2)]
        feats = feats[:, keep]
        mono_local = [mono_local[i] for i in keep]
        m = feats.shape[1]
        if m == 0:
            return None

    def build_A(ka, kb):
        cols = []
        if ka.any():
            cols.append(feats[:, ka])
        cols.append(np.ones((n, 1)))
        if kb.any():
            cols.append(-feats[:, kb] * y[:, None])
        return np.hstack(cols)

    kept_a = np.ones(m, dtype=bool)
    kept_b = np.ones(m, dtype=bool)
    for _ in range(max_iter):
        A = build_A(kept_a, kept_b)
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        ka = int(kept_a.sum())
        kb = int(kept_b.sum())
        a_k = sol[:ka]
        b_k = sol[ka + 1: ka + 1 + kb]
        all_abs = np.concatenate([np.abs(a_k), np.abs(b_k)]) if ka + kb else np.array([0.0])
        thresh = max(rel_thresh * float(all_abs.max()), abs_thresh)
        na = np.abs(a_k) >= thresh
        nb = np.abs(b_k) >= thresh
        if na.all() and nb.all():
            break
        new_ka = np.zeros(m, dtype=bool)
        new_ka[np.where(kept_a)[0][na]] = True
        new_kb = np.zeros(m, dtype=bool)
        new_kb[np.where(kept_b)[0][nb]] = True
        if not new_ka.any() and not new_kb.any():
            return None
        kept_a, kept_b = new_ka, new_kb

    A = build_A(kept_a, kept_b)
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    ka = int(kept_a.sum())
    kb = int(kept_b.sum())
    a = np.zeros(m)
    a[kept_a] = sol[:ka]
    a0 = float(sol[ka])
    b = np.zeros(m)
    b[kept_b] = sol[ka + 1: ka + 1 + kb]
    numer = feats @ a + a0
    denom = feats @ b + 1.0
    if not np.all(np.abs(denom) > 1e-10):
        return None
    pred = numer / denom
    if not np.all(np.isfinite(pred)):
        return None
    mse = float(np.mean((y - pred) ** 2))
    if not np.isfinite(mse) or mse >= mse_floor:
        return None
    mono_global = [tuple(int(sel[i]) for i in mn) for mn in mono_local]
    return dict(mse=mse, a=a, a0=a0, b=b, b0=1.0,
                kept_a=kept_a, kept_b=kept_b,
                monomials=mono_global, pred=pred, max_degree=max_degree)
