"""Robust least-squares solvers for DRAGON symbolic regression.

Provides ``safe_lstsq`` (weighted least squares with bias) and
``sparse_lstsq`` (L1-regularised via iterative masking).
"""

from __future__ import annotations

import numpy as np


def safe_lstsq(P, y_t):
    """Solve ``P @ w + b ≈ y_t`` via least squares.

    Returns
    -------
    weights : np.ndarray or None
    bias : float or None
    predicted : np.ndarray or None
        All ``None`` when the system is underdetermined or fails.
    """
    if P.shape[0] < P.shape[1] + 1:
        return None, None, None
    A = np.hstack([P, np.ones((P.shape[0], 1))])
    try:
        sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    return sol[:-1], sol[-1], A @ sol


def sparse_lstsq(P, y_t, rel_thresh=1e-3, abs_thresh=1e-8, max_iter=10):
    """Iterative hard-thresholding sparse least squares.

    Solves, prunes small-magnitude weights, re-solves, repeats until stable.

    Returns
    -------
    weights : np.ndarray or None
    bias : float or None
    predicted : np.ndarray or None
    kept : np.ndarray[bool] or None
        Mask of which columns in *P* were kept.
    """
    n, k = P.shape
    if n < k + 1:
        return None, None, None, None
    kept = np.ones(k, dtype=bool)
    for _ in range(max_iter):
        Pk = P[:, kept]
        if Pk.shape[1] == 0:
            return None, None, None, None
        A = np.hstack([Pk, np.ones((n, 1))])
        try:
            sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
        except np.linalg.LinAlgError:
            return None, None, None, None
        w_k = sol[:-1]
        thresh = max(rel_thresh * np.max(np.abs(w_k)), abs_thresh)
        mask_k = np.abs(w_k) >= thresh
        if mask_k.all():
            break
        idx_keep = np.where(kept)[0][mask_k]
        new_kept = np.zeros(k, dtype=bool)
        new_kept[idx_keep] = True
        if new_kept.sum() == 0:
            return None, None, None, None
        kept = new_kept
    Pk = P[:, kept]
    A = np.hstack([Pk, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
    return sol[:-1], float(sol[-1]), A @ sol, kept
