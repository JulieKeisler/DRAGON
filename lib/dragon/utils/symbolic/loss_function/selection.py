"""Model selection with parsimony tolerance for DRAGON symbolic regression.

Provides ``eff_params`` for effective-parameter counting and ``select_model``
for choosing the best candidate (channel / OLS / nested / rational) while
penalising excessive complexity via relative tolerance.
"""

from __future__ import annotations

import numpy as np


_POST_ORDER = {"channel": 0, "linear": 0, "ols": 1, "nested": 2, "rational": 3}


def eff_params(cand: dict, min_w: float = 1e-4) -> int:
    """Count the effective number of parameters in a candidate model.

    Parameters
    ----------
    cand : dict
        Must contain ``"kind"`` and kind-specific fields.
    min_w : float
        Weight threshold below which a parameter is considered zero.
    """
    kind = cand["kind"]
    if kind in ("channel", "linear"):
        return 1
    if kind == "ols":
        w = cand.get("ols_weights")
        return int(np.sum(np.abs(w) >= min_w)) if w is not None else 1
    if kind == "nested":
        nd = cand["nested"]
        return int(np.sum(nd["valid"] & (np.abs(nd["w"]) >= min_w)))
    if kind == "rational":
        r = cand["rational"]
        return int(np.sum(np.abs(r["a"]) >= min_w) + np.sum(np.abs(r["b"]) >= min_w))
    return 1


def select_model(cands: list, parsimony_rel_tol: float = 0.02,
                 complexity_max=None) -> dict:
    """Parsimony-aware model selection.

    Among candidates within ``parsimony_rel_tol`` of the best MSE,
    selects the one with fewest effective parameters.

    Parameters
    ----------
    cands : list[dict]
        Each candidate must have ``"mse_norm"`` and ``"kind"`` keys.
    parsimony_rel_tol : float
        Relative tolerance for considering a candidate "near" the best.
    complexity_max : int or None
        Hard cap on the number of effective parameters.
    """
    for c in cands:
        c["_k"] = eff_params(c)
    pool = ([c for c in cands if c["_k"] <= complexity_max]
            if complexity_max is not None else cands)
    if not pool:
        pool = [min(cands, key=lambda c: c["_k"])]
    best_mse = min(c["mse_norm"] for c in pool)
    thresh = best_mse * (1.0 + parsimony_rel_tol)
    near = [c for c in pool if c["mse_norm"] <= thresh]
    return min(near, key=lambda c: (c["_k"], _POST_ORDER.get(c["kind"], 9), c["mse_norm"]))
