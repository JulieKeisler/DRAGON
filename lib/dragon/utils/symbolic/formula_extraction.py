"""Formula extraction utilities for DRAGON.

Pure functions that convert OLS analysis results into human-readable
formula strings. All functions operate on plain dicts — no class state.
"""

from __future__ import annotations

import numpy as np

from dragon.utils.symbolic.dag_to_formula import format_nested, format_rational


# ══════════════════════════════════════════════════════════════════════════════
#  FORMULA BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def channel_formula(
    formulas: list[str],
    selected_c: int | None,
    pred_all,
    y_np: np.ndarray,
    search_loss: str = "corr",
) -> str | None:
    """Build the channel formula string, applying affine scaling if corr loss."""
    if not formulas or selected_c is None or selected_c >= len(formulas):
        return None
    raw = str(formulas[selected_c])
    if search_loss != "corr":
        return raw
    ch = (pred_all[:, selected_c].numpy().ravel()
          if pred_all.ndim > 1 and pred_all.shape[-1] > 1
          else pred_all.squeeze().numpy().ravel())
    A = np.column_stack([ch, np.ones_like(ch)])
    c, *_ = np.linalg.lstsq(A, y_np.ravel(), rcond=None)
    a1, a0 = float(c[0]), float(c[1])
    if abs(a1 - 1.0) > 1e-4 or abs(a0) > 1e-4:
        return f"{a1:.6e} * ({raw}) + {a0:.6e}"
    return raw


def all_channel_formulas(
    formulas: list[str] | None,
    analysis: dict,
    selected_c: int | None,
) -> list[dict]:
    """Return a list of dicts with idx, formula, loss, selected for each channel."""
    ch_map = {int(c): (float(v) if v is not None and np.isfinite(v) else None)
              for c, v in analysis.get("channel_results", [])}
    return [{"idx": ci, "formula": str(f), "loss": ch_map.get(ci),
             "selected": ci == selected_c}
            for ci, f in enumerate(formulas or [])]


def ols_formula(formulas: list[str] | None, analysis: dict) -> str | None:
    """Build the OLS-combined formula string."""
    wts = analysis.get("ols_weights")
    bias = analysis.get("ols_bias", 0.0)
    if wts is None or not formulas:
        return None
    terms = [f"{w:+.4f}*({formulas[i]})"
             for i, w in enumerate(wts) if abs(w) > 1e-6 and i < len(formulas)]
    if abs(bias) > 1e-6:
        terms.append(f"{bias:+.4f}")
    return " ".join(terms) if terms else None


def polyrat_formulas(
    formulas: list[str] | None,
    valid_idx_global,
    analysis: dict,
) -> dict[str, str]:
    """Build polynomial-rational formula strings keyed by degree."""
    out = {}
    for rc in analysis.get("rational_candidates", []):
        dk = str(rc.get("max_degree", "?"))
        if valid_idx_global is not None and formulas:
            out[dk] = format_rational(rc, formulas, valid_idx_global)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  STATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_best_formula(state: dict, winner_type: str, rat_degree=None) -> None:
    """Pick the best formula string into ``state['best_formula']`` based on winner type."""
    if winner_type == "rational":
        key = str(rat_degree) if rat_degree is not None else next(iter(state.get("formula_polyrat", {})), None)
        state["best_formula"] = state.get("formula_polyrat", {}).get(key) or state.get("best_formula", "N/A")
    elif winner_type == "nested":
        state["best_formula"] = state.get("formula_nested") or state.get("best_formula", "N/A")
    elif winner_type == "ols":
        state["best_formula"] = state.get("formula_ols") or state.get("best_formula", "N/A")
    else:
        state["best_formula"] = state.get("formula_channel") or state.get("best_formula", "N/A")


def update_losses(state: dict, analysis: dict, selected_c: int | None) -> None:
    """Update ``state`` with channel / OLS / nested / rational losses."""
    var_y = analysis.get("var_y", 1.0) or 1.0
    ch_map = {c: v for c, v in analysis.get("channel_results", [])}

    state["loss_channel"] = (float(ch_map[selected_c])
                             if selected_c in ch_map and ch_map[selected_c] is not None else None)
    state["mse_channel"] = (state["loss_channel"] * var_y
                            if state["loss_channel"] is not None else None)

    ols_nm = analysis.get("ols_mse_norm")
    state["loss_ols"] = (float(ols_nm) if ols_nm is not None and np.isfinite(ols_nm) else None)
    state["mse_ols"] = (float(ols_nm) * var_y
                        if state["loss_ols"] is not None else None)

    nest = analysis.get("nested")
    state["loss_nested"] = (float(nest["mse"] / var_y) if nest and np.isfinite(nest["mse"]) else None)
    state["mse_nested"] = (float(nest["mse"]) if nest and np.isfinite(nest["mse"]) else None)

    lp, mp = {}, {}
    for rc in analysis.get("rational_candidates", []):
        dk = str(rc.get("max_degree", "?"))
        nm = rc["mse"] / var_y
        if np.isfinite(nm):
            lp[dk] = float(nm)
            mp[dk] = float(rc["mse"])
    state["loss_polyrat"] = lp
    state["mse_polyrat"] = mp
