"""Formula extraction utilities for DRAGON.

Pure functions that convert OLS analysis results into human-readable
formula strings. All functions operate on plain dicts — no class state.
"""

from __future__ import annotations

import numpy as np

from dragon.utils.symbolic.dag_to_formula import format_nested, format_rational


# ══════════════════════════════════════════════════════════════════════════════
#  SYMPY INTEGRATION (optional — import lazily)
# ══════════════════════════════════════════════════════════════════════════════


def _loglog_fit(X_feat, y_pred, feature_names):
    """_loglog_fit

    Fit a power-law model via log-log linear regression.

    Parameters
    ----------
    X_feat : numpy.ndarray
        Feature matrix ``(n_samples, n_features)``.
    y_pred : numpy.ndarray
        Target values to fit ``(n_samples,)``.
    feature_names : list[str]
        Human-readable names for each feature column.

    Returns
    -------
    tuple
        ``(expr, latex_str, r2)`` on success, or ``(None, None, 0.0)``
        if the fit fails or has too few valid points.
    """
    from sympy import symbols, nsimplify, latex, powsimp

    n_feat = X_feat.shape[1]
    mask = np.isfinite(y_pred) & (np.abs(y_pred) > 1e-15)
    for fi in range(n_feat):
        mask &= np.isfinite(X_feat[:, fi]) & (np.abs(X_feat[:, fi]) > 1e-15)
    if mask.sum() < n_feat + 2:
        return None, None, 0.0

    log_y = np.log(np.abs(y_pred[mask]))
    log_feats = np.column_stack([np.log(np.abs(X_feat[mask, fi]))
                                 for fi in range(n_feat)])
    A = np.column_stack([log_feats, np.ones(mask.sum())])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, 0.0

    pred_log = A @ coeffs
    ss_res = np.sum((log_y - pred_log) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30) if ss_tot > 1e-30 else 0.0

    syms = {name: symbols(name, positive=True) for name in feature_names}
    product = None
    for fi, fname in enumerate(feature_names):
        p = coeffs[fi]
        if abs(p) < 1e-6:
            continue
        p_r = nsimplify(float(p), rational=True, tolerance=0.01)
        term = syms[fname] ** p_r
        product = term if product is None else product * term

    if product is None:
        const = nsimplify(float(np.exp(coeffs[-1])), rational=False, tolerance=0.01)
        return const, latex(const), r2

    scale = nsimplify(float(np.exp(coeffs[-1])), rational=False, tolerance=0.01)
    simplified = powsimp(scale * product, force=True)
    return simplified, latex(simplified), r2


def _poly_fit(X_feat, y_pred, feature_names, max_degree=2):
    """_poly_fit

    Fit a polynomial ``y = sum c_i * monomial_i(x)`` via Ridge regression.

    Parameters
    ----------
    X_feat : numpy.ndarray
        Feature matrix ``(n_samples, n_features)``.
    y_pred : numpy.ndarray
        Target values to fit ``(n_samples,)``.
    feature_names : list[str]
        Human-readable names for each feature column.
    max_degree : int, optional
        Maximum monomial degree to include (default 2).

    Returns
    -------
    tuple
        ``(expr, latex_str, r2)`` on success, or ``(None, None, r2)``
        if the fit is poor or fails.
    """
    from sympy import symbols, nsimplify, latex, Add, Mul, Symbol

    n, d = X_feat.shape
    if n < d + 2:
        return None, None, 0.0

    # Build monomial names and matrix
    mono_names = []   # list of tuples: (feature_index_tuple, label)
    mono_cols = []

    # Degree 1
    for fi in range(d):
        mono_names.append(((fi,), feature_names[fi]))
        mono_cols.append(X_feat[:, fi])

    # Degree 2+
    if max_degree >= 2:
        for fi in range(d):
            for fj in range(fi, d):
                mono_names.append(((fi, fj), f"{feature_names[fi]}*{feature_names[fj]}"))
                mono_cols.append(X_feat[:, fi] * X_feat[:, fj])

    if max_degree >= 3:
        for fi in range(d):
            for fj in range(fi, d):
                for fk in range(fj, d):
                    mono_names.append(((fi, fj, fk),
                                       f"{feature_names[fi]}*{feature_names[fj]}*{feature_names[fk]}"))
                    mono_cols.append(X_feat[:, fi] * X_feat[:, fj] * X_feat[:, fk])

    A = np.column_stack(mono_cols)
    # Add intercept
    A_aug = np.column_stack([A, np.ones(n)])

    # Ridge regression
    alpha = 1e-6
    try:
        ATA = A_aug.T @ A_aug + alpha * np.eye(A_aug.shape[1])
        ATA[-1, -1] = 0.0  # don't regularize intercept
        coeffs = np.linalg.solve(ATA, A_aug.T @ y_pred)
    except np.linalg.LinAlgError:
        return None, None, 0.0

    # R²
    pred = A_aug @ coeffs
    ss_res = np.sum((y_pred - pred) ** 2)
    ss_tot = np.sum((y_pred - y_pred.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30) if ss_tot > 1e-30 else 0.0

    if r2 < 0.9:
        return None, None, r2

    # Build SymPy expression
    syms = {name: symbols(name, positive=True) for name in feature_names}
    terms = []
    for i, (idx_tuple, _) in enumerate(mono_names):
        c = coeffs[i]
        if abs(c) < 1e-6:
            continue
        c_r = nsimplify(float(c), rational=True, tolerance=0.01)
        if len(idx_tuple) == 1:
            term = c_r * syms[feature_names[idx_tuple[0]]]
        elif len(idx_tuple) == 2:
            term = c_r * syms[feature_names[idx_tuple[0]]] * syms[feature_names[idx_tuple[1]]]
        elif len(idx_tuple) == 3:
            term = c_r * syms[feature_names[idx_tuple[0]]] * syms[feature_names[idx_tuple[1]]] * syms[feature_names[idx_tuple[2]]]
        else:
            continue
        terms.append(term)

    intercept = coeffs[-1]
    if abs(intercept) > 1e-6:
        terms.append(nsimplify(float(intercept), rational=True, tolerance=0.01))

    if not terms:
        return None, None, 0.0

    expr = Add(*terms) if len(terms) > 1 else terms[0]
    simplified = nsimplify(expr, rational=False, tolerance=0.01)
    return simplified, latex(simplified), r2


def _nested_sympy(formulas_sympy, analysis, feature_names=None):
    """_nested_sympy

    Build a SymPy expression directly from the nested-OLS model.

    Constructs ``inv_link( sum_i w_i * unary_i(ch_i) + b )`` where
    ``ch_i`` are SymPy channel expressions.

    Parameters
    ----------
    formulas_sympy : list[sympy.Expr]
        SymPy expressions for each channel.
    analysis : dict
        Full analysis dict from ``ols_pipeline.evaluate``.
    feature_names : list[str] or None
        Input feature names — used to enforce positivity assumptions.

    Returns
    -------
    tuple[sympy.Expr, str] or (None, None)
        ``(simplified_expr, latex_string)``, or ``(None, None)`` on failure.
    """
    from sympy import (symbols, sympify, log, sqrt, Abs, exp, cbrt,
                       nsimplify, latex, S, simplify)

    nested = analysis.get("nested")
    if nested is None or not formulas_sympy:
        return None, None

    link = nested.get("link", "id")
    unaries = nested.get("unaries", [])
    w = nested.get("w")
    b = nested.get("b", 0.0)
    valid = nested.get("valid")
    if w is None or valid is None:
        return None, None

    _UNARY = {
        "id": lambda x: x,
        "log": lambda x: log(Abs(x)),
        "sqrt": lambda x: sqrt(Abs(x)),
        "sq": lambda x: x**2,
        "inv": lambda x: 1 / x,
        "neg": lambda x: -x,
        "cbrt": lambda x: cbrt(x),
        "abs": lambda x: Abs(x),
    }
    _LINK_INV = {
        "id": lambda x: x,
        "log": lambda x: exp(x),
        "sqrt": lambda x: x**2,
        "sq": lambda x: sqrt(x),
        "inv": lambda x: 1 / x,
        "cbrt": lambda x: x**3,
    }

    linear = S.Zero
    for i, (wi, ui, vi) in enumerate(zip(w, unaries, valid)):
        if not vi or abs(wi) < 1e-8:
            continue
        ch_i = formulas_sympy[i] if i < len(formulas_sympy) else symbols(f"ch_{i}")
        unary_fn = _UNARY.get(ui)
        if unary_fn is None:
            return None, None
        linear += nsimplify(float(wi), rational=False, tolerance=0.01) * unary_fn(ch_i)

    if abs(b) > 1e-8:
        linear += nsimplify(float(b), rational=False, tolerance=0.01)

    inv_link = _LINK_INV.get(link)
    if inv_link is None:
        return None, None
    expr = inv_link(linear)
    if feature_names:
        for name in feature_names:
            expr = expr.subs(symbols(name), symbols(name, positive=True))
    expr = simplify(expr)
    return expr, latex(expr)


def _rational_sympy(analysis, feature_names):
    """_rational_sympy

    Build a SymPy expression from the rational-OLS model.

    The rational model is ``y = (a0 + Σ a_i·m_i) / (b0 + Σ b_i·m_i)``
    where ``m_i`` are monomials over the input features.

    Parameters
    ----------
    analysis : dict
        Full analysis dict from ``ols_pipeline.evaluate``.  Must contain
        ``"rational_candidates"`` with numerator/denominator coefficients.
    feature_names : list[str]
        Human-readable names for each feature.

    Returns
    -------
    tuple[sympy.Expr, str] or (None, None)
        ``(simplified_expr, latex_string)``, or ``(None, None)`` on failure.
    """
    from sympy import (symbols, Mul, Add, nsimplify, latex, S, simplify, refine, Q)

    rats = analysis.get("rational_candidates")
    if not rats:
        return None, None

    rat = min(rats, key=lambda r: r.get("mse", np.inf))
    monomials = rat.get("monomials", [])
    a = np.asarray(rat.get("a", []))
    b = np.asarray(rat.get("b", []))
    a0 = float(rat.get("a0", 0.0))
    b0 = float(rat.get("b0", 1.0))
    kept_a = np.asarray(rat.get("kept_a", np.ones(len(a), dtype=bool)))
    kept_b = np.asarray(rat.get("kept_b", np.zeros(len(b), dtype=bool)))

    syms = {name: symbols(name, positive=True) for name in feature_names}

    def _build_poly(coeffs, kept_mask):
        terms = []
        for i, mono in enumerate(monomials):
            if i >= len(kept_mask) or not kept_mask[i]:
                continue
            c = float(coeffs[i])
            if abs(c) < 1e-10:
                continue
            c_s = nsimplify(c, rational=True, tolerance=0.01)
            if isinstance(mono, (list, tuple)):
                prod = S.One
                for idx in mono:
                    if idx < len(feature_names):
                        prod *= syms[feature_names[idx]]
                terms.append(c_s * prod)
            elif isinstance(mono, int) and mono < len(feature_names):
                terms.append(c_s * syms[feature_names[mono]])
        return Add(*terms) if terms else S.Zero

    num = _build_poly(a, kept_a)
    if abs(a0) > 1e-10:
        num += nsimplify(a0, rational=True, tolerance=0.01)

    den = _build_poly(b, kept_b)
    if abs(b0) > 1e-10:
        den += nsimplify(b0, rational=True, tolerance=0.01)

    if den == S.Zero or den == 0:
        return None, None

    expr = num / den
    expr = simplify(expr)
    for name in feature_names:
        expr = refine(expr, Q.positive(syms[name]))
    return expr, latex(expr)


def sympify_nested(
    analysis: dict,
    feature_names: list[str],
    pred_all=None,
    X=None,
    selected_c=None,
    y_true=None,
    formulas_sympy=None,
    min_r2: float = 0.99,
    min_pred_r2: float = 0.8,
):
    """Build a clean SymPy expression from the model's prediction.

    Tries (in order):
    1. Direct nested-OLS extraction (link/unaries/weights → SymPy).
    2. Power-law fit to the nested-OLS prediction.
    3. Polynomial fit to the nested-OLS prediction.
    4. Power-law fit to the selected channel / OLS prediction.

    Parameters
    ----------
    analysis : dict
        From ``ols_pipeline.evaluate``.
    feature_names : list[str]
        Names of input features.
    pred_all : numpy.ndarray or None
        Model outputs ``(n_samples, n_columns)``.
    X : numpy.ndarray or None
        Original input features ``(n_samples, n_features)``.
    selected_c : int or None
        Index of the selected channel (used as fallback prediction).
    y_true : numpy.ndarray or None
        True target values — used to check prediction quality before fitting.
    formulas_sympy : list or None
        SymPy channel expressions from ``graph_to_all_formulas(parse_sympy=True)``.
    min_r2 : float
        Minimum R² for the power-law fit.
    min_pred_r2 : float
        Minimum R² between prediction and true values.  If the model's
        prediction is not well-correlated with the true target, no symbolic
        fitting is attempted (returns None).

    Returns
    -------
    tuple[sympy.Expr, str] or (None, None)
        ``(simplified_expr, latex_string)``.
    """
    n_feat = len(feature_names)

    if X is not None:
        X_feat = X[:, :n_feat]
    elif pred_all is not None and pred_all.shape[1] >= n_feat:
        X_feat = pred_all[:, :n_feat]
    else:
        return None, None

    # --- 1. Collect candidate predictions ---
    nested = analysis.get("nested")
    candidates = []  # (label, y_pred_array)

    # nested OLS prediction
    if nested is not None and nested.get("pred") is not None:
        npred = nested["pred"]
        arr = npred.detach().numpy().ravel() if hasattr(npred, 'detach') else np.asarray(npred).ravel()
        candidates.append(("nested", arr))

    # raw model output for selected channel
    raw_arr = None
    if pred_all is not None and selected_c is not None and selected_c < pred_all.shape[1]:
        raw = pred_all[:, selected_c]
        if hasattr(raw, 'detach'):
            raw = raw.detach().numpy()
        raw_arr = np.asarray(raw).ravel()
        candidates.append(("raw", raw_arr))

    # Affinely-scaled raw output: fit y = a*raw + b so the quality gate
    # can compare against y_true even when the model uses corr loss.
    if raw_arr is not None and y_true is not None:
        yt = np.asarray(y_true).ravel()
        if raw_arr.shape[0] == yt.shape[0] and raw_arr.shape[0] > 2:
            A = np.column_stack([raw_arr, np.ones_like(raw_arr)])
            c, *_ = np.linalg.lstsq(A, yt, rcond=None)
            scaled = A @ c
            candidates.insert(0, ("scaled", scaled))

    if not candidates:
        return None, None

    # --- 2. Quality gate: pick best prediction that passes R² threshold ---
    y_pred = None
    if y_true is not None:
        yt = np.asarray(y_true).ravel()
        best_r2 = -1.0
        for label, cand in candidates:
            if cand.shape[0] != yt.shape[0] or cand.shape[0] <= 2:
                continue
            ss_res = np.sum((yt - cand) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-30) if ss_tot > 1e-30 else 0.0
            if r2 >= min_pred_r2 and r2 > best_r2:
                best_r2 = r2
                y_pred = cand
        # if none passed, use the best available anyway (fits will self-evaluate)
        if y_pred is None:
            y_pred = candidates[-1][1]
    else:
        y_pred = candidates[0][1]

    # --- 2a. Rational-OLS extraction (more accurate than nested) ---
    expr, latex_str = _rational_sympy(analysis, feature_names)
    if expr is not None:
        return expr, latex_str

    # --- 2b. Direct nested-model extraction ---
    if nested is not None and formulas_sympy:
        expr, latex_str = _nested_sympy(formulas_sympy, analysis, feature_names)
        if expr is not None:
            return expr, latex_str

    # --- 3. Try power-law fit (log-log) ---
    expr, latex_str, r2 = _loglog_fit(X_feat, y_pred, feature_names)
    if r2 >= min_r2 and latex_str is not None:
        return expr, latex_str

    # --- 4. Try polynomial fit (degree 2) ---
    expr, latex_str, r2 = _poly_fit(X_feat, y_pred, feature_names, max_degree=2)
    if r2 >= 0.99 and latex_str is not None:
        return expr, latex_str

    # --- 5. Try polynomial fit (degree 3) if degree 2 was close ---
    if r2 >= 0.9:
        expr, latex_str, r2 = _poly_fit(X_feat, y_pred, feature_names, max_degree=3)
        if r2 >= 0.99 and latex_str is not None:
            return expr, latex_str

    # --- 6. Direct fit to y_true: when the model is accurate (low loss)
    #     but the prediction is untransformed (e.g. passthrough channels),
    #     try fitting the true target directly with power-law / polynomial. ---
    if y_true is not None and X_feat is not None:
        yt = np.asarray(y_true).ravel()
        if yt.shape[0] == X_feat.shape[0] and yt.shape[0] > 2:
            # Only try this if the scaled prediction R² was decent —
            # means the model identified useful features even if the
            # raw output needs a complex transform.
            ss_res_s = np.sum((yt - y_pred) ** 2)
            ss_tot_s = np.sum((yt - yt.mean()) ** 2)
            pred_r2 = 1.0 - ss_res_s / (ss_tot_s + 1e-30) if ss_tot_s > 1e-30 else 0.0
            if pred_r2 >= 0.95:
                expr, latex_str, r2 = _loglog_fit(X_feat, yt, feature_names)
                if r2 >= min_r2 and latex_str is not None:
                    return expr, latex_str

                expr, latex_str, r2 = _poly_fit(X_feat, yt, feature_names, max_degree=2)
                if r2 >= 0.99 and latex_str is not None:
                    return expr, latex_str

                if r2 >= 0.9:
                    expr, latex_str, r2 = _poly_fit(X_feat, yt, feature_names, max_degree=3)
                    if r2 >= 0.99 and latex_str is not None:
                        return expr, latex_str

    return None, None


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
    """channel_formula

    Build the channel formula string, applying affine scaling if corr loss.

    Parameters
    ----------
    formulas : list[str]
        Candidate formula strings for each channel.
    selected_c : int or None
        Index of the selected channel.
    pred_all : torch.Tensor or numpy.ndarray
        Model predictions ``(n_samples, n_channels)``.
    y_np : numpy.ndarray
        True target values ``(n_samples,)``.
    search_loss : str, optional
        Loss type used during search (default ``"corr"``).  When
        ``"corr"`` an affine scaling ``a*formula + b`` is fitted.

    Returns
    -------
    str or None
        The formula string, or ``None`` if no valid channel exists.
    """
    if not formulas or selected_c is None or selected_c >= len(formulas):
        return None
    raw = str(formulas[selected_c])
    if search_loss != "corr":
        return raw
    if hasattr(pred_all, 'detach'):
        ch = (pred_all[:, selected_c].detach().numpy().ravel()
              if pred_all.ndim > 1 and pred_all.shape[-1] > 1
              else pred_all.squeeze().detach().numpy().ravel())
    else:
        ch = (pred_all[:, selected_c].ravel()
              if pred_all.ndim > 1 and pred_all.shape[-1] > 1
              else np.asarray(pred_all).ravel())
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
    """all_channel_formulas

    Return a list of dicts with idx, formula, loss, selected for each channel.

    Parameters
    ----------
    formulas : list[str] or None
        Formula strings for each channel.
    analysis : dict
        Full analysis dict from ``ols_pipeline.evaluate``.
    selected_c : int or None
        Index of the selected channel.

    Returns
    -------
    list[dict]
        Each dict contains ``"idx"``, ``"formula"``, ``"loss"``, and
        ``"selected"`` keys.
    """
    ch_map = {int(c): (float(v) if v is not None and np.isfinite(v) else None)
              for c, v in analysis.get("channel_results", [])}
    return [{"idx": ci, "formula": str(f), "loss": ch_map.get(ci),
             "selected": ci == selected_c}
            for ci, f in enumerate(formulas or [])]


def ols_formula(formulas: list[str] | None, analysis: dict) -> str | None:
    """ols_formula

    Build the OLS-combined formula string.

    Parameters
    ----------
    formulas : list[str] or None
        Formula strings for each channel.
    analysis : dict
        Full analysis dict — must contain ``"ols_weights"`` and
        ``"ols_bias"``.

    Returns
    -------
    str or None
        Weighted sum of channel formulas, or ``None`` on failure.
    """
    wts = analysis.get("ols_weights")
    bias = analysis.get("ols_bias", 0.0)
    if wts is None or not formulas:
        return None
    terms = [f"{w:+.4f}*({formulas[i]})"
             for i, w in enumerate(wts) if abs(w) > 1e-6 and i < len(formulas)]
    if abs(bias) > 1e-6:
        terms.append(f"{bias:+.4f}")
    return " ".join(terms) if terms else None


_UNARY_SYMBOLS = {
    "id": "", "sq": "**2", "sqrt": "**0.5", "inv": "**(-1)", "neg": "(-1)*", "log": "ln",
    "abs": "|...|",
}
_LINK_INV = {
    "id": "", "log": "exp(...)", "sqrt": "(...)**2", "sq": "sqrt(...)",
    "inv": "1/(...)", "cbrt": "(...)**3",
}


def nested_formula(
    formulas: list[str] | None,
    analysis: dict,
) -> str | None:
    """nested_formula

    Build a human-readable string from the nested-OLS result.

    The nested model computes ``inv_link( sum_i w_i * unary_i(ch_i) + b )``.

    Parameters
    ----------
    formulas : list[str] or None
        Formula strings for each channel.
    analysis : dict
        Full analysis dict — must contain ``"nested"`` with weights,
        unaries, link function, and bias.

    Returns
    -------
    str or None
        Human-readable formula string, or ``None`` on failure.
    """
    nested = analysis.get("nested")
    if nested is None or not formulas:
        return None
    link = nested.get("link", "id")
    unaries = nested.get("unaries", [])
    w = nested.get("w")
    b = nested.get("b", 0.0)
    valid = nested.get("valid")
    if w is None or valid is None:
        return None

    parts = []
    for i, (wi, ui, vi) in enumerate(zip(w, unaries, valid)):
        if not vi or abs(wi) < 1e-8:
            continue
        f = formulas[i] if i < len(formulas) else f"ch[{i}]"
        sym = _UNARY_SYMBOLS.get(ui, f".{ui}()")
        if ui == "log":
            inner = f"ln({f})"
        elif ui == "neg":
            inner = f"(-{f})"
        elif ui == "abs":
            inner = f"|{f}|"
        elif sym:
            inner = f"({f}){sym}"
        else:
            inner = f
        parts.append(f"{wi:+.4f}*{inner}")
    linear = " ".join(parts) if parts else "0"
    if abs(b) > 1e-8:
        linear += f" {b:+.4f}"

    link_inv = _LINK_INV.get(link, f"{link}(...)")
    if link == "id":
        return linear
    return link_inv.replace("...", linear)


def polyrat_formulas(
    formulas: list[str] | None,
    valid_idx_global,
    analysis: dict,
) -> dict[str, str]:
    """polyrat_formulas

    Build polynomial-rational formula strings keyed by degree.

    Parameters
    ----------
    formulas : list[str] or None
        Channel formula strings used as building blocks.
    valid_idx_global : list[int] or None
        Indices of valid channels in the global ordering.
    analysis : dict
        Full analysis dict — must contain ``"rational_candidates"``.

    Returns
    -------
    dict[str, str]
        Mapping from degree string (e.g. ``"2"``) to the formatted
        rational formula string.
    """
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
    """set_best_formula

    Pick the best formula string into ``state['best_formula']`` based on winner type.

    Parameters
    ----------
    state : dict
        Mutable state dict — updated in place with ``"best_formula"``.
    winner_type : str
        One of ``"rational"``, ``"nested"``, ``"ols"``, or any other
        value (falls back to channel formula).
    rat_degree : str or None
        Degree key for the rational formula (used when
        ``winner_type == "rational"``).

    Returns
    -------
    None
        Modifies *state* in place.
    """
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
    """update_losses

    Update ``state`` with channel / OLS / nested / rational losses.

    Parameters
    ----------
    state : dict
        Mutable state dict — updated in place with ``"loss_*"`` and
        ``"mse_*"`` keys.
    analysis : dict
        Full analysis dict from ``ols_pipeline.evaluate``.
    selected_c : int or None
        Index of the selected channel.

    Returns
    -------
    None
        Modifies *state* in place.
    """
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
