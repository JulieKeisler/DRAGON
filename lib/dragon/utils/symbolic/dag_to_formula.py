"""Symbolic formula extraction and compilation utilities for DragonSR.

This module provides the core functions for:
  * Extracting symbolic expressions from Dragon DAGs
  * Compiling SymPy expressions back into minimal Dragon DAGs
  * Formatting formula strings with correct precedence
  * Formatting OLS-fitted model results as human-readable formulas
"""

from collections import Counter

import numpy as np
import torch.nn as nn
from sympy import Symbol, Add, Mul, Pow, sympify, Integer, Rational, Float

from dragon.search_space.dag_encoding import AdjMatrix, SymbolicNode, fill_adj_matrix
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import (
    Negate, Inverse, SelectFeatures, ConstantBrick, ChannelBoost,
)


# ══════════════════════════════════════════════════════════════════════════════
#  FORMULA STRING FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

_FMT_FUNCS = {"sqrt", "ln", "sin", "cos", "exp", "abs", "log"}

# Fitted coefficients used to be printed with a fixed number of decimals
# ("%+.4f"), which keeps two significant figures below 0.01 and one below
# 0.001: a fitted 0.000817 printed as 0.0008, whose cube is 6% off.  coef()
# keeps at least six significant figures, and never fewer decimals than the
# fixed spec it replaces.  LEGACY_FORMAT = True prints exactly as the code of
# the paper's runs did (the equivalence tests compare against that code).
LEGACY_FORMAT = False


def coef(x, spec="+.4f"):
    """Print the number *x*; *spec* is the fixed format it used to be printed with."""
    if LEGACY_FORMAT:
        return format(x, spec)
    x = float(x)
    sign = "+" if spec.startswith("+") else ""
    if x == 0.0 or not np.isfinite(x):
        return format(x, sign + "g")
    if spec.endswith("e"):
        return format(x, sign + ".6e")
    sig = max(6, int(np.floor(np.log10(abs(x)))) + 1 + 4)
    return format(x, f"{sign}.{sig}g")


def _is_wrapped(expr):
    """Return True if *expr* is entirely enclosed by a matching pair of parentheses."""
    if len(expr) < 2 or expr[0] != "(" or expr[-1] != ")":
        return False
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(expr) - 1:
                return False
    return depth == 0


def _strip_outer_parens(expr):
    """Remove all layers of outer parentheses from *expr*."""
    expr = expr.strip()
    while _is_wrapped(expr):
        expr = expr[1:-1].strip()
    return expr


def _find_top_level_op(expr):
    """Find the lowest-precedence top-level operator in *expr*."""
    depth = 0
    best = None
    best_prec = 99
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            if expr.startswith("**", i):
                if best is None or 2 < best_prec:
                    best = (i, "**")
                    best_prec = 2
                i += 1
            elif ch in "+-":
                if i == 0 or expr[i - 1] in "eE*/+-(":
                    i += 1
                    continue
                if 0 < best_prec:
                    best = (i, ch)
                    best_prec = 0
            elif ch in "*/":
                if i == 0 or expr[i - 1] in "eE*/+-(":
                    i += 1
                    continue
                if 1 < best_prec:
                    best = (i, ch)
                    best_prec = 1
        i += 1
    return best


def _wrap_if_needed(expr, child_prec, parent_prec, *, right=False, op=None):
    """Wrap *expr* in parentheses if its precedence is below *parent_prec*."""
    need = child_prec < parent_prec
    if right and op in {"-", "/"} and child_prec == parent_prec:
        need = True
    return f"({expr})" if need else expr


def _format_expr(expr):
    """Recursively format an expression string with minimal parentheses."""
    expr = _strip_outer_parens(str(expr).strip())
    if not expr:
        return expr, 9

    op_info = _find_top_level_op(expr)
    if op_info is not None:
        idx, op = op_info
        left = expr[:idx].strip()
        right = expr[idx + len(op):].strip()
        prec = {"+": 0, "-": 0, "*": 1, "/": 1, "**": 2}[op]
        left_s, left_p = _format_expr(left)
        right_s, right_p = _format_expr(right)
        if op == "**":
            left_s = _wrap_if_needed(left_s, left_p, prec)
            if right_p <= prec:
                right_s = f"({right_s})"
            return f"{left_s}**{right_s}", prec
        left_s = _wrap_if_needed(left_s, left_p, prec)
        right_s = _wrap_if_needed(right_s, right_p, prec, right=True, op=op)
        return f"{left_s} {op} {right_s}", prec

    if expr.startswith("-"):
        inner_s, inner_p = _format_expr(expr[1:].strip())
        return f"-{_wrap_if_needed(inner_s, inner_p, 3)}", 3

    p = expr.find("(")
    if p > 0 and expr.endswith(")"):
        name = expr[:p].strip()
        inner = expr[p + 1:-1]
        if name in _FMT_FUNCS:
            inner_s, _ = _format_expr(inner)
            return f"{name}({inner_s})", 4

    return expr, 5


def format_formula_string(expr):
    """Cheap parenthesis cleanup for generated formula strings.

    This is a pure string formatter: no SymPy parsing, no algebra, only a
    recursive precedence-aware re-rendering of the generated expression.

    Parameters
    ----------
    expr : str
        Raw expression string produced by :func:`graph_to_formula`.

    Returns
    -------
    str
        Cleaned-up expression string with minimal parentheses.
    """
    return _format_expr(expr)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  FORMULA EXTRACTION FROM DAG
# ══════════════════════════════════════════════════════════════════════════════

def op_tensors(inputs, combiner):
    """Combine lists of symbolic feature strings according to *combiner*.

    Parameters
    ----------
    inputs : list[list[str]]
        Per-parent lists of symbolic feature expressions.
    combiner : str
        One of ``'add'``, ``'mul'``, ``'concat'``.

    Returns
    -------
    list[str]
        Combined symbolic expressions.
    """
    if combiner == "add":
        op = "+"
        neutral = "0"
    elif combiner == "mul":
        op = "*"
        neutral = "1"
    elif combiner == "concat":
        out = []
        for l in inputs:
            out.extend(l)
        return out
    else:
        raise ValueError(combiner)

    n = max(len(l) for l in inputs)

    padded = []
    for l in inputs:
        if len(l) == n:
            padded.append(l)
        else:
            pad = [neutral] * (n - len(l))
            padded.append(pad + l)

    result = []
    for j in range(n):
        expr = padded[0][j]
        for i in range(1, len(padded)):
            expr = f"({expr}) {op} ({padded[i][j]})"
        result.append(expr)

    return result


def apply_operation(out, node):
    """Apply a node's operation symbolically to the feature expressions.

    Parameters
    ----------
    out : list[str]
        Current list of symbolic feature expressions entering the node.
    node : Node or SymbolicNode
        The DAG node whose operation should be applied.

    Returns
    -------
    list[str]
        Updated list of symbolic expressions after the operation.
    """
    # String dispatch avoids fragile isinstance chains across brick class hierarchies.
    op = node.operation
    name = op.__class__.__name__

    if name == "Identity":
        return out
    if name == "Negate":
        return [f"-({x})" for x in out]
    if name == "Inverse":
        # forward: 1 / (X + 1e-8)  -- the epsilon is printed, not hidden
        return [f"1/(({x})+1e-08)" for x in out]
    if name == "SumFeatures":
        if len(out) == 0:
            return ["0"]
        return [f"({'+'.join(out)})"]
    if name == "SelectFeatures":
        idx = node.hp.get("feature_indices", None)
        if idx is None:
            return out
        if any(i >= len(out) for i in idx):
            return out
        try:
            return [out[i] for i in idx]
        except IndexError:
            return out
    if name == "ConstantBrick":
        value = getattr(op, "value", None)
        if value is not None and hasattr(value, "item"):
            value = value.item()
        elif value is None:
            value = 0.0
        return [str(round(value, 6)) if LEGACY_FORMAT else coef(value, "")]
    if name == "Power":
        exp = getattr(op, "exponent", 2.0)
        if hasattr(exp, "item"):
            exp = exp.item()
        exp = float(exp)
        if exp.is_integer():
            exp = int(exp)
        # forward: torch.pow(|X| + 1e-30, p) * sign(X).  For odd integer p this
        # equals X**p, but for even or fractional p it does NOT -- printing
        # "(x)**2" where sign(x)*|x|**2 was evaluated is a different function
        # on negative inputs, so the sign form is printed verbatim.
        return [f"sign({x})*(abs({x})+1e-30)**{exp}" for x in out]
    if name == "Sqrt":
        # forward: sqrt(clamp(X, min=0))
        return [f"sqrt(maximum({x}, 0))" for x in out]
    if name == "Ln":
        # forward: where(X > 0, log(X + 1e-30), X)  -- the pass-through for
        # non-positive X is a different function from ln and is printed.
        return [f"where(({x})>0, ln({x}), {x})" for x in out]
    if name == "Sin":
        return [f"sin({x})" for x in out]
    if name == "Cos":
        return [f"cos({x})" for x in out]
    if name == "Exp":
        # forward: exp(clamp(X, max=20)).  Above 20 the brick returns a
        # CONSTANT exp(20)=4.8517e8, so "exp(x)" is a different function
        # wherever the input exceeds 20 (e.g. remote-sensing DN values).
        return [f"exp(minimum({x}, 20))" for x in out]
    if name == "ExpAffine":
        a = getattr(op, "a", 1.0)
        if hasattr(a, "item"):
            a = float(a.item())
        return [f"exp(({a:.6g})*({x}))" for x in out]
    if name == "ChannelBoost":
        if len(out) == 0:
            return out
        if len(out) == 1:
            # ChannelBoost.forward duplicates the single channel unchanged
            # (`derived = X[..., 0:1]`); the mode is NOT applied when there is
            # only one channel. Applying it here rendered (x)+(x), (x)-(x),
            # (x)*(x) or (x)/(x) -- verified wrong for all four modes against
            # forward() -- turning correct models into formulas that score
            # R^2 < 0 for both the verifier and any human reading the string.
            return out + [out[0]]
        a, b = out[0], out[-1]
        mode = getattr(op, "mode", "add")
        if mode == "add":
            derived = f"({a})+({b})"
        elif mode == "sub":
            derived = f"({a})-({b})"
        elif mode == "mul":
            derived = f"({a})*({b})"
        elif mode == "div":
            derived = f"({a})/({b})"
        else:
            derived = f"({a})+({b})"
        return out + [derived]

    return out


def graph_to_formula(adj_matrix, X, nodes, channel=None, parse_sympy=True):
    """Extract a symbolic formula from the DAG.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of the DAG.
    X : np.ndarray
        Array of feature names.
    nodes : list
        List of DAG operation nodes.
    channel : int or None, default=None
        Which output channel to extract the formula for.
        If None, defaults to channel 0 (legacy behaviour).
    parse_sympy : bool, default=True
        If True, parse the output string into a SymPy expression. If False,
        return the raw expression string for speed.

    Returns
    -------
    sympy.Expr or str
        Symbolic expression for the requested channel.
    """
    n = adj_matrix.shape[0]

    d = X.shape[-1]
    out_dict = {}
    out_dict[0] = [f"{i}" for i in X]

    for i in range(1, n):
        parents = [j for j in range(i) if adj_matrix[j, i] == 1]
        inputs = [out_dict[j] for j in parents]

        out = op_tensors(inputs, nodes[i].combiner)
        out = apply_operation(out, nodes[i])
        out_dict[i] = out

    selected = channel if channel is not None else 0
    if selected >= len(out_dict[n - 1]):
        selected = 0
    expr_str = out_dict[n - 1][selected]
    return sympify(expr_str) if parse_sympy else format_formula_string(expr_str)


def graph_to_all_formulas(adj_matrix, X, nodes, parse_sympy=True):
    """Extract symbolic formulas for ALL output channels of the DAG.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of the DAG.
    X : np.ndarray
        Array of feature names.
    nodes : list
        List of DAG operation nodes.
    parse_sympy : bool, default=True
        If True, parse each channel expression into SymPy. If False, return
        raw expression strings (much faster for large DAGs).

    Returns
    -------
    list[sympy.Expr | str]
        One expression per output channel.
    """
    n = adj_matrix.shape[0]

    out_dict = {}
    out_dict[0] = [f"{i}" for i in X]

    for i in range(1, n):
        parents = [j for j in range(i) if adj_matrix[j, i] == 1]
        inputs = [out_dict[j] for j in parents]

        out = op_tensors(inputs, nodes[i].combiner)
        out = apply_operation(out, nodes[i])
        out_dict[i] = out

    formulas = []
    for elem in out_dict[n - 1]:
        if parse_sympy:
            try:
                formulas.append(sympify(elem))
            except Exception:
                formulas.append(None)
        else:
            formulas.append(format_formula_string(elem))
    return formulas


# ══════════════════════════════════════════════════════════════════════════════
#  SYMPY EXPRESSION → MINI DAG
# ══════════════════════════════════════════════════════════════════════════════

def expr_to_mini_dag(expr, input_names, max_nodes=None):
    """Compile a SymPy expression into a minimal, valid Dragon DAG.

    Parameters
    ----------
    expr : sympy.Expr
        The symbolic expression to compile.
    input_names : list[str]
        Feature names matching the input tensor columns.
    max_nodes : int or None, default=None
        Maximum number of nodes allowed in the resulting DAG.
        If the compiled expression would exceed this limit, a
        ``ValueError`` is raised so the caller can skip the
        mini-dag replacement.

    Returns
    -------
    AdjMatrix
        A valid Dragon DAG that evaluates to the given expression.
    """
    nodes = []
    edges = []
    cache = {}

    def _check_budget():
        if max_nodes is not None and len(nodes) >= max_nodes:
            raise ValueError(
                f"expr_to_mini_dag: expression requires more than "
                f"{max_nodes} nodes -- skipping mini-dag conversion."
            )

    root = SymbolicNode(
        combiner="add",
        operation=Identity,
        hp={},
        activation=nn.Identity()
    )
    nodes.append(root)

    def create_symbol_copy(symbol_name):
        _check_budget()
        i = input_names.index(symbol_name)
        node = SymbolicNode(
            combiner="add",
            operation=SelectFeatures,
            hp={"feature_indices": [i]},
            activation=nn.Identity()
        )
        idx = len(nodes)
        nodes.append(node)
        edges.append((0, idx))
        return idx

    def multiply_chain(base, exponent):
        assert isinstance(exponent, int) and exponent > 0
        factors = []
        for _ in range(exponent):
            if isinstance(base, Symbol):
                factor_idx = create_symbol_copy(str(base))
                factors.append(factor_idx)
            else:
                factor_idx = compile_expr(base)
                factors.append(factor_idx)

        current_idx = factors[0]
        for factor_idx in factors[1:]:
            _check_budget()
            mul_node = SymbolicNode("mul", Identity, {}, nn.Identity())
            mul_idx = len(nodes)
            nodes.append(mul_node)
            edges.append((current_idx, mul_idx))
            edges.append((factor_idx, mul_idx))
            current_idx = mul_idx

        return current_idx

    def compile_expr(e):
        if not isinstance(e, Symbol) and e in cache:
            return cache[e]

        if isinstance(e, (int, float, Integer, Rational, Float)):
            _check_budget()
            value = float(e)
            const_node = SymbolicNode(
                combiner="add",
                operation=ConstantBrick,
                hp={"value": abs(value)},
                activation=nn.Identity()
            )
            const_idx = len(nodes)
            nodes.append(const_node)
            edges.append((0, const_idx))

            if value < 0:
                _check_budget()
                neg_node = SymbolicNode(
                    combiner="add",
                    operation=Negate,
                    hp={},
                    activation=nn.Identity()
                )
                neg_idx = len(nodes)
                nodes.append(neg_node)
                edges.append((const_idx, neg_idx))
                cache[e] = neg_idx
                return neg_idx
            else:
                cache[e] = const_idx
                return const_idx

        if isinstance(e, Symbol):
            idx = create_symbol_copy(str(e))
            return idx

        if isinstance(e, Add):
            children = [compile_expr(a) for a in e.args]
            _check_budget()
            node = SymbolicNode("add", Identity, {}, nn.Identity())
            idx = len(nodes)
            nodes.append(node)
            for c in children:
                edges.append((c, idx))
            cache[e] = idx
            return idx

        if isinstance(e, Mul):
            negate = False
            children_expr = []
            for a in e.args:
                if a == -1:
                    negate = True
                else:
                    children_expr.append(a)

            children = [compile_expr(a) for a in children_expr]
            _check_budget()
            node = SymbolicNode("mul", Identity, {}, nn.Identity())
            idx = len(nodes)
            nodes.append(node)
            for c in children:
                edges.append((c, idx))

            out = idx
            if negate:
                _check_budget()
                neg_node = SymbolicNode("add", Negate, {}, nn.Identity())
                neg_idx = len(nodes)
                nodes.append(neg_node)
                edges.append((out, neg_idx))
                out = neg_idx

            cache[e] = out
            return out

        if isinstance(e, Pow):
            base, exponent = e.args

            if exponent == -1:
                child = compile_expr(base)
                _check_budget()
                node = SymbolicNode("add", Inverse, {}, nn.Identity())
                idx = len(nodes)
                nodes.append(node)
                edges.append((child, idx))
                cache[e] = idx
                return idx

            elif isinstance(exponent, (int, Integer)) and exponent > 0:
                idx = multiply_chain(base, int(exponent))
                cache[e] = idx
                return idx

            elif isinstance(exponent, (int, Integer)) and exponent < 0:
                pos_exp = -int(exponent)
                idx = multiply_chain(base, pos_exp)
                _check_budget()
                inv_node = SymbolicNode("add", Inverse, {}, nn.Identity())
                inv_idx = len(nodes)
                nodes.append(inv_node)
                edges.append((idx, inv_idx))
                cache[e] = inv_idx
                return inv_idx

            else:
                raise NotImplementedError(f"Unsupported exponent: {e}")

        raise NotImplementedError(f"Unsupported expr: {e}")

    output_expr_idx = compile_expr(expr)

    _check_budget()
    output_node = SymbolicNode(
        combiner="add",
        operation=Identity,
        hp={},
        activation=nn.Identity()
    )
    output_idx = len(nodes)
    nodes.append(output_node)
    edges.append((output_expr_idx, output_idx))

    n = len(nodes)
    matrix = np.zeros((n, n), dtype=int)
    for i, j in edges:
        assert i < j, f"Non topological edge {i}->{j}"
        matrix[i, j] = 1

    dag = AdjMatrix(nodes, matrix)
    dag.set(input_shape=(len(input_names),))
    return dag


# ══════════════════════════════════════════════════════════════════════════════
#  OLS FORMULA FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

_U_WRAP = {
    'id'  : lambda f: f,
    'sq'  : lambda f: f"({f})**2",
    'sqrt': lambda f: f"sqrt(abs({f}))",
    'inv' : lambda f: f"1/({f})",
    'neg' : lambda f: f"-({f})",
    'log' : lambda f: f"log(abs({f}))",
}

_LINK_WRAP = {
    'id'  : lambda s: s,
    'log' : lambda s: f"exp({s})",
    'sqrt': lambda s: f"({s})**2",
    'sq'  : lambda s: f"sqrt(abs({s}))",
    'inv' : lambda s: f"1/({s})",
    'cbrt': lambda s: f"({s})**3",
}


def format_nested(nested: dict, formulas: list, min_weight: float = 1e-4) -> str:
    """Format a nested-OLS result as a human-readable formula string.

    Parameters
    ----------
    nested : dict
        Must contain ``link``, ``unaries``, ``w``, ``b``, ``valid`` keys.
    formulas : list[str]
        Channel formula strings.
    min_weight : float
        Minimum absolute weight to include a term.
    """
    link    = nested["link"]
    unaries = nested["unaries"]
    w       = nested["w"]
    b       = nested["b"]
    valid   = nested["valid"]
    valid_w = [(i, wi) for i, wi in enumerate(w)
               if i < len(valid) and valid[i] and abs(wi) >= min_weight and i < len(formulas)]
    if not valid_w:
        return f"{coef(b, '.4f')}"
    used_u = {unaries[i] for i, _ in valid_w}
    if link == "log" and used_u == {"log"}:
        terms = [f"({formulas[i]})**({coef(wi, '.4f')})" for i, wi in valid_w]
        return f"{coef(np.exp(b), '.4e')} * " + " * ".join(terms)
    if link == "log":
        inner = " ".join(f"{coef(wi, '+.4f')}*{_U_WRAP[unaries[i]](formulas[i])}" for i, wi in valid_w)
        return f"{coef(np.exp(b), '.4e')} * exp({inner})"
    if link == "inv" and used_u == {"inv"}:
        denom = " ".join(f"{coef(wi, '+.4f')}/({formulas[i]})" for i, wi in valid_w)
        if abs(b) > min_weight:
            denom += f" {coef(b, '+.4f')}"
        return f"1 / ({denom})"
    inner = " ".join(f"{coef(wi, '+.4f')}*{_U_WRAP[unaries[i]](formulas[i])}" for i, wi in valid_w)
    if abs(b) > min_weight:
        inner += f" {coef(b, '+.4f')}"
    return _LINK_WRAP[link](inner)


def format_rational(rational: dict, formulas: list, valid_idx_global,
                    min_weight: float = 1e-4) -> str:
    """Format a poly-rational OLS result as a human-readable formula string.

    Parameters
    ----------
    rational : dict
        Must contain ``a``, ``a0``, ``b``, ``b0``, ``kept_a``, ``kept_b``,
        ``monomials`` keys.
    formulas : list[str]
        Channel formula strings.
    valid_idx_global : np.ndarray
        Global indices of valid channels.
    min_weight : float
        Minimum absolute weight to include a term.
    """
    a, a0  = rational["a"], rational["a0"]
    b, b0  = rational["b"], rational["b0"]
    kept_a = rational["kept_a"]
    kept_b = rational["kept_b"]
    monos  = rational["monomials"]

    def mono_str(mono):
        counts = Counter(mono)
        parts  = []
        for li, exp in counts.items():
            gi    = int(valid_idx_global[li])
            f_str = formulas[gi] if gi < len(formulas) else f"ch{gi}"
            parts.append(f"({f_str})" if exp == 1 else f"({f_str})**{exp}")
        return "*".join(parts) if parts else "1"

    num = [f"{coef(a[k], '+.4f')}*{mono_str(m)}" for k, m in enumerate(monos)
           if kept_a[k] and abs(a[k]) >= min_weight]
    if abs(a0) >= min_weight or not num:
        num.append(f"{coef(a0, '+.4f')}")
    den = [f"{coef(b[k], '+.4f')}*{mono_str(m)}" for k, m in enumerate(monos)
           if kept_b[k] and abs(b[k]) >= min_weight]
    den.append(f"{coef(b0, '+.4f')}")
    return f"({' '.join(num)}) / ({' '.join(den)})"
