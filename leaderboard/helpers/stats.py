# stats.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_svg(fig) -> str:
    """Serialize a Matplotlib figure to a self-contained inlinable SVG string."""
    import io as _io
    import matplotlib.pyplot as _plt
    buf = _io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    _plt.close(fig)
    s = buf.getvalue()
    i = s.find("<svg")
    return s[i:] if i >= 0 else s


def _plotly_to_html(fig) -> str:
    """Render a Plotly figure as a self-contained inlinable HTML snippet."""
    import plotly.io as _pio
    return _pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def _matplotlib_to_svg(fig) -> str:
    """Render a Matplotlib figure as an inline SVG string."""
    svg = _fig_to_svg(fig)
    return f"<div style='max-width:100%;overflow:auto'>{svg}</div>"


def _make_dragon_landscape_svg(comp_csv_path: str):
    """Return Plotly or Matplotlib HTML snippet (3-panel: scatter+best, histogram, convergence).

    Returns None on failure.
    """
    if not os.path.exists(comp_csv_path):
        return None

    df = pd.read_csv(comp_csv_path)
    if "Loss" not in df.columns:
        return None

    df["Loss"] = df["Loss"].replace([np.inf, -np.inf], np.nan)
    df["BestSoFar"] = df["Loss"].expanding().min()
    finite = df["Loss"].dropna()
    if finite.empty:
        return None
    best = float(finite.min())

    elapsed = None
    if "TimeStamp" in df.columns:
        try:
            ts = pd.to_datetime(df["TimeStamp"], errors="coerce")
            if ts.notna().any():
                elapsed = (ts - ts.iloc[0]).dt.total_seconds() / 60.0
        except Exception:
            elapsed = None
    x_conv = elapsed if elapsed is not None else df["Idx"]
    x_conv_label = "Wall-clock time (min)" if elapsed is not None else "Iteration index"

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Loss landscape (all evaluations)",
                "Loss distribution",
                f"Convergence ({x_conv_label.split(' ')[0].lower()})",
            ),
            horizontal_spacing=0.08,
        )
        fig.add_trace(go.Scattergl(
            x=df["Idx"], y=df["Loss"], mode="markers",
            marker=dict(size=4, color="#185fa5", opacity=0.4),
            name="Individual loss",
            hovertemplate="iter=%{x}<br>loss=%{y:.3e}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Idx"], y=df["BestSoFar"], mode="lines",
            line=dict(color="#a32d2d", width=1.8),
            name="Best so far",
            hovertemplate="iter=%{x}<br>best=%{y:.3e}<extra></extra>",
        ), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1, title_text="Loss (1−|corr|)")
        fig.update_xaxes(title_text="Iteration", row=1, col=1)

        fig.add_trace(go.Histogram(
            x=finite, nbinsx=60, marker_color="#3b6d11",
            opacity=0.8, name="Loss histogram",
            hovertemplate="loss∈[%{x}]<br>count=%{y}<extra></extra>",
        ), row=1, col=2)
        fig.add_vline(x=best, line=dict(color="#a32d2d", dash="dash"),
                      annotation_text=f"min = {best:.3e}",
                      annotation_position="top right", row=1, col=2)
        fig.update_xaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.add_trace(go.Scatter(
            x=x_conv, y=df["BestSoFar"], mode="lines",
            line=dict(color="#a32d2d", width=1.8), name="Best so far",
            hovertemplate=(x_conv_label + "=%{x:.2f}<br>best=%{y:.3e}<extra></extra>"),
            showlegend=False,
        ), row=1, col=3)
        fig.update_yaxes(type="log", title_text="Best loss", row=1, col=3)
        fig.update_xaxes(title_text=x_conv_label, row=1, col=3)

        fig.update_layout(
            template="plotly_white",
            height=460, margin=dict(l=60, r=30, t=80, b=60),
            title=dict(
                text=f"DragonSR search landscape — {len(df)} evaluations · best = {best:.3e}",
                font=dict(size=13)),
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.14, font=dict(size=11)),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(size=11, color="#1a1a1a"),
        )
        return _plotly_to_html(fig)
    except Exception:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            fig.patch.set_facecolor('white')

            axes[0].scatter(df["Idx"], df["Loss"], s=16, c="#185fa5", alpha=0.4)
            axes[0].plot(df["Idx"], df["BestSoFar"], color="#a32d2d", linewidth=1.8)
            axes[0].set_yscale('log')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss (1−|corr|)')
            axes[0].set_title('Loss landscape (all evaluations)')
            axes[0].grid(True, which='both', linestyle='--', alpha=0.25)

            axes[1].hist(finite, bins=60, color="#3b6d11", alpha=0.8)
            axes[1].axvline(best, color="#a32d2d", linestyle='--')
            axes[1].set_xlabel('Loss')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Loss distribution')
            axes[1].grid(True, linestyle='--', alpha=0.25)
            axes[1].text(0.95, 0.95, f'min = {best:.3e}', transform=axes[1].transAxes,
                         ha='right', va='top', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            axes[2].plot(x_conv, df["BestSoFar"], color="#a32d2d", linewidth=1.8)
            axes[2].set_yscale('log')
            axes[2].set_xlabel(x_conv_label)
            axes[2].set_ylabel('Best loss')
            axes[2].set_title(f'Convergence ({x_conv_label.split(" ")[0].lower()})')
            axes[2].grid(True, which='both', linestyle='--', alpha=0.25)

            fig.suptitle(f"DragonSR search landscape — {len(df)} evaluations · best = {best:.3e}", fontsize=13)
            return _matplotlib_to_svg(fig)
        except Exception:
            return None



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
        landscape_svg = _make_dragon_landscape_svg(comp_csv)
    except Exception:
        pass
    return actual_T, loss_history, landscape_svg

def _compute_pysr_scores(hof: list) -> list:
    """Add a 'score' field (parsimony score) to each HoF entry. Mutates and returns the list."""
    if not hof:
        return hof
    try:
        s         = sorted(hof, key=lambda h: (h.get("complexity") or 0))
        prev_loss = None; prev_cplx = None
        for h in s:
            c = h.get("complexity"); l = h.get("loss")
            if (prev_loss is None or prev_cplx is None
                    or l is None or c is None
                    or l <= 0 or prev_loss <= 0 or c == prev_cplx):
                h["score"] = None
            else:
                try:
                    h["score"] = float(-(np.log(l) - np.log(prev_loss)) / (c - prev_cplx))
                except Exception:
                    h["score"] = None
            prev_loss = l if (l is not None and l > 0) else prev_loss
            prev_cplx = c if c is not None else prev_cplx
        return s
    except Exception:
        return hof


def _make_pysr_pareto_svg(hof: list):
    """Plotly Pareto frontier (complexity vs loss, log-y) with hover tooltips."""
    try:
        if not hof:
            return None
        import plotly.graph_objects as go
        pts = [(h.get("complexity"), h.get("loss"),
                h.get("formula") or "", h.get("score"))
               for h in hof
               if h.get("complexity") is not None and h.get("loss") is not None
               and h.get("loss") > 0]
        if not pts:
            return None
        pts.sort(key=lambda p: p[0])
        cs = [p[0] for p in pts]; ls = [p[1] for p in pts]
        forms = [p[2] for p in pts]; scores = [p[3] for p in pts]
        hov = [
            f"complexity={c}<br>loss={l:.3e}<br>"
            f"score={('—' if s is None else f'{s:.3f}')}<br>"
            f"expr: {f if len(f) < 60 else f[:57] + '...'}"
            for c, l, f, s in zip(cs, ls, forms, scores)
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cs, y=ls, mode="lines+markers",
            line=dict(color="#185fa5", width=1.5),
            marker=dict(size=8, color="#185fa5", line=dict(width=1, color="#0a3460")),
            text=hov, hoverinfo="text", name="Pareto frontier",
        ))
        fig.update_layout(
            template="plotly_white",
            height=420, margin=dict(l=60, r=30, t=60, b=55),
            title=dict(text="PySR Pareto frontier (complexity vs loss)", font=dict(size=13)),
            xaxis=dict(title="Complexity"),
            yaxis=dict(title="Loss (PySR MSE)", type="log"),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(size=11, color="#1a1a1a"),
        )
        return _plotly_to_html(fig)
    except Exception:
        return None


def _make_pysr_tree_svg(formula_str: str):
    """Parse the PySR formula string via sympy and render the binary AST as a Graphviz SVG."""
    try:
        if not formula_str or formula_str.strip() in ("N/A", ""):
            return None
        import shutil as _shutil
        if _shutil.which("dot") is None:
            return None
        import sympy as sp
        import graphviz as _gv

        cleaned = formula_str
        if "=" in cleaned and cleaned.split("=")[0].strip().isidentifier():
            cleaned = cleaned.split("=", 1)[1].strip()
        try:
            expr = sp.sympify(cleaned, evaluate=False)
        except Exception:
            try:
                from sympy.parsing.sympy_parser import (
                    parse_expr, standard_transformations,
                    implicit_multiplication_application, convert_xor)
                tr   = (standard_transformations
                        + (implicit_multiplication_application, convert_xor))
                expr = parse_expr(cleaned, evaluate=False, transformations=tr)
            except Exception:
                return None

        G = _gv.Digraph(
            format="svg",
            node_attr={"shape": "ellipse", "fontsize": "11",
                       "fontname": "monospace", "style": "filled"},
            graph_attr={"rankdir": "TB", "bgcolor": "transparent",
                        "nodesep": "0.20", "ranksep": "0.30"},
        )
        counter = [0]

        def add(node):
            i   = counter[0]; counter[0] += 1
            nid = f"n{i}"
            if node.is_Atom:
                lbl  = str(node); fill = "#3b6d11"; fc = "#ECECEC"
            else:
                lbl  = type(node).__name__
                lbl  = {"Add": "+", "Mul": "×", "Pow": "^",
                        "exp": "exp", "log": "log",
                        "sin": "sin", "cos": "cos",
                        "sqrt": "√", "Abs": "|·|"}.get(lbl, lbl)
                fill = "#ffa600"; fc = "#1a1a1a"
            G.node(nid, label=lbl, fillcolor=fill, fontcolor=fc, color="black")
            for child in getattr(node, "args", ()):
                cid = add(child)
                G.edge(nid, cid)
            return nid

        add(expr)
        svg = G.pipe(format="svg").decode("utf-8", errors="ignore")
        i   = svg.find("<svg")
        return svg[i:] if i >= 0 else svg
    except Exception:
        return None


def _make_pysr_tree_data(formula_str: str):
    """Parse a PySR formula string via sympy and return a JSON-serialisable node list."""
    try:
        if not formula_str or formula_str.strip() in ("N/A", ""):
            return None
        import sympy as sp
        cleaned = formula_str
        if "=" in cleaned and cleaned.split("=")[0].strip().isidentifier():
            cleaned = cleaned.split("=", 1)[1].strip()
        try:
            expr = sp.sympify(cleaned, evaluate=False)
        except Exception:
            try:
                from sympy.parsing.sympy_parser import (
                    parse_expr, standard_transformations,
                    implicit_multiplication_application, convert_xor)
                tr   = (standard_transformations
                        + (implicit_multiplication_application, convert_xor))
                expr = parse_expr(cleaned, evaluate=False, transformations=tr)
            except Exception:
                return None

        _LABELS = {"Add": "+", "Mul": "\u00d7", "Pow": "^",
                   "exp": "exp", "log": "log",
                   "sin": "sin", "cos": "cos",
                   "sqrt": "\u221a", "Abs": "|x|"}
        nodes = []

        def _visit(node):
            idx     = len(nodes)
            is_atom = node.is_Atom
            if is_atom and node.is_Number:
                try:
                    lbl = f"{float(node):.4g}"
                except Exception:
                    lbl = str(node)
            elif is_atom:
                lbl = str(node)
            else:
                lbl  = _LABELS.get(type(node).__name__, type(node).__name__)
            kind = ("var" if (is_atom and not node.is_Number)
                    else "const" if is_atom else "op")
            nodes.append({"id": idx, "label": lbl, "kind": kind, "children": []})
            for child in getattr(node, "args", ()):
                child_idx = _visit(child)
                nodes[idx]["children"].append(child_idx)
            return idx

        _visit(expr)
        if not nodes:
            return None

        bin_nodes: list = []

        def _binarize(old_idx: int) -> int:
            nd     = nodes[old_idx]
            new_ch = [_binarize(c) for c in nd["children"]]
            if len(new_ch) <= 2:
                idx = len(bin_nodes)
                bin_nodes.append({"id": idx, "label": nd["label"],
                                   "kind": nd["kind"], "children": new_ch})
                return idx
            chain = new_ch[-1]
            for c in reversed(new_ch[:-1]):
                idx = len(bin_nodes)
                bin_nodes.append({"id": idx, "label": nd["label"],
                                   "kind": nd["kind"], "children": [c, chain]})
                chain = idx
            return chain

        _binarize(0)
        return bin_nodes if bin_nodes else None
    except Exception:
        return None


def _make_formula_stats_svg(per_cell):
    """Per-formula across-method×runs interactive stats panel (2 subplots).

    `per_cell` is a list of {method, run, loss, runtime} dicts.
    """
    try:
        if not per_cell:
            return None
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from collections import OrderedDict
        groups  = OrderedDict()
        for d in per_cell:
            m = d.get("method") or "?"
            groups.setdefault(m, []).append(d)
        methods = list(groups.keys())
        if not methods:
            return None

        palette = ["#185fa5", "#a32d2d", "#3b6d11", "#ba7517", "#6e3a8a",
                   "#117a8b", "#c2185b", "#7d6608", "#0a3460", "#4d4d4d"]
        colour  = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Best vs median loss per method",
                "Loss vs runtime (cost / quality)",
            ),
            horizontal_spacing=0.18,
        )

        names_x, best_vals, med_vals, bar_cols = [], [], [], []
        for m in methods:
            vs = [d["loss"] for d in groups[m]
                  if d.get("loss") is not None and np.isfinite(d["loss"]) and d["loss"] > 0]
            if not vs:
                continue
            names_x.append(m); best_vals.append(min(vs))
            med_vals.append(float(np.median(vs))); bar_cols.append(colour[m])
        if names_x:
            fig.add_trace(go.Bar(
                x=names_x, y=best_vals, name="best",
                marker=dict(color=bar_cols, line=dict(width=0.6, color="#222")),
                hovertemplate="%{x}<br>best 1−R² = %{y:.3e}<extra></extra>",
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=names_x, y=med_vals, name="median",
                marker=dict(color=bar_cols, opacity=0.45, line=dict(width=0.6, color="#222")),
                hovertemplate="%{x}<br>median 1−R² = %{y:.3e}<extra></extra>",
                showlegend=False,
            ), row=1, col=1)

        for m in methods:
            xs = [d.get("runtime") for d in groups[m]
                  if d.get("runtime") is not None
                  and d.get("loss") is not None and np.isfinite(d["loss"]) and d["loss"] > 0]
            ys = [d["loss"] for d in groups[m]
                  if d.get("runtime") is not None
                  and d.get("loss") is not None and np.isfinite(d["loss"]) and d["loss"] > 0]
            if not xs:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=m,
                marker=dict(color=colour[m], size=10, line=dict(width=0.6, color="#222")),
                hovertemplate=f"{m}<br>runtime=%{{x:.1f}}s<br>1−R²=%{{y:.3e}}<extra></extra>",
                legendgroup=m, showlegend=True,
            ), row=1, col=2)

        fig.update_yaxes(type="log", title_text="1−R² (log)", row=1, col=1)
        fig.update_xaxes(title_text="Method", tickangle=-25, row=1, col=1)
        fig.update_xaxes(title_text="Runtime (s)", row=1, col=2)
        fig.update_yaxes(type="log", title_text="1−R² (log)", row=1, col=2)
        fig.update_layout(
            template="plotly_white",
            barmode="group",
            height=460, margin=dict(l=70, r=40, t=90, b=80),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(size=11, color="#1a1a1a"),
            legend=dict(orientation="h", x=0, y=1.14, font=dict(size=10)),
        )
        return _plotly_to_html(fig)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  DAG INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

class DAGInspector:
    """Stateless DAG introspection utilities (all methods are static)."""

    @staticmethod
    def summarize(adj, nodes) -> tuple:
        """Return (ops_used_str, has_const_str, dag_size, dag_text, dag_svg, dag_data)."""
        try:
            from dragon.utils.plot_functions import str_operations
            descs_raw = str_operations(nodes)
        except Exception:
            descs_raw = [[str(n)] for n in nodes]

        op_names = [str(row[1]) if len(row) >= 2 else str(row[0]) for row in descs_raw]
        op_disp  = op_names[1:] if op_names else []
        ops_used_str  = ", ".join(op_disp) if op_disp else "—"
        const_count   = sum(1 for n in op_disp if "const" in n.lower())
        has_const_str = f"yes ({const_count})" if const_count else "no (0)"
        dag_size      = int(getattr(adj, "shape", (0,))[0]) if hasattr(adj, "shape") else len(nodes)

        descs = []
        for row in descs_raw:
            try:
                descs.append(" | ".join(str(x) for x in row[:-1]) if len(row) > 1 else str(row[0]))
            except Exception:
                descs.append("?")

        dag_text = DAGInspector._make_text(adj, descs)
        dag_svg  = DAGInspector._make_svg(adj, descs)
        dag_data = DAGInspector._make_data(adj, descs)
        return ops_used_str, has_const_str, dag_size, dag_text, dag_svg, dag_data

    @staticmethod
    def _make_text(adj, descs) -> str:
        try:
            n        = adj.shape[0]
            parents  = [[str(j) for j in range(n) if adj[j, i]] for i in range(n)]
            children = [[str(j) for j in range(n) if adj[i, j]] for i in range(n)]
            idx_w    = max(3, len(str(n - 1)) + 2)
            child_w  = max(8, max((len(",".join(c)) for c in children), default=1) + 2)
            par_w    = max(8, max((len(",".join(p)) for p in parents),  default=1) + 2)
            header   = (f"{'Idx'.ljust(idx_w)}| {'Children'.ljust(child_w)}| "
                        f"{'Parents'.ljust(par_w)}| Description")
            sep      = "-" * (len(header) + 10)
            lines    = [sep, header, sep]
            for i in range(n):
                c = ",".join(children[i]) or "-"
                p = ",".join(parents[i])  or "-"
                d = (descs[i] if i < len(descs) else "?")[:80]
                lines.append(f"[{str(i).rjust(idx_w-2)}] | {c.ljust(child_w)} | {p.ljust(par_w)} | {d}")
            lines.append(sep)
            return "\n".join(lines)
        except Exception:
            return "\n".join(f"[{i}] {d}" for i, d in enumerate(descs))

    @staticmethod
    def _make_svg(adj, descs) -> str | None:
        try:
            import shutil
            if shutil.which("dot") is None:
                return None
            import graphviz as gv
            G = gv.Digraph(
                format="svg",
                node_attr={"shape": "box", "fontsize": "11",
                           "fontname": "sans-serif", "style": "rounded,filled"},
                graph_attr={"rankdir": "TB", "bgcolor": "transparent",
                            "nodesep": "0.25", "ranksep": "0.35"},
            )
            for i, d in enumerate(descs):
                fill  = "#3b6d11" if i == 0 else "#ffa600"
                fc    = "#ECECEC" if i == 0 else "#1a1a1a"
                label = f"[{i}] {d}".replace("\\", "\\\\").replace('"', '\\"')
                G.node(str(i), label=label, fillcolor=fill, color="black", fontcolor=fc)
            n = adj.shape[0]
            for i in range(n):
                for j in range(n):
                    if adj[i, j]:
                        G.edge(str(i), str(j))
            svg = G.pipe(format="svg").decode("utf-8", errors="ignore")
            ix  = svg.find("<svg")
            return svg[ix:] if ix >= 0 else svg
        except Exception:
            return None

    @staticmethod
    def _make_data(adj, descs) -> list:
        try:
            n = adj.shape[0]
            return [
                {"idx": i,
                 "children": [int(j) for j in range(n) if adj[i, j]],
                 "desc": descs[i] if i < len(descs) else "?"}
                for i in range(n)
            ]
        except Exception:
            return []
