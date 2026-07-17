"""DAG introspection utilities for DRAGON.

Provides ``DAGInspector`` for summarising DAG structure, generating
text tables, Graphviz SVGs, and JSON-serialisable data.
"""

from __future__ import annotations


class DAGInspector:
    """Stateless DAG introspection utilities (all methods are static)."""

    @staticmethod
    def summarize(adj, nodes) -> tuple:
        """Return ``(ops_used_str, has_const_str, dag_size, dag_text, dag_svg, dag_data)``."""
        try:
            from dragon.utils.plot_functions import str_operations
            descs_raw = str_operations(nodes)
        except Exception:
            descs_raw = [[str(n)] for n in nodes]

        op_names = [str(row[1]) if len(row) >= 2 else str(row[0]) for row in descs_raw]
        op_disp = op_names[1:] if op_names else []
        ops_used_str = ", ".join(op_disp) if op_disp else "—"
        const_count = sum(1 for n in op_disp if "const" in n.lower())
        has_const_str = f"yes ({const_count})" if const_count else "no (0)"
        dag_size = int(getattr(adj, "shape", (0,))[0]) if hasattr(adj, "shape") else len(nodes)

        descs = []
        for row in descs_raw:
            try:
                descs.append(" | ".join(str(x) for x in row[:-1]) if len(row) > 1 else str(row[0]))
            except Exception:
                descs.append("?")

        dag_text = DAGInspector._make_text(adj, descs)
        dag_svg = DAGInspector._make_svg(adj, descs)
        dag_data = DAGInspector._make_data(adj, descs)
        return ops_used_str, has_const_str, dag_size, dag_text, dag_svg, dag_data

    @staticmethod
    def _make_text(adj, descs) -> str:
        try:
            n = adj.shape[0]
            parents = [[str(j) for j in range(n) if adj[j, i]] for i in range(n)]
            children = [[str(j) for j in range(n) if adj[i, j]] for i in range(n)]
            idx_w = max(3, len(str(n - 1)) + 2)
            child_w = max(8, max((len(",".join(c)) for c in children), default=1) + 2)
            par_w = max(8, max((len(",".join(p)) for p in parents), default=1) + 2)
            header = (f"{'Idx'.ljust(idx_w)}| {'Children'.ljust(child_w)}| "
                      f"{'Parents'.ljust(par_w)}| Description")
            sep = "-" * (len(header) + 10)
            lines = [sep, header, sep]
            for i in range(n):
                c = ",".join(children[i]) or "-"
                p = ",".join(parents[i]) or "-"
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
                fill = "#3b6d11" if i == 0 else "#ffa600"
                fc = "#ECECEC" if i == 0 else "#1a1a1a"
                label = f"[{i}] {d}".replace("\\", "\\\\").replace('"', '\\"')
                G.node(str(i), label=label, fillcolor=fill, color="black", fontcolor=fc)
            n = adj.shape[0]
            for i in range(n):
                for j in range(n):
                    if adj[i, j]:
                        G.edge(str(i), str(j))
            svg = G.pipe(format="svg").decode("utf-8", errors="ignore")
            ix = svg.find("<svg")
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
