"""
leaderboard.py — DragonSR vs PySR Leaderboard
Runs DragonSR (curriculum search + full OLS pipeline) and PySR on multiple
targets, then outputs dragonfsr_leaderboard_v2 HTML with injected results.

Launch with:  python -u leaderboard.py
"""

# ══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS  — edit this section only
# ══════════════════════════════════════════════════════════════════════════════

# ── Formula IDs — must match the HTML FORMULAS[].id list exactly ─────────────
TARGETS = [
    # ── TEST RUN: only n4 and ndvi ──────────────────────────────────────────────────────
    "n4", "ndvi", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12",
    # # Nguyen benchmarks (synthetic)
    # "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12",
    # # Physics (synthetic)
     "newton", "rydberg", "idealgas", "kepler", "schechter", "bode", "leavitt", "planck", #"hubble",
    # # Remote sensing (from data/6000_points.csv)
    "wi2015", "awei_sh", "bai", "ndvi", "savi", "bsi", "evi2", "mndwi", "vari", "nirv",
]

# ── Runs per target (R1–R5 in the HTML; strategies match HTML pill labels) ────
N_RUNS          = 2              # todo: change back to 2
INIT_STRATEGIES = ["random", "diverse", "xgboost", "warmstart", "adversarial"]

# ── DragonSR search budget ────────────────────────────────────────────────────
# DRAGON_N_ITERATIONS : hard cap on the *total* number of iterations across all
#   complexity levels (curriculum) or for the single flat run.  Each complexity
#   level receives up to DRAGON_T_PER_LEVEL evaluations; once the cumulative
#   total would exceed DRAGON_N_ITERATIONS the remaining budget is trimmed and
#   the search stops early.
DRAGON_N_ITERATIONS   = 10000      # 10000
DRAGON_K_INIT         = 500       # 500
DRAGON_MAX_COMPLEXITY = 10        # 10
DRAGON_T_PER_LEVEL    = 1000       # 1000
DRAGON_LOSS_THRESHOLD = 1e-30   # stop early if loss ≤ this

# ── Post-processing (OLS / nested / poly-rational) complexity control ─────────
# The OLS pipeline (_ols_eval) recombines DAG channels into models whose
# complexity is otherwise UNBOUNDED: nested links, and poly-rational fractions
# with many numerator/denominator terms.  A high-degree rational almost always
# shaves a sliver of MSE off a simpler OLS fit on noisy data and therefore wins
# the plain argmin(loss) — adding complexity that is not justified by the data.
# These two knobs bound that extra complexity:
#   POST_PARSIMONY_REL_TOL : a more complex model wins only if it beats the
#       simplest near-tied candidate by MORE than this RELATIVE MSE margin.
#       0.0   → disabled (plain argmin loss, legacy behaviour).
#       0.02  → a rational/nested must improve MSE by >2 % over OLS to be kept;
#               marginal noise-fitting gains no longer flip the winner.
#       Bounded by construction: the reported loss is at most (1+tol)× the best,
#       so the search fitness signal is changed by at most this margin.
#   POST_COMPLEXITY_MAX    : hard cap on the number of fitted terms a post-
#       processing model may use (None = no cap).  Candidates above the cap are
#       rejected; the simplest candidate is always kept as a fallback.
POST_PARSIMONY_REL_TOL = 0.02
POST_COMPLEXITY_MAX    = None

# ── Search loss kind ─────────────────────────────────────────────────────────
#   SEARCH_LOSS = "mse"   : classical normalised MSE  (default, sensitive to
#                            outliers because residuals are squared).
#               = "huber" : normalised Huber loss     (quadratic for |r|<δ,
#                            linear beyond → noisy/outlier-heavy targets
#                            stop dominating the fitness signal).
#   HUBER_DELTA_FRAC : Huber threshold as a fraction of std(y).  0.1– 0.3 is a
#                      standard range; smaller → more robust.
SEARCH_LOSS      = "huber"
HUBER_DELTA_FRAC = 0.20

# Gaussian noise level applied to y for the +Noise ablation method
# (relative to std(y)).  Only used when method_cfg["add_noise"] is True.
NOISE_STD = 0.01

# ── Smart-parallel operator subsets  ───────────────────────────────────────────
# Used by the "spar" DragonSR method: 4 streams launched in parallel via
# ThreadPoolExecutor, each Mutant_UCB run scoped to a different operator
# subset.  Best loss across the 4 streams wins and is reported as "spar".
#   - all          : every operator (reference)
#   - alg          : pure algebraic (select / unary / power)
#   - alg_trig     : algebraic + trigonometric (sin, cos)
#   - alg_explog   : algebraic + exponential / logarithm (exp, ln)
SPAR_OP_GROUPS = {
    "all":        ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    "alg":        ["select", "unary", "power"],
    "alg_trig":   ["select", "unary", "power", "sin", "cos"],
    "alg_explog": ["select", "unary", "power", "ln", "exp"],
}

# ── DragonSR — single method config  (id must match an HTML METHODS entry) ────

DRAGON_METHOD_CONFIGS = [

    # {
    #     "id":          "allops",
    #     "description": "DragonSR — reference method (all ops, full OLS, var-aug, no noise)",
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"], 
    #     "curriculum":  True,
    #     "parallel_N":  1,
    #     "loss_mode":   "full",
    #     "var_aug":     True,
    #     "add_noise":   False,
    # },
    # {
    #     "id":          "spar",
    #     "description": ("DragonSR — smart-parallel: 4 op-subset streams "
    #                     "(all / alg / alg+trig / alg+exp,ln) run in parallel "
    #                     "via ThreadPoolExecutor; best loss wins."),
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    #     "curriculum":  True,
    #     "parallel_N":  1,
    #     "loss_mode":   "full",
    #     "var_aug":     True,
    #     "add_noise":   False,
    #     "smart_parallel": True,
    # },
    # {
    #     "id":          "boosted_spar",
    #     "description": ("DragonSR — boosted smart-parallel: same 4 op-subset "
    #                     "streams as 'spar' run concurrently, then their "
    #                     "winning predictions ŷ_stream are stacked and "
    #                     "meta-combined via sparse-OLS / nested-OLS / "
    #                     "poly-rational-OLS (whichever fits best)."),
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    #     "curriculum":  True,
    #     "parallel_N":  1,
    #     "loss_mode":   "full",
    #     "var_aug":     True,
    #     "add_noise":   False,
    #     "smart_parallel": True,
    #     "boosted":        True,
    # },
    {
        "id":          "spar_denoise",
        "description": ("DragonSR — smart-parallel + stochastic subsampling "
                        "denoising: each candidate formula is evaluated on a "
                        "fresh random subset of the data at every iteration.  "
                        "Noisy samples never consistently dominate the loss "
                        "signal; their influence is diluted by implicit "
                        "averaging across iterations (SSD)."),
        "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
        "curriculum":  True,
        "parallel_N":  1,
        "loss_mode":   "full",
        "var_aug":     True,
        "add_noise":   True,
        "smart_parallel":  True,
        # Self-validating GPR-Matérn smoother applied to the noisy y BEFORE
        # the stoch_sub in-loop step.  Estimates the noise floor (σ̂²) by
        # marginal likelihood and only denoises when a calibration check
        # confirms it helps; on fast-varying targets (e.g. ndvi) it falls back
        # to raw y automatically.  y_noisy → auto-denoise → y_clean.
        "pre_denoise_method": "auto",
        # subsample_ratio: fraction of the dataset drawn without replacement
        # at each loss evaluation.  0.5 = 50 % of rows per call.
        "denoise_method":  "stoch_sub",
        "subsample_ratio": 0.1,
        # MC-Dropout uncertainty weighting: train a small MLP on (X, y_noisy),
        # run 50 stochastic forward passes to estimate per-sample aleatoric
        # uncertainty, downweight noisy/uncertain points in both subsampling
        # and the final SR loss.  Works on top of stoch_sub: clean points
        # are drawn more often AND count more in the loss.
        "mc_dropout":           False,
        "mc_dropout_n_forward": 50,
        "mc_dropout_p":         0.15,
        "mc_dropout_epochs":    300,
        "mc_dropout_hidden":    64,
    },
    # # ── Ablations of the reference method (allops) ────────────────────────
    # {
    #     "id":          "noolsratn",
    #     "description": "Ablation of allops — NO OLS / rat / nested (channel-only loss)",
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos"],
    #     "curriculum":  True,
    #     "parallel_N":  1,
    #     "loss_mode":   "channel",
    #     "var_aug":     True,
    #     "add_noise":   False,
    # },
    # {
    #     "id":          "allops_const",
    #     "description": "DragonSR — +ConstantBrick (Adam-optimized constants)",
    #     "operators":   ["select", "unary", "power", "ln", "exp", "sin", "cos", "const"],
    #     "curriculum":  True,
    #     "parallel_N":  1,
    #     "loss_mode":   "channel",
    #     "var_aug":     True,
    #     "add_noise":   False,
    #     "optimize_constants": True,
    # }
]

# ── PySR config ───────────────────────────────────────────────────────────────
PYSR_NITERATIONS     = 4000   # 4000
PYSR_POPULATIONS     = 5      # 5
PYSR_POPULATION_SIZE = 120     # 120
PYSR_MAXSIZE         = 30     # 30
PYSR_JULIA_PROJECT   = "/Users/elyaschikhaoui/Desktop/dragon/.dragonenv/julia_env"
# PySR's operators:

PYSR_BINARY_OPERATORS = ["+", "-", "*", "/"]
PYSR_UNARY_OPERATORS  = ["log", "exp", "sin", "cos", "sqrt", "abs"]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/6000_points.csv"   # remote sensing CSV
OUTPUT_DIR     = "leaderboard_runs"
HTML_OUTPUT    = "dragonfsr_leaderboard_v2.html"
LOG_SUFFIX     = "_found_formulas.txt"

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
N_TOP_FEATURES = 10
N_SYNTH_SAMPLES = 6000   # samples for physics/Nguyen synthetic datasets


# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys

# Allow overriding Julia config via environment variables (cluster usage)
if os.environ.get("PYSR_JULIA_PROJECT"):
    PYSR_JULIA_PROJECT = os.environ["PYSR_JULIA_PROJECT"]
import time
import json
import shutil
import subprocess
import traceback
import multiprocessing as mp
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from pathlib import Path

# dragon path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib.dragon
sys.modules['dragon'] = sys.modules['lib.dragon']

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

from dragon.search_space.bricks_variables import operations_var
from dragon.search_space.base_variables import CatVar, Constant, ArrayVar
from dragon.search_space.dag_encoding import AdjMatrix, SymbolicNode
from dragon.search_space.dag_variables import HpVar, EvoDagVariable
from dragon.search_operators.base_neighborhoods import (
    CatInterval, ConstantInterval, ArrayInterval, FloatInterval,
)
from dragon.search_operators.dag_neighborhoods import EvoDagInterval, HpInterval
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import (
    SelectFeatures, Inverse, Negate, Power, SumFeatures, ConstantBrick,
    Ln, Sin, Cos, Exp
)

from dragon.search_algorithm.mutant_ucb import Mutant_UCB
from dragon.utils.plot_functions import graph_to_all_formulas, str_operations

# ── Thread-safe SIGALRM patch ─────────────────────────────────────────────────
# DRAGON's `timed_evaluation` installs a SIGALRM handler, which raises
# `ValueError: signal only works in main thread of the main interpreter` when
# called from a worker thread (smart-parallel branch).  We patch it to fall
# back to a plain call (no timeout) when not on the main thread.
import threading as _threading
import dragon.search_algorithm.search_algorithm as _dragon_sa

_original_timed_evaluation = _dragon_sa.timed_evaluation


def _timed_evaluation_thread_safe(x, idx, max_duration, evaluation):
    if _threading.current_thread() is _threading.main_thread():
        return _original_timed_evaluation(x, idx, max_duration, evaluation)
    # Non-main thread: SIGALRM unavailable, just call the evaluation directly.
    return evaluation(x, idx)


_dragon_sa.timed_evaluation = _timed_evaluation_thread_safe

# ══════════════════════════════════════════════════════════════════════════════
#  DAG INTROSPECTION HELPERS  — for the leaderboard modal
# ══════════════════════════════════════════════════════════════════════════════

def _dag_summary(adj, nodes):
    """Return (ops_used_str, has_const_str, dag_size, dag_text, dag_svg)."""
    try:
        descs_raw = str_operations(nodes)
    except Exception:
        descs_raw = [[str(n)] for n in nodes]

    # Op-name list: take the second token of each row (the brick class name).
    op_names = []
    for row in descs_raw:
        if len(row) >= 2:
            op_names.append(str(row[1]))
        elif row:
            op_names.append(str(row[0]))
    # Drop the synthetic "Input" entry on row 0 if present.
    op_names_disp = op_names[1:] if op_names else []
    ops_used_str = ", ".join(op_names_disp) if op_names_disp else "—"

    # Constants: any brick whose name contains "Const".
    const_count = sum(1 for n in op_names_disp if "const" in n.lower())
    has_const_str = f"yes ({const_count})" if const_count > 0 else "no (0)"

    dag_size = int(getattr(adj, "shape", (0,))[0]) if hasattr(adj, "shape") else len(nodes)

    # Compact one-line description per node (combiner | brick | hp..., no activation).
    descs = []
    for row in descs_raw:
        try:
            descs.append(" | ".join(str(x) for x in row[:-1]) if len(row) > 1 else str(row[0]))
        except Exception:
            descs.append("?")

    # Build textual DAG table (children + parents per node).
    try:
        n = adj.shape[0]
        parents  = [[str(j) for j in range(n) if adj[j, i]] for i in range(n)]
        children = [[str(j) for j in range(n) if adj[i, j]] for i in range(n)]
        idx_w   = max(3, len(str(n - 1)) + 2)
        child_w = max(8, max((len(",".join(c)) for c in children), default=1) + 2)
        par_w   = max(8, max((len(",".join(p)) for p in parents),  default=1) + 2)
        header  = (f"{'Idx'.ljust(idx_w)}| {'Children'.ljust(child_w)}| "
                   f"{'Parents'.ljust(par_w)}| Description")
        sep     = "-" * (len(header) + 10)
        lines   = [sep, header, sep]
        for i in range(n):
            c = ",".join(children[i]) or "-"
            p = ",".join(parents[i])  or "-"
            d = (descs[i] if i < len(descs) else "?")[:80]
            lines.append(f"[{str(i).rjust(idx_w-2)}] | {c.ljust(child_w)} | {p.ljust(par_w)} | {d}")
        lines.append(sep)
        dag_text = "\n".join(lines)
    except Exception:
        dag_text = "\n".join(f"[{i}] {d}" for i, d in enumerate(descs))

    # ── Graphviz SVG (best-effort; degrades to None on failure) ────────────
    dag_svg = None
    try:
        import shutil as _shutil
        if _shutil.which("dot") is not None:
            import graphviz as _gv
            G = _gv.Digraph(
                format="svg",
                node_attr={"shape": "box", "fontsize": "11",
                           "fontname": "sans-serif", "style": "rounded,filled"},
                graph_attr={"rankdir": "TB", "bgcolor": "transparent",
                            "nodesep": "0.25", "ranksep": "0.35"},
            )
            for i, d in enumerate(descs):
                fill = "#3b6d11" if i == 0 else "#ffa600"
                fc   = "#ECECEC" if i == 0 else "#1a1a1a"
                label = f"[{i}] {d}".replace("\\", "\\\\").replace('"', '\\"')
                G.node(str(i), label=label, fillcolor=fill, color="black",
                       fontcolor=fc)
            n = adj.shape[0]
            for i in range(n):
                for j in range(n):
                    if adj[i, j]:
                        G.edge(str(i), str(j))
            svg_bytes = G.pipe(format="svg")
            svg_str = svg_bytes.decode("utf-8", errors="ignore")
            # Strip XML/DOCTYPE prologue so the SVG can be inlined cleanly.
            _ix = svg_str.find("<svg")
            if _ix >= 0:
                svg_str = svg_str[_ix:]
            dag_svg = svg_str
    except Exception:
        dag_svg = None

    # ── Structured DAG data (JSON-serialisable) for JS fallback renderer ─────
    dag_data = []
    try:
        _n = adj.shape[0]
        dag_data = [
            {"idx": i,
             "children": [int(j) for j in range(_n) if adj[i, j]],
             "desc": descs[i] if i < len(descs) else "?"}
            for i in range(_n)
        ]
    except Exception:
        dag_data = []

    return ops_used_str, has_const_str, dag_size, dag_text, dag_svg, dag_data


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL VISUALISATIONS  (matplotlib SVG, embedded inline in modal)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_svg(fig) -> str:
    """Serialize a Matplotlib figure to a self-contained inlinable SVG string."""
    import io as _io
    buf = _io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    s = buf.getvalue()
    i = s.find("<svg")
    return s[i:] if i >= 0 else s


# ── Plotly helpers ───────────────────────────────────────────────────────────
def _plotly_to_html(fig) -> str:
    """Render a Plotly figure as a self-contained inlinable HTML snippet
    (a <div> + a <script> calling Plotly.newPlot). The Plotly.js library
    itself is loaded once via the CDN tag injected in <head>."""
    import plotly.io as _pio
    return _pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def _make_dragon_landscape_svg(comp_csv_path: str):
    """Return Plotly HTML snippet (3-panel: scatter+best, histogram, convergence).

    The snippet is interactive (zoom, hover, toggle traces).  Returns None on failure.
    """
    try:
        if not os.path.exists(comp_csv_path):
            return None
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        df = pd.read_csv(comp_csv_path)
        if "Loss" not in df.columns:
            return None
        df["Loss"] = df["Loss"].replace([np.inf, -np.inf], np.nan)
        df["BestSoFar"] = df["Loss"].expanding().min()
        finite = df["Loss"].dropna()
        if finite.empty:
            return None
        best = float(finite.min())

        # Wall-clock minutes (if TimeStamp present)
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

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Loss landscape (all evaluations)",
                "Loss distribution",
                f"Convergence ({x_conv_label.split(' ')[0].lower()})",
            ),
            horizontal_spacing=0.08,
        )
        # 1) Scatter + best-so-far
        fig.add_trace(go.Scattergl(
            x=df["Idx"], y=df["Loss"], mode="markers",
            marker=dict(size=4, color="#185fa5", opacity=0.4),
            name="Individual loss", hovertemplate="iter=%{x}<br>loss=%{y:.3e}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Idx"], y=df["BestSoFar"], mode="lines",
            line=dict(color="#a32d2d", width=1.8),
            name="Best so far", hovertemplate="iter=%{x}<br>best=%{y:.3e}<extra></extra>",
        ), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1, title_text="Loss (1−|corr|)")
        fig.update_xaxes(title_text="Iteration", row=1, col=1)

        # 2) Histogram
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

        # 3) Convergence
        fig.add_trace(go.Scatter(
            x=x_conv, y=df["BestSoFar"], mode="lines",
            line=dict(color="#a32d2d", width=1.8),
            name="Best so far",
            hovertemplate=(x_conv_label + "=%{x:.2f}<br>best=%{y:.3e}<extra></extra>"),
            showlegend=False,
        ), row=1, col=3)
        fig.update_yaxes(type="log", title_text="Best loss", row=1, col=3)
        fig.update_xaxes(title_text=x_conv_label, row=1, col=3)

        fig.update_layout(
            template="plotly_white",
            height=460, margin=dict(l=60, r=30, t=80, b=60),
            title=dict(text=f"DragonSR search landscape — {len(df)} evaluations · best = {best:.3e}",
                       font=dict(size=13)),
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.14, font=dict(size=11)),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(size=11, color="#1a1a1a"),
        )
        return _plotly_to_html(fig)
    except Exception:
        return None


def _compute_pysr_scores(hof: list) -> list:
    """Add a 'score' field to each HoF entry (PySR parsimony score = −Δln(loss)/Δcomplexity).

    Sorted by ascending complexity beforehand.  The first entry has score=None.
    Mutates and returns the same list.
    """
    if not hof:
        return hof
    try:
        s = sorted(hof, key=lambda h: (h.get("complexity") or 0))
        prev_loss = None
        prev_cplx = None
        for h in s:
            c = h.get("complexity")
            l = h.get("loss")
            if (prev_loss is None or prev_cplx is None
                    or l is None or c is None
                    or l <= 0 or prev_loss <= 0
                    or c == prev_cplx):
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
    """Plotly Pareto frontier (complexity vs loss, log-y) with hover tooltips
    showing the expression and parsimony score."""
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
        forms = [p[2] for p in pts]
        scores = [p[3] for p in pts]
        # Custom hover: include formula + score
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
            marker=dict(size=8, color="#185fa5",
                        line=dict(width=1, color="#0a3460")),
            text=hov, hoverinfo="text", name="Pareto frontier",
        ))
        fig.update_layout(
            template="plotly_white",
            height=420, margin=dict(l=60, r=30, t=60, b=55),
            title=dict(text="PySR Pareto frontier (complexity vs loss)",
                       font=dict(size=13)),
            xaxis=dict(title="Complexity"),
            yaxis=dict(title="Loss (PySR MSE)", type="log"),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(size=11, color="#1a1a1a"),
        )
        return _plotly_to_html(fig)
    except Exception:
        return None


def _make_pysr_tree_svg(formula_str: str):
    """Parse the PySR string formula via sympy and render the binary AST as
    a Graphviz SVG tree.  Returns None on any failure."""
    try:
        if not formula_str or formula_str.strip() in ("N/A", ""):
            return None
        import shutil as _shutil
        if _shutil.which("dot") is None:
            return None
        import sympy as sp
        import graphviz as _gv

        # Replace common PySR/SR.jl operator strings sympy doesn't grok.
        cleaned = formula_str
        # Remove a possible leading "y = " etc.
        if "=" in cleaned and cleaned.split("=")[0].strip().isidentifier():
            cleaned = cleaned.split("=", 1)[1].strip()
        try:
            expr = sp.sympify(cleaned, evaluate=False)
        except Exception:
            try:
                from sympy.parsing.sympy_parser import (
                    parse_expr, standard_transformations,
                    implicit_multiplication_application,
                    convert_xor)
                tr = (standard_transformations
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
            i = counter[0]; counter[0] += 1
            nid = f"n{i}"
            if node.is_Atom:
                lbl = str(node)
                fill = "#3b6d11"; fc = "#ECECEC"
            else:
                lbl = type(node).__name__
                # Friendlier op symbols
                lbl = {"Add": "+", "Mul": "×", "Pow": "^",
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
        i = svg.find("<svg")
        return svg[i:] if i >= 0 else svg
    except Exception:
        return None


def _make_pysr_tree_data(formula_str: str):
    """Parse a PySR formula string via sympy and return a JSON-serialisable
    node list [{id, label, kind, children}] for the JS _renderFormulaTree
    renderer.  Works without graphviz.  Returns None on failure."""
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
                tr = (standard_transformations
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
            idx = len(nodes)
            is_atom = node.is_Atom
            if is_atom and node.is_Number:
                try:
                    lbl = f"{float(node):.4g}"
                except Exception:
                    lbl = str(node)
            elif is_atom:
                lbl = str(node)
            else:
                lbl = _LABELS.get(type(node).__name__, type(node).__name__)
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
        # Binarize: sympy flattens Add/Mul into n-ary nodes; convert to binary
        # right-leaning chains so the tree always has at most 2 children/node.
        bin_nodes: list = []

        def _binarize(old_idx: int) -> int:
            nd = nodes[old_idx]
            old_ch = nd["children"]
            new_ch = [_binarize(c) for c in old_ch]
            if len(new_ch) <= 2:
                idx = len(bin_nodes)
                bin_nodes.append({"id": idx, "label": nd["label"],
                                   "kind": nd["kind"], "children": new_ch})
                return idx
            # Right-leaning: Add(a,b,c,d) -> Add(a, Add(b, Add(c,d)))
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
    """Per-formula across-method×runs interactive stats panel.

    Returns a Plotly HTML snippet with three subplots:
      (a) Box-plot of 1−R² per method  → spread + median + outliers
      (b) ECDF (= performance profile) → "fraction of runs reaching loss ≤ τ"
      (c) Scatter loss vs runtime      → cost/quality trade-off

    `per_cell` is a list of {method, run, loss, runtime} dicts.
    """
    try:
        if not per_cell:
            return None
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from collections import OrderedDict
        groups = OrderedDict()
        for d in per_cell:
            m = d.get("method") or "?"
            groups.setdefault(m, []).append(d)
        methods = list(groups.keys())
        if not methods:
            return None

        # Stable per-method colour
        palette = ["#185fa5", "#a32d2d", "#3b6d11", "#ba7517", "#6e3a8a",
                   "#117a8b", "#c2185b", "#7d6608", "#0a3460", "#4d4d4d"]
        colour = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Best vs median loss per method",
                "Loss vs runtime (cost / quality)",
            ),
            horizontal_spacing=0.18,
        )

        # (a) Best vs median loss per method — grouped bars
        names_x, best_vals, med_vals, bar_cols = [], [], [], []
        for m in methods:
            vs = [d["loss"] for d in groups[m]
                  if d.get("loss") is not None and np.isfinite(d["loss"])
                  and d["loss"] > 0]
            if not vs:
                continue
            names_x.append(m)
            best_vals.append(min(vs))
            med_vals.append(float(np.median(vs)))
            bar_cols.append(colour[m])
        if names_x:
            fig.add_trace(go.Bar(
                x=names_x, y=best_vals, name="best",
                marker=dict(color=bar_cols, line=dict(width=0.6, color="#222")),
                hovertemplate="%{x}<br>best 1−R² = %{y:.3e}<extra></extra>",
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=names_x, y=med_vals, name="median",
                marker=dict(color=bar_cols, opacity=0.45,
                            line=dict(width=0.6, color="#222")),
                hovertemplate="%{x}<br>median 1−R² = %{y:.3e}<extra></extra>",
                showlegend=False,
            ), row=1, col=1)

        # (b) Scatter loss vs runtime
        for m in methods:
            xs = [d.get("runtime") for d in groups[m]
                  if d.get("runtime") is not None
                  and d.get("loss") is not None and np.isfinite(d["loss"])
                  and d["loss"] > 0]
            ys = [d["loss"] for d in groups[m]
                  if d.get("runtime") is not None
                  and d.get("loss") is not None and np.isfinite(d["loss"])
                  and d["loss"] > 0]
            if not xs:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=m,
                marker=dict(color=colour[m], size=10,
                            line=dict(width=0.6, color="#222")),
                hovertemplate=f"{m}<br>runtime=%{{x:.1f}}s<br>1−R²=%{{y:.3e}}<extra></extra>",
                legendgroup=m, showlegend=True,
            ), row=1, col=2)

        # Axes
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
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATORS (physics + Nguyen)
# ══════════════════════════════════════════════════════════════════════════════

_SYNTH_CATEGORIES = {
    # physics
    "hubble", "newton", "rydberg", "idealgas", "kepler",
    "bode", "schechter", "leavitt", "planck",
    # nguyen
    "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12",
    # other
    "expreal",
}


def _gen_physics(formula_id: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    if formula_id == "hubble":
        d = rng.uniform(1, 1000, n)
        y = 70.0 * d                          # v = H0·d, H0=70 km/s/Mpc
        return pd.DataFrame({"d": d}), pd.Series(y, name="v")
    elif formula_id == "newton":
        G = 6.674e-11
        m1 = rng.uniform(1e24, 1e30, n)
        m2 = rng.uniform(1e24, 1e30, n)
        r  = rng.uniform(1e8,  1e12, n)
        F  = G * m1 * m2 / r**2
        return pd.DataFrame({"m1": m1, "m2": m2, "r": r}), pd.Series(F, name="F")
    elif formula_id == "rydberg":
        R = 1.097e7
        n1 = rng.integers(1, 5, n).astype(float)
        n2 = n1 + rng.integers(1, 5, n).astype(float)
        y  = R * (1.0/n1**2 - 1.0/n2**2)
        return pd.DataFrame({"n1": n1, "n2": n2}), pd.Series(y, name="inv_lambda")
    elif formula_id == "idealgas":
        R = 8.314
        P = rng.uniform(1e4, 1e6, n)
        nc = rng.uniform(0.1, 10, n)
        T  = rng.uniform(200, 1000, n)
        V  = nc * R * T / P
        return pd.DataFrame({"P": P, "n": nc, "T": T}), pd.Series(V, name="V")
    elif formula_id == "kepler":
        G = 6.674e-11; M_sun = 1.989e30
        a = rng.uniform(0.1, 50, n) * 1.496e11
        T = 2 * np.pi * np.sqrt(a**3 / (G * M_sun))
        return pd.DataFrame({"a": a}), pd.Series(T, name="T")
    elif formula_id == "bode":
        nv = np.arange(0, 9, dtype=float)
        a  = 0.4 + 0.3 * 2**nv
        return pd.DataFrame({"n": nv}), pd.Series(a, name="a")
    elif formula_id == "schechter":
        phi_star = 1.5e-2; L_star = 1e10; alpha = -1.1
        L = rng.exponential(L_star, n)
        L = np.clip(L, 1e6, 1e13)
        phi = phi_star * (L/L_star)**alpha * np.exp(-L/L_star)
        mask = np.isfinite(phi) & (phi > 0)
        return pd.DataFrame({"L": L[mask]}), pd.Series(phi[mask], name="phi")
    elif formula_id == "leavitt":
        P = rng.uniform(1, 100, n)
        M = -2.81 * np.log10(P) - 1.43
        return pd.DataFrame({"P": P}), pd.Series(M, name="M")
    elif formula_id == "planck":
        h = 6.626e-34; c = 3e8; k = 1.381e-23
        nu = rng.uniform(1e11, 3e14, n)
        T  = rng.uniform(1000, 30000, n)
        exp_arg = np.clip(h * nu / (k * T), 0, 700)
        B = 2 * h * nu**3 / c**2 / (np.expm1(exp_arg) + 1e-300)
        mask = np.isfinite(B) & (B > 0)
        return pd.DataFrame({"nu": nu[mask], "T": T[mask]}), pd.Series(B[mask], name="B")
    elif formula_id == "expreal":
        x = rng.uniform(0, 2, n)
        y = np.exp(x) + np.exp(-x) + x**2
        return pd.DataFrame({"x": x}), pd.Series(y, name="y")
    raise ValueError(f"Unknown physics formula: {formula_id}")


def _gen_nguyen(formula_id: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    if formula_id == "n1":
        x = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x}), pd.Series(x**3+x**2+x, name="y")
    elif formula_id == "n2":
        x = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x}), pd.Series(x**4+x**3+x**2+x, name="y")
    elif formula_id == "n3":
        x = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x}), pd.Series(x**5+x**4+x**3+x**2+x, name="y")
    elif formula_id == "n4":
        x = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x}), pd.Series(x**6+x**5+x**4+x**3+x**2+x, name="y")
    elif formula_id == "n5":
        x = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x}), pd.Series(np.sin(x**2)*np.cos(x)-1, name="y")
    elif formula_id == "n6":
        x = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x}), pd.Series(np.sin(x)+np.sin(x+x**2), name="y")
    elif formula_id == "n7":
        x = rng.uniform(0, 2, n)
        return pd.DataFrame({"x": x}), pd.Series(np.log(x+1)+np.log(x**2+1), name="y")
    elif formula_id == "n8":
        x = rng.uniform(0, 4, n)
        return pd.DataFrame({"x": x}), pd.Series(np.sqrt(x), name="y")
    elif formula_id == "n9":
        x = rng.uniform(-1, 1, n); yv = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(np.sin(x)+np.sin(yv**2), name="z")
    elif formula_id == "n10":
        x = rng.uniform(-1, 1, n); yv = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(2*np.sin(x)*np.cos(yv), name="z")
    elif formula_id == "n11":
        x = rng.uniform(1, 2, n); yv = rng.uniform(1, 2, n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(x**yv, name="z")
    elif formula_id == "n12":
        x = rng.uniform(-1, 1, n); yv = rng.uniform(-1, 1, n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(x**4-x**3+yv**2/2-yv, name="z")
    raise ValueError(f"Unknown Nguyen formula: {formula_id}")


def load_dataset(target: str, run_id: int = 0, strategy: str = "xgboost"):
    """Master dispatcher: returns (X_df, y_series) for any target."""
    seed = RANDOM_SEED + run_id
    if target in _SYNTH_CATEGORIES:
        if target.startswith("n") and target[1:].isdigit():
            return _gen_nguyen(target, N_SYNTH_SAMPLES, seed)
        return _gen_physics(target, N_SYNTH_SAMPLES, seed)
    # Remote sensing — CSV-based
    return _build_remote_dataset(target, seed, strategy)


def _build_remote_dataset(target: str, seed: int, strategy: str):
    """Load CSV, compute remote sensing target column."""
    return build_dataset(target)


def build_dataset(target: str, data_path: str = DATA_PATH):
    """Load CSV, compute target column, return (X_df, y_series)."""
    df = pd.read_csv(data_path)
    # drop non-numeric metadata
    for col in ['system:index', 'QA60', '.geo', 'date']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Compute target if not already present
    if target == "NDVI":
        df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    elif target == "SAVI":
        df['SAVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5) * 1.5
    elif target == "BSI":
        df['BSI'] = (df['B11'] + df['B4'] - df['B8'] - df['B2']) / \
                    (df['B11'] + df['B4'] + df['B8'] + df['B2'])
    elif target == "MNDWI":
        df['MNDWI'] = (df['B3'] - df['B11']) / (df['B3'] + df['B11'])
    elif target == "NDMI":
        df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'])
    elif target == "MSI":
        df['MSI'] = df['B11'] / df['B8']
    elif target == "NDWI_McFeeters":
        df['NDWI_McFeeters'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'])
    elif target == "BAI":
        df['BAI'] = 1 / ((0.1 - df['B4'])**2 + (0.06 - df['B8'])**2)
    elif target == "AWEI_sh":
        df['AWEI_sh'] = df['B2'] + 2.5*df['B3'] - 1.5*(df['B11'] + df['B12']) - 0.25*df['B8']
    elif target == "AWEI_nsh":
        df['AWEI_nsh'] = 4*(df['B3'] - df['B11']) - (0.25*df['B8'] + 2.75*df['B12'])
    elif target == "WI2015":
        df['WI2015'] = (1.7204 + 171*(df['B2'] + df['B3'] + df['B4'])
                        - 3*(df['B2']*df['B3']) - 1.8*(df['B2']*df['B4'])
                        - 48*(df['B3']*df['B4']) - 0.8*(df['B8']*df['B11']))
    elif target == "NDRE_B5":
        df['NDRE_B5'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'])
    elif target == "NDRE_B6":
        df['NDRE_B6'] = (df['B8A'] - df['B6']) / (df['B8A'] + df['B6'])
    elif target == "evi2":
        df['evi2'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 2.4*df['B4'] + 1)
    elif target == "vari":
        denom = df['B3'] + df['B4'] - df['B2']
        df['vari'] = (df['B3'] - df['B4']) / denom.replace(0, np.nan)
    # lowercase aliases for HTML IDs
    elif target == "ndvi":
        df['ndvi'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    elif target == "bsi":
        df['bsi'] = (df['B11'] + df['B4'] - df['B8'] - df['B2']) / \
                    (df['B11'] + df['B4'] + df['B8'] + df['B2'])
    elif target == "savi":
        df['savi'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5) * 1.5
    elif target == "mndwi":
        df['mndwi'] = (df['B3'] - df['B11']) / (df['B3'] + df['B11'])
    elif target == "bai":
        df['bai'] = 1 / ((0.1 - df['B4'])**2 + (0.06 - df['B8'])**2)
    elif target == "awei_sh":
        df['awei_sh'] = df['B2'] + 2.5*df['B3'] - 1.5*(df['B11'] + df['B12']) - 0.25*df['B8']
    elif target == "awei_nsh":
        df['awei_nsh'] = 4*(df['B3'] - df['B11']) - (0.25*df['B8'] + 2.75*df['B12'])
    elif target == "wi2015":
        df['wi2015'] = (1.7204 + 171*(df['B2'] + df['B3'] + df['B4'])
                        - 3*(df['B2']*df['B3']) - 1.8*(df['B2']*df['B4'])
                        - 48*(df['B3']*df['B4']) - 0.8*(df['B8']*df['B11']))
    elif target == "nirv":
        df['nirv'] = df['B8'] * ((df['B8'] - df['B4']) / (df['B8'] + df['B4']))

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found / not defined in build_dataset().")

    df = df.dropna(subset=[target])
    df = df[np.isfinite(df[target])]

    y = df[target].copy()
    X = df.drop(columns=[target])
    # keep only numeric
    X = X.select_dtypes(include=[np.number])
    return X, y


def _auto_denoise(X: pd.DataFrame, y: pd.Series, max_samples: int = 600, nu: float = 2.5):
    """Self-validating GPR-Matérn denoiser (recommended default).

    Pipeline
    --------
    1. Fit a Gaussian Process with a (Constant × Matérn) + WhiteKernel kernel.
       The Matérn kernel (nu=2.5) suits analytic targets better than RBF, and
       the WhiteKernel term estimates the noise variance sigma_hat^2 by marginal
       likelihood — this *is* the noise floor used for early stopping.
    2. Calibration guard (no clean signal needed): via K-fold, measure the
       held-out predictive RMSE on the NOISY data.  If the smoother is
       unbiased it satisfies  RMSE_holdout ~= sigma_hat.  A ratio >> 1 means the
       smoother is oversmoothing real structure (e.g. fast-varying remote-
       sensing ratios) → denoising would *hurt*, so we fall back to raw y.

    Benchmarked recovery (RMSE_to_clean / RMSE_noise, lower = better):
        n4   : 0.15  (denoise applied)    vs  0.26 for XGBoost
        ndvi : guard SKIPs (any smoother >1.0, i.e. would corrupt the signal)

    Returns
    -------
    y_clean : pd.Series
    info    : dict — {"method", "applied", "ratio", "noise_var", "noise_sigma",
                      "snr_db", "nu"}
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        Matern, WhiteKernel, ConstantKernel)
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.model_selection import KFold

    n = len(y)
    std_y = float(y.std())
    if n < 8 or std_y < 1e-12:
        return y.copy(), {"method": "auto", "applied": False,
                          "reason": "too few samples / constant y",
                          "noise_var": 0.0}

    Xs = _SS().fit_transform(X.values.astype(float))
    yv = y.values.astype(float)

    def _make_gp():
        kern = (ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=nu)
                + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e2)))
        return GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=2,
                                        normalize_y=True, random_state=RANDOM_SEED)

    def _fit_predict(Xtr, ytr, Xte):
        gp = _make_gp()
        if len(ytr) > max_samples:
            sub = np.random.default_rng(RANDOM_SEED).choice(
                len(ytr), max_samples, replace=False)
            gp.fit(Xtr[sub], ytr[sub])
        else:
            gp.fit(Xtr, ytr)
        return gp.predict(Xte)

    # -- Calibration guard: held-out NOISY RMSE via K-fold -------------------
    kf = KFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    fold_err = []
    for tr, te in kf.split(Xs):
        try:
            yh_te = _fit_predict(Xs[tr], yv[tr], Xs[te])
            fold_err.append(np.mean((yh_te - yv[te]) ** 2))
        except Exception:
            fold_err.append(np.var(yv))
    rmse_holdout = float(np.sqrt(np.mean(fold_err)))

    # -- Full fit (for sigma_hat and the denoised signal) --------------------
    gp_full = _make_gp()
    try:
        if n > max_samples:
            sub = np.random.default_rng(RANDOM_SEED).choice(n, max_samples, replace=False)
            gp_full.fit(Xs[sub], yv[sub])
        else:
            gp_full.fit(Xs, yv)
        y_hat = gp_full.predict(Xs)
        noise_level = float(gp_full.kernel_.k2.noise_level)   # normalised-y units
    except Exception as e:
        return y.copy(), {"method": "auto", "applied": False,
                          "reason": f"gpr failed ({e})", "noise_var": 0.0}

    sigma_hat = np.sqrt(max(noise_level, 0.0)) * std_y
    ratio     = rmse_holdout / max(sigma_hat, 1e-12)
    noise_var = float(sigma_hat ** 2)
    var_y     = float(np.var(yv))
    snr_db    = 10.0 * np.log10(max(var_y - noise_var, 1e-30) / max(noise_var, 1e-30))

    # Unbiased smoother => RMSE_holdout ~= sigma_hat.  Excess => oversmoothing => skip.
    applied = bool(ratio < 1.15)
    info = {"method": "auto", "applied": applied, "ratio": float(ratio),
            "noise_var": noise_var, "noise_sigma": float(sigma_hat),
            "snr_db": float(snr_db), "nu": nu}
    if applied:
        return pd.Series(y_hat, index=y.index, name=y.name), info
    return y.copy(), info


def _apply_denoise(X: pd.DataFrame, y: pd.Series, method: str):
    """Dispatch to the requested denoising helper.

    Parameters
    ----------
    method : "auto"      → ``_auto_denoise`` (self-validating GPR-Matérn)
             "stoch_sub" → no-op preprocessing (in-loop subsampling)

    Returns
    -------
    (y_clean, info) — same convention as ``_auto_denoise``.
    """
    if method == "auto":
        return _auto_denoise(X, y)
    elif method == "stoch_sub":
        # Not a pre-processing step: the actual subsampling happens inside
        # loss_function at each Dragon evaluation.  Return y unchanged so
        # the run_all precompute loop is a transparent no-op for this method.
        return y.copy(), {"method": "stoch_sub", "note": "in-loop subsampling"}
    else:
        raise ValueError(f"Unknown denoise_method: {method!r}. "
                         f"Supported values: 'auto', 'stoch_sub'.")


def _mc_dropout_weights(
        X: pd.DataFrame, y: pd.Series,
        n_forward: int = 50,
        dropout_p: float = 0.15,
        n_epochs: int = 300,
        hidden: int = 64,
        random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Estimate per-sample confidence via a small MLP + Monte-Carlo Dropout.

    Pipeline
    --------
    1. Standardise (X, y) and train a small 2-hidden-layer MLP with Dropout
       for ``n_epochs`` steps (Adam, weight-decay).
    2. Run ``n_forward`` stochastic forward passes (dropout active, model.train())
       on the full dataset.
    3. Compute per-sample variance of predictions → aleatoric uncertainty.
    4. Convert to confidence weights: w_i = 1 / (1 + σ²_i / μ_σ²).
       High-variance (likely noisy) samples get lower weight.
    5. Normalise so that weights sum to n (unbiased expectation).

    Returns
    -------
    weights : np.ndarray shape (n,), float32
        Confidence weights summing to n.  Pass to ``make_loss_function`` as
        ``_sample_weights`` to guide both stochastic subsampling (sampling
        probability proportional to weight) and full-dataset loss reweighting.
    """
    n, d = len(y), X.shape[1]
    if n < 8:
        return np.ones(n, dtype=np.float32)

    Xv = X.values.astype(np.float32)
    yv = y.values.astype(np.float32).ravel()
    xstd  = Xv.std(0) + 1e-8
    xmean = Xv.mean(0)
    ystd  = float(yv.std()) + 1e-8
    ymean = float(yv.mean())
    Xs = (Xv - xmean) / xstd
    ys = (yv - ymean) / ystd

    Xt = torch.tensor(Xs)
    yt = torch.tensor(ys).unsqueeze(1)

    torch.manual_seed(random_seed)
    h = hidden
    net = nn.Sequential(
        nn.Linear(d, h),     nn.ReLU(), nn.Dropout(p=dropout_p),
        nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p=dropout_p),
        nn.Linear(h // 2, 1),
    )
    opt   = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    mse_f = nn.MSELoss()

    net.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        mse_f(net(Xt), yt).backward()
        opt.step()

    # MC inference: keep dropout active (model.train()) for stochasticity
    net.train()
    with torch.no_grad():
        preds = np.stack([
            net(Xt).squeeze(1).numpy()
            for _ in range(n_forward)
        ])  # (n_forward, n)

    variances = preds.var(axis=0)          # (n,)
    mean_var  = float(variances.mean()) + 1e-30
    confidence = 1.0 / (1.0 + variances / mean_var)  # ∈ (0.5, 1.0]
    weights    = confidence * n / confidence.sum()
    return weights.astype(np.float32)


def xgboost_feature_selection(X: pd.DataFrame, y: pd.Series, n_top: int = N_TOP_FEATURES):
    """Return top-n feature names ranked by XGBoost importance."""
    scaler_X = StandardScaler()
    X_sc = scaler_X.fit_transform(X)
    scaler_z = StandardScaler()
    z_sc = scaler_z.fit_transform(y.values.reshape(-1, 1)).ravel()
    mdl = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=RANDOM_SEED, verbosity=0)
    mdl.fit(X_sc, z_sc)
    fi = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
    scores = fi.to_dict()
    top_feats = fi.head(n_top).index.tolist()
    return top_feats, scores


def build_feature_combinations(feature_names, feature_scores, max_combo=6,
                                budget_per_size=None):
    """Build feature index combinations weighted by XGBoost importance."""
    n = len(feature_names)
    if n == 1:
        return [[0]]

    probs = np.array([feature_scores.get(c, 1.0/n) for c in feature_names])
    probs = probs / probs.sum()
    log_probs = np.log(probs + 1e-300)

    max_combo = min(max_combo, n)
    TOP_M = {3: 30, 4: 20, 5: 14, 6: 12}
    default_budget = {1: n, 2: n*(n-1)//2}
    for s in range(3, max_combo+1):
        default_budget[s] = max(50, 200//(s-1))
    if budget_per_size:
        default_budget.update(budget_per_size)

    all_combos = []
    for size in range(1, max_combo+1):
        budget = default_budget[size]
        if size <= 2:
            pool = list(combinations(range(n), size))
        else:
            M = min(TOP_M.get(size, 12), n)
            top_idx = np.argsort(probs)[-M:]
            pool = list(combinations(sorted(top_idx.tolist()), size))
        scored = [(sum(log_probs[i] for i in combo), list(combo)) for combo in pool]
        scored.sort(key=lambda x: -x[0])
        all_combos.extend(combo for _, combo in scored[:budget])
    return all_combos


# ══════════════════════════════════════════════════════════════════════════════
#  DRAGON COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class MetaArchi(nn.Module):
    def __init__(self, args, input_shape):
        super().__init__()
        self.input_shape = input_shape
        assert isinstance(args['Dag'], AdjMatrix)
        self.dag = args['Dag']
        self.dag.set(input_shape)

    def forward(self, X):
        return self.dag(X)

    def set_prediction_to_save(self, name, df):
        if hasattr(self, "prediction"):
            self.prediction[name] = df
        else:
            self.prediction = {name: df}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "best_model.pth"))
        if hasattr(self, "prediction"):
            for k, v in self.prediction.items():
                v.to_csv(os.path.join(path, f"best_model_{k}_outputs.csv"))


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.DoubleTensor(X.values)
        self.y = torch.DoubleTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def correl(pred, true):
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(true, torch.Tensor):
        true = true.numpy()
    pred = pred.ravel()
    true = true.ravel()
    pd_ = pred - pred.mean()
    td_ = true - true.mean()
    cov = np.sum(pd_ * td_)
    sp = np.sqrt(np.sum(pd_**2))
    st = np.sqrt(np.sum(td_**2))
    if sp < 1e-12 or st < 1e-12:
        return 0.0
    return float(cov / (sp * st + 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
#  OLS HELPERS  — ported from ref.py (nested + poly-rational post-processing)
# ══════════════════════════════════════════════════════════════════════════════

_EPS = 1e-8
RAT_MAX_DEGREE      = 2
RAT_MAX_BASIS_CHANS = 3
RAT_MAX_FEATURES    = 4  # max monomials kept after corr-ranking

_LINKS = {
    'id'  : (lambda y: y,                                         lambda p: p),
    'log' : (lambda y: np.log(np.abs(y) + _EPS),                 lambda p: np.exp(np.clip(p, -50, 50))),
    'sqrt': (lambda y: np.sqrt(np.abs(y)),                        lambda p: p ** 2),
    'sq'  : (lambda y: y ** 2,                                    lambda p: np.sqrt(np.abs(p))),
    'inv' : (lambda y: 1.0 / (np.abs(y) + _EPS),                 lambda p: 1.0 / (np.abs(p) + _EPS)),
    'cbrt': (lambda y: np.sign(y) * np.abs(y) ** (1.0 / 3.0),    lambda p: p ** 3),
}
_UNARIES = {
    'id'  : lambda x: x,
    'sq'  : lambda x: x ** 2,
    'sqrt': lambda x: np.sqrt(np.abs(x)),
    'inv' : lambda x: 1.0 / (np.abs(x) + _EPS),
    'neg' : lambda x: -x,
    'log' : lambda x: np.log(np.abs(x) + _EPS),
}
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


def _safe_lstsq(P, y_t):
    if P.shape[0] < P.shape[1] + 1:
        return None, None, None
    A = np.hstack([P, np.ones((P.shape[0], 1))])
    try:
        sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    return sol[:-1], sol[-1], A @ sol


def _sparse_lstsq(P, y_t, rel_thresh=1e-3, abs_thresh=1e-8, max_iter=10):
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
        new_keep_local = np.abs(w_k) >= thresh
        if new_keep_local.all():
            break
        idx_keep = np.where(kept)[0][new_keep_local]
        new_kept = np.zeros(k, dtype=bool)
        new_kept[idx_keep] = True
        if new_kept.sum() == 0:
            return None, None, None, None
        kept = new_kept
    Pk = P[:, kept]
    A  = np.hstack([Pk, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
    return sol[:-1], float(sol[-1]), A @ sol, kept


def _nested_ols(channels, y, mse_floor):
    n_ch = channels.shape[1]
    best = None
    for link_name, (fwd, inv) in _LINKS.items():
        with np.errstate(all='ignore'):
            y_t = fwd(y)
        if not np.all(np.isfinite(y_t)) or np.std(y_t) < 1e-12:
            continue
        y_t_c = y_t - y_t.mean()
        unaries_pick = []
        chans_t  = np.empty_like(channels)
        valid_mask = np.zeros(n_ch, dtype=bool)
        for j in range(n_ch):
            best_corr, best_u, best_vals = -1.0, 'id', channels[:, j]
            for u_name, u_fn in _UNARIES.items():
                with np.errstate(all='ignore'):
                    v = u_fn(channels[:, j])
                if not np.all(np.isfinite(v)) or np.std(v) < 1e-12:
                    continue
                v_c   = v - v.mean()
                denom = np.sqrt((v_c ** 2).sum() * (y_t_c ** 2).sum()) + 1e-30
                c     = abs((v_c * y_t_c).sum() / denom)
                if np.isfinite(c) and c > best_corr:
                    best_corr, best_u, best_vals = c, u_name, v
            unaries_pick.append(best_u)
            if best_corr > 0:
                chans_t[:, j] = best_vals
                valid_mask[j] = True
        if valid_mask.sum() == 0:
            continue
        w_kept, b, pred_t, kept_local = _sparse_lstsq(
            chans_t[:, valid_mask], y_t, rel_thresh=1e-3, abs_thresh=1e-8, max_iter=10)
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


def _build_poly_features(channels, max_degree=2, include_cross=True):
    n, k = channels.shape
    feats, names = [], []
    for j in range(k):
        feats.append(channels[:, j]); names.append((j,))
    if max_degree >= 2:
        for j in range(k):
            feats.append(channels[:, j] ** 2); names.append((j, j))
        if include_cross:
            for i in range(k):
                for j in range(i + 1, k):
                    feats.append(channels[:, i] * channels[:, j]); names.append((i, j))
    if not feats:
        return np.empty((n, 0)), []
    return np.column_stack(feats), names


def _poly_rational_ols(channels, y, mse_floor,
                       max_degree=RAT_MAX_DEGREE,
                       max_basis_channels=RAT_MAX_BASIS_CHANS,
                       max_features=RAT_MAX_FEATURES,
                       rel_thresh=1e-3, abs_thresh=1e-8, max_iter=10):
    n, k = channels.shape
    if k == 0 or n < 4:
        return None
    y_c = y - y.mean()
    sy  = (y_c ** 2).sum() ** 0.5 + 1e-30
    corrs = np.zeros(k)
    for j in range(k):
        v = channels[:, j]; v_c = v - v.mean()
        sv = (v_c ** 2).sum() ** 0.5 + 1e-30
        corrs[j] = abs((v_c * y_c).sum() / (sv * sy))
    sel = np.argsort(-corrs)[:min(k, max_basis_channels)]
    chans_sel = channels[:, sel]
    feats, mono_local = _build_poly_features(chans_sel, max_degree=max_degree, include_cross=True)
    if feats.shape[1] == 0:
        return None
    ok = np.array([np.all(np.isfinite(feats[:, j])) and np.std(feats[:, j]) > 1e-15
                   for j in range(feats.shape[1])])
    if ok.sum() == 0:
        return None
    feats      = feats[:, ok]
    mono_local = [m for m, k_ok in zip(mono_local, ok) if k_ok]
    if feats.shape[1] > max_features:
        scores = np.zeros(feats.shape[1])
        for j in range(feats.shape[1]):
            f   = feats[:, j]; f_c = f - f.mean()
            sf  = (f_c ** 2).sum() ** 0.5 + 1e-30
            scores[j] = abs((f_c * y_c).sum() / (sf * sy))
        keep = np.argsort(-scores)[:max_features]
        feats      = feats[:, keep]
        mono_local = [mono_local[i] for i in keep]
    m = feats.shape[1]
    if n < 2 * m + 2:
        keep = np.argsort(-np.array([
            abs(np.corrcoef(feats[:, j], y)[0, 1]) if np.std(feats[:, j]) > 1e-15 else 0.0
            for j in range(m)]))[: max(1, (n - 2) // 2)]
        feats      = feats[:, keep]
        mono_local = [mono_local[i] for i in keep]
        m = feats.shape[1]
        if m == 0:
            return None

    def build_A(kept_a, kept_b):
        cols = []
        if kept_a.any(): cols.append(feats[:, kept_a])
        cols.append(np.ones((n, 1)))
        if kept_b.any(): cols.append(-feats[:, kept_b] * y[:, None])
        return np.hstack(cols)

    kept_a = np.ones(m, dtype=bool)
    kept_b = np.ones(m, dtype=bool)
    for _ in range(max_iter):
        A = build_A(kept_a, kept_b)
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        ka = int(kept_a.sum()); kb = int(kept_b.sum())
        a_k = sol[:ka]; a_0 = float(sol[ka]); b_k = sol[ka + 1: ka + 1 + kb]
        all_abs = np.concatenate([np.abs(a_k), np.abs(b_k)]) if (ka + kb) > 0 else np.array([0.0])
        max_w  = float(all_abs.max()) if all_abs.size else 0.0
        thresh = max(rel_thresh * max_w, abs_thresh)
        na = np.abs(a_k) >= thresh; nb = np.abs(b_k) >= thresh
        if na.all() and nb.all():
            break
        idx_a = np.where(kept_a)[0][na]; idx_b = np.where(kept_b)[0][nb]
        new_kept_a = np.zeros(m, dtype=bool); new_kept_a[idx_a] = True
        new_kept_b = np.zeros(m, dtype=bool); new_kept_b[idx_b] = True
        if not new_kept_a.any() and not new_kept_b.any():
            return None
        kept_a, kept_b = new_kept_a, new_kept_b

    A = build_A(kept_a, kept_b)
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    ka = int(kept_a.sum()); kb = int(kept_b.sum())
    a_k = sol[:ka]; a_0 = float(sol[ka]); b_k = sol[ka + 1: ka + 1 + kb]
    a = np.zeros(m); a[kept_a] = a_k
    b = np.zeros(m); b[kept_b] = b_k
    numer = feats @ a + a_0
    denom = feats @ b + 1.0
    if not np.all(np.abs(denom) > 1e-10):
        return None
    pred = numer / denom
    if not np.all(np.isfinite(pred)):
        return None
    mse = float(np.mean((y - pred) ** 2))
    if not np.isfinite(mse) or mse >= mse_floor:
        return None
    mono_global = [tuple(int(sel[idx]) for idx in mn) for mn in mono_local]
    return dict(mse=mse, a=a, a0=a_0, b=b, b0=1.0,
                kept_a=kept_a, kept_b=kept_b,
                monomials=mono_global, pred=pred, max_degree=max_degree)


def _format_nested_smart(nested, formulas, min_weight=1e-4):
    from collections import Counter as _Ctr
    link    = nested['link']
    unaries = nested['unaries']
    w       = nested['w']
    b       = nested['b']
    valid   = nested['valid']
    valid_w = [(i, wi) for i, wi in enumerate(w)
               if valid[i] and abs(wi) >= min_weight and i < len(formulas)]
    if not valid_w:
        return f"{b:.4f}"
    used_unaries = {unaries[i] for i, _ in valid_w}
    if link == 'log' and used_unaries == {'log'}:
        coef  = np.exp(b)
        terms = [f"({formulas[i]})**({wi:.4f})" for i, wi in valid_w]
        return f"{coef:.4e} * " + " * ".join(terms)
    if link == 'log':
        coef  = np.exp(b)
        inner = " ".join(f"{wi:+.4f}*{_U_WRAP[unaries[i]](formulas[i])}" for i, wi in valid_w)
        return f"{coef:.4e} * exp({inner})"
    if link == 'inv' and used_unaries == {'inv'}:
        terms = [f"{wi:+.4f}/({formulas[i]})" for i, wi in valid_w]
        denom = " ".join(terms)
        if abs(b) > min_weight:
            denom += f" {b:+.4f}"
        return f"1 / ({denom})"
    terms = [f"{wi:+.4f}*{_U_WRAP[unaries[i]](formulas[i])}" for i, wi in valid_w]
    inner = " ".join(terms)
    if abs(b) > min_weight:
        inner += f" {b:+.4f}"
    return _LINK_WRAP[link](inner)


def _mono_to_str(mono, formulas, valid_idx_global):
    from collections import Counter as _Ctr
    counts = _Ctr(mono)
    parts  = []
    for local_idx, exponent in counts.items():
        global_idx = int(valid_idx_global[local_idx])
        f_str = formulas[global_idx] if global_idx < len(formulas) else f"ch{global_idx}"
        parts.append(f"({f_str})" if exponent == 1 else f"({f_str})**{exponent}")
    return "*".join(parts) if parts else "1"


def _format_poly_rational_smart(rational, formulas, valid_idx_global, min_weight=1e-4):
    a, a0  = rational['a'], rational['a0']
    b, b0  = rational['b'], rational['b0']
    kept_a = rational['kept_a']
    kept_b = rational['kept_b']
    monos  = rational['monomials']
    num_terms = []
    for k, mono in enumerate(monos):
        if kept_a[k] and abs(a[k]) >= min_weight:
            num_terms.append(f"{a[k]:+.4f}*{_mono_to_str(mono, formulas, valid_idx_global)}")
    if abs(a0) >= min_weight or not num_terms:
        num_terms.append(f"{a0:+.4f}")
    den_terms = []
    for k, mono in enumerate(monos):
        if kept_b[k] and abs(b[k]) >= min_weight:
            den_terms.append(f"{b[k]:+.4f}*{_mono_to_str(mono, formulas, valid_idx_global)}")
    den_terms.append(f"{b0:+.4f}")
    return f"({' '.join(num_terms)}) / ({' '.join(den_terms)})"


def _normalized_mse(pred, y, var_y):
    if var_y < 1e-30:
        return 1.0
    return float(np.mean((y - pred) ** 2) / var_y)


def _normalized_huber(pred, y, var_y, delta_frac: float = None):
    """Normalised Huber loss — robust drop-in replacement for normMSE.

        H_δ(r) = ½·r²                 if |r| ≤ δ
               = δ·(|r| - ½δ)         otherwise

    We return  mean(H_δ(y - pred)) / (½·var_y), so the value matches the
    normMSE scale at the MSE-regime limit (δ → ∞ gives exactly normMSE) and
    stays in the same units used by the search comparator.
    """
    if var_y < 1e-30:
        return 1.0
    df = HUBER_DELTA_FRAC if delta_frac is None else delta_frac
    delta = float(df) * float(np.sqrt(var_y))
    r = y - pred
    abs_r = np.abs(r)
    quad  = np.minimum(abs_r, delta)
    lin   = abs_r - quad
    h     = 0.5 * quad * quad + delta * lin
    return float(np.mean(h) / (0.5 * var_y))


def _normalized_loss(pred, y, var_y):
    """Dispatch to the configured search loss (MSE or Huber)."""
    if SEARCH_LOSS == "huber":
        return _normalized_huber(pred, y, var_y)
    return _normalized_mse(pred, y, var_y)


# ── Post-processing model selection under parsimony control ───────────────────
# Structural ordering used only as a tie-break (simpler kind preferred).
_POST_ORDER = {"channel": 0, "linear": 0, "ols": 1, "nested": 2, "rational": 3}


def _post_eff_params(cand, min_w=1e-4):
    """Number of fitted terms a post-processing candidate actually uses."""
    kind = cand["kind"]
    if kind in ("channel", "linear"):
        return 1
    if kind == "ols":
        w = cand.get("ols_weights")
        return int(np.sum(np.abs(w) >= min_w)) if w is not None else 1
    if kind == "nested":
        nd = cand["nested"]; w = nd["w"]; valid = nd["valid"]
        return int(np.sum(valid & (np.abs(w) >= min_w)))
    if kind == "rational":
        r = cand["rational"]
        return int(np.sum(np.abs(r["a"]) >= min_w) + np.sum(np.abs(r["b"]) >= min_w))
    return 1


def _select_post_model(cands):
    """Pick the winning post-processing candidate under parsimony control.

    A more complex model wins only if it beats the simplest near-tied candidate
    by more than POST_PARSIMONY_REL_TOL (relative MSE).  Candidates exceeding
    POST_COMPLEXITY_MAX terms are rejected (the simplest is always kept as a
    fallback).  With POST_PARSIMONY_REL_TOL == 0 and POST_COMPLEXITY_MAX is None
    this reduces exactly to argmin(mse_norm) (legacy behaviour).
    """
    for c in cands:
        c["_k"] = _post_eff_params(c)
    cap  = POST_COMPLEXITY_MAX
    pool = [c for c in cands if cap is None or c["_k"] <= cap]
    if not pool:
        pool = [min(cands, key=lambda c: c["_k"])]
    best_mse = min(c["mse_norm"] for c in pool)
    thresh   = best_mse * (1.0 + POST_PARSIMONY_REL_TOL)
    near     = [c for c in pool if c["mse_norm"] <= thresh]
    # Among statistically-tied candidates, prefer the fewest terms, then the
    # simplest kind, then the lowest loss.
    return min(near, key=lambda c: (c["_k"], _POST_ORDER.get(c["kind"], 9), c["mse_norm"]))


def _ols_eval(pred_all, true_all, loss_mode="full"):
    """
    Dispatch OLS evaluation based on loss_mode:
      'full'    -- best channel + sparse OLS + nested OLS + poly-rational OLS (all degrees)
      'ols'     -- best channel + sparse OLS only
      'channel' -- best channel only (no OLS post-processing)
    Returns: (mse_norm, selected_c, ols_weights, best_channel_loss,
              lr_stub, nested, rational, valid_idx_global, analysis)
    analysis is a dict with all per-candidate data for verbose printing.
    """
    y     = true_all.squeeze().numpy()
    var_y = float(np.var(y))
    if var_y < 1e-30:
        return 1.0, 0, None, 1.0, None, None, None, None, {}

    # -- Single-channel path --
    if pred_all.ndim == 1 or pred_all.shape[-1] == 1:
        ch = pred_all.squeeze().numpy()
        if not np.all(np.isfinite(ch)) or np.std(ch) < 1e-12:
            return 1.0, 0, None, 1.0, None, None, None, np.array([0]), {}
        if loss_mode == "channel":
            r = correl(ch, y)
            ch_loss = float(1 - r ** 2) if np.isfinite(r) else 1.0
            return ch_loss, 0, None, ch_loss, None, None, None, np.array([0]), {}
        w, b, pred = _safe_lstsq(ch.reshape(-1, 1), y)
        if w is None:
            r = correl(ch, y); ch_loss = float(1 - r ** 2) if np.isfinite(r) else 1.0
            return ch_loss, 0, None, ch_loss, None, None, None, np.array([0]), {}
        mse_lin = _normalized_loss(pred, y, var_y)
        if loss_mode == "ols":
            analysis = {'var_y': var_y, 'channel_results': [(0, mse_lin)], 'n_ch': 1,
                        'ols_mse_norm': mse_lin, 'ols_kept_global': None,
                        'ols_bias': float(b) if b is not None else 0.0,
                        'ols_weights': None, 'nested': None, 'rational_candidates': []}
            return mse_lin, 0, None, mse_lin, None, None, None, np.array([0]), analysis
        # full: try nested + all rational degrees
        ch_mat  = ch.reshape(-1, 1)
        nested  = _nested_ols(ch_mat, y, mse_floor=mse_lin * var_y)
        floor   = min(mse_lin, (nested["mse"] / var_y) if nested else mse_lin) * var_y
        rat_candidates = []
        rational = None; best_rat_mse = np.inf
        for _deg in range(1, RAT_MAX_DEGREE + 1):
            _rc = _poly_rational_ols(ch_mat, y, mse_floor=floor, max_degree=_deg)
            if _rc is not None:
                rat_candidates.append(_rc)
                if rational is None or _rc["mse"] < best_rat_mse:
                    rational = _rc; best_rat_mse = _rc["mse"]
        cands = [{"kind": "linear", "mse_norm": mse_lin}]
        if nested is not None:
            cands.append({"kind": "nested", "mse_norm": nested["mse"] / var_y,
                          "nested": nested})
        if rational is not None:
            cands.append({"kind": "rational", "mse_norm": rational["mse"] / var_y,
                          "rational": rational})
        winner = _select_post_model(cands)
        analysis = {"var_y": var_y, "channel_results": [(0, mse_lin)], "n_ch": 1,
                    "ols_mse_norm": mse_lin, "ols_kept_global": None,
                    "ols_bias": float(b) if b is not None else 0.0,
                    "ols_weights": None, "nested": nested,
                    "rational_candidates": rat_candidates}
        if winner["kind"] == "rational":
            return rational["mse"] / var_y, 0, None, mse_lin, None, None, rational, np.array([0]), analysis
        if winner["kind"] == "nested":
            return nested["mse"] / var_y, 0, None, mse_lin, None, nested, None, np.array([0]), analysis
        return mse_lin, 0, None, mse_lin, None, None, None, np.array([0]), analysis

    # -- Multi-channel path --
    P    = pred_all.numpy()
    n_ch = P.shape[1]

    # Best single channel -- collect per-channel results
    best_channel_loss = 1.0
    selected_c        = 0
    channel_results   = []
    for c in range(n_ch):
        if not np.all(np.isfinite(P[:, c])) or np.std(P[:, c]) < 1e-12:
            channel_results.append((c, None))
            continue
        w, b, pred = _safe_lstsq(P[:, c:c+1], y)
        if w is None:
            channel_results.append((c, None))
            continue
        loss = _normalized_loss(pred, y, var_y)
        channel_results.append((c, loss))
        if loss < best_channel_loss:
            best_channel_loss = loss
            selected_c = c

    if loss_mode == "channel":
        analysis = {"var_y": var_y, "channel_results": channel_results, "n_ch": n_ch,
                    "ols_mse_norm": None, "ols_kept_global": None, "ols_bias": None,
                    "ols_weights": None, "nested": None, "rational_candidates": []}
        return best_channel_loss, selected_c, None, best_channel_loss, None, None, None, None, analysis

    # Sparse OLS via Ridge regression matching the notebook loss logic.
    mask = np.array([np.all(np.isfinite(P[:, c])) and np.std(P[:, c]) > 1e-15
                     for c in range(n_ch)])
    valid_idx_global = np.where(mask)[0]
    ols_loss    = 1.0
    ols_weights = None
    ols_bias    = 0.0
    ols_kept    = None
    lr_obj      = None
    if mask.sum() > 0:
        P_clean = P[:, mask]
        try:
            lr = Ridge(alpha=1e-8).fit(P_clean, y)
            r2 = lr.score(P_clean, y)
            if np.isfinite(r2) and r2 > 0:
                ols_loss = float(1.0 - r2)
                kept_global = np.zeros(n_ch, dtype=bool)
                kept_global[valid_idx_global] = True
                ols_kept = kept_global
                ols_weights = np.zeros(n_ch)
                ols_weights[mask] = lr.coef_
                ols_bias = float(lr.intercept_)
                class _LRStub: pass
                lr_obj            = _LRStub()
                lr_obj.coef_      = ols_weights
                lr_obj.intercept_ = lr.intercept_
                lr_obj._kept      = kept_global
        except Exception:
            pass

    if loss_mode == "ols":
        analysis = {"var_y": var_y, "channel_results": channel_results, "n_ch": n_ch,
                    "ols_mse_norm": ols_loss, "ols_kept_global": ols_kept, "ols_bias": ols_bias,
                    "ols_weights": ols_weights, "nested": None, "rational_candidates": []}
        if ols_weights is not None and ols_loss <= best_channel_loss:
            return ols_loss, selected_c, ols_weights, best_channel_loss, lr_obj, None, None, valid_idx_global, analysis
        return best_channel_loss, selected_c, None, best_channel_loss, None, None, None, valid_idx_global, analysis

    # Full: also try nested + rational (all degrees)
    nested = None
    if mask.sum() > 0:
        # Pre-select top RAT_MAX_BASIS_CHANS channels by |Pearson r| to keep nested OLS fast
        _P_valid = P[:, mask]
        _y_c = y - y.mean(); _sy = np.sqrt((_y_c**2).sum()) + 1e-30
        _corrs = np.array([
            abs(((_P_valid[:, j] - _P_valid[:, j].mean()) * _y_c).sum() /
                ((np.sqrt(((_P_valid[:, j] - _P_valid[:, j].mean())**2).sum()) + 1e-30) * _sy))
            if np.all(np.isfinite(_P_valid[:, j])) and np.std(_P_valid[:, j]) > 1e-15
            else 0.0
            for j in range(_P_valid.shape[1])
        ])
        _top_local  = np.argsort(-_corrs)[:RAT_MAX_BASIS_CHANS]
        _top_global = valid_idx_global[_top_local]
        nested_sub = _nested_ols(
            P[:, _top_global], y,
            mse_floor=min(ols_loss, best_channel_loss) * var_y)
        if nested_sub is not None:
            w_full = np.zeros(n_ch); w_full[_top_global] = nested_sub["w"]
            unaries_full = ["id"] * n_ch
            for k_i, j in enumerate(_top_global):
                unaries_full[j] = nested_sub["unaries"][k_i]
            valid_full = np.zeros(n_ch, dtype=bool)
            valid_full[_top_global] = nested_sub["valid"]
            nested = dict(mse=nested_sub["mse"], link=nested_sub["link"],
                          unaries=unaries_full, w=w_full, b=nested_sub["b"],
                          valid=valid_full, pred=nested_sub["pred"])

    # Poly-rational: loop through all degrees 1..RAT_MAX_DEGREE (fixed floor per ref notebook)
    rat_candidates = []
    rational       = None
    best_rat_mse   = np.inf
    if mask.sum() > 0:
        floor = min(ols_loss, best_channel_loss,
                    (nested["mse"] / var_y) if nested else np.inf) * var_y
        for _deg in range(1, RAT_MAX_DEGREE + 1):
            _rc = _poly_rational_ols(P[:, mask], y, mse_floor=floor, max_degree=_deg)
            if _rc is not None:
                rat_candidates.append(_rc)
                if rational is None or _rc["mse"] < best_rat_mse:
                    rational = _rc; best_rat_mse = _rc["mse"]

    analysis = {"var_y": var_y, "channel_results": channel_results, "n_ch": n_ch,
                "ols_mse_norm": ols_loss, "ols_kept_global": ols_kept, "ols_bias": ols_bias,
                "ols_weights": ols_weights, "nested": nested,
                "rational_candidates": rat_candidates}

    cands = [{"kind": "channel", "mse_norm": best_channel_loss}]
    if ols_weights is not None:
        cands.append({"kind": "ols", "mse_norm": ols_loss, "ols_weights": ols_weights})
    if nested is not None:
        cands.append({"kind": "nested", "mse_norm": nested["mse"] / var_y, "nested": nested})
    if rational is not None:
        cands.append({"kind": "rational", "mse_norm": rational["mse"] / var_y, "rational": rational})
    winner = _select_post_model(cands)

    if winner["kind"] == "rational":
        return (rational["mse"] / var_y, selected_c, ols_weights, best_channel_loss,
                lr_obj, None, rational, valid_idx_global, analysis)
    if winner["kind"] == "nested":
        return (nested["mse"] / var_y, selected_c, ols_weights, best_channel_loss,
                lr_obj, nested, None, valid_idx_global, analysis)
    if winner["kind"] == "ols":
        return (ols_loss, selected_c, ols_weights, best_channel_loss,
                lr_obj, None, None, valid_idx_global, analysis)
    return (best_channel_loss, selected_c, None, best_channel_loss,
            None, None, None, valid_idx_global, analysis)


def _print_ols_analysis(analysis, all_formulas, valid_idx_global, selected_c, f=None):
    """Print full OLS analysis like the reference notebook (steps 3a-3e).

    analysis         -- dict returned by _ols_eval (9th element)
    all_formulas     -- list of formula strings from graph_to_all_formulas
    valid_idx_global -- array mapping local OLS index -> global channel index
    selected_c       -- best single channel index
    f                -- optional open file object to mirror output
    """
    def _p(msg):
        print(msg)
        if f is not None:
            f.write(msg + "\n")

    var_y           = analysis.get("var_y", 1.0)
    channel_results = analysis.get("channel_results", [])
    n_ch            = analysis.get("n_ch", len(channel_results))
    ols_mse_norm    = analysis.get("ols_mse_norm", None)
    ols_kept        = analysis.get("ols_kept_global", None)
    ols_bias        = analysis.get("ols_bias", 0.0)
    ols_wts         = analysis.get("ols_weights", None)
    nested          = analysis.get("nested", None)
    rat_candidates  = analysis.get("rational_candidates", [])

    # -- 3a. Per-channel --
    _p(f"\n-- Single-channel analysis ({n_ch} channels) --")
    best_ch_norm = np.inf
    for c, mse_norm_c in channel_results:
        f_str = all_formulas[c] if c < len(all_formulas) else "?"
        if mse_norm_c is None:
            _p(f"  ch[{c}]: constant or NaN (skipped)  formula={f_str}")
        else:
            tag = " <<< SELECTED" if c == selected_c else ""
            _p(f"  ch[{c}]: normMSE={mse_norm_c:.10f}  R2={1.0 - mse_norm_c:.8f}  "
               f"formula={f_str}{tag}")
            if mse_norm_c < best_ch_norm:
                best_ch_norm = mse_norm_c

    # -- 3b. Sparse OLS --
    if ols_mse_norm is not None and ols_wts is not None:
        _p(f"\n-- Sparse OLS fit --")
        _p(f"  normMSE={ols_mse_norm:.10f}  R2={1.0 - ols_mse_norm:.8f}")
        _p("  Weights (kept after pruning):")
        for vi, w in enumerate(ols_wts):
            if abs(w) > 1e-6:
                f_i = all_formulas[vi] if vi < len(all_formulas) else "?"
                _p(f"    w={w:+.6f}  [{vi}] {f_i}")
        _p(f"    bias = {ols_bias:+.6f}")
    elif ols_mse_norm is not None:
        _p(f"\n-- Sparse OLS: normMSE={ols_mse_norm:.10f}  R2={1.0 - ols_mse_norm:.8f} --")

    # -- 3c. Nested OLS --
    if nested is not None:
        norm_nested = nested["mse"] / var_y
        _link = nested["link"]
        _p(f"\n-- Nested OLS --")
        _p(f"  link={_link}")
        _p("  Weights (kept after pruning):")
        valid_arr = nested.get("valid", np.zeros(n_ch, dtype=bool))
        w_arr     = nested.get("w", np.zeros(n_ch))
        u_arr     = nested.get("unaries", ["id"] * n_ch)
        for j in range(n_ch):
            if valid_arr[j]:
                f_j = all_formulas[j] if j < len(all_formulas) else "?"
                _p(f"    w={w_arr[j]:+.6f}  unary={u_arr[j]:5s}  [{j}] {f_j}")
        _p(f"    bias = {nested['b']:+.6f}")
        _p(f"  normMSE={norm_nested:.10f}  R2={1.0 - norm_nested:.8f}")
        nest_str = _format_nested_smart(nested, all_formulas)
        _p(f"  Formula = {nest_str}")
    else:
        _p("\n-- Nested OLS: no improvement over linear/channel baseline --")

    # -- 3d. Poly-Rational candidates (one per degree) --
    _p(f"\n-- Poly-Rational OLS (exploration deg 1 to {RAT_MAX_DEGREE}) --")
    best_rat      = None
    best_rat_norm = np.inf
    for rc in rat_candidates:
        deg     = rc.get("max_degree", "?")
        norm_rc = rc["mse"] / var_y
        rat_str = _format_poly_rational_smart(rc, all_formulas, valid_idx_global)
        _p(f"\n  > Degre {deg}:")
        _p(f"    normMSE={norm_rc:.10f}  R2={1.0 - norm_rc:.8f}")
        _p(f"    Formula = {rat_str}")
        if norm_rc < best_rat_norm:
            best_rat_norm = norm_rc
            best_rat      = rc
    if not rat_candidates:
        _p("  No rational formulation improved over baseline.")

    # -- 3e. Comparison table --
    _p("\n-- Comparison --")
    if best_ch_norm < np.inf:
        _p(f"  Best channel [{selected_c}]:  normMSE={best_ch_norm:.10f}  R2={1.0 - best_ch_norm:.8f}")
    if ols_mse_norm is not None and ols_wts is not None:
        _p(f"  Sparse OLS:              normMSE={ols_mse_norm:.10f}  R2={1.0 - ols_mse_norm:.8f}")
    if nested is not None:
        norm_n = nested["mse"] / var_y
        _link2 = nested["link"]
        _p(f"  Nested OLS:              normMSE={norm_n:.10f}  R2={1.0 - norm_n:.8f}  link={_link2}")
    if best_rat is not None:
        _deg2 = best_rat["max_degree"]
        _p(f"  Best Poly-Rational:      normMSE={best_rat_norm:.10f}  R2={1.0 - best_rat_norm:.8f}  deg={_deg2}")



def build_search_space(feature_names, feature_scores, operator_keys, all_combos=None):
    """Construct DRAGON search space for a given operator pool."""
    n = len(feature_names)
    if all_combos is None:
        all_combos = build_feature_combinations(feature_names, feature_scores)

    combo_weights = SelectFeatures.combination_weights(
        [feature_scores.get(c, 1.0/n) for c in feature_names],
        all_combos
    )

    # ── Node variable builders ────────────────────────────────────────
    select_var = HpVar(
        "SelectFeatures",
        Constant("SelectFeaturesOp", SelectFeatures, neighbor=ConstantInterval()),
        hyperparameters={
            "feature_indices": CatVar(
                "feature_indices",
                features=all_combos,
                weights=combo_weights,
                neighbor=CatInterval(),
            )
        },
        neighbor=HpInterval(),
    )
    unary_var = HpVar(
        "UnaryOp",
        CatVar("UnaryOpType", features=[Identity, Inverse, Negate], neighbor=CatInterval()),
        hyperparameters={},
        neighbor=HpInterval(),
    )
    power_var = HpVar(
        "Power",
        Constant("PowerOp", Power, neighbor=ConstantInterval()),
        hyperparameters={
            "exponent": CatVar("exponent", features=[-3, -2, -1, 1, 2, 3],
                               neighbor=CatInterval())
        },
        neighbor=HpInterval(),
    )
    sum_var = HpVar(
        "Sum", Constant("SumOp", SumFeatures, neighbor=ConstantInterval()),
        hyperparameters={}, neighbor=HpInterval(),
    )
    ln_var = HpVar(
        "Ln", Constant("LnOp", Ln, neighbor=ConstantInterval()),
        hyperparameters={}, neighbor=HpInterval(),
    )
    sin_var = HpVar(
        "Sin", Constant("SinOp", Sin, neighbor=ConstantInterval()),
        hyperparameters={}, neighbor=HpInterval(),
    )
    cos_var = HpVar(
        "Cos", Constant("CosOp", Cos, neighbor=ConstantInterval()),
        hyperparameters={}, neighbor=HpInterval(),
    )
    exp_var = HpVar(
        "Exp", Constant("ExpOp", Exp, neighbor=ConstantInterval()),
        hyperparameters={}, neighbor=HpInterval(),
    )

    key_map = {
        "select": select_var,
        "unary":  unary_var,
        "power":  power_var,
        "sum":    sum_var,
        "ln":     ln_var,
        "sin":    sin_var,
        "cos":    cos_var,
        "exp":    exp_var,
        # ConstantBrick (+const method)
        "const": HpVar(
            "ConstantBrick",
            Constant("ConstOp", ConstantBrick, neighbor=ConstantInterval()),
            hyperparameters={}, neighbor=HpInterval(),
        ),
        # Dilation = power with fixed positive exponent (proxy for dilat+offset)
        "dilation": HpVar(
            "Dilation",
            Constant("DilOp", Power, neighbor=ConstantInterval()),
            hyperparameters={
                "exponent": CatVar("exponent", features=[0.5, 1, 2, 3], neighbor=CatInterval())
            },
            neighbor=HpInterval(),
        ),
    }
    candidates = [key_map[k] for k in operator_keys if k in key_map]

    cand_ops = operations_var(
        "CandidateOperations",
        size=6,
        candidates=candidates,
        combiner_features=['add', 'mul'],
        activations=Constant("id", value=nn.Identity(), neighbor=ConstantInterval()),
        node_type=SymbolicNode,
    )
    dag = EvoDagVariable(
        label="Dag",
        operations=cand_ops,
        init_complexity=4,
        neighbor=EvoDagInterval(nb_mutations=2),
    )
    return ArrayVar(dag, label="Search Space", neighbor=ArrayInterval()), dag


def make_loss_function(search_space, train_loader, device, num_features,
                       feature_names, log_path, loss_mode="full",
                       optimize_constants=False,
                       subsample_ratio=1.0, _X_np=None, _y_np=None,
                       _sample_weights=None):
    """Factory returning (loss_function, state_dict) for DRAGON.

    loss_mode: 'full' = nested+rational OLS | 'ols' = sparse linear | 'channel' = best channel
    optimize_constants: if True, run Adam on the model when it has exactly 1 trainable
        parameter (i.e. a single ConstantBrick) so the constant gets fitted before the
        OLS evaluation step.
    state_dict is mutated in-place as new best solutions are found.
    """
    _state = {
        "best_loss":           np.inf,
        "winner_type":         "channel",
        "corr_value":          0.0,
        "alignment_loss":      1.0,
        "rat_degree":          None,
        "best_formula":        "N/A",
        # ── All formula variants for the best solution ──────────────────
        "formula_channel":     None,   # best single channel formula
        "formula_ols":         None,   # sparse OLS combination
        "formula_nested":      None,   # nested OLS (link + unaries)
        "formula_polyrat":     {},     # {deg_str: formula_str} all degrees tried
        # ── Losses per method (normMSE = 1-R²) ───────────────────────────────
        "loss_channel":        None,   # normMSE of best channel
        "loss_ols":            None,   # normMSE of sparse OLS
        "loss_nested":         None,   # normMSE of nested OLS
        "loss_polyrat":        {},     # {deg_str: normMSE}
    }

    def loss_function(args, idx, *kwargs):
        labels = [e.label for e in search_space]
        # Robust unpacking: seed DAGs are passed as bare AdjMatrix instead of a list
        if isinstance(args, AdjMatrix):
            args_dict = {labels[0]: args}
        else:
            args_dict = dict(zip(labels, args))

        model = MetaArchi(args_dict, input_shape=(num_features,)).to(device)

        # ── Optional: optimize ConstantBrick scalar via Adam ─────────
        if optimize_constants:
            params = sum(p.numel() for p in model.parameters())
            if params == 1:
                # Make sure model & data share dtype (some synthetic targets
                # produce float64 tensors → backward() throws "Found dtype
                # Double but expected Float").
                model = model.float()
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                mse_fn = nn.MSELoss()
                eps = 1e-12
                max_initial_loss = 1.0
                model.train()
                for epoch in range(100):
                    epoch_loss = 0.0
                    n_seen = 0
                    for Xb, yb in train_loader:
                        Xb = Xb.to(device).float()
                        yb = yb.to(device).float()
                        opt.zero_grad()
                        pred = model(Xb)
                        loss = mse_fn(pred, yb)
                        loss.backward()
                        opt.step()
                        epoch_loss += loss.item() * Xb.size(0)
                        n_seen += Xb.size(0)
                    epoch_loss /= max(n_seen, 1)
                    if epoch == 0 and (epoch_loss > max_initial_loss or np.isnan(epoch_loss)):
                        break
                    if epoch_loss < eps:
                        break

        model.eval()
        # ── Stochastic subsampling: draw a fresh random subset each call ──
        # Each DAG evaluation sees a different slice of the data; noisy
        # samples never consistently dominate the loss signal.  Over many
        # iterations the noise averages out (implicit denoising).
        if subsample_ratio < 1.0 and _X_np is not None and _y_np is not None:
            _n_total = len(_y_np)
            _n_sub   = max(int(_n_total * subsample_ratio), min(_n_total, 20))
            # Sampling probabilities: MC-Dropout confidence if available
            _sub_prob = None
            if _sample_weights is not None and len(_sample_weights) == _n_total:
                _w_pos = np.maximum(_sample_weights, 0.0)
                _w_sum = _w_pos.sum()
                _sub_prob = (_w_pos / _w_sum) if _w_sum > 0 else None
            _sub_rng = np.random.default_rng(int(idx) % (2 ** 31))
            _sub_idx = _sub_rng.choice(_n_total, _n_sub, replace=False, p=_sub_prob)
            _Xb_sub  = torch.tensor(_X_np[_sub_idx], dtype=torch.float32).to(device)
            _yb_sub  = torch.tensor(_y_np[_sub_idx], dtype=torch.float32).reshape(-1, 1).to(device)
            with torch.no_grad():
                pred_all = model(_Xb_sub).detach().cpu()
            true_all = _yb_sub.detach().cpu()
        else:
            all_pred, all_true = [], []
            with torch.no_grad():
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    all_pred.append(model(Xb).detach().cpu())
                    all_true.append(yb.detach().cpu())
            pred_all = torch.cat(all_pred)
            true_all = torch.cat(all_true)

        (mse, selected_c, ols_weights, alignment_loss,
         _lr, nested, rational, valid_idx_global, _analysis) = _ols_eval(pred_all, true_all, loss_mode)

        # Determine winner type and build prediction array
        y_np = true_all.squeeze().numpy()
        if rational is not None:
            winner_type = "rational"
            rat_degree  = rational.get("max_degree", RAT_MAX_DEGREE)
            y_pred      = rational["pred"]
        elif nested is not None:
            winner_type = "nested"
            rat_degree  = None
            y_pred      = nested["pred"]
        elif ols_weights is not None and np.any(ols_weights != 0):
            winner_type = "ols"
            rat_degree  = None
            P    = pred_all.numpy()
            kept = (ols_weights != 0)
            y_pred = P[:, kept] @ ols_weights[kept]
            y_pred = y_pred + float(np.mean(y_np - y_pred))
        else:
            winner_type = "channel"
            rat_degree  = None
            if pred_all.ndim > 1 and pred_all.shape[-1] > 1:
                _ch_raw = pred_all[:, selected_c].numpy()
            else:
                _ch_raw = pred_all.squeeze().numpy()
            # Linear post-processing alignment:  y ≈ a1 * channel + a0
            _A = np.column_stack([_ch_raw, np.ones_like(_ch_raw)])
            _coeffs, _, _, _ = np.linalg.lstsq(_A, y_np, rcond=None)
            _a1, _a0 = float(_coeffs[0]), float(_coeffs[1])
            y_pred = _a1 * _ch_raw + _a0
            _state["_channel_a1"] = _a1
            _state["_channel_a0"] = _a0

        # ── MC-Dropout weighted loss (full-data path only) ──────────────
        # Recompute mse using per-sample confidence weights so that
        # high-uncertainty (noisy) points contribute less to the score.
        # This overrides the OLS-internal mse only when weights are
        # available and we evaluated the full dataset (not a subsample).
        if (_sample_weights is not None
                and subsample_ratio >= 1.0
                and y_pred is not None):
            try:
                _w  = np.asarray(_sample_weights, dtype=np.float64)
                _w  = np.maximum(_w, 0.0)
                _ws = _w.sum()
                if _ws > 0 and len(_w) == len(y_np):
                    _w  = _w / _ws
                    _ym_w  = float(np.dot(_w, y_np))
                    _var_w = float(np.dot(_w, (y_np - _ym_w) ** 2))
                    if _var_w > 1e-30:
                        _resid = y_np - np.asarray(y_pred).ravel()
                        if SEARCH_LOSS == "huber":
                            _delta = HUBER_DELTA_FRAC * float(np.sqrt(_var_w))
                            _abs_r = np.abs(_resid)
                            _quad  = np.minimum(_abs_r, _delta)
                            _lin   = _abs_r - _quad
                            _h     = 0.5 * _quad ** 2 + _delta * _lin
                            mse    = float(np.dot(_w, _h) / (0.5 * _var_w))
                        else:
                            mse = float(np.dot(_w, _resid ** 2) / _var_w)
            except Exception:
                pass

        corr_val = float(correl(
            torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred,
            true_all))

        if mse < _state["best_loss"]:
            _state["best_loss"]      = mse
            _state["winner_type"]    = winner_type
            _state["corr_value"]     = corr_val
            _state["alignment_loss"] = float(alignment_loss)
            _state["rat_degree"]     = rat_degree
            # ── Stash the winning prediction (used by 'boosted_spar' to
            #    OLS-combine the 4 stream-best ŷ across threads).
            try:
                _y_pred_np = (y_pred.detach().cpu().numpy()
                              if isinstance(y_pred, torch.Tensor)
                              else np.asarray(y_pred, dtype=np.float64))
                _state["best_pred_np"] = np.asarray(_y_pred_np, dtype=np.float64).ravel()
                _state["best_y_np"]    = np.asarray(y_np, dtype=np.float64).ravel()
            except Exception:
                pass
            try:
                adj      = model.dag.matrix
                nodes    = model.dag.operations
                formulas = graph_to_all_formulas(adj, feature_names, nodes)
                _state["best_formula"] = str(formulas[selected_c]) if formulas else "N/A"

                # ── DAG meta (ops list, const presence, size, text view) ────
                try:
                    _ops_str, _const_str, _dag_size, _dag_text, _dag_svg, _dag_data = _dag_summary(adj, nodes)
                    _state["ops_used"]   = _ops_str
                    _state["has_const"]  = _const_str
                    _state["dag_size"]   = _dag_size
                    _state["dag_text"]   = _dag_text
                    _state["dag_svg"]    = _dag_svg
                    _state["dag_data"]   = _dag_data
                except Exception:
                    pass

                # ── Extract all per-method formula variants ─────────────────
                # Channel formula — always wrap with the SELECTED channel's
                # own linear alignment  a1*(channel) + a0, computed on the
                # channel itself (independent of the winning method) so the
                # HTML "Channel" tab shows the usable aligned formula.
                _f_ch_raw = (str(formulas[selected_c])
                             if formulas and selected_c < len(formulas) else None)
                if _f_ch_raw is not None:
                    if pred_all.ndim > 1 and pred_all.shape[-1] > 1:
                        _ch_only = pred_all[:, selected_c].numpy().ravel()
                    else:
                        _ch_only = pred_all.squeeze().numpy().ravel()
                    _A_ch = np.column_stack([_ch_only, np.ones_like(_ch_only)])
                    _co_ch, _, _, _ = np.linalg.lstsq(_A_ch, y_np.ravel(), rcond=None)
                    _ch_a1, _ch_a0 = float(_co_ch[0]), float(_co_ch[1])
                    if abs(_ch_a1 - 1.0) > 1e-4 or abs(_ch_a0) > 1e-4:
                        _f_ch = f"{_ch_a1:.6e} * ({_f_ch_raw}) + {_ch_a0:.6e}"
                    else:
                        _f_ch = _f_ch_raw
                else:
                    _f_ch = None
                _state["formula_channel"] = _f_ch

                # ── All channel formulas (one per output channel) ───────────
                _ch_results = _analysis.get("channel_results", []) or []
                _ch_loss_map = {int(c): (float(lv) if lv is not None and np.isfinite(lv) else None)
                                for c, lv in _ch_results}
                _all_chs = []
                for _ci, _f in enumerate(formulas or []):
                    _all_chs.append({
                        "idx": _ci,
                        "formula": str(_f),
                        "loss": _ch_loss_map.get(_ci),
                        "selected": (_ci == selected_c),
                    })
                _state["all_channel_formulas"] = _all_chs

                # Sparse OLS formula
                _ols_wts  = _analysis.get("ols_weights")
                _ols_bias = _analysis.get("ols_bias", 0.0)
                if _ols_wts is not None and formulas:
                    _nz = [(i, w) for i, w in enumerate(_ols_wts) if abs(w) > 1e-6]
                    _terms = [f"{w:+.4f}*({formulas[i]})"
                              for i, w in _nz if i < len(formulas)]
                    if abs(_ols_bias) > 1e-6:
                        _terms.append(f"{_ols_bias:+.4f}")
                    _state["formula_ols"] = " ".join(_terms) if _terms else None
                else:
                    _state["formula_ols"] = None

                # Nested OLS formula
                _nested_ana = _analysis.get("nested")
                _state["formula_nested"] = (
                    _format_nested_smart(_nested_ana, formulas)
                    if _nested_ana is not None and formulas else None)

                # Poly-rational per degree
                _polyrat_dict = {}
                for _rc in _analysis.get("rational_candidates", []):
                    _deg_key = str(_rc.get("max_degree", "?"))
                    if valid_idx_global is not None and formulas:
                        _polyrat_dict[_deg_key] = _format_poly_rational_smart(
                            _rc, formulas, valid_idx_global)
                _state["formula_polyrat"] = _polyrat_dict

                # ── Per-method losses (normMSE) ─────────────────────────────
                _var_y = _analysis.get("var_y", 1.0) or 1.0
                _state["loss_channel"] = float(_analysis["channel_results"][selected_c][1]) \
                    if _analysis.get("channel_results") and selected_c < len(_analysis["channel_results"]) \
                    and _analysis["channel_results"][selected_c][1] is not None else None
                _state["mse_channel"] = (float(_analysis["channel_results"][selected_c][1]) * _var_y) \
                    if _analysis.get("channel_results") and selected_c < len(_analysis["channel_results"]) \
                    and _analysis["channel_results"][selected_c][1] is not None else None
                _ols_nm = _analysis.get("ols_mse_norm")
                _state["loss_ols"] = float(_ols_nm) if _ols_nm is not None and np.isfinite(_ols_nm) else None
                _state["mse_ols"] = (float(_ols_nm) * _var_y) if _ols_nm is not None and np.isfinite(_ols_nm) else None
                _nest = _analysis.get("nested")
                _state["loss_nested"] = float(_nest["mse"] / _var_y) \
                    if _nest is not None and np.isfinite(_nest["mse"]) else None
                _state["mse_nested"] = float(_nest["mse"]) \
                    if _nest is not None and np.isfinite(_nest["mse"]) else None
                _loss_polyrat = {}
                _mse_polyrat = {}
                for _rc in _analysis.get("rational_candidates", []):
                    _dk = str(_rc.get("max_degree", "?"))
                    _nm = _rc["mse"] / _var_y
                    if np.isfinite(_nm):
                        _loss_polyrat[_dk] = float(_nm)
                        _mse_polyrat[_dk] = float(_rc["mse"])
                _state["loss_polyrat"] = _loss_polyrat
                _state["mse_polyrat"] = _mse_polyrat

                # ── Set best_formula to the actual winning method's formula ──
                if winner_type == "rational":
                    _rat_key = str(rat_degree) if rat_degree is not None else next(iter(_polyrat_dict), None)
                    _state["best_formula"] = _polyrat_dict.get(_rat_key) or _state["best_formula"]
                elif winner_type == "nested":
                    _state["best_formula"] = _state["formula_nested"] or _state["best_formula"]
                elif winner_type == "ols":
                    _state["best_formula"] = _state["formula_ols"] or _state["best_formula"]
                else:  # channel: formula_channel already carries a1*(.)+a0
                    _state["best_formula"] = _state.get("formula_channel") or _state["best_formula"]

                with open(log_path, "a") as f:
                    f.write(f"=== NEW BEST  Idx={idx}  Loss={mse:.10f}  "
                            f"Winner={winner_type.upper()}  "
                            f"alignment_loss={alignment_loss:.10f}\n")
                    _print_ols_analysis(_analysis, formulas, valid_idx_global, selected_c, f=f)
                    f.write("\n")

                print(f"\n  >> NEW BEST loss={mse:.10f} [{winner_type.upper()}]  Idx={idx}")
                _print_ols_analysis(_analysis, formulas, valid_idx_global, selected_c)
            except Exception as e:
                print(f"  >> Could not extract formula: {e}")

        print(f"Idx={idx}, Loss = {mse:.10f}")
        df_pred = pd.DataFrame({"pred": y_pred, "true": y_np})
        model.set_prediction_to_save("prediction", df_pred)
        return float(mse), model

    return loss_function, _state


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM INIT — seed DAGs (chain / fan / skip / rand topologies)
# ══════════════════════════════════════════════════════════════════════════════

def _build_random_seed_dags(feature_names, operator_keys, seed=0, n_seeds=200):
    """Build a population of seed DAGs for the 'random' init strategy.

    Mirrors the reference notebook seed_dags scheme: chain / fan / skip / random
    topologies built from a small op_pool of {Identity, Inverse, Negate,
    SumFeatures, Power(±1..3), SelectFeatures(singles + pairs)} with combiner
    patterns (all-add / all-mul / alternating).
    """
    import random as _rnd
    rng = _rnd.Random(seed)
    n_feat = len(feature_names)
    if n_feat == 0:
        return None

    # Build op_pool ---------------------------------------------------------
    op_pool = []
    def _add(name, op_cls, hp=None, act=None):
        op_pool.append((name, op_cls, hp or {}, act or nn.Identity()))

    _add("Identity",   Identity)
    _add("Inverse",    Inverse)
    _add("Negate",     Negate)
    _add("SumFeatures", SumFeatures)
    for exp in (-3, -2, -1, 1, 2, 3):
        _add(f"Power_{exp}", Power, hp={"exponent": exp})
    # SelectFeatures singles + pairs
    for i in range(n_feat):
        _add(f"Sel_{i}", SelectFeatures, hp={"indices": [i]})
    for i, j in combinations(range(n_feat), 2):
        _add(f"Sel_{i}_{j}", SelectFeatures, hp={"indices": [i, j]})

    # Filter op_pool by available operators (best-effort)
    if "const" in operator_keys:
        _add("Const", ConstantBrick)

    combiner_patterns = [
        lambda k: "add",
        lambda k: "mul",
        lambda k: "add" if k % 2 == 0 else "mul",
        lambda k: "mul" if k % 2 == 0 else "add",
    ]
    topologies = ["chain", "fan", "skip", "rand"]

    seed_dags = []
    for s in range(n_seeds):
        size = rng.randint(3, 7)
        topo = topologies[s % len(topologies)]
        comb_fn = combiner_patterns[s % len(combiner_patterns)]

        nodes = []
        for k in range(size):
            name, op_cls, hp, act = op_pool[rng.randrange(len(op_pool))]
            nodes.append(SymbolicNode(
                combiner=comb_fn(k), operation=op_cls,
                hp=dict(hp), activation=act,
            ))

        # Build adjacency matrix
        M = np.zeros((size, size), dtype=int)
        if topo == "chain":
            for k in range(size - 1):
                M[k, k + 1] = 1
        elif topo == "fan":
            for k in range(1, size):
                M[0, k] = 1
        elif topo == "skip":
            for k in range(size - 1):
                M[k, k + 1] = 1
            for k in range(size - 2):
                M[k, k + 2] = 1
        else:  # rand
            for k in range(size - 1):
                M[k, k + 1] = 1  # ensure connectivity
            for i in range(size):
                for j in range(i + 1, size):
                    if rng.random() < 0.3:
                        M[i, j] = 1

        try:
            seed_dags.append(AdjMatrix(operations=nodes, matrix=M))
        except Exception:
            continue

    # Wrap each AdjMatrix in a list to match the ArrayVar([dag]) configuration format
    return [[m] for m in seed_dags] if seed_dags else None


# ══════════════════════════════════════════════════════════════════════════════
#  DRAGON WORKER — runs in a subprocess
# ══════════════════════════════════════════════════════════════════════════════

def dragon_worker(method_cfg: dict, target: str, run_id: int,
                  *, _y_predenoised=None, _denoise_info=None,
                  _max_iters: int = None,
                  _X_preloaded=None, _y_preloaded=None) -> dict:
    """Entry point for each parallel DRAGON process.

    method_cfg : one element of DRAGON_METHOD_CONFIGS
    run_id     : 0..N_RUNS-1, also indexes INIT_STRATEGIES
    """
    method_id   = method_cfg["id"]

    # ── Smart-parallel dispatcher ────────────────────────────────────
    # When `smart_parallel` is enabled and we are NOT already inside a
    # spawned stream, fan out 4 worker streams (each with its own operator
    # subset) using a ThreadPoolExecutor and return the best result.
    if method_cfg.get("smart_parallel") and not method_cfg.get("_in_smart_stream"):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        stream_results = []
        with ThreadPoolExecutor(max_workers=len(SPAR_OP_GROUPS)) as ex:
            futs = {}
            # Divide the hard cap evenly across parallel streams so the
            # total actual_T stays within DRAGON_N_ITERATIONS.
            _n_streams      = len(SPAR_OP_GROUPS)
            _iter_cap       = _max_iters if _max_iters is not None else DRAGON_N_ITERATIONS
            _stream_budget  = max(1, _iter_cap // _n_streams)
            for stream_id, ops in SPAR_OP_GROUPS.items():
                sub_cfg = dict(method_cfg)
                sub_cfg["operators"]         = list(ops)
                sub_cfg["_in_smart_stream"]  = True
                sub_cfg["_stream_id"]        = stream_id
                futs[ex.submit(dragon_worker, sub_cfg, target, run_id,
                               _y_predenoised=_y_predenoised,
                               _denoise_info=_denoise_info,
                               _max_iters=_stream_budget,
                               _X_preloaded=_X_preloaded,
                               _y_preloaded=_y_preloaded)] = stream_id
            for f in as_completed(futs):
                try:
                    stream_results.append(f.result())
                except Exception:
                    print(f"[spar/{target}/run{run_id}] stream {futs[f]} ERROR:")
                    traceback.print_exc()
        if not stream_results:
            return {
                "target": target, "method": method_id, "run_id": run_id,
                "strategy": INIT_STRATEGIES[run_id],
                "loss": float(np.inf), "r2": 0.0,
                "formula": "ERROR: all smart-parallel streams failed",
                "time_s": 0.0, "log_path": "",
                "description": method_cfg["description"],
                "search_space_ops": list(method_cfg.get("operators", [])),
            }
        best = min(stream_results, key=lambda r: r.get("loss", float("inf")))
        # Re-brand as the parent method ("spar") and attach a per-stream summary.
        best["method"]            = method_id
        best["description"]       = method_cfg["description"]
        best["search_space_ops"]  = list(method_cfg.get("operators", []))
        # ── Total search effort accounting ──────────────────────────────
        # `actualT` from each stream = #rows in its computation_file.csv,
        # i.e. the number of model evaluations actually performed by that
        # Mutant_UCB run.  For a fair comparison with serial methods we
        # report the *sum* across all streams as the total search budget
        # consumed (wall-clock / CPU-equivalent).  We also keep the
        # winner's own count under `actualT_winner` and the per-stream
        # breakdown under `actualT_per_stream` for diagnostics.
        per_stream_T = {
            r.get("_stream_id", "?"): int(r["actualT"])
            for r in stream_results if r.get("actualT") is not None
        }
        best["actualT_per_stream"] = per_stream_T
        best["actualT_winner"]     = best.get("actualT")
        if per_stream_T:
            best["actualT"] = int(sum(per_stream_T.values()))
        best["smart_parallel_streams"] = [
            {"stream":  r.get("_stream_id", "?"),
             "ops":     r.get("search_space_ops", []),
             "loss":    r.get("loss"),
             "formula": r.get("formula"),
             "time_s":  r.get("time_s"),
             "actualT": r.get("actualT")}
            for r in stream_results
        ]
        # ── Boosted combination across streams (boosted_spar) ───────────
        # Stack each stream's winning ŷ as columns of a [n × K] matrix and
        # let `_ols_eval` pick the best meta-combiner among
        # sparse-OLS / nested-OLS / poly-rational-OLS.  Replaces the best
        # only if the boosted loss is strictly lower than the best stream.
        if method_cfg.get("boosted"):
            try:
                # Collect aligned (ŷ, y) pairs.  Streams may have permuted
                # the data differently (per-thread DataLoader shuffle); we
                # re-align by stably sorting on y so column indices match.
                preds, sids, y_ref = [], [], None
                ref_order = None
                for r in stream_results:
                    p  = r.get("_best_pred_np")
                    yy = r.get("_best_y_np")
                    if p is None or yy is None:
                        continue
                    p  = np.asarray(p,  dtype=np.float64).ravel()
                    yy = np.asarray(yy, dtype=np.float64).ravel()
                    if p.shape != yy.shape or p.size < 4:
                        continue
                    if not np.all(np.isfinite(p)):
                        continue
                    if y_ref is None:
                        y_ref     = yy.copy()
                        ref_order = np.argsort(y_ref, kind="stable")
                        y_ref_sorted = y_ref[ref_order]
                    else:
                        if yy.size != y_ref.size:
                            continue
                        # Same multiset of y values? (sanity check)
                        if not np.allclose(np.sort(yy), np.sort(y_ref),
                                           rtol=1e-8, atol=1e-12):
                            continue
                    self_order = np.argsort(yy, kind="stable")
                    # p in y_ref order: place p[self_order[i]] at ref_order[i]
                    p_aligned = np.empty_like(p)
                    p_aligned[ref_order] = p[self_order]
                    preds.append(p_aligned)
                    sids.append(r.get("_stream_id", "?"))
                if len(preds) >= 2:
                    P = np.column_stack(preds).astype(np.float64)
                    (mse_b, sel_c_b, w_b, ch_loss_b,
                     _lr_b, nested_b, rational_b, _vidx_b, ana_b) = _ols_eval(
                        torch.tensor(P), torch.tensor(y_ref).reshape(-1, 1),
                        loss_mode="full")
                    if np.isfinite(mse_b) and mse_b < best.get("loss", float("inf")):
                        # Build a concise textual formula
                        if rational_b is not None:
                            wtype = "boosted-polyrat"
                            f_str = (f"poly-rational(deg≤{rational_b.get('max_degree','?')}) "
                                     f"over {len(sids)} streams")
                        elif nested_b is not None:
                            wtype = "boosted-nested"
                            f_str = f"nested-OLS over {len(sids)} streams"
                        elif w_b is not None and np.any(np.asarray(w_b) != 0):
                            wtype = "boosted-ols"
                            wts   = np.asarray(w_b).ravel()
                            bias  = float(ana_b.get("ols_bias", 0.0)) if ana_b else 0.0
                            terms = []
                            for sid, w in zip(sids, wts):
                                if abs(float(w)) > 1e-8:
                                    terms.append(f"{float(w):+.4g}·ŷ_{sid}")
                            if abs(bias) > 1e-8:
                                terms.append(f"{bias:+.4g}")
                            f_str = " ".join(terms) if terms else "(boosted-OLS)"
                        else:
                            wtype = "boosted-channel"
                            f_str = f"best ŷ_{sids[sel_c_b]} (boosted)"
                        best["loss"]           = float(mse_b)
                        best["r2"]             = float(1.0 - mse_b) if mse_b <= 1.0 else 0.0
                        best["formula"]        = "boosted: " + f_str
                        best["winner_type"]    = wtype
                        best["alignment_loss"] = float(ch_loss_b) if ch_loss_b is not None else float(mse_b)
                        best["boosted_streams"] = list(sids)
            except Exception as _e:
                print(f"[{method_id}/{target}/run{run_id}] boost FAILED: {_e}")
        # Strip numpy arrays before returning (not JSON-serialisable).
        for r in stream_results:
            r.pop("_best_pred_np", None)
            r.pop("_best_y_np",    None)
        best.pop("_best_pred_np", None)
        best.pop("_best_y_np",    None)
        return best

    strategy    = INIT_STRATEGIES[run_id]
    if method_cfg.get("_in_smart_stream"):
        run_dir = os.path.join(OUTPUT_DIR, target, method_id, f"run_{run_id}",
                               f"stream_{method_cfg['_stream_id']}")
    else:
        run_dir = os.path.join(OUTPUT_DIR, target, method_id, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    log_path    = os.path.join(run_dir, f"{target}_{method_id}{LOG_SUFFIX}")
    save_dir    = os.path.join(run_dir, "save")

    t_start   = time.time()
    best_loss = np.inf
    best_formula = "N/A"
    result_phase = None
    loss_state = {}
    loss_history = []
    actual_T = None
    landscape_svg = None

    try:
        # ── Data ─────────────────────────────────────────────────────
        if _X_preloaded is not None and _y_preloaded is not None:
            X_df = _X_preloaded
            y    = _y_preloaded.copy()
        else:
            X_df, y = load_dataset(target, run_id=run_id, strategy=strategy)
        seed = RANDOM_SEED + run_id
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── Optional: add Gaussian noise to y ────────────────────────
        if method_cfg.get("add_noise", False):
            noise_rng = np.random.default_rng(seed + 9999)
            y = y + noise_rng.normal(0, NOISE_STD * float(y.std()), len(y))

        # ── Optional: GP pre-denoising on the (possibly noisy) y ─────
        # Applied AFTER add_noise so GPR smooths the noisy signal.
        # Intended to be paired with denoise_method="stoch_sub" (in-loop).
        pre_dm = method_cfg.get("pre_denoise_method")
        if pre_dm:
            try:
                y, _pre_dn_info = _apply_denoise(X_df, y, pre_dm)
                print(f"[{method_id}/{target}/run{run_id}] pre_denoise({pre_dm}): "
                      f"{_pre_dn_info}")
            except Exception as _e:
                print(f"[{method_id}/{target}/run{run_id}] pre_denoise({pre_dm}) "
                      f"FAILED ({_e}); continuing with raw noisy y")

        # ── Optional: blind denoising of y ──────────────────────────
        # The preferred path uses a precomputed (y_clean, info) passed in
        # from run_all(), where denoising is done ONCE per (target, run_id)
        # and shared across all Dragon methods that request the same method.
        # Fallback: compute here if no precomputed value was provided.
        denoise_info = None
        dm = method_cfg.get("denoise_method")
        if dm:
            if dm == "stoch_sub":
                # stoch_sub is purely in-loop (random subsampling inside
                # loss_function at each evaluation) — y is never preprocessed.
                # Keeping y as-is preserves any noise added by add_noise=True.
                denoise_info = {"method": "stoch_sub", "note": "in-loop subsampling"}
            elif _y_predenoised is not None:
                # Use the precomputed denoised y (fast path — no recomputation)
                y = _y_predenoised
                denoise_info = _denoise_info
            else:
                # Fallback: compute denoising here (e.g. called standalone)
                try:
                    y, denoise_info = _apply_denoise(X_df, y, dm)
                    print(f"[{method_id}/{target}/run{run_id}] denoise({dm}): "
                          f"{denoise_info}")
                except Exception as _e:
                    print(f"[{method_id}/{target}/run{run_id}] denoise({dm}) "
                          f"FAILED ({_e}); continuing with raw y")
        elif method_cfg.get("denoise", False):
            # Legacy key for backward compatibility → fall back to "auto".
            try:
                y, denoise_info = _apply_denoise(X_df, y, "auto")
                print(f"[{method_id}/{target}/run{run_id}] legacy denoise(auto): "
                      f"{denoise_info}")
            except Exception as _e:
                print(f"[{method_id}/{target}/run{run_id}] denoise FAILED ({_e}); "
                      f"continuing with raw y")

        # ── Variable augmentation for synthetic targets ───────────────
        # Progressive template-matching: rank all unary/binary candidate
        # features by Pearson r² with y; add only the single best feature
        # (or the two features that form the best binary pair when neither
        # a single candidate nor a raw feature already explains y well).
        use_var_aug = method_cfg.get("var_aug", True)
        if use_var_aug and target in _SYNTH_CATEGORIES:
            from itertools import combinations as _combos
            _F32_MAX  = np.finfo(np.float32).max
            _THRESHOLD = 0.999
            _raw_set  = set(X_df.columns.tolist())
            _y_arr    = y.values.ravel().astype(np.float64)
            # ── Candidate generation ──────────────────────────────────
            _cands = {c: X_df[c].values.astype(np.float64) for c in X_df.columns}
            for _c in list(X_df.columns):
                _v = _cands[_c]
                _cands[f"({_c})**2"]            = _v ** 2
                _cands[f"({_c})**3"]            = _v ** 3
                with np.errstate(divide='ignore', invalid='ignore'):
                    _cands[f"1/({_c})"]         = np.where(np.abs(_v) > 1e-12, 1.0 / _v, 0.0)
                _cands[f"sqrt(abs({_c}))"]      = np.sqrt(np.abs(_v))
                _cands[f"-({_c})"]              = -_v
                _cands[f"log(abs({_c})+1e-12)"] = np.log(np.abs(_v) + 1e-12)
            for _ci, _cj in _combos(list(X_df.columns), 2):
                _vi = X_df[_ci].values.astype(np.float64)
                _vj = X_df[_cj].values.astype(np.float64)
                _cands[f"({_ci}+{_cj})"]  = _vi + _vj
                _cands[f"({_ci}-{_cj})"]  = _vi - _vj
                _cands[f"({_cj}-{_ci})"]  = _vj - _vi
                _cands[f"({_ci}*{_cj})"]  = np.clip(_vi * _vj, -_F32_MAX, _F32_MAX)
                with np.errstate(divide='ignore', invalid='ignore'):
                    _cands[f"({_ci}/{_cj})"] = np.where(np.abs(_vj) > 1e-12, _vi / _vj, 0.0)
                    _cands[f"({_cj}/{_ci})"] = np.where(np.abs(_vi) > 1e-12, _vj / _vi, 0.0)
            # ── Clean (clip, drop constant non-raw features) ─────────
            _cands = {
                _nm: np.clip(np.where(np.isfinite(_v), _v, 0.0), -_F32_MAX, _F32_MAX)
                for _nm, _v in _cands.items()
                if _nm in _raw_set or np.std(_v) > 1e-15
            }
            _cand_names = list(_cands.keys())
            _cand_vals  = np.array([_cands[_nm] for _nm in _cand_names], dtype=np.float64)
            # ── Vectorised Pearson r² (batch over N candidates × n_samples) ──
            _n   = len(_y_arr)
            _y_c = _y_arr - _y_arr.mean()
            _y_s = _y_arr.std()

            def _batch_r2(_T):
                _T  = np.where(np.isfinite(_T), _T, 0.0)
                _Tc = _T - _T.mean(axis=1, keepdims=True)
                _Ts = np.sqrt((_Tc ** 2).mean(axis=1))
                _d  = _Tc @ _y_c / _n
                return np.divide(_d, _Ts * _y_s,
                                 where=_Ts > 1e-15,
                                 out=np.zeros(_T.shape[0])) ** 2

            # ── Phase 1: best single candidate ───────────────────────
            _r2_s     = _batch_r2(_cand_vals)
            _bi       = int(np.argmax(_r2_s))
            _best_s_nm, _best_s_r2 = _cand_names[_bi], float(_r2_s[_bi])
            # ── Phase 2: best binary pair ─────────────────────────────
            _N    = len(_cand_names)
            _bp_r2 = 0.0
            _bp    = (None, None)
            for _i in range(_N):
                _V2 = _cand_vals[_i + 1:]
                if _V2.shape[0] == 0:
                    break
                _v1 = _cand_vals[_i]
                _br = np.zeros(_V2.shape[0])
                with np.errstate(divide='ignore', invalid='ignore'):
                    _templates = [
                        _v1 + _V2,
                        _v1 - _V2,
                        _V2 - _v1,
                        np.clip(_v1 * _V2, -_F32_MAX, _F32_MAX),
                        np.where(np.abs(_V2) > 1e-12, _v1 / _V2, 0.0),
                        np.where(np.abs(_v1) > 1e-12, _V2 / _v1, 0.0),
                    ]
                for _T in _templates:
                    _r2 = _batch_r2(np.atleast_2d(_T))
                    _m  = _r2 > _br
                    _br[_m] = _r2[_m]
                for _k in range(_V2.shape[0]):
                    if _br[_k] > _bp_r2:
                        _bp_r2 = _br[_k]
                        _bp    = (_cand_names[_i], _cand_names[_i + 1 + _k])
            # ── Decision: prefer single if r² is high enough ─────────
            _s_new = 0 if _best_s_nm in _raw_set else 1
            _p_new = sum(1 for _nm in _bp if _nm and _nm not in _raw_set)
            if _best_s_r2 >= _THRESHOLD and _s_new <= _p_new:
                _to_add = [] if _best_s_nm in _raw_set else [_best_s_nm]
                print(f"[var_aug/{target}] single: {_best_s_nm}  r²={_best_s_r2:.4f}")
            else:
                _to_add = [_nm for _nm in _bp if _nm and _nm not in _raw_set]
                print(f"[var_aug/{target}] pair: {_bp}  r²={_bp_r2:.4f}")
            if _to_add:
                X_df = pd.concat(
                    [X_df, pd.DataFrame({_nm: _cands[_nm] for _nm in _to_add},
                                        index=X_df.index)], axis=1)
            print(f"[var_aug/{target}] {X_df.shape[1]} features after augmentation")

        # ── Feature selection / init strategy ────────────────────────
        if use_var_aug and strategy in ("xgboost", "diverse"):
            top_feats, feat_scores = xgboost_feature_selection(X_df, y, N_TOP_FEATURES)
            feature_names = top_feats
            X_sel = X_df[feature_names]
        elif strategy == "warmstart" and use_var_aug:
            # Use XGBoost but take a slightly wider feature set
            top_feats, feat_scores = xgboost_feature_selection(
                X_df, y, min(N_TOP_FEATURES + 4, X_df.shape[1])
            )
            feature_names = top_feats
            X_sel = X_df[feature_names]
        elif strategy == "adversarial" and use_var_aug:
            # Pick features with high variance (adversarial: harder inputs first)
            variances = X_df.var()
            feature_names = variances.nlargest(N_TOP_FEATURES).index.tolist()
            feat_scores = (variances / variances.sum()).to_dict()
            X_sel = X_df[feature_names]
        else:
            feature_names = X_df.columns.tolist()[:N_TOP_FEATURES]
            X_sel = X_df[feature_names]
            feat_scores = {c: 1.0 / len(feature_names) for c in feature_names}

        num_features = X_sel.shape[1]

        # ── DataLoader ────────────────────────────────────────────────
        ds     = RegressionDataset(X_sel, y.to_frame())
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Search space ──────────────────────────────────────────────
        all_combos = build_feature_combinations(feature_names, feat_scores)
        search_space, dag = build_search_space(
            feature_names, feat_scores, method_cfg["operators"], all_combos
        )

        loss_mode = method_cfg.get("loss_mode", "full")
        # ── Stochastic subsampling: pass raw arrays to make_loss_function ──
        # When denoise_method == "stoch_sub" the actual denoising is done
        # inside each loss_function call (fresh random subset per eval).
        # No y pre-processing needed — just capture X_sel / y as numpy arrays.
        _subsample_ratio = 1.0
        _X_np_sub        = None
        _y_np_sub        = None
        if method_cfg.get("denoise_method") == "stoch_sub":
            _subsample_ratio = float(method_cfg.get("subsample_ratio", 0.5))
            _X_np_sub = X_sel.values.astype(np.float32)
            _y_np_sub = y.values.astype(np.float32).ravel()

        # ── MC-Dropout uncertainty weights ───────────────────────────────
        # Train a small MLP with dropout on the (possibly noisy) data,
        # run MC inference to estimate per-sample aleatoric uncertainty,
        # and convert to confidence weights.  High-noise samples are
        # downweighted in both stochastic subsampling and the final loss.
        _mc_sample_weights = None
        if method_cfg.get("mc_dropout", False):
            try:
                _mc_kw = dict(
                    n_forward   = method_cfg.get("mc_dropout_n_forward",  50),
                    dropout_p   = method_cfg.get("mc_dropout_p",          0.15),
                    n_epochs    = method_cfg.get("mc_dropout_epochs",     300),
                    hidden      = method_cfg.get("mc_dropout_hidden",      64),
                    random_seed = seed,
                )
                _mc_sample_weights = _mc_dropout_weights(X_sel, y, **_mc_kw)
                _w_min = float(_mc_sample_weights.min())
                _w_max = float(_mc_sample_weights.max())
                print(f"[{method_id}/{target}/run{run_id}] MC-Dropout weights: "
                      f"min={_w_min:.3f}  max={_w_max:.3f}  "
                      f"(low = uncertain/noisy points downweighted)")
            except Exception as _mc_e:
                print(f"[{method_id}/{target}/run{run_id}] MC-Dropout FAILED "
                      f"({_mc_e}); continuing without sample weights")

        loss_fn, loss_state = make_loss_function(
            search_space, loader, device, num_features,
            feature_names, log_path, loss_mode=loss_mode,
            optimize_constants=method_cfg.get("optimize_constants", False),
            subsample_ratio=_subsample_ratio,
            _X_np=_X_np_sub, _y_np=_y_np_sub,
            _sample_weights=_mc_sample_weights,
        )

        # ── Diverse init: build seed_dags (chain/fan/skip/rand topologies)
        seed_models = None
        if strategy == "diverse":
            seed_models = _build_random_seed_dags(
                feature_names, method_cfg["operators"], seed=run_id
            )

        # ── Run Mutant_UCB (curriculum or flat) ───────────────────────
        os.makedirs(save_dir, exist_ok=True)
        parallel_N = method_cfg.get("parallel_N", 1)
        use_curriculum = method_cfg.get("curriculum", False)

        if use_curriculum:
            global_best   = np.inf
            budget_mode   = method_cfg.get("budget_mode", "default")
            total_iters   = 0  # cumulative iterations across all levels
            _iter_cap     = _max_iters if _max_iters is not None else DRAGON_N_ITERATIONS
            for complexity in range(1, DRAGON_MAX_COMPLEXITY + 1):
                # ── How many iterations remain in the global budget? ──
                remaining = _iter_cap - total_iters
                if remaining <= 0:
                    break  # global cap exhausted
                dag.complexity = complexity
                clean = (complexity == 1)
                _csv_path = os.path.join(save_dir, "computation_file.csv")
                pop_size  = (len(pd.read_csv(_csv_path))
                             if not clean and os.path.exists(_csv_path) else 0)
                if clean or budget_mode == "complexity":
                    T_level = pop_size + complexity * DRAGON_T_PER_LEVEL
                    extra   = {}
                else:
                    extra   = {"pop_path": save_dir}
                    T_level = pop_size + complexity * DRAGON_T_PER_LEVEL
                # Cap to the remaining global budget
                T_level = min(T_level, remaining)
                sa_kwargs = dict(
                    search_space=search_space,
                    evaluation=loss_fn,
                    T=T_level,
                    K=DRAGON_K_INIT,
                    N=parallel_N, E=1000,
                    save_dir=save_dir,
                    clean_all=clean,
                    verbose=True,
                    loss_threshold=DRAGON_LOSS_THRESHOLD,
                    **extra,
                )
                if clean and seed_models is not None:
                    sa_kwargs["models"] = seed_models
                sa = Mutant_UCB(**sa_kwargs)
                sa.run()
                # Update cumulative budget from the CSV (actual evaluations done)
                try:
                    _csv = os.path.join(save_dir, "computation_file.csv")
                    total_iters = len(pd.read_csv(_csv))
                except Exception:
                    total_iters += T_level  # fallback estimate
                level_best = sa.min_loss
                prev_plateau = (level_best >= global_best - 1e-6)
                global_best  = min(global_best, level_best)
                if global_best <= DRAGON_LOSS_THRESHOLD:
                    break
            best_loss = global_best
        else:
            _iter_cap = _max_iters if _max_iters is not None else DRAGON_N_ITERATIONS
            sa_kwargs = dict(
                search_space=search_space,
                evaluation=loss_fn,
                T=_iter_cap,
                K=DRAGON_K_INIT,
                N=parallel_N, E=1000,
                save_dir=save_dir,
                clean_all=True,
                verbose=True,
                loss_threshold=DRAGON_LOSS_THRESHOLD,
            )
            if seed_models is not None:
                sa_kwargs["models"] = seed_models
            sa = Mutant_UCB(**sa_kwargs)
            sa.run()
            best_loss = sa.min_loss

        best_formula = loss_state["best_formula"]
        # Fall back to log extraction if state formula is N/A
        if best_formula == "N/A":
            best_formula = _extract_best_formula_from_log(log_path)

        # ── Read loss landscape from computation_file.csv ─────────────
        comp_csv = os.path.join(save_dir, "computation_file.csv")
        if os.path.exists(comp_csv):
            try:
                df_comp = pd.read_csv(comp_csv)
                actual_T = len(df_comp)
                if "Loss" in df_comp.columns:
                    losses = df_comp["Loss"].replace([np.inf, -np.inf], np.nan).dropna().values
                    n_pts  = min(len(losses), 80)
                    idx_pts = np.linspace(0, len(losses) - 1, n_pts, dtype=int)
                    loss_history = [[int(i), float(losses[i])] for i in idx_pts]
            except Exception:
                pass
            # Statistical landscape SVG (3-panel matplotlib).
            try:
                landscape_svg = _make_dragon_landscape_svg(comp_csv)
            except Exception:
                landscape_svg = None
        else:
            landscape_svg = None

    except Exception:
        tb = traceback.format_exc()
        best_formula = f"ERROR: {tb[:200]}"
        print(f"[{method_id}/{target}/run{run_id}] ERROR:\n{tb}")

    elapsed = time.time() - t_start

    result = {
        "target":         target,
        "method":         method_id,
        "run_id":         run_id,
        "strategy":       strategy,
        # Carry the stream id so the smart-parallel dispatcher can build
        # a per-stream summary; harmless / absent otherwise.
        "_stream_id":     method_cfg.get("_stream_id"),
        "loss":           float(best_loss),
        "r2":             float(1.0 - best_loss) if best_loss <= 1.0 else 0.0,
        "formula":        str(best_formula),
        "time_s":         elapsed,
        "log_path":       log_path,
        "description":    method_cfg["description"],
        "search_space_ops": list(method_cfg.get("operators", [])),
        # ── rich OLS stats ─────────────────────────────────────────────
        "loss_mode":        method_cfg.get("loss_mode", "full"),
        "winner_type":      loss_state.get("winner_type", "channel"),
        "corr_value":       float(loss_state.get("corr_value", 0.0)),
        "alignment_loss":   float(loss_state.get("alignment_loss", 1.0)),
        "rat_degree":       loss_state.get("rat_degree"),
        "loss_history":     loss_history,
        "actualT":         int(actual_T) if actual_T is not None else None,
        # ── All formula variants ────────────────────────────────────────
        "all_channel_formulas": loss_state.get("all_channel_formulas", []),
        "ops_used":         loss_state.get("ops_used"),
        "has_const":        loss_state.get("has_const"),
        "dag_size":         loss_state.get("dag_size"),
        "dag_text":         loss_state.get("dag_text"),
        "dag_svg":          loss_state.get("dag_svg"),
        "dag_data":         loss_state.get("dag_data", []),
        "formula_channel":  loss_state.get("formula_channel"),
        "formula_ols":      loss_state.get("formula_ols"),
        "formula_nested":   loss_state.get("formula_nested"),
        "formula_polyrat":  loss_state.get("formula_polyrat", {}),
        # ── Per-method losses ───────────────────────────────────────────────
        "loss_channel":     loss_state.get("loss_channel"),
        "loss_ols":         loss_state.get("loss_ols"),
        "loss_nested":      loss_state.get("loss_nested"),
        "loss_polyrat":     loss_state.get("loss_polyrat", {}),
        # ── Per-method raw MSE (absolute) ───────────────────────────────────
        "mse_channel":      loss_state.get("mse_channel"),
        "mse_ols":          loss_state.get("mse_ols"),
        "mse_nested":       loss_state.get("mse_nested"),
        "mse_polyrat":      loss_state.get("mse_polyrat", {}),
        # Statistical search-landscape SVG (3-panel: scatter+best, hist, conv vs time)
        "landscape_svg":    landscape_svg,
    }
    # Pass raw winning prediction to the smart-parallel dispatcher when
    # running inside a stream (numpy arrays — never JSON-serialised).
    if method_cfg.get("_in_smart_stream"):
        result["_best_pred_np"] = loss_state.get("best_pred_np")
        result["_best_y_np"]    = loss_state.get("best_y_np")
    return result


def _extract_best_formula_from_log(log_path: str) -> str:
    """Return the formula with the smallest logged loss."""
    if not os.path.exists(log_path):
        return "N/A"
    best_loss = np.inf
    best_formula = "N/A"
    current_formula = None
    current_loss = None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- Idx="):
                    # parse loss
                    try:
                        current_loss = float(line.split("Loss=")[1].split()[0].rstrip("---").strip())
                    except Exception:
                        current_loss = None
                    current_formula = None
                elif "<<< SELECTED" in line and current_loss is not None:
                    try:
                        formula_part = line.split("Formula[")[1].split("]: ", 1)[1]
                        formula_part = formula_part.replace(" <<< SELECTED", "").strip()
                        current_formula = formula_part
                    except Exception:
                        pass
                    if current_loss is not None and current_loss < best_loss:
                        best_loss = current_loss
                        best_formula = current_formula or "N/A"
    except Exception:
        pass
    return best_formula


# ══════════════════════════════════════════════════════════════════════════════
#  PySR WORKER  — calls Julia subprocess
# ══════════════════════════════════════════════════════════════════════════════

def run_pysr(target: str, run_id: int, add_noise: bool = False,
             _X_preloaded=None, _y_clean_preloaded=None,
             _y_noisy_preloaded=None) -> dict:
    """Run PySR via the PySRRegressor Python API (SymbolicRegression.jl backend).

    add_noise=True → pass the pre-built noisy y (same as Dragon) to PySRRegressor.
    Method id is then "pysr_noise".
    """
    from pysr import PySRRegressor
    method_id = "pysr_noise" if add_noise else "pysr"
    run_dir = os.path.join(OUTPUT_DIR, target, method_id, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"{target}_{method_id}{LOG_SUFFIX}")

    t_start = time.time()
    best_loss = np.inf
    best_formula = "N/A"
    hall_of_fame = []

    try:
        # ── Load data ────────────────────────────────────────────────────────
        if _X_preloaded is not None and _y_clean_preloaded is not None:
            X_df = _X_preloaded
            y_series = (
                _y_noisy_preloaded
                if (add_noise and _y_noisy_preloaded is not None)
                else _y_clean_preloaded
            )
        else:
            X_df, y_series = load_dataset(target, run_id)
        X = X_df.values.astype(np.float64)
        y = y_series.values.ravel().astype(np.float64)
        feature_names = list(X_df.columns)

        # ── Fit ──────────────────────────────────────────────────────────────
        model = PySRRegressor(
            niterations=PYSR_NITERATIONS,
            populations=PYSR_POPULATIONS,
            population_size=PYSR_POPULATION_SIZE,
            maxsize=PYSR_MAXSIZE,
            binary_operators=PYSR_BINARY_OPERATORS,
            unary_operators=PYSR_UNARY_OPERATORS,
            denoise=add_noise,
            julia_project=PYSR_JULIA_PROJECT,
            verbosity=0,
            random_state=RANDOM_SEED + run_id,
        )
        model.fit(X, y, variable_names=feature_names)

        # ── Write run log ────────────────────────────────────────────────────
        with open(log_path, "w") as _lf:
            _lf.write(
                f"target={target} method={method_id} run_id={run_id}\n"
                f"niterations={PYSR_NITERATIONS} populations={PYSR_POPULATIONS} "
                f"population_size={PYSR_POPULATION_SIZE} maxsize={PYSR_MAXSIZE} "
                f"denoise={add_noise}\n"
            )

        # ── Extract results ──────────────────────────────────────────────────
        df = model.equations_
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                hall_of_fame.append({
                    "complexity": int(row["complexity"]),
                    "loss":       float(row["loss"]),
                    "formula":    str(row["equation"]),
                })
            best_loss    = min(h["loss"] for h in hall_of_fame if np.isfinite(h["loss"]))
            best_row     = df.loc[df["loss"].idxmin()]
            best_formula = str(best_row["equation"])

    except Exception:
        tb = traceback.format_exc()
        best_formula = f"ERROR: {tb[:200]}"
        print(f"[PySR/{target}] ERROR:\n{tb}")

    elapsed = time.time() - t_start
    if not np.isfinite(best_loss):
        best_loss = 1.0
        if best_formula in ("N/A", ""):
            best_formula = "PySR: no result"
    hall_of_fame = _compute_pysr_scores(hall_of_fame)
    pareto_svg   = _make_pysr_pareto_svg(hall_of_fame)
    tree_svg     = _make_pysr_tree_svg(best_formula)
    tree_data    = _make_pysr_tree_data(best_formula)
    return {
        "target":      target,
        "method":      method_id,
        "run_id":      run_id,
        "loss":        float(best_loss),
        "r2":          float(1.0 - best_loss) if best_loss <= 1.0 else 0.0,
        "formula":     str(best_formula),
        "time_s":      elapsed,
        "log_path":    log_path,
        "description": (
            "PySR +Noise with GP denoising step (built-in PySR denoise=True)"
            if add_noise else "PySR (SymbolicRegression.jl)"
        ),
        "hall_of_fame": hall_of_fame,
        "pareto_svg":   pareto_svg,
        "tree_svg":     tree_svg,
        "tree_data":    tree_data,
    }





# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_all(targets=TARGETS, n_runs=N_RUNS, resume=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "results.json")

    # ── Resume: reload previously completed results (only when --continue is passed) ──
    results = []
    if resume and os.path.exists(results_path):
        try:
            with open(results_path) as _f:
                results = json.load(_f)
            print(f"[resume] Loaded {len(results)} existing results from {results_path}")
        except Exception as _e:
            print(f"[resume] Could not load existing results ({_e}); starting fresh")
            results = []
    elif not resume:
        print("[run_all] Starting fresh (use --continue / -continue to resume).")
    _done = {(r["target"], int(r["run_id"]), r["method"]) for r in results}

    for target in targets:
        print(f"\n{'='*70}")
        print(f"  TARGET: {target}")
        print(f"{'='*70}")

        for run_id in range(n_runs):
            strategy = INIT_STRATEGIES[run_id]
            print(f"\n--- Run {run_id+1}/{n_runs}  strategy={strategy} ---")

            # ── Pre-build shared X and y (clean + noisy) once per (target, run_id) ──
            # Both Dragon and PySR receive the identical arrays → fair comparison.
            _seed_data = RANDOM_SEED + run_id
            _X_base, _y_clean = load_dataset(target, run_id=run_id, strategy=strategy)
            _noise_rng = np.random.default_rng(_seed_data + 9999)
            _y_noisy   = _y_clean + _noise_rng.normal(
                0, NOISE_STD * float(_y_clean.std()), len(_y_clean)
            )

            # ── Precompute denoised y once per (target, run_id) ──────────
            # Collect all unique denoise_method values requested by any
            # Dragon config.  Each unique method is computed exactly once
            # and shared across every Dragon method that requests it.
            _denoise_cache: dict = {}  # denoise_method → (y_clean, info)
            _needed_dm = set(
                cfg.get("denoise_method")
                for cfg in DRAGON_METHOD_CONFIGS
                if cfg.get("denoise_method")
            )
            if _needed_dm:
                for _dm in sorted(_needed_dm):
                    try:
                        _y_dn, _dn_info = _apply_denoise(_X_base, _y_clean, _dm)
                        _denoise_cache[_dm] = (_y_dn, _dn_info)
                        print(f"  [denoise/{_dm}/{target}/run{run_id}] "
                              f"info={_dn_info}")
                    except Exception as _de:
                        print(f"  [denoise/{_dm}/{target}/run{run_id}] "
                              f"FAILED ({_de}); affected methods will use raw y")
                        _denoise_cache[_dm] = (None, None)

            # Run DRAGON method configs sequentially (parallelization added later)
            for cfg in DRAGON_METHOD_CONFIGS:
                _key = (target, run_id, cfg["id"])
                if _key in _done:
                    print(f"  [{cfg['id']}] SKIPPED (already done)")
                    continue
                _dm = cfg.get("denoise_method")
                _y_pre, _info_pre = _denoise_cache.get(_dm, (None, None)) if _dm else (None, None)
                r = dragon_worker(cfg, target, run_id,
                                  _y_predenoised=_y_pre, _denoise_info=_info_pre,
                                  _X_preloaded=_X_base, _y_preloaded=_y_clean)
                results.append(r)
                _done.add(_key)
                print(f"  [{r['method']}] loss={r['loss']:.6f}  r2={r['r2']:.6f}  "
                      f"t={r['time_s']:.0f}s  strategy={r.get('strategy','?')}")
                print(f"           formula: {r['formula'][:100]}")

            # PySR baseline (sequential)
            if (target, run_id, "pysr") in _done:
                print(f"  [pysr] SKIPPED (already done)")
            else:
                r = run_pysr(target, run_id,
                             _X_preloaded=_X_base, _y_clean_preloaded=_y_clean,
                             _y_noisy_preloaded=_y_noisy)
                results.append(r)
                _done.add((target, run_id, "pysr"))
                print(f"  [{r['method']}] loss={r['loss']:.6f}  r2={r['r2']:.6f}  "
                      f"t={r['time_s']:.0f}s  strategy={r.get('strategy','?')}")
                print(f"           formula: {r['formula'][:100]}")

            # PySR +Noise (mirror of Dragon `+Noise` ablation)
            if (target, run_id, "pysr_noise") in _done:
                print(f"  [pysr_noise] SKIPPED (already done)")
            else:
                r = run_pysr(target, run_id, add_noise=True,
                             _X_preloaded=_X_base, _y_clean_preloaded=_y_clean,
                             _y_noisy_preloaded=_y_noisy)
                results.append(r)
                _done.add((target, run_id, "pysr_noise"))
                print(f"  [{r['method']}] loss={r['loss']:.6f}  r2={r['r2']:.6f}  "
                      f"t={r['time_s']:.0f}s  strategy={r.get('strategy','?')}")
                print(f"           formula: {r['formula'][:100]}")

        # Checkpoint after each target
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        build_html(results)   # live update after every target

    return results



# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDER  — uses dragonfsr_leaderboard_v2 design (attachment)
# ══════════════════════════════════════════════════════════════════════════════

_WINNER_REMAP = {
    "rational": "polyrat",
    "nested":   "nested",
    "ols":      "ols",
    "channel":  "channel",
}


def _result_to_db_entry(r):
    """Convert a dragon_worker / run_pysr result dict to HTML DB entry format."""
    loss = r.get("loss")
    finite_loss = (float(loss) if loss is not None and np.isfinite(float(loss)) else None)
    wt   = _WINNER_REMAP.get(r.get("winner_type", ""), None)
    rd   = r.get("rat_degree")
    is_pysr = r.get("method") in ("pysr", "pysr_noise")
    if is_pysr:
        total_t = PYSR_NITERATIONS
        pop_k   = PYSR_POPULATION_SIZE
        # Max complexity actually reached on the Pareto frontier.
        _hof    = r.get("hall_of_fame") or []
        _cplx   = [h.get("complexity") for h in _hof if h.get("complexity") is not None]
        max_c   = max(_cplx) if _cplx else 20  # 20 = PySR options.maxsize cap
    else:
        total_t = int(r.get("actualT") if r.get("actualT") is not None else DRAGON_N_ITERATIONS)
        pop_k   = DRAGON_K_INIT
        max_c   = DRAGON_MAX_COMPLEXITY
    # Smart-parallel extras (only present when method == 'spar').
    actual_t_winner     = r.get("actualT_winner")
    actual_t_per_stream = r.get("actualT_per_stream") or {}
    return {
        "oneMinusR2": round(finite_loss, 8) if finite_loss is not None else None,
        "mse":        None,
        "winner":     wt,
        "bestCh":     None,
        "polyDeg":    int(rd) if rd is not None and pd.notna(rd) else None,
        "nestedLink": None,
        "finalExpr":  str(r.get("formula", "")),
        "runtime":    round(float(r.get("time_s", 0)), 2),
        "totalT":     total_t,
        "actualTWinner":   int(actual_t_winner) if actual_t_winner is not None else None,
        "actualTPerStream": {str(k): int(v) for k, v in actual_t_per_stream.items()},
        "K":          pop_k,
        "maxComp":    max_c,
        "phase":      None,
        # Min search loss reached (the actual quantity the search optimised).
        # Dragon -> alignment_loss (1 - |corr|);  PySR -> raw MSE returned per member.
        "searchLoss": (
            (lambda _v: float(_v) if _v is not None and np.isfinite(_v) else None)(
                min((h["loss"] for h in (r.get("hall_of_fame") or [])
                     if h.get("loss") is not None and np.isfinite(h["loss"])),
                    default=None)
            )
            if is_pysr else
            (float(r["alignment_loss"]) if r.get("alignment_loss") is not None
             and np.isfinite(float(r["alignment_loss"])) else None)
        ),
        "searchLossKind": "PySR MSE" if is_pysr else "1\u2212|corr|",
        "ops":        (", ".join(r["search_space_ops"]) if r.get("search_space_ops")
                       else ("+, -, *, /, exp, sqrt, abs, sin, cos, log" if is_pysr else None)),
        "nconst":     r.get("has_const") or ("yes (PySR constants always optimised)" if is_pysr else None),
        "dagSize":    r.get("dag_size"),
        "dagText":    r.get("dag_text"),
        "dagSvg":     r.get("dag_svg"),
        "dagData":    r.get("dag_data") or [],
        "hallOfFame": r.get("hall_of_fame") or [],
        # ── Statistical visualisations (inline SVG) ─────────────────
        "landscapeSvg": r.get("landscape_svg"),     # Dragon: 3-panel landscape
        "paretoSvg":    r.get("pareto_svg"),        # PySR: Pareto frontier
        "treeSvg":      r.get("tree_svg"),          # PySR: sympy/Graphviz AST
        "treeData":     r.get("tree_data"),         # PySR: JSON tree for JS renderer
        "channels":      [
            {
                "tag":  f"ch[{c['idx']}]" + (" *" if c.get('selected') else ""),
                "text": (f"1\u2212R\u00b2={c['loss']:.3e}  \u00b7  {c['formula']}"
                          if c.get('loss') is not None else str(c['formula'])),
            }
            for c in (r.get("all_channel_formulas") or [])
        ],
        "notes":         str(r.get("description", "")) or None,
        # ── Per-method formula variants ─────────────────────────────────
        "formulaChannel":  str(r["formula_channel"]) if r.get("formula_channel") else None,
        "formulaOls":      str(r["formula_ols"]) if r.get("formula_ols") else None,
        "formulaNested":   str(r["formula_nested"]) if r.get("formula_nested") else None,
        "formulaPolyrat":  {str(k): str(v) for k, v in (r.get("formula_polyrat") or {}).items()},
        # ── Per-method losses ────────────────────────────────────────────────
        "lossChannel": float(r["loss_channel"]) if r.get("loss_channel") is not None else None,
        "lossOls":     float(r["loss_ols"])     if r.get("loss_ols")     is not None else None,
        "lossNested":  float(r["loss_nested"])  if r.get("loss_nested")  is not None else None,
        "lossPolyrat": {str(k): float(v) for k, v in (r.get("loss_polyrat") or {}).items()},
        # ── Per-method raw MSE (absolute) ────────────────────────────────────
        "mseCh":      float(r["mse_channel"]) if r.get("mse_channel") is not None else None,
        "mseOls":     float(r["mse_ols"])     if r.get("mse_ols")     is not None else None,
        "mseNested":  float(r["mse_nested"])  if r.get("mse_nested")  is not None else None,
        "msePolyrat": {str(k): float(v) for k, v in (r.get("mse_polyrat") or {}).items()},
    }


def build_html(results, notebook_path="dragonfsr_leaderboard_v2.html"):
    """
    1. Inject PRELOADED_DB into the VS Code notebook template (if it exists).
    2. Write leaderboard_standalone.html using the dragonfsr_leaderboard_v2 design.
    """
    import re as _re

    # ── Build DB dict ────────────────────────────────────────────────────────
    db = {}
    for r in results:
        key      = f"{r['target']}__{r['method']}__{int(r.get('run_id', 0))}"
        db[key]  = _result_to_db_entry(r)
    db_json = json.dumps(db, ensure_ascii=False).replace("</", "<\\/")

    # ── Per-formula aggregate stats (boxplot + heatmap across method × run) ─
    formula_stats = {}
    by_target = {}
    for r in results:
        by_target.setdefault(r["target"], []).append({
            "method": r.get("method"),
            "run":    int(r.get("run_id", 0)),
            "loss":   (float(r["loss"]) if r.get("loss") is not None
                       and np.isfinite(float(r["loss"])) else None),
            "runtime": (float(r["time_s"]) if r.get("time_s") is not None else None),
        })
    for tgt, cells in by_target.items():
        svg = _make_formula_stats_svg(cells)
        if svg:
            formula_stats[tgt] = svg
    formula_stats_json = json.dumps(formula_stats, ensure_ascii=False).replace("</", "<\\/")

    # ── Update notebook template (inject PRELOADED_DB constant) ─────────────
    tmpl = Path(__file__).parent / notebook_path
    if tmpl.exists():
        html = tmpl.read_text(encoding="utf-8")
        new_decl = f"const PRELOADED_DB = {db_json};"
        patched  = _re.sub(
            r"const PRELOADED_DB\s*=\s*\{[^;]*\};",
            new_decl, html, count=1, flags=_re.DOTALL)
        if patched == html:
            # pattern not found — append declaration before first </script>
            patched = html.replace("</script>", f"const PRELOADED_DB = {db_json};\n</script>", 1)
        tmpl.write_text(patched, encoding="utf-8")
        print(f"✓ Template updated ({len(db)} cells): {tmpl}")

    # ── Write standalone HTML ────────────────────────────────────────────────
    _build_standalone_html(db, db_json, formula_stats_json)


def _build_standalone_html(db: dict, db_json: str, formula_stats_json: str = "{}") -> None:
    """
    Write leaderboard_standalone.html using the exact dragonfsr_leaderboard_v2
    design from the attachment.  PRELOADED_DB is injected as a JS constant;
    manual additions in VS Code notebook are merged on top via window.storage.
    """

    # ── CSS variable defaults (standalone browser fallback) ──────────────────
    CSS_VARS = """\
:root{
  --color-text-primary:#1a1a1a;--color-text-secondary:#6b6b6b;
  --color-text-tertiary:#9a9a9a;--color-background-primary:#ffffff;
  --color-background-secondary:#f7f7f7;--color-border-secondary:#e0e0e0;
  --color-border-tertiary:#ebebeb;
  --font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --font-mono:"SF Mono","Fira Code",Consolas,monospace;
}
@media(prefers-color-scheme:dark){
  :root{
    --color-text-primary:#e8e8e8;--color-text-secondary:#a0a0a0;
    --color-text-tertiary:#6a6a6a;--color-background-primary:#1e1e1e;
    --color-background-secondary:#252525;--color-border-secondary:#3d3d3d;
    --color-border-tertiary:#303030;
  }
}"""

    # ── Attachment CSS (verbatim) ─────────────────────────────────────────────
    CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
.root{padding:1rem 0;color:var(--color-text-primary);font-family:var(--font-sans)}
.title{font-size:19px;font-weight:500;letter-spacing:-0.02em;margin-bottom:3px}
.subtitle{font-size:11px;color:var(--color-text-secondary);margin-bottom:1.2rem}
.statbar{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1.2rem}
.sc{background:var(--color-background-secondary);border-radius:8px;padding:9px 11px;text-align:center}
.sc .v{font-size:18px;font-weight:500}
.sc .l{font-size:9px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.tabs{display:flex;gap:5px;margin-bottom:.9rem;flex-wrap:wrap}
.tab{font-size:10px;padding:3px 9px;border:0.5px solid var(--color-border-secondary);border-radius:5px;cursor:pointer;background:var(--color-background-secondary);color:var(--color-text-secondary);transition:all .12s}
.tab.active{background:var(--color-text-primary);color:var(--color-background-primary);border-color:var(--color-text-primary)}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:.9rem;font-size:10px;color:var(--color-text-secondary);align-items:center}
.ld{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:3px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:1200px;font-size:10px;table-layout:auto}
thead th{font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;color:var(--color-text-secondary);padding:5px 7px;text-align:center;border-bottom:1px solid var(--color-border-secondary);white-space:nowrap;background:var(--color-background-primary)}
thead th.fcol{text-align:left;min-width:145px;position:sticky;left:0;z-index:3;background:var(--color-background-primary)}
.gh-pysr{background:#E6F1FB!important;color:#0C447C!important}
.gh-drag{background:#EEEDFE!important;color:#3C3489!important}
tbody td{padding:4px 6px;border-bottom:0.5px solid var(--color-border-tertiary);vertical-align:top;text-align:center}
tbody td.fcol{text-align:left;font-weight:500;font-size:10px;position:sticky;left:0;background:var(--color-background-primary);z-index:1}
tbody td.fcol .ftex{font-size:9px;color:var(--color-text-tertiary);font-weight:400;margin-top:1px;font-style:italic}
tbody tr:hover td{background:var(--color-background-secondary)}
tbody tr:hover td.fcol{background:var(--color-background-secondary)}
.secrow td{background:var(--color-background-secondary)!important;font-size:9px;font-weight:500;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.07em;padding:4px 7px}
.cg{display:flex;flex-direction:column;gap:2px;align-items:center;min-width:52px}
.runrow{display:flex;gap:2px;justify-content:center}
.pill{display:inline-flex;align-items:center;justify-content:center;height:17px;min-width:22px;padding:0 4px;border-radius:3px;font-size:8px;font-weight:500;cursor:pointer;transition:transform .08s;white-space:nowrap}
.pill:hover{transform:scale(1.08)}
.pill.empty{background:var(--color-background-secondary);color:var(--color-text-tertiary);border:0.5px dashed var(--color-border-tertiary)}
.pill.perfect{background:#EAF3DE;color:#27500A;border:0.5px solid #97C459}
.pill.great{background:#E6F1FB;color:#0C447C;border:0.5px solid #85B7EB}
.pill.ok{background:#FAEEDA;color:#633806;border:0.5px solid #EF9F27}
.pill.fail{background:#FCEBEB;color:#791F1F;border:0.5px solid #F09595}
.smini{font-size:8px;color:var(--color-text-tertiary);text-align:center;line-height:1.5;margin-top:1px}
.winner-badge{font-size:7px;padding:1px 3px;border-radius:2px;font-weight:500}
.wb-polyrat{background:#EEEDFE;color:#3C3489}
.wb-nested{background:#E1F5EE;color:#085041}
.wb-ols{background:#E6F1FB;color:#0C447C}
.wb-ch{background:#F1EFE8;color:#444441}
.modal-wrap{display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.55);align-items:flex-start;justify-content:center;padding:24px 0;overflow-y:auto;backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px)}
.modal{position:relative;background:var(--color-background-primary);border:0.5px solid var(--color-border-secondary);border-radius:12px;padding:1.25rem;width:1100px;max-width:98vw;max-height:calc(100vh - 48px);overflow-y:auto;box-shadow:0 18px 48px rgba(0,0,0,0.35)}
.modal-close{position:absolute;top:10px;right:14px;width:30px;height:30px;border-radius:50%;border:0.5px solid var(--color-border-secondary);background:var(--color-background-secondary);color:var(--color-text-primary);cursor:pointer;font-size:18px;line-height:1;display:flex;align-items:center;justify-content:center;font-family:inherit}
.modal-close:hover{background:var(--color-background-tertiary,#eee)}
.modal h3{font-size:14px;font-weight:500;margin-bottom:1rem}
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:.8rem}
.mf{display:flex;flex-direction:column;gap:3px}
.mf label{font-size:10px;color:var(--color-text-secondary)}
.mf input,.mf select,.mf textarea{font-size:11px;padding:5px 7px;border:0.5px solid var(--color-border-secondary);border-radius:6px;background:var(--color-background-secondary);color:var(--color-text-primary);width:100%}
.mf textarea{resize:vertical;min-height:50px}
.mf.full{grid-column:1/-1}
.mactions{display:flex;gap:7px;justify-content:flex-end;margin-top:.8rem}
.btn-p{font-size:11px;padding:5px 13px;border-radius:6px;border:0.5px solid var(--color-text-primary);background:var(--color-text-primary);color:var(--color-background-primary);cursor:pointer}
.btn-s{font-size:11px;padding:5px 13px;border-radius:6px;border:0.5px solid var(--color-border-secondary);background:transparent;color:var(--color-text-secondary);cursor:pointer}
.section-divider{border-top:0.5px solid var(--color-border-tertiary);margin:10px 0 6px;padding-top:6px}
.sec-label{font-size:9px;font-weight:500;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
.ch-row{display:flex;align-items:center;gap:5px;margin-bottom:4px}
.ch-tag{font-size:9px;background:var(--color-background-secondary);border-radius:3px;padding:1px 4px;color:var(--color-text-secondary);min-width:28px;text-align:center}
.add-ch-btn{font-size:9px;padding:2px 6px;border:0.5px dashed var(--color-border-secondary);border-radius:4px;cursor:pointer;background:transparent;color:var(--color-text-secondary)}
.rm-ch{font-size:9px;cursor:pointer;color:var(--color-text-tertiary);background:transparent;border:none;padding:0 2px}
.ftabs{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px}
.ftab{font-size:9px;padding:2px 8px;border-radius:12px;border:0.5px solid var(--color-border-secondary);background:transparent;color:var(--color-text-secondary);cursor:pointer;transition:background .15s,color .15s}
.ftab:hover{background:var(--color-background-secondary)}
.ftab.ftab-active{background:var(--color-text-primary);color:var(--color-background-primary);border-color:var(--color-text-primary)}
.ftab.ftab-winner{border-color:#3b6d11;color:#3b6d11}
.ftab.ftab-winner.ftab-active{background:#3b6d11;color:#fff}
.formula-view{font-size:10px;font-family:var(--font-mono);padding:7px 9px;border:0.5px solid var(--color-border-secondary);border-radius:6px;background:var(--color-background-secondary);color:var(--color-text-primary);width:100%;resize:vertical;min-height:60px;box-sizing:border-box}
.tt{background:var(--color-background-primary);border:0.5px solid var(--color-border-secondary);border-radius:8px;padding:9px 11px;font-size:10px;z-index:999;pointer-events:none;min-width:210px;position:absolute;display:none}
.tt .tttitle{font-weight:500;margin-bottom:5px;font-size:11px}
.tt .ttr{display:flex;justify-content:space-between;gap:14px;margin-bottom:2px}
.tt .ttr span:last-child{color:var(--color-text-primary);font-weight:500}
.tt .ttr span:first-child{color:var(--color-text-secondary)}
.phase-pills{display:flex;gap:2px;flex-wrap:wrap;justify-content:center;margin-top:2px}
.pp{font-size:7px;padding:1px 4px;border-radius:2px}
.pp-alg{background:#F1EFE8;color:#444441}
.pp-ln{background:#FAEEDA;color:#633806}
.pp-tri{background:#EEEDFE;color:#3C3489}
.pp-expl{background:#E6F1FB;color:#0C447C}
.fn{margin-top:1.2rem;font-size:9px;color:var(--color-text-tertiary);border-top:0.5px solid var(--color-border-tertiary);padding-top:8px;line-height:1.7}"""

    # ── HTML body (verbatim from attachment) ─────────────────────────────────
    BODY = """\
<div class="root">
<div class="title">Symbolic Regression Leaderboard</div>
<div class="subtitle">PySR vs DragonSR — 5 runs × formula × method · metric: 1−R² (lower is better) · v0.2</div>

<div class="statbar" id="statbar">
  <div class="sc"><div class="v" id="s-runs">0</div><div class="l">Runs logged</div></div>
  <div class="sc"><div class="v" id="s-best">—</div><div class="l">Best 1−R²</div></div>
  <div class="sc"><div class="v" id="s-fills">0/32</div><div class="l">Formulas hit</div></div>
  <div class="sc"><div class="v" id="s-polyrat">0</div><div class="l">Poly-rat wins</div></div>
  <div class="sc"><div class="v" id="s-rt">—</div><div class="l">Avg runtime</div></div>
</div>

<div class="tabs" id="tabs">
  <div class="tab active" data-cat="all">All</div>
  <div class="tab" data-cat="physics">Physics</div>
  <div class="tab" data-cat="nguyen">Nguyen 1–12</div>
  <div class="tab" data-cat="remote">Remote sensing</div>
  <div class="tab" data-cat="other">Other</div>
</div>

<div class="legend">
  <span><span class="ld" style="background:#3b6d11"></span>exact (1−R²&lt;1e-6)</span>
  <span><span class="ld" style="background:#185fa5"></span>great (&lt;1e-3)</span>
  <span><span class="ld" style="background:#ba7517"></span>ok (&lt;0.1)</span>
  <span><span class="ld" style="background:#a32d2d"></span>fail (≥0.1)</span>
  <span style="margin-left:8px;color:var(--color-text-tertiary)">winner badge:</span>
  <span class="winner-badge wb-polyrat">P-RAT</span>
  <span class="winner-badge wb-nested">NEST</span>
  <span class="winner-badge wb-ols">OLS</span>
  <span class="winner-badge wb-ch">CH</span>
</div>

<div class="scroll">
<table id="tbl">
<thead>
<tr>
  <th class="fcol" rowspan="2" style="vertical-align:bottom">Formula</th>
  <th class="gh-pysr" colspan="2">PySR</th>
  <th class="gh-drag" colspan="6">DragonSR</th>
</tr>
<tr>
  <th class="gh-pysr">Baseline</th>
  <th class="gh-pysr">+Noise (GP denoising)</th>
  <th class="gh-drag">All ops ★ (ref)</th>
  <th class="gh-drag">Smart par***</th>
  <th class="gh-drag">Boosted spar***</th>
  <th class="gh-drag">+ConstBrick</th>
  <th class="gh-drag">No OLS/rat/nest</th>
  <th class="gh-drag">Smart par w/ denoise***</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<div id="ttbox" class="tt"></div>

<div class="modal-wrap" id="mwrap" style="display:none" onclick="if(event.target===this)closeModal()">
<div class="modal" id="mbox">
  <button type="button" class="modal-close" onclick="closeModal()" title="Close (Esc)" aria-label="Close">×</button>
  <h3 id="mtitle">Log run</h3>
  <div class="mgrid">
    <div class="mf full"><label>Formula · method · run</label><input id="mflabel" readonly style="opacity:.6"/></div>
    <div class="mf full"><label>Init strategy (read-only)</label><input id="minit" readonly style="opacity:.6"/></div>
    <div style="grid-column:1/-1" class="section-divider" id="sec-channels">
      <div class="sec-label" id="sec-channels-label">Output channels — all formulas (one per channel)</div>
      <div id="chlist" style="max-height:260px;overflow-y:auto"></div>
    </div>
    <div style="grid-column:1/-1" class="section-divider">
      <div class="sec-label">Symbolic formulas <span style="font-weight:400;opacity:.7">(toggle to view each method's output)</span></div>
      <div class="ftabs" id="ftabs"></div>
      <div id="ftab-lossinfo" style="font-size:9px;color:var(--color-text-secondary);margin-bottom:4px;font-family:var(--font-mono)"></div>
      <textarea id="mfexpr" class="formula-view" placeholder="(no formula)" readonly></textarea>
    </div>
    <div style="grid-column:1/-1" class="section-divider">
      <div class="sec-label">Budget &amp; search space</div>
      <div class="mgrid" style="margin:0">
        <div class="mf"><label>Runtime (s)</label><input id="mrt" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Total iterations (T)</label><input id="miter" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Population K</label><input id="mpop" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Max complexity reached</label><input id="mcomp" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Min search loss reached</label><input id="mphase" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Ops used (list)</label><input id="mops" readonly style="opacity:.6"/></div>
        <div class="mf"><label>n_const (constants optimised)</label><input id="mnconst" readonly style="opacity:.6"/></div>
        <div class="mf"><label>DAG tree size (nodes)</label><input id="mdag" readonly style="opacity:.6"/></div>
      </div>
      <div id="mdagviz" style="margin-top:8px;font-family:var(--font-mono);font-size:10px;white-space:pre;overflow:auto;max-height:520px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:6px;display:none"></div>
    </div>
    <!-- ── Statistical visualisations (per run) ──────────────────────── -->
    <div style="grid-column:1/-1" class="section-divider">
      <div class="sec-label">Search statistics &amp; landscape</div>
      <div id="mlandscape" style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow-x:auto"></div>
      <div id="mpareto"    style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow:auto;max-height:480px"></div>
      <div id="mtree"      style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow:auto;max-height:480px">
        <div style="font-size:10px;color:var(--color-text-secondary);margin-bottom:4px;text-align:left">Best PySR formula — syntactic AST (sympy parse tree)</div>
        <div id="mtree-body"></div>
      </div>
      <div id="mformstats" style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow-x:auto">
        <div style="font-size:10px;color:var(--color-text-secondary);margin-bottom:4px;text-align:left">Aggregated stats for this formula — box-plot per method · ECDF performance profile · loss-vs-runtime trade-off</div>
        <div id="mformstats-body"></div>
      </div>
    </div>
    <div class="mf full" style="margin-top:8px"><label>Notes</label><input id="mnotes" readonly style="opacity:.6"/></div>
  </div>
  <div class="mactions">
    <button class="btn-s" onclick="closeModal()">Close</button>
  </div>
</div>
</div>

<div class="fn">
  <p>* Curriculum: 3 op-phases (algebraic → +ln/exp → +sin/cos), progressive complexity 1→11, warm-start population carries over.</p>
  <p>** Curriculum + palier: curriculum with plateau-based difficulty/lr scheduling between phases.</p>
  <p>*** Intelligent parallelization: adaptive island worker allocation based on partial solutions.</p>
  <p>Init strategies (DragonSR only) — R1: random uniform · R2: diverse population (seed DAGs) · R3: XGBoost feature select · R4: warm-start (PySR) · R5: adversarial init. <em>PySR baseline uses one fixed configuration; columns 1..N are repeated runs differing only by data seed for synthetic targets.</em></p>
  <p>Metric: 1−R² = normMSE = MSE/Var(y). Winner: best of {best-channel, sparse OLS, nested OLS, poly-rational OLS}.</p>
</div>
</div>"""

    # ── JavaScript (attachment verbatim, with PRELOADED_DB injection) ─────────
    JS = f"""\
const PRELOADED_DB={db_json};
const FORMULA_STATS_SVG={formula_stats_json};
const FORMULAS=[
  {{id:'hubble',name:'Hubble',cat:'physics',tex:'v = H_0 \\\\cdot d'}},
  {{id:'newton',name:'Newton gravity',cat:'physics',tex:'F = G m_1 m_2 / r^2'}},
  {{id:'rydberg',name:'Rydberg',cat:'physics',tex:'1/\\\\lambda = R(1/n_1^2 - 1/n_2^2)'}},
  {{id:'idealgas',name:'Ideal Gas',cat:'physics',tex:'PV = nRT'}},
  {{id:'kepler',name:"Kepler 3rd",cat:'physics',tex:'T^2 = (4\\\\pi^2/GM)\\\\,a^3'}},
  {{id:'bode',name:"Bode's law",cat:'physics',tex:'a_n = 0.4 + 0.3 \\\\cdot 2^n'}},
  {{id:'schechter',name:'Schechter',cat:'physics',tex:'\\\\phi(L)=\\\\phi^*(L/L^*)^\\\\alpha e^{{-L/L^*}}'}},
  {{id:'leavitt',name:'Leavitt',cat:'physics',tex:'M = a\\\\log P + b'}},
  {{id:'planck',name:"Planck's law",cat:'physics',tex:'B(\\\\nu,T)=\\\\frac{{2h\\\\nu^3}}{{c^2}}\\\\frac{{1}}{{e^{{h\\\\nu/kT}}-1}}'}},
  {{id:'n4',name:'Nguyen 4',cat:'nguyen',tex:'x^6+x^5+x^4+x^3+x^2+x'}},
  {{id:'n5',name:'Nguyen 5',cat:'nguyen',tex:'\\\\sin(x^2)\\\\cos(x)-1'}},
  {{id:'n6',name:'Nguyen 6',cat:'nguyen',tex:'\\\\sin(x)+\\\\sin(x+x^2)'}},
  {{id:'n7',name:'Nguyen 7',cat:'nguyen',tex:'\\\\ln(x+1)+\\\\ln(x^2+1)'}},
  {{id:'n8',name:'Nguyen 8',cat:'nguyen',tex:'\\\\sqrt{{x}}'}},
  {{id:'n9',name:'Nguyen 9',cat:'nguyen',tex:'\\\\sin(x)+\\\\sin(y^2)'}},
  {{id:'n10',name:'Nguyen 10',cat:'nguyen',tex:'2\\\\sin(x)\\\\cos(y)'}},
  {{id:'n11',name:'Nguyen 11',cat:'nguyen',tex:'x^y'}},
  {{id:'n12',name:'Nguyen 12',cat:'nguyen',tex:'x^4-x^3+y^2/2-y'}},
  {{id:'ndvi',name:'NDVI',cat:'remote',tex:'\\\\frac{{B_8-B_4}}{{B_8+B_4}}'}},
  {{id:'wi2015',name:'WI2015',cat:'remote',tex:'1.72+171(B_2+B_3+B_4)-3B_2B_3-1.8B_2B_4-48B_3B_4-0.8B_8B_{{11}}'}},
  {{id:'awei_sh',name:'AWEI_sh',cat:'remote',tex:'B_2+2.5B_3-1.5(B_{{11}}+B_{{12}})-0.25B_8'}},
  {{id:'bai',name:'BAI',cat:'remote',tex:'\\\\frac{{1}}{{(0.1-B_4)^2+(0.06-B_8)^2}}'}},
  {{id:'bsi',name:'BSI',cat:'remote',tex:'\\\\frac{{B_{{11}}+B_4-B_8-B_2}}{{B_{{11}}+B_4+B_8+B_2}}'}},
  {{id:'evi2',name:'EVI2',cat:'remote',tex:'\\\\frac{{2.5(B_8-B_4)}}{{B_8+2.4B_4+1}}'}},
  {{id:'vari',name:'VARI',cat:'remote',tex:'\\\\frac{{B_3-B_4}}{{B_3+B_4-B_2}}'}},
  {{id:'savi',name:'SAVI',cat:'remote',tex:'\\\\frac{{1.5(B_8-B_4)}}{{B_8+B_4+0.5}}'}},
  {{id:'nirv',name:'NIRv',cat:'remote',tex:'B_8\\\\cdot\\\\frac{{B_8-B_4}}{{B_8+B_4}}'}},
];
const METHODS=['pysr','pysr_noise','allops','spar','boosted_spar','allops_const','noolsratn','spar_denoise'];
const MINIT=['Random uniform','Diverse population (seed DAGs)','XGBoost feature select','Warm-start (PySR)','Adversarial init'];
const PHASE_LABELS={{alg:'Algebraic',ln:'+Ln/Exp',trig:'+Sin/Cos',exploit:'Exploit'}};
const PHASE_CLS={{alg:'pp-alg',ln:'pp-ln',trig:'pp-trig',exploit:'pp-expl'}};

let DB={{}};
let activeCat='all';
let pending=null;
let chCount=0;

function key(f,m,r){{return `${{f}}__${{m}}__${{r}}`}}

function lossClass(v){{
  if(v===null||v===undefined) return 'empty';
  if(v<1e-6) return 'perfect';
  if(v<1e-3) return 'great';
  if(v<0.1) return 'ok';
  return 'fail';
}}
function fmtLoss(v){{
  if(v===null||v===undefined) return '—';
  if(v<0.001) return v.toExponential(2);
  return v.toFixed(4);
}}
function winnerBadge(w){{
  if(!w) return '';
  const map={{polyrat:'wb-polyrat',nested:'wb-nested',ols:'wb-ols',channel:'wb-ch'}};
  const lbl={{polyrat:'P-RAT',nested:'NEST',ols:'OLS',channel:'CH'}};
  return `<span class="winner-badge ${{map[w]||''}}">${{lbl[w]||w}}</span>`;
}}
function phasePill(ph){{
  if(!ph) return '';
  return `<span class="pp ${{PHASE_CLS[ph]||''}}">${{PHASE_LABELS[ph]||ph}}</span>`;
}}

function renderCell(fid,mid){{
  let html='<div class="cg"><div class="runrow">';
  let vals=[],rts=[],polyrats=0;
  for(let r=0;r<5;r++){{
    const k=key(fid,mid,r);
    const d=DB[k];
    const v=d?d.oneMinusR2:null;
    const cls=lossClass(v);
    const lbl=v!==null&&v!==undefined?fmtLoss(v):(mid==='pysr'?String(r+1):['R','D','X','W','A'][r]||String(r+1));
    html+=`<div class="pill ${{cls}}" data-fid="${{fid}}" data-mid="${{mid}}" data-run="${{r}}"
      onmouseenter="showTT(event,'${{fid}}','${{mid}}',${{r}})"
      onmouseleave="hideTT()"
      onclick="openModal('${{fid}}','${{mid}}',${{r}})">${{lbl}}</div>`;
    if(d&&v!==null){{vals.push(v);if(d.runtime)rts.push(d.runtime);if(d.winner==='polyrat')polyrats++;}}
  }}
  html+='</div>';
  if(vals.length>=2){{
    const mu=vals.reduce((a,b)=>a+b,0)/vals.length;
    const sig=Math.sqrt(vals.reduce((a,b)=>a+(b-mu)**2,0)/vals.length);
    html+=`<div class="smini">μ=${{fmtLoss(mu)}} σ=${{sig.toExponential(1)}}</div>`;
  }}
  const winners=[0,1,2,3,4].map(r=>DB[key(fid,mid,r)]?.winner);
  const anyWin=winners.some(Boolean);
  if(anyWin){{
    const badges=winners.map((w,i)=>{{
      if(!w) return `<span class="winner-badge" style="opacity:.25" title="Run ${{i+1}}: —">R${{i+1}}</span>`;
      const b=winnerBadge(w);
      return b.replace('<span class="winner-badge', `<span title="Run ${{i+1}}: ${{w}}" class="winner-badge`);
    }}).join(' ');
    html+=`<div style="margin-top:2px;display:flex;gap:2px;flex-wrap:wrap">${{badges}}</div>`;
  }}
  const phases=[0,1,2,3,4].map(r=>DB[key(fid,mid,r)]?.phase).filter(Boolean);
  if(phases.length){{
    html+=`<div class="phase-pills">${{[...new Set(phases)].map(p=>phasePill(p)).join('')}}</div>`;
  }}
  // Per-run runtimes (R1..RN) listed individually
  const allRts=[0,1,2,3,4].map(r=>{{const dd=DB[key(fid,mid,r)];return dd&&dd.runtime?dd.runtime:null;}});
  if(allRts.some(t=>t!==null)){{
    const parts=allRts.map((t,i)=>t!==null?`R${{i+1}}:\u202f${{t.toFixed(1)}}s`:`R${{i+1}}:\u202f\u2014`);
    html+=`<div class="smini" style="white-space:normal;line-height:1.3">${{parts.join(' \u00b7 ')}}</div>`;
  }}
  html+='</div>';
  return html;
}}

function buildTable(){{
  const tbody=document.getElementById('tbody');
  tbody.innerHTML='';
  const cats=['physics','nguyen','remote','other'];
  const cnames={{physics:'Physics laws',nguyen:'Nguyen benchmark (1–12)',remote:'Remote sensing indices',other:'Other'}};
  cats.forEach(cat=>{{
    const rows=FORMULAS.filter(f=>f.cat===cat&&(activeCat==='all'||activeCat===cat));
    if(!rows.length) return;
    const sr=document.createElement('tr');sr.className='secrow';
    sr.innerHTML=`<td class="fcol">${{cnames[cat]}}</td>${{METHODS.map(()=>'<td></td>').join('')}}`;
    tbody.appendChild(sr);
    rows.forEach(f=>{{
      const tr=document.createElement('tr');
      let cells=`<td class="fcol"><div>${{f.name}}</div><div class="ftex">$${{f.tex}}$</div></td>`;
      METHODS.forEach(m=>{{cells+=`<td id="cell_${{f.id}}_${{m}}">${{renderCell(f.id,m)}}</td>`;}});
      tr.innerHTML=cells;tbody.appendChild(tr);
    }});
  }});
  updateStats();
  // Render LaTeX in the formula column (KaTeX auto-render).
  if(window.renderMathInElement){{
    try{{
      renderMathInElement(document.getElementById('tbl'),{{
        delimiters:[{{left:'$',right:'$',display:false}}],
        throwOnError:false,
      }});
    }}catch(e){{}}
  }}
}}

function updateStats(){{
  let runs=0,fills=new Set(),best=null,polyrats=0,rts=[];
  Object.entries(DB).forEach(([k,d])=>{{
    if(!d) return;
    if(d.oneMinusR2!==null&&d.oneMinusR2!==undefined){{
      runs++;
      const fid=k.split('__')[0];fills.add(fid);
      if(best===null||d.oneMinusR2<best) best=d.oneMinusR2;
      if(d.winner==='polyrat') polyrats++;
    }}
    if(d.runtime) rts.push(d.runtime);
  }});
  document.getElementById('s-runs').textContent=runs;
  document.getElementById('s-fills').textContent=`${{fills.size}}/${{FORMULAS.length}}`;
  document.getElementById('s-best').textContent=best!==null?fmtLoss(best):'—';
  document.getElementById('s-polyrat').textContent=polyrats;
  document.getElementById('s-rt').textContent=rts.length?(rts.reduce((a,b)=>a+b,0)/rts.length).toFixed(1)+'s':'—';
}}

function _renderChannelsRO(channels){{
  const div=document.getElementById('chlist');
  div.innerHTML='';
  if(!channels||!channels.length){{
    div.innerHTML='<div style="font-size:10px;opacity:.6;padding:4px">(no channel info)</div>';
    return;
  }}
  channels.forEach((c,i)=>{{
    const row=document.createElement('div');row.className='ch-row';
    row.style.cssText='display:flex;align-items:flex-start;gap:6px;padding:3px 4px;border-bottom:0.5px solid var(--color-border-tertiary)';
    row.innerHTML=`<span class="ch-tag" style="font-family:var(--font-mono);font-size:10px;min-width:60px;color:var(--color-text-secondary)">${{c.tag||('ch['+i+']')}}</span>
      <span style="flex:1;font-family:var(--font-mono);font-size:10px;white-space:pre-wrap;word-break:break-all;color:var(--color-text-primary)">${{(c.text||c).toString().replace(/</g,'&lt;')}}</span>`;
    div.appendChild(row);
  }});
}}
function _renderHallOfFame(hof){{
  const div=document.getElementById('chlist');
  div.innerHTML='';
  if(!hof||!hof.length){{
    div.innerHTML='<div style="font-size:10px;opacity:.6;padding:4px">(no Pareto frontier available)</div>';
    return;
  }}
  const tbl=document.createElement('table');
  tbl.style.cssText='width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:10px';
  tbl.innerHTML=`<thead><tr style="text-align:left;border-bottom:0.5px solid var(--color-border-secondary)">
    <th style="padding:3px 6px;width:30px">#</th>
    <th style="padding:3px 6px;width:60px">cmplx</th>
    <th style="padding:3px 6px;width:110px">loss</th>
    <th style="padding:3px 6px;width:90px" title="PySR parsimony score = -Δln(loss)/Δcomplexity (higher = better gain per added complexity unit)">score</th>
    <th style="padding:3px 6px">expression</th></tr></thead>`;
  const tb=document.createElement('tbody');
  hof.forEach((m,i)=>{{
    const tr=document.createElement('tr');
    tr.style.borderBottom='0.5px solid var(--color-border-tertiary)';
    const lossStr=(m.loss!==null&&m.loss!==undefined)?Number(m.loss).toExponential(3):'—';
    const scoreStr=(m.score!==null&&m.score!==undefined&&isFinite(m.score))?Number(m.score).toFixed(3):'—';
    tr.innerHTML=`<td style="padding:3px 6px;color:var(--color-text-secondary)">${{i+1}}</td>
      <td style="padding:3px 6px">${{m.complexity??'—'}}</td>
      <td style="padding:3px 6px">${{lossStr}}</td>
      <td style="padding:3px 6px;color:var(--color-text-secondary)">${{scoreStr}}</td>
      <td style="padding:3px 6px;white-space:pre-wrap;word-break:break-all">${{(m.formula||'').replace(/</g,'&lt;')}}</td>`;
    tb.appendChild(tr);
  }});
  tbl.appendChild(tb);
  div.appendChild(tbl);
}}

function _buildFormulaTabs(d, isPysr){{
  const container=document.getElementById('ftabs');
  const ta=document.getElementById('mfexpr');
  container.innerHTML='';

  function lossBadge(v){{
    if(v===null||v===undefined) return '';
    let col='#ba7517';
    if(v<1e-6) col='#3b6d11';
    else if(v<1e-3) col='#185fa5';
    else if(v>=0.1) col='#a32d2d';
    return ` <span style="font-size:8px;opacity:.85;color:${{col}}">${{v.toExponential(3)}}</span>`;
  }}

  // Collect all available method formulas (no synthetic "Winner" tab)
  const tabs=[];
  if(d&&d.formulaChannel) tabs.push({{id:'ch',   label:'Channel',    expr:d.formulaChannel, loss:d.lossChannel, mse:d.mseCh}});
  if(d&&d.formulaOls)     tabs.push({{id:'ols',  label:'Sparse OLS', expr:d.formulaOls,     loss:d.lossOls,     mse:d.mseOls}});
  if(d&&d.formulaNested)  tabs.push({{id:'nest', label:'Nested OLS', expr:d.formulaNested,  loss:d.lossNested,  mse:d.mseNested}});
  if(d&&d.formulaPolyrat){{
    Object.keys(d.formulaPolyrat).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(deg=>{{
      const expr=d.formulaPolyrat[deg];
      const lv=(d.lossPolyrat||{{}})[deg]??null;
      const mv=(d.msePolyrat||{{}})[deg]??null;
      if(expr) tabs.push({{id:'rat'+deg, label:'P-RAT deg\u202f'+deg, expr, loss:lv, mse:mv}});
    }});
  }}

  // Hide the lossInfo line for PySR up-front (no R²/MSE applicable; hidden
  // even on early-return below when there are no per-method tabs).
  const lossInfo=document.getElementById('ftab-lossinfo');
  if(isPysr&&lossInfo){{ lossInfo.textContent=''; lossInfo.style.display='none'; }}
  else if(lossInfo){{ lossInfo.style.display=''; lossInfo.textContent=''; }}

  // Fallback: if no per-method formulas, show the single finalExpr
  if(!tabs.length){{
    ta.value=d?.finalExpr??'(no formula)';
    return;
  }}

  // Auto-select the tab with the lowest loss (= best method)
  let activeIdx=0;
  let bestLoss=Infinity;
  tabs.forEach((tab,i)=>{{
    if(tab.loss!==null&&tab.loss!==undefined&&tab.loss<bestLoss){{
      bestLoss=tab.loss; activeIdx=i;
    }}
  }});

  function activate(idx){{
    activeIdx=idx;
    container.querySelectorAll('.ftab').forEach((b,i)=>{{
      b.classList.toggle('ftab-active',i===idx);
      b.classList.toggle('ftab-winner',i===idx&&tabs[i].loss===bestLoss);
    }});
    ta.value=tabs[idx].expr||'(empty)';
    if(lossInfo){{
      const lv=tabs[idx].loss;
      const mv=tabs[idx].mse;
      const parts=[];
      if(lv!==null&&lv!==undefined) parts.push('1\u2212R\u00b2\u202f=\u202f'+lv.toExponential(6));
      if(mv!==null&&mv!==undefined) parts.push('MSE\u202f=\u202f'+mv.toExponential(6));
      lossInfo.textContent=parts.join('\u2003');
    }}
  }}

  tabs.forEach((tab,i)=>{{
    const btn=document.createElement('button');
    btn.className='ftab';
    btn.innerHTML=tab.label+lossBadge(tab.loss);
    btn.onclick=()=>activate(i);
    container.appendChild(btn);
  }});
  activate(activeIdx);
}}

// ── JS-based DAG renderer (fallback when graphviz 'dot' is unavailable) ──────
function _renderDagSvg(dagData){{
  const n=(dagData||[]).length;
  if(!n) return null;
  // Build parents list from children
  const parents=Array.from({{length:n}},()=>[]);
  dagData.forEach((node,i)=>{{
    (node.children||[]).forEach(j=>{{ if(j>=0&&j<n) parents[j].push(i); }});
  }});
  // Depth = longest path from root (DP on topological order 0..n-1)
  const depth=new Array(n).fill(0);
  for(let i=1;i<n;i++){{
    if(parents[i].length) depth[i]=Math.max(...parents[i].map(p=>depth[p]))+1;
  }}
  const maxD=Math.max(...depth);
  const layers=Array.from({{length:maxD+1}},()=>[]);
  depth.forEach((d,i)=>layers[d].push(i));
  // Layout constants
  const NW=240,NH=54,HGAP=18,VGAP=40,PAD=20;
  const layerW=layers.map(l=>l.length*(NW+HGAP)-HGAP);
  const totalW=Math.max(...layerW)+2*PAD;
  const totalH=(maxD+1)*(NH+VGAP)-VGAP+2*PAD;
  // Node positions (top-left corner)
  const pos=new Array(n);
  layers.forEach((layer,d)=>{{
    const lw=layer.length*(NW+HGAP)-HGAP;
    const sx=(totalW-lw)/2;
    layer.forEach((nid,k)=>{{ pos[nid]={{x:sx+k*(NW+HGAP),y:PAD+d*(NH+VGAP)}}; }});
  }});
  const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let p=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${{totalW}} ${{totalH}}" width="${{totalW}}" height="${{totalH}}" style="max-width:100%;height:auto;background:transparent">`,
    `<defs><marker id="dagar" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0,0 8,3 0,6" fill="#777"/></marker></defs>`];
  // Edges
  dagData.forEach((node,i)=>{{
    (node.children||[]).forEach(j=>{{
      if(j<0||j>=n) return;
      const x1=pos[i].x+NW/2,y1=pos[i].y+NH;
      const x2=pos[j].x+NW/2,y2=pos[j].y;
      const ym=(y1+y2)/2;
      p.push(`<path d="M${{x1}},${{y1}} C${{x1}},${{ym}} ${{x2}},${{ym}} ${{x2}},${{y2}}" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#dagar)"/>`);
    }});
  }});
  // Nodes — two-line label: line1=[idx] ClassName, line2=hyperparameters
  dagData.forEach((node,i)=>{{
    const {{x,y}}=pos[i];
    const isRoot=(i===0),isLeaf=!(node.children||[]).length;
    const fill=isRoot?'#3b6d11':isLeaf?'#185fa5':'#ffa600';
    const fc=(isRoot||isLeaf)?'#ececec':'#1a1a1a';
    const fullDesc=`[${{i}}] ${{node.desc||''}}`;
    const parts=fullDesc.split(' | ');
    const line1=esc(parts[0].slice(0,34));
    const line2=parts.length>1?esc(parts.slice(1).join(' | ').slice(0,38)):'';
    const ty1=y+(line2?NH/2-5:NH/2+4);
    p.push(
      `<rect x="${{x}}" y="${{y}}" width="${{NW}}" height="${{NH}}" rx="5" fill="${{fill}}" stroke="#444" stroke-width="1"><title>${{esc(fullDesc)}}</title></rect>`,
      `<text text-anchor="middle" font-size="10" font-family="monospace" fill="${{fc}}">`,
      `<tspan x="${{x+NW/2}}" y="${{ty1}}">${{line1}}</tspan>`,
      line2?`<tspan x="${{x+NW/2}}" dy="13">${{line2}}</tspan>`:'',
      `</text>`
    );
  }});
  p.push('</svg>');
  return p.join('');
}}

function _renderFormulaTree(nodes){{
  /* Render a sympy AST as an SVG tree.
     nodes = [{{id, label, kind:"op"|"var"|"const", children:[idx,...]}}]
     Root = node not referenced as any other node's child (auto-detected). */
  if(!nodes||!nodes.length) return null;
  // Auto-detect root (the binarizer appends root last, not at index 0)
  const _allCh=new Set(nodes.flatMap(nd=>(nd.children||[])));
  const root=Math.max(0,nodes.findIndex((_,i)=>!_allCh.has(i)));
  const NW=66,NH=34,HGAP=8,VGAP=38,PAD=18;
  // Compute subtree pixel widths
  const w=new Array(nodes.length).fill(NW);
  function calcW(i){{
    const ch=nodes[i].children||[];
    if(!ch.length){{w[i]=NW;return;}}
    ch.forEach(c=>calcW(c));
    w[i]=Math.max(NW,ch.reduce((s,c)=>s+w[c],0)+Math.max(0,ch.length-1)*HGAP);
  }}
  calcW(root);
  // Assign center-x and top-y for each node
  const cx=new Array(nodes.length).fill(0),cy=new Array(nodes.length).fill(0);
  function layout(i,left,depth){{
    cy[i]=PAD+depth*(NH+VGAP);
    const ch=nodes[i].children||[];
    if(!ch.length){{cx[i]=left+NW/2;return;}}
    let x=left;
    ch.forEach(c=>{{layout(c,x,depth+1);x+=w[c]+HGAP;}});
    cx[i]=(cx[ch[0]]+cx[ch[ch.length-1]])/2;
  }}
  layout(root,PAD,0);
  const totalW=w[root]+2*PAD;
  const totalH=Math.max(...cy)+NH+PAD;
  const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const parts=[
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${{totalW}} ${{totalH}}" width="${{totalW}}" height="${{totalH}}" style="max-width:100%;height:auto;background:transparent">`,
    `<defs><marker id="tarr" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto"><polygon points="0,0 6,2.5 0,5" fill="#888"/></marker></defs>`
  ];
  // Edges (curved bezier)
  nodes.forEach((nd,i)=>{{
    (nd.children||[]).forEach(j=>{{
      const x1=cx[i],y1=cy[i]+NH,x2=cx[j],y2=cy[j],ym=(y1+y2)/2;
      parts.push(`<path d="M${{x1}},${{y1}} C${{x1}},${{ym}} ${{x2}},${{ym}} ${{x2}},${{y2}}" fill="none" stroke="#999" stroke-width="1.4" marker-end="url(#tarr)"/>`);
    }});
  }});
  // Nodes
  nodes.forEach((nd,i)=>{{
    const x=cx[i]-NW/2,y=cy[i];
    const fill=nd.kind==='op'?'#ffa600':nd.kind==='const'?'#185fa5':'#3b6d11';
    const fc=nd.kind==='op'?'#1a1a1a':'#ececec';
    const lbl=esc(String(nd.label||'').slice(0,14));
    parts.push(
      `<rect x="${{x}}" y="${{y}}" width="${{NW}}" height="${{NH}}" rx="5" fill="${{fill}}" stroke="#444" stroke-width="1"/>`,
      `<text x="${{cx[i]}}" y="${{y+NH/2+4}}" text-anchor="middle" font-size="11" font-family="monospace" fill="${{fc}}">${{lbl}}</text>`
    );
  }});
  parts.push('</svg>');
  return parts.join('');
}}

function openModal(fid,mid,run){{
  const f=FORMULAS.find(x=>x.id===fid);
  const midx=METHODS.indexOf(mid);
  const _isPysr = mid==='pysr' || mid==='pysr_noise';
  const mnames=['PySR baseline','PySR +Noise','All ops ★ (ref)','Smart parallel***','Boosted spar***','+ConstantBrick','No OLS/rat/nest','Smart par w/ denoise***'];
  pending={{fid,mid,run}};
  document.getElementById('mtitle').textContent='Log run result';
  document.getElementById('mflabel').value=`${{f.name}}  ·  ${{mnames[midx]}}  ·  Run ${{run+1}}`;
  if(_isPysr){{
    const pysrLbl = mid==='pysr_noise'
      ? `Run ${{run+1}} — PySR +Noise (GP denoising pre-step, denoise=True)`
      : `Run ${{run+1}} — PySR baseline (single deterministic config; only the data seed varies between runs for synthetic targets)`;
    document.getElementById('minit').value=pysrLbl;
  }} else {{
    document.getElementById('minit').value=`R${{run+1}}: ${{MINIT[run]}}`;
  }}
  const k=key(fid,mid,run);const d=DB[k]||{{}};
  // ── Formula tabs (suppress R²/MSE info-line for PySR which uses raw MSE) ──
  _buildFormulaTabs(d, _isPysr);
  document.getElementById('mrt').value=d.runtime??'';
  // Total iterations (T): for smart-parallel ('spar') show
  // total = sum(streams)  +  per-stream breakdown  +  winner count.
  {{
    let tval = (d.totalT==null) ? '' : Number(d.totalT).toLocaleString();
    if (d.actualTPerStream && Object.keys(d.actualTPerStream).length){{
      const parts = Object.entries(d.actualTPerStream)
        .map(([k,v])=>`${{k}}=${{Number(v).toLocaleString()}}`).join(', ');
      const winStr = (d.actualTWinner!=null)
        ? `  ·  winner=${{Number(d.actualTWinner).toLocaleString()}}` : '';
      tval = `${{Number(d.totalT).toLocaleString()}}  (Σ streams: ${{parts}}${{winStr}})`;
    }}
    document.getElementById('miter').value = tval;
  }}
  document.getElementById('mpop').value=d.K??'';
  document.getElementById('mcomp').value=d.maxComp??'';
  // Min search loss reached (Dragon: alignment loss = 1−corr; PySR: best MSE)
  {{
    const sl=d.searchLoss;
    document.getElementById('mphase').value=(sl===null||sl===undefined)?'':
      (Number(sl).toExponential(4)+(d.searchLossKind?'  ('+d.searchLossKind+')':''));
  }}
  document.getElementById('mops').value=d.ops??'';
  document.getElementById('mnconst').value=d.nconst??'';
  document.getElementById('mdag').value=d.dagSize??'';
  document.getElementById('mnotes').value=d.notes??'';
  // ── Section: channels (DRAGON) or hall-of-fame (PySR) ──────────────
  const secLbl=document.getElementById('sec-channels-label');
  if(_isPysr){{
    if(secLbl) secLbl.textContent='PySR hall of fame — Pareto frontier (complexity vs loss)';
    _renderHallOfFame(d.hallOfFame||[]);
  }} else {{
    if(secLbl) secLbl.textContent='Output channels — all formulas (one per channel, read-only)';
    _renderChannelsRO(d.channels||[]);
  }}
  // ── DAG visualisation (SVG if available, otherwise textual table) ──
  const dviz=document.getElementById('mdagviz');
  if(dviz){{
    if(d.dagSvg){{
      dviz.style.display='block';
      dviz.style.whiteSpace='normal';
      dviz.style.textAlign='center';
      dviz.innerHTML=d.dagSvg;
      const svgEl=dviz.querySelector('svg');
      if(svgEl){{ svgEl.style.maxWidth='100%'; svgEl.style.height='auto'; }}
    }} else if(d.dagData&&d.dagData.length){{
      // JS-based SVG fallback (graphviz 'dot' not in PATH on this machine)
      const jsSvg=_renderDagSvg(d.dagData);
      if(jsSvg){{
        dviz.style.display='block';
        dviz.style.whiteSpace='normal';
        dviz.style.textAlign='center';
        dviz.innerHTML=jsSvg;
        const svgEl=dviz.querySelector('svg');
        if(svgEl){{ svgEl.style.maxWidth='100%'; svgEl.style.height='auto'; }}
      }} else {{
        dviz.style.display='block';
        dviz.style.whiteSpace='pre';
        dviz.style.textAlign='left';
        dviz.textContent=d.dagText||'';
      }}
    }} else if(d.dagText){{
      dviz.style.display='block';
      dviz.style.whiteSpace='pre';
      dviz.style.textAlign='left';
      dviz.textContent=d.dagText;
    }} else {{
      dviz.style.display='none';
      dviz.innerHTML='';
    }}
  }}
  // ── Statistical visualisations (per-run + per-formula) ─────────────
  function _execScripts(container){{
    // <script> tags inserted via innerHTML do not auto-execute. Re-create them
    // so embedded Plotly.newPlot() calls actually run.
    container.querySelectorAll('script').forEach(old=>{{
      const s=document.createElement('script');
      if(old.src) s.src=old.src; else s.textContent=old.textContent;
      old.parentNode.replaceChild(s,old);
    }});
  }}
  function _setSvgBlock(elId, svg, isWrappedBody){{
    const el=document.getElementById(elId);
    if(!el) return;
    if(svg){{
      el.style.display='block';
      const tgt=isWrappedBody?el.querySelector('#'+elId+'-body'):el;
      if(tgt){{
        tgt.innerHTML=svg;
        const s=tgt.querySelector('svg');
        if(s){{s.style.maxWidth='100%';s.style.height='auto';}}
        _execScripts(tgt);
      }}
    }} else {{
      el.style.display='none';
      const tgt=isWrappedBody?el.querySelector('#'+elId+'-body'):el;
      if(tgt) tgt.innerHTML='';
    }}
  }}
  // IMPORTANT: show the modal BEFORE injecting Plotly content. Plotly reads
  // the container's clientWidth at render time; if the modal is still
  // display:none the width is 0 and the chart is rendered squashed.
  const mwrap=document.getElementById('mwrap');
  mwrap.style.display='flex';
  document.body.style.overflow='hidden';
  // Dragon: 3-panel landscape; PySR: Pareto + AST tree.
  _setSvgBlock('mlandscape', _isPysr ? null : (d.landscapeSvg||null), false);
  _setSvgBlock('mpareto',    _isPysr ? (d.paretoSvg||null) : null,   false);
  if(_isPysr){{
    let _treeSvgContent=d.treeSvg||null;
    if(!_treeSvgContent&&d.treeData&&d.treeData.length)
      _treeSvgContent=_renderFormulaTree(d.treeData)||null;
    _setSvgBlock('mtree',_treeSvgContent,true);
  }}else{{
    _setSvgBlock('mtree',null,true);
  }}
  // Per-formula aggregate stats (boxplot + heatmap), shared across runs/methods.
  _setSvgBlock('mformstats', (typeof FORMULA_STATS_SVG!=='undefined' && FORMULA_STATS_SVG[fid]) || null, true);
  // Force Plotly to re-fit each visible chart now that the modal width is set.
  setTimeout(()=>{{
    if(window.Plotly){{
      ['mlandscape','mpareto','mformstats'].forEach(id=>{{
        const el=document.getElementById(id);
        if(!el||el.style.display==='none') return;
        el.querySelectorAll('.js-plotly-plot').forEach(p=>{{
          try{{ window.Plotly.Plots.resize(p); }}catch(e){{}}
        }});
      }});
    }}
  }},80);
  setTimeout(()=>{{ const mb=document.getElementById('mbox'); if(mb) mb.scrollTop=0; }},20);
}}
function closeModal(){{
  document.getElementById('mwrap').style.display='none';
  document.body.style.overflow='';
  pending=null;
}}
// Esc closes the modal popup
document.addEventListener('keydown',function(e){{
  if(e.key==='Escape'){{
    const mw=document.getElementById('mwrap');
    if(mw&&mw.style.display!=='none') closeModal();
  }}
}});
function saveRun(){{
  if(!pending) return;
  const {{fid,mid,run}}=pending;
  const k=key(fid,mid,run);
  DB[k]={{
    oneMinusR2:DB[k]?.oneMinusR2??null,
    mse:DB[k]?.mse??null,
    winner:DB[k]?.winner??null,
    bestCh:DB[k]?.bestCh??null,
    polyDeg:DB[k]?.polyDeg??null,
    nestedLink:DB[k]?.nestedLink??null,
    finalExpr:DB[k]?.finalExpr??null,
    runtime:DB[k]?.runtime??null,
    totalT:DB[k]?.totalT??null,
    actualTWinner:DB[k]?.actualTWinner??null,
    actualTPerStream:DB[k]?.actualTPerStream??{{}},
    K:DB[k]?.K??null,
    maxComp:DB[k]?.maxComp??null,
    phase:DB[k]?.phase??null,
    ops:DB[k]?.ops??null,
    nconst:DB[k]?.nconst??null,
    dagSize:DB[k]?.dagSize??null,
    channels:DB[k]?.channels||[],
    notes:DB[k]?.notes??null,
  }};
  const cell=document.getElementById(`cell_${{fid}}_${{mid}}`);
  if(cell) cell.innerHTML=renderCell(fid,mid);
  closeModal();updateStats();
  try{{window.storage.set('lb_v2',JSON.stringify(DB),false);}}catch(e){{}}
}}

function showTT(e,fid,mid,run){{
  const k=key(fid,mid,run);const d=DB[k];
  const f=FORMULAS.find(x=>x.id===fid);
  const mnames=['PySR','PySR +Noise','All ops ★ (ref)','Smart par','Boosted spar','ConstBrick','No OLS','Smart par+denoise'];
  const midx=METHODS.indexOf(mid);
  const tt=document.getElementById('ttbox');
  let h=`<div class="tttitle">${{f.name}} · ${{mnames[midx]}} · R${{run+1}}</div>`;
  h+=`<div class="ttr"><span>Init</span><span>${{MINIT[run]}}</span></div>`;
  if(d){{
    if(d.oneMinusR2!==null&&d.oneMinusR2!==undefined) h+=`<div class="ttr"><span>1−R²</span><span>${{fmtLoss(d.oneMinusR2)}}</span></div>`;
    if(d.mse!==null&&d.mse!==undefined) h+=`<div class="ttr"><span>MSE</span><span>${{d.mse.toExponential(3)}}</span></div>`;
    if(d.winner) h+=`<div class="ttr"><span>Winner</span><span>${{d.winner}}</span></div>`;
    if(d.polyDeg) h+=`<div class="ttr"><span>P-RAT degree</span><span>≤${{d.polyDeg}}</span></div>`;
    if(d.nestedLink) h+=`<div class="ttr"><span>Nested link</span><span>${{d.nestedLink}}</span></div>`;
    // Per-method 1−R² breakdown
    function ttloss(v){{ return (v!==null&&v!==undefined)?v.toExponential(3):'—'; }}
    if(d.lossChannel!==undefined||d.lossOls!==undefined||d.lossNested!==undefined){{
      h+=`<div style="border-top:0.5px solid var(--color-border-tertiary);margin:4px 0 3px"></div>`;
      h+=`<div style="font-size:8px;font-weight:500;color:var(--color-text-secondary);margin-bottom:2px">1\u2212R\u00b2 per method</div>`;
      if(d.lossChannel!==null&&d.lossChannel!==undefined) h+=`<div class="ttr"><span>Channel</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossChannel)}}</span></div>`;
      if(d.lossOls!==null&&d.lossOls!==undefined)         h+=`<div class="ttr"><span>Sparse OLS</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossOls)}}</span></div>`;
      if(d.lossNested!==null&&d.lossNested!==undefined)   h+=`<div class="ttr"><span>Nested OLS</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossNested)}}</span></div>`;
      if(d.lossPolyrat){{
        Object.keys(d.lossPolyrat).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(deg=>{{
          h+=`<div class="ttr"><span>P-RAT deg ${{deg}}</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossPolyrat[deg])}}</span></div>`;
        }});
      }}
    }}
    if(d.channels&&d.channels.length) h+=`<div class="ttr"><span>Channels</span><span>${{d.channels.length}}</span></div>`;
    if(d.phase) h+=`<div class="ttr"><span>Phase</span><span>${{PHASE_LABELS[d.phase]||d.phase}}</span></div>`;
    if(d.maxComp) h+=`<div class="ttr"><span>Max complexity</span><span>${{d.maxComp}}</span></div>`;
    if(d.totalT) h+=`<div class="ttr"><span>Total iters T</span><span>${{d.totalT.toLocaleString()}}</span></div>`;
    if(d.actualTPerStream && Object.keys(d.actualTPerStream).length){{
      const parts=Object.entries(d.actualTPerStream)
        .map(([k,v])=>`${{k}}:${{Number(v).toLocaleString()}}`).join('  ');
      h+=`<div class="ttr"><span>Streams T</span><span style="font-size:9px">${{parts}}</span></div>`;
      if(d.actualTWinner!=null) h+=`<div class="ttr"><span>Winner T</span><span>${{Number(d.actualTWinner).toLocaleString()}}</span></div>`;
    }}
    if(d.K) h+=`<div class="ttr"><span>Pop K</span><span>${{d.K}}</span></div>`;
    if(d.runtime) h+=`<div class="ttr"><span>Runtime</span><span>${{d.runtime.toFixed(2)}}s</span></div>`;
    if(d.ops) h+=`<div class="ttr"><span>Ops</span><span style="font-size:9px">${{d.ops}}</span></div>`;
    if(d.finalExpr) h+=`<div class="ttr" style="flex-direction:column;gap:2px"><span>Winner formula</span><span style="font-family:var(--font-mono);font-size:8px;word-break:break-all">${{d.finalExpr}}</span></div>`;
    const altCount=[d.formulaOls,d.formulaNested,...Object.values(d.formulaPolyrat||{{}})].filter(Boolean).length;
    if(altCount>0) h+=`<div style="font-size:8px;color:var(--color-text-tertiary);margin-top:2px">+${{altCount}} alt formula${{altCount>1?'s':''}} (click to toggle)</div>`;
    if(d.notes) h+=`<div class="ttr"><span>Notes</span><span>${{d.notes}}</span></div>`;
  }} else {{
    h+=`<div style="color:var(--color-text-tertiary);font-size:9px;margin-top:4px">Not run yet — click to log</div>`;
  }}
  tt.innerHTML=h;tt.style.display='block';
  const rect=e.target.getBoundingClientRect();
  const container=document.querySelector('.root').getBoundingClientRect();
  tt.style.left=Math.min(rect.left-container.left+20,container.width-220)+'px';
  tt.style.top=(rect.bottom-container.top+4)+'px';
}}
function hideTT(){{document.getElementById('ttbox').style.display='none';}}

document.getElementById('tabs').addEventListener('click',e=>{{
  const t=e.target.closest('.tab');if(!t)return;
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');activeCat=t.dataset.cat;buildTable();
}});

async function init(){{
  // Merge PRELOADED_DB (from Python run) with any manual additions stored in VS Code notebook
  let stored={{}};
  try{{
    const s=await window.storage.get('lb_v2',false);
    if(s&&s.value) stored=JSON.parse(s.value);
  }}catch(e){{}}
  DB=Object.assign({{}},PRELOADED_DB,stored);
  buildTable();
}}
init();"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DragonSR vs PySR — Leaderboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"
  onload="if(typeof buildTable==='function')buildTable();"></script>
<!-- Plotly (for interactive Search statistics & landscape charts) -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
{CSS_VARS}
{CSS}
</style>
</head>
<body>
{BODY}
<script>
{JS}
</script>
</body>
</html>"""

    out = Path(__file__).parent / "leaderboard_standalone.html"
    out.write_text(html, encoding="utf-8")
    print(f"✓ Standalone HTML written ({len(db)} entries): {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pysr_only(targets=TARGETS, n_runs=N_RUNS):
    """Complete missing PySR entries in an existing results.json, then rebuild HTML.

    Entries with formula starting with 'ERROR:' or 'SKIP' are treated as
    failed and will be re-run (removed from results before re-running).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_path}")
    else:
        results = []
        print("No existing results.json found — starting from scratch.")

    def _is_failed(r):
        f = str(r.get("formula", "")).strip()
        if f in ("", "N/A", "TIMEOUT", "PySR: no result"):
            return True
        return f.startswith(("ERROR:", "SKIP", "PySR: no result"))

    # Remove failed PySR entries so they can be re-run
    results = [r for r in results
               if r.get("method") not in ("pysr", "pysr_noise") or not _is_failed(r)]

    done = {(r["target"], int(r["run_id"]), r["method"]) for r in results}

    for target in targets:
        for run_id in range(n_runs):
            # Pre-build shared data so both pysr and pysr_noise use the same arrays
            _seed_data = RANDOM_SEED + run_id
            _X_base, _y_clean = load_dataset(target, run_id=run_id)
            _noise_rng = np.random.default_rng(_seed_data + 9999)
            _y_noisy   = _y_clean + _noise_rng.normal(
                0, NOISE_STD * float(_y_clean.std()), len(_y_clean)
            )
            for add_noise in (False, True):
                method_id = "pysr_noise" if add_noise else "pysr"
                if (target, run_id, method_id) in done:
                    print(f"  [SKIP] {target}/run_{run_id}/{method_id} already present")
                    continue
                print(f"  [RUN]  {target}/run_{run_id}/{method_id}")
                r = run_pysr(target, run_id, add_noise=add_noise,
                             _X_preloaded=_X_base, _y_clean_preloaded=_y_clean,
                             _y_noisy_preloaded=_y_noisy)
                results.append(r)
                done.add((target, run_id, method_id))
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                build_html(results)
    return results


def run_dragon_only(targets=TARGETS, n_runs=N_RUNS):
    """Complete missing Dragon entries in an existing results.json, then rebuild HTML.

    Entries with formula 'N/A', '' or starting with 'ERROR:' are treated as
    failed and will be re-run.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_path}")
    else:
        results = []
        print("No existing results.json found — starting from scratch.")

    dragon_method_ids = {cfg["id"] for cfg in DRAGON_METHOD_CONFIGS}

    def _is_failed(r):
        f = str(r.get("formula", "")).strip()
        return f in ("", "N/A") or f.startswith("ERROR:")

    # Remove failed Dragon entries so they can be re-run
    results = [r for r in results
               if r.get("method") not in dragon_method_ids or not _is_failed(r)]

    done = {(r["target"], int(r["run_id"]), r["method"]) for r in results}

    for target in targets:
        for run_id in range(n_runs):
            for cfg in DRAGON_METHOD_CONFIGS:
                method_id = cfg["id"]
                if (target, run_id, method_id) in done:
                    print(f"  [SKIP] {target}/run_{run_id}/{method_id} already present")
                    continue
                print(f"  [RUN]  {target}/run_{run_id}/{method_id}")
                r = dragon_worker(cfg, target, run_id)
                results.append(r)
                done.add((target, run_id, method_id))
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                build_html(results)
    return results


if __name__ == "__main__":
    import argparse
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pysr-only", action="store_true",
                        help="Skip DragonSR; only run missing PySR entries and rebuild HTML")
    parser.add_argument("--dragon-only", action="store_true",
                        help="Skip PySR; only run missing Dragon entries and rebuild HTML")
    parser.add_argument("--continue", "-continue", dest="resume", action="store_true",
                        help="Resume run_all() from an existing results.json instead of starting fresh")
    args = parser.parse_args()
    if args.pysr_only:
        run_pysr_only()
    elif args.dragon_only:
        run_dragon_only()
    else:
        run_all(resume=args.resume)
