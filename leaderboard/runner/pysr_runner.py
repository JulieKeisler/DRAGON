# pysr_runner.py
from __future__ import annotations

import os
import time
import traceback

import numpy as np

from Config import PySR as _CfgPySR, Paths as _CfgPaths, Experiment as _CfgExp
from dataprocessing.Data import DatasetLoader
from helpers.stats import (
    _compute_pysr_scores,
    _make_pysr_pareto_svg,
    _make_pysr_tree_svg,
    _make_pysr_tree_data,
)

_dataset_loader = DatasetLoader()


# ══════════════════════════════════════════════════════════════════════════════
#  PySR RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  PySR RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _configure_pysr_environment() -> None:
    """Configure the Julia project environment used by PySR/Juliacall."""
    julia_project = os.environ.get("PYSR_JULIA_PROJECT") or _CfgPySR.JULIA_PROJECT
    if not julia_project:
        return
    os.environ["PYSR_JULIA_PROJECT"] = julia_project
    os.environ["PYTHON_JULIAPKG_PROJECT"] = julia_project
    os.environ["JULIA_PROJECT"] = julia_project

def run_pysr(
    target: str,
    run_id: int,
    add_noise: bool = False,
    _X_preloaded=None,
    _y_clean_preloaded=None,
    _y_noisy_preloaded=None,
) -> dict:
    """Run PySR via the PySRRegressor Python API (SymbolicRegression.jl backend).

    add_noise=True → pass the pre-built noisy y (same as Dragon) to PySRRegressor.
    Method id is then "pysr_noise".
    """
    _configure_pysr_environment()
    from pysr import PySRRegressor

    method_id = "pysr_noise" if add_noise else "pysr"
    run_dir   = os.path.join(_CfgPaths.OUTPUT_DIR, target, method_id, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    log_path  = os.path.join(run_dir, f"{target}_{method_id}{_CfgPaths.LOG_SUFFIX}")

    t_start      = time.time()
    best_loss    = np.inf
    best_formula = "N/A"
    hall_of_fame = []

    try:
        # ── Load data ──────────────────────────────────────────────────────────
        if _X_preloaded is not None and _y_clean_preloaded is not None:
            X_df     = _X_preloaded
            y_series = (
                _y_noisy_preloaded
                if (add_noise and _y_noisy_preloaded is not None)
                else _y_clean_preloaded
            )
        else:
            X_df, y_series = _dataset_loader.load(target, run_id=run_id)

        # ── Denoise (same fast GPR as DragonSR, not PySR's slow built-in) ──────
        # PySR's denoise=True fits a GP on ALL points with n_restarts_optimizer=50,
        # which is orders of magnitude slower than DragonSR's GPRDenoiser (subsampled,
        # n_restarts=2). When add_noise=True we pre-denoise y here and feed the clean
        # signal to PySR with denoise=False.
        denoise_info = None
        if add_noise:
            from dataprocessing.Denoise import GPRDenoiser
            denoiser = GPRDenoiser(seed=_CfgExp.RANDOM_SEED + run_id)
            y_series, denoise_info = denoiser.transform(X_df, y_series)

        X             = X_df.values.astype(np.float64)
        y             = y_series.values.ravel().astype(np.float64)
        feature_names = list(X_df.columns)

        # ── Fit ────────────────────────────────────────────────────────────────
        model = PySRRegressor(
            niterations=_CfgPySR.N_ITERATIONS,
            populations=_CfgPySR.POPULATIONS,
            population_size=_CfgPySR.POPULATION_SIZE,
            maxsize=_CfgPySR.MAXSIZE,
            binary_operators=_CfgPySR.BINARY_OPS,
            unary_operators=_CfgPySR.UNARY_OPS,
            should_optimize_constants=_CfgPySR.SHOULD_OPTIMIZE_CONSTANTS,
            denoise=False,
            julia_project=_CfgPySR.JULIA_PROJECT,
            verbosity=0,
            random_state=_CfgExp.RANDOM_SEED + run_id,
        )
        model.fit(X, y, variable_names=feature_names)

        # ── Write run log ──────────────────────────────────────────────────────
        with open(log_path, "w") as _lf:
            _lf.write(
                f"target={target} method={method_id} run_id={run_id}\n"
                f"niterations={_CfgPySR.N_ITERATIONS} populations={_CfgPySR.POPULATIONS} "
                f"population_size={_CfgPySR.POPULATION_SIZE} maxsize={_CfgPySR.MAXSIZE} "
                f"should_optimize_constants={_CfgPySR.SHOULD_OPTIMIZE_CONSTANTS} "
                f"denoise={add_noise} denoise_backend={'GPRDenoiser' if add_noise else 'none'}\n"
            )
            if denoise_info is not None:
                _lf.write(f"denoise_info={denoise_info}\n")

        # ── Extract results ────────────────────────────────────────────────────
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
            "PySR +Noise with DragonSR GPRDenoiser (subsampled Matern GP) pre-applied"
            if add_noise else "PySR (SymbolicRegression.jl)"
        ),
        "hall_of_fame": hall_of_fame,
        "pareto_svg":   pareto_svg,
        "tree_svg":     tree_svg,
        "tree_data":    tree_data,
    }
