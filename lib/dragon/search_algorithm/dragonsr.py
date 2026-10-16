"""DragonSR: symbolic regression by DAG search, as a scikit-learn style estimator.

Ported from the frozen experiment code (CODE_VERSION AGGREGATE 2846c144):
the stream loop and tie-break of ``leaderboard/runner/Worker.py``
(``_run_smart_parallel``), the curriculum of
``leaderboard/pipeline/DragonOrchestrator.py`` (``run_search``,
``run_worker_pipeline``) and the preprocessing of
``leaderboard/dataprocessing/Preprocessing.py``.  The defaults are the
configuration of the paper's runs (``config_fixed_noise*.txt``); the search
logic is unchanged.  What differs from the experiment harness:

* the data are given to ``fit`` -- no target is loaded, no noise is injected
  and no denoiser is applied (the paper's DragonSR runs used none);
* the operator families never depend on the target (the paper's runs set
  ``PER_TARGET_OPS = False``);
* the search state is written to a temporary directory, removed after the
  search, unless ``save_dir`` is given;
* ``predict`` evaluates the reported expression itself, so a prediction is
  exactly what the printed formula says.

Usage
-----
>>> model = DragonSR(n_iterations=2000).fit(X, y)       # X: DataFrame
>>> model.formula_
>>> model.predict(X_new)
"""
from __future__ import annotations

import keyword
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dragon.search_algorithm.mutant_ucb import Mutant_UCB
from dragon.search_space.symbolic import (
    CombinationBuilder, SymbolicSearchSpace, operator_streams, MAX_NODES,
)
from dragon.utils.symbolic.evaluation import (
    DragonSearcher, RegressionDataset, COMPOSITION_PENALTY,
)
from dragon.utils.symbolic.preprocessing import (
    VarAugmentor, XGBoostSelector, N_TOP_FEATURES,
)

# Streams whose loss lies within this window of the lowest are tied
# (frozen Worker._run_smart_parallel: _TIE_REL, _TIE_ABS).
TIE_REL, TIE_ABS = 0.02, 1e-10
# Among tied streams: simplest model class first (frozen _ORDER).
CLASS_ORDER = {"channel": 0, "linear": 0, "ols": 1, "nested": 2, "rational": 3}

# Functions a reported expression may contain, for ``predict``.
_FORMULA_NAMESPACE = {
    "exp": np.exp, "ln": np.log, "log": np.log, "sin": np.sin, "cos": np.cos,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign, "where": np.where,
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "pi": np.pi,
}


def _seed_everything(seed: int) -> None:
    """Same effect as the frozen ``dragon.utils.tools.set_seed(seed, change_random=True)``."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _computed_rows(save_dir: str) -> int:
    try:
        return len(pd.read_csv(os.path.join(save_dir, "computation_file.csv")))
    except Exception:
        return 0


def _recovered_models(save_dir: str) -> int:
    """Number of models Mutant-UCB recovers from ``save_dir`` (frozen ``_recovered_models``)."""
    import glob
    return len(glob.glob(os.path.join(save_dir, "x_*.pkl")))


def _has_var(r: dict) -> bool:
    # A bare constant can land in the tie window on targets with a huge dynamic range.
    return any(c.isalpha() for c in str(r.get("formula") or ""))


def _loss_of(r: dict) -> float:
    v = r.get("loss")
    return float("inf") if v is None else v


def _simplicity(r: dict) -> tuple:
    return (0 if _has_var(r) else 1,
            CLASS_ORDER.get(r.get("winner_type"), 9),
            len(str(r.get("formula") or "")),
            _loss_of(r))


def select_stream(stream_results: list[dict]) -> dict:
    """The stream whose expression a run reports (frozen tie-break)."""
    lo = min((_loss_of(r) for r in stream_results), default=float("inf"))
    if lo == float("inf"):
        near = list(stream_results)
    else:
        cut = lo + max(abs(lo) * TIE_REL, TIE_ABS)
        near = [r for r in stream_results if _loss_of(r) <= cut] or list(stream_results)
    return min(near, key=_simplicity)


class DragonSR:
    """Symbolic regression by search over directed acyclic graphs.

    The defaults are the configuration of the paper's runs.

    Parameters
    ----------
    n_iterations : int
        Evaluations of one run, shared equally by its streams (20 000).
    families : list of str, optional
        Operator families searched, one stream per family and objective
        (default: all six of ``OPERATOR_FAMILIES``).
    objectives : sequence of str
        Stream objectives: ``"channel"`` fits one channel affinely rescaled,
        ``"full"`` the four model classes.
    k_init : int
        Random single-node graphs drawn per stream (30).
    t_per_level : int
        Evaluations added per curriculum level (30).
    max_complexity : int
        Number of curriculum levels (55).
    max_nodes : int
        Maximum number of nodes of a graph (15).
    n_top_features : int
        Features kept by the XGBoost ranking (14).
    var_aug : bool
        Add the engineered feature(s) most correlated with ``y``.
    subsample : float
        Fraction of the observations each candidate is scored on; the paper
        used 0.1 on noisy targets and 1.0 otherwise.
    search_loss : str
        ``"corr"`` (1 - Pearson r^2, the paper's), or one of
        ``SearchLoss.available()``.
    composition_penalty : float
        Weight of the penalty on nested same-family functions (0.01).
    optimize_constants : bool
        Fit constants by the scale-free line search.
    seed : int
        Seeds the starting state.  The streams are threads that share one
        generator, so a run is not reproducible bit for bit.
    n_jobs : int, optional
        Threads (default: one per stream).
    save_dir : str, optional
        Keep the search state there instead of in a temporary directory.
    device : str, optional
        Torch device (default: CUDA when available).
    verbose : bool
        Print the search's progress.

    Attributes
    ----------
    formula_ : str
        The reported expression, in terms of the raw inputs.
    winner_type_ : str
        Its model class: ``channel``, ``ols`` (linear), ``nested``
        (transformed) or ``rational``.
    stream_ : str
        The stream that reported it, ``"<family>|<objective>"``.
    streams_ : list of dict
        Loss, expression, class and evaluations of every stream.
    """

    def __init__(
        self,
        n_iterations: int = 20_000,
        families=None,
        objectives=("channel", "full"),
        k_init: int = 30,
        t_per_level: int = 30,
        max_complexity: int = 55,
        max_nodes: int = MAX_NODES,
        n_top_features: int = N_TOP_FEATURES,
        var_aug: bool = True,
        subsample: float = 1.0,
        search_loss: str = "corr",
        composition_penalty: float = COMPOSITION_PENALTY,
        optimize_constants: bool = True,
        seed: int = 42,
        n_jobs: int | None = None,
        save_dir: str | None = None,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.n_iterations = n_iterations
        self.families = families
        self.objectives = objectives
        self.k_init = k_init
        self.t_per_level = t_per_level
        self.max_complexity = max_complexity
        self.max_nodes = max_nodes
        self.n_top_features = n_top_features
        self.var_aug = var_aug
        self.subsample = subsample
        self.search_loss = search_loss
        self.composition_penalty = composition_penalty
        self.optimize_constants = optimize_constants
        self.seed = seed
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.device = device
        self.verbose = verbose

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names=None):
        X = self._as_frame(X, feature_names)
        y = pd.Series(np.asarray(y, dtype=np.float64).ravel(), index=X.index, name="y")
        self.n_features_in_ = X.shape[1]
        self.raw_feature_names_ = list(X.columns)
        t0 = time.time()

        # Preprocessing, as in PreprocessingPipeline.transform (no noise, no denoiser).
        X_aug = VarAugmentor().transform(X, y) if self.var_aug else X
        names, scores = XGBoostSelector(n_top=self.n_top_features).transform(X_aug, y)
        X_sel = X_aug[names]
        self.feature_names_ = list(names)
        self.feature_scores_ = dict(scores)

        streams = operator_streams(self.families)
        objectives = list(self.objectives) if self.objectives else [None]
        n_streams = len(streams) * len(objectives)
        stream_budget = max(1, int(self.n_iterations) // n_streams)
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # One seeding per run, before any stream starts (frozen _run_smart_parallel).
        _seed_everything(int(self.seed))

        root = self.save_dir or tempfile.mkdtemp(prefix="dragonsr_")
        jobs = [(fam, ops, obj) for fam, ops in streams.items() for obj in objectives]
        results = []
        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs or n_streams) as ex:
                futs = {ex.submit(self._run_stream, fam, ops, obj, X_sel, y, device,
                                  stream_budget, root): (fam, obj)
                        for fam, ops, obj in jobs}
                for f in as_completed(futs):
                    results.append(f.result())
        finally:
            if self.save_dir is None:
                shutil.rmtree(root, ignore_errors=True)

        best = select_stream(results)
        self.formula_ = best["formula"]
        self.loss_ = best["loss"]
        self.winner_type_ = best["winner_type"]
        self.stream_ = best["stream"]
        self.model_ = best.pop("_model", None)
        self.streams_ = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        self.n_evaluations_ = int(sum(r["actualT"] for r in results))
        self.time_s_ = time.time() - t0
        return self

    def _run_stream(self, family, ops, objective, X_sel, y, device, budget, root):
        sid = family if objective is None else f"{family}|{objective}"
        save_dir = os.path.join(root, f"stream_{sid.replace('|', '_')}", "save")
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.time()

        names = list(X_sel.columns)
        loader = DataLoader(RegressionDataset(X_sel, y), batch_size=32, shuffle=True)
        combos = CombinationBuilder().build(names, self.feature_scores_)
        search_space, dag = SymbolicSearchSpace(
            names, self.feature_scores_, list(ops), max_nodes=self.max_nodes).build(combos)
        sub = float(self.subsample) < 1.0
        searcher = DragonSearcher(
            search_space, loader, device, X_sel.shape[1], names, log_path=None,
            loss_mode=objective or "full", search_loss=self.search_loss,
            optimize_constants=self.optimize_constants, constants_optimizer="dichotomy",
            subsample_ratio=float(self.subsample) if sub else 1.0,
            X_np=X_sel.values.astype(np.float32) if sub else None,
            y_np=y.values.astype(np.float32).ravel() if sub else None,
            sample_weights=None, composition_penalty=self.composition_penalty,
            verbose=self.verbose)

        best_loss, reached = self._curriculum(search_space, dag, searcher, save_dir, budget)
        searcher.finalize_constants()
        refined = searcher.state.get("best_loss")
        if refined is not None and np.isfinite(refined):
            best_loss = float(refined)
        searcher.finalize_ols_postprocessing()
        s = searcher.state
        return {
            "stream": sid, "family": family, "objective": objective,
            "loss": float(best_loss), "formula": str(s.get("best_formula", "N/A")),
            "winner_type": s.get("winner_type", "channel"),
            "actualT": _computed_rows(save_dir), "max_complexity_reached": reached,
            "time_s": time.time() - t0, "_model": s.get("best_model"),
        }

    def _curriculum(self, search_space, dag, searcher, save_dir, budget):
        """Frozen DragonOrchestrator.run_search: levels of ``t_per_level`` evaluations."""
        global_best, reached, total = np.inf, 0, 0
        for complexity in range(1, int(self.max_complexity) + 1):
            remaining = budget - total
            if remaining <= 0:
                break
            dag.complexity = complexity
            first = complexity == 1
            pop = 0 if first else _recovered_models(save_dir)
            T = pop + min(int(self.t_per_level), remaining)
            sa = Mutant_UCB(search_space=search_space, evaluation=searcher, T=T,
                            K=int(self.k_init), N=1, E=1000, save_dir=save_dir,
                            clean_all=False, verbose=self.verbose, loss_threshold=None,
                            **({} if first else {"pop_path": save_dir}))
            sa.run()
            reached = complexity
            try:
                total = len(pd.read_csv(os.path.join(save_dir, "computation_file.csv")))
            except Exception:
                total += T
            global_best = min(global_best, sa.min_loss)
        # Disk safeguard of the frozen code, run once after the levels: keep the
        # best_model/ directories and the csv files only (a run writes ~10^5 files).
        for name in os.listdir(save_dir):
            full = os.path.join(save_dir, name)
            if os.path.isdir(full):
                if "best_model" not in name:
                    shutil.rmtree(full, ignore_errors=True)
            elif ".csv" not in name:
                try:
                    os.remove(full)
                except OSError:
                    pass
        return global_best, reached

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X, feature_names=None) -> np.ndarray:
        """Evaluate the reported expression on the raw inputs."""
        if not hasattr(self, "formula_"):
            raise RuntimeError("DragonSR is not fitted")
        if self.formula_.startswith(("ERROR", "N/A")):
            raise RuntimeError(f"no expression to evaluate: {self.formula_[:80]}")
        X = self._as_frame(X, feature_names or self.raw_feature_names_)
        env = dict(_FORMULA_NAMESPACE)
        env.update({c: X[c].to_numpy(dtype=np.float64) for c in X.columns})
        with np.errstate(all="ignore"):
            out = eval(self.formula_, {"__builtins__": {}}, env)
        return np.broadcast_to(np.asarray(out, dtype=np.float64), (len(X),)).copy()

    def score(self, X, y) -> float:
        """Coefficient of determination of ``predict(X)``."""
        y = np.asarray(y, dtype=np.float64).ravel()
        r = y - self.predict(X)
        return float(1.0 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2))

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _as_frame(X, feature_names=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            names = list(feature_names) if feature_names is not None else \
                [f"x{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=names)
        bad = [c for c in df.columns
               if not str(c).isidentifier() or keyword.iskeyword(str(c))
               or str(c) in _FORMULA_NAMESPACE]
        if bad:
            raise ValueError(f"feature names must be Python identifiers that are not "
                             f"function names of the printed expressions: {bad}")
        return df.astype(np.float64)
