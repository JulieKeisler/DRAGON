"""Denoising and sample-weighting utilities for DRAGON.

Provides self-validating GPR and LinearGAM denoisers, MC-Dropout confidence
weighting, noise injection, and a sampling context. All config-dependent
defaults are exposed as constructor parameters so the module has no
dependency on any application-specific Config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
#  GPR DENOISER  (self-validating Matérn smoother)
# ══════════════════════════════════════════════════════════════════════════════

class GPRDenoiser:
    """Self-validating GPR-Matérn denoiser.

    Fits a GP with (Constant × Matérn) + WhiteKernel on a subsample,
    estimates the noise floor σ̂² by marginal likelihood, then applies
    a calibration guard: if held-out RMSE / σ̂ >= 1.15 the smoother is
    oversmoothing real structure and denoising is skipped.

    Parameters
    ----------
    max_samples : int
        Maximum subsample size for GP fitting.
    nu : float
        Matérn smoothness parameter.
    seed : int
        Random seed for subsampling and GP fitting.
    """

    def __init__(self, max_samples: int = 600, nu: float = 2.5, guard_threshold: float = 1.15, seed: int = 42):
        self.max_samples = max_samples
        self.nu = nu
        self.guard_threshold = guard_threshold
        self.seed = seed

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, dict]:
        from sklearn.model_selection import KFold

        std_y = float(y.std())
        if len(y) < 8 or std_y < 1e-12:
            return y.copy(), {"applied": False, "reason": "too few samples / constant y"}

        Xs = self._scale(X.values.astype(float))
        yv = y.values.astype(float)

        fold_err = []
        for tr, te in KFold(n_splits=4, shuffle=True, random_state=self.seed).split(Xs):
            try:
                yh = self._fit_predict(Xs[tr], yv[tr], Xs[te])
                fold_err.append(np.mean((yh - yv[te]) ** 2))
            except Exception:
                fold_err.append(np.var(yv))
        rmse_holdout = float(np.sqrt(np.mean(fold_err)))

        gp = self._make_gp()
        try:
            if len(y) > self.max_samples:
                sub = np.random.default_rng(self.seed).choice(len(y), self.max_samples, replace=False)
                gp.fit(Xs[sub], yv[sub])
            else:
                gp.fit(Xs, yv)
            y_hat = gp.predict(Xs)
            noise_level = float(gp.kernel_.k2.noise_level)
        except Exception as e:
            return y.copy(), {"applied": False, "reason": f"gpr failed: {e}"}

        sigma_hat = np.sqrt(max(noise_level, 0.0)) * std_y
        ratio = rmse_holdout / max(sigma_hat, 1e-12)
        noise_var = float(sigma_hat ** 2)
        snr_db = 10.0 * np.log10(max(np.var(yv) - noise_var, 1e-30) / max(noise_var, 1e-30))

        applied = bool(ratio < self.guard_threshold)
        info = {
            "applied": applied,
            "ratio": float(ratio),
            "noise_var": noise_var,
            "noise_sigma": float(sigma_hat),
            "snr_db": float(snr_db),
        }
        if applied:
            return pd.Series(y_hat, index=y.index, name=y.name), info
        return y.copy(), info

    def _make_gp(self):
        """Build a new GaussianProcessRegressor with Constant × Matérn + WhiteKernel."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        kern = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=self.nu)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e2))
        )
        return GaussianProcessRegressor(
            kernel=kern, n_restarts_optimizer=2,
            normalize_y=True, random_state=self.seed,
        )

    def _fit_predict(self, Xtr, ytr, Xte):
        """Fit a GP on the training split and predict on the test split."""
        gp = self._make_gp()
        if len(ytr) > self.max_samples:
            sub = np.random.default_rng(self.seed).choice(len(ytr), self.max_samples, replace=False)
            gp.fit(Xtr[sub], ytr[sub])
        else:
            gp.fit(Xtr, ytr)
        return gp.predict(Xte)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Standardize features to zero mean and unit variance."""
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  LinGAM DENOISER
# ══════════════════════════════════════════════════════════════════════════════

class LinGAMDenoiser:
    """Denoiser using pygam's LinearGAM; returns original y on import failure, too few samples, or constant y.

    Parameters
    ----------
    transform(X, y) -> tuple[pd.Series, dict]
        The returned dict contains ``"applied"`` (bool) and, when True, ``"model"`` (str).
    """

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, dict]:
        try:
            from pygam import LinearGAM
        except Exception as e:
            return y.copy(), {"applied": False, "reason": f"pygam import failed: {e}"}

        std_y = float(y.std())
        if len(y) < 8 or std_y < 1e-12:
            return y.copy(), {"applied": False, "reason": "too few samples / constant y"}

        try:
            model = LinearGAM().fit(X.values.astype(float), y.values.astype(float))
            y_hat = model.predict(X.values.astype(float))
            return pd.Series(y_hat, index=y.index, name=y.name), {"applied": True, "model": "LinearGAM"}
        except Exception as e:
            return y.copy(), {"applied": False, "reason": f"LinearGAM failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
#  MC-DROPOUT WEIGHTER
# ══════════════════════════════════════════════════════════════════════════════

class MCDropoutWeighter:
    """Estimate per-sample confidence weights via MC-Dropout.

    Parameters
    ----------
    n_forward : int
        Number of stochastic forward passes.
    dropout_p : float
        Dropout probability during training and inference.
    n_epochs : int
        Training epochs.
    hidden : int
        Hidden layer width.
    random_seed : int
        RNG seed.
    """

    def __init__(
        self,
        n_forward: int = 20,
        dropout_p: float = 0.15,
        n_epochs: int = 60,
        hidden: int = 32,
        random_seed: int = 42,
    ):
        self.n_forward = n_forward
        self.dropout_p = dropout_p
        self.n_epochs = n_epochs
        self.hidden = hidden
        self.random_seed = random_seed

    @classmethod
    def get_sample_weights(
        cls,
        method_cfg: dict,
        X: pd.DataFrame,
        y: pd.Series,
        seed: int,
        method_id: str | None = None,
    ) -> np.ndarray | None:
        if not method_cfg.get("mc_dropout", False):
            return None
        try:
            sample_weights = cls(random_seed=seed).compute_weights(X, y)
            if method_id is not None:
                print(f"[{method_id}] MC-Dropout: min={sample_weights.min():.3f} max={sample_weights.max():.3f}")
            return sample_weights
        except Exception as e:
            if method_id is not None:
                print(f"[{method_id}] MC-Dropout FAILED ({e})")
            return None

    def compute_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Return per-sample confidence weights (shape: (n,), dtype: float32)."""
        n, d = len(y), X.shape[1]
        if n < 8:
            return np.ones(n, dtype=np.float32)

        Xv = X.values.astype(np.float32)
        yv = y.values.astype(np.float32).ravel()
        xstd = Xv.std(0) + 1e-8
        xmean = Xv.mean(0)
        ystd = float(yv.std()) + 1e-8
        ymean = float(yv.mean())
        Xs = (Xv - xmean) / xstd
        ys = (yv - ymean) / ystd

        Xt = torch.tensor(Xs)
        yt = torch.tensor(ys).unsqueeze(1)
        torch.manual_seed(self.random_seed)
        h = self.hidden
        net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p=self.dropout_p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p=self.dropout_p),
            nn.Linear(h // 2, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
        mse_f = nn.MSELoss()
        net.train()
        for _ in range(self.n_epochs):
            opt.zero_grad()
            mse_f(net(Xt), yt).backward()
            opt.step()

        net.train()
        with torch.no_grad():
            preds = np.stack([net(Xt).squeeze(1).detach().numpy() for _ in range(self.n_forward)])

        variances = preds.var(axis=0)
        mean_var = float(variances.mean()) + 1e-30
        confidence = 1.0 / (1.0 + variances / mean_var)
        weights = confidence * n / confidence.sum()
        return weights.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE INJECTOR
# ══════════════════════════════════════════════════════════════════════════════

class NoiseInjector:
    """Adds Gaussian noise to y (relative to std(y)).

    Parameters
    ----------
    noise_std : float or None
        Standard deviation of noise as a fraction of ``std(y)``.
        If ``None``, defaults to ``0.05`` (5% of std(y)).
    seed : int
        Base RNG seed.
    """

    def __init__(self, noise_std: float | None = None, seed: int = 42):
        self.noise_std = 0.05 if noise_std is None else noise_std
        self.seed = seed

    def transform(self, y: pd.Series, run_id: int = 0) -> pd.Series:
        rng = np.random.default_rng(self.seed + run_id + 9999)
        noise = rng.normal(0, self.noise_std * float(y.std()), len(y))
        return y + noise


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLING CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

class SamplingContext:
    """Holds subsampling configuration for stochastic evaluation.

    Parameters
    ----------
    subsample_ratio : float
        Fraction of the dataset to use per evaluation.
    X_np : np.ndarray or None
        Feature matrix as numpy (for subsampling).
    y_np : np.ndarray or None
        Target vector as numpy (for subsampling).
    """

    def __init__(self, subsample_ratio: float = 1.0, X_np=None, y_np=None):
        self.subsample_ratio = subsample_ratio
        self.X_np = X_np
        self.y_np = y_np

    @classmethod
    def get_sampling(cls, method_cfg: dict, X_sel: pd.DataFrame, y: pd.Series,
                     subsample_ratio: float = 0.5):
        """Build a SamplingContext from a method config dict.

        Parameters
        ----------
        method_cfg : dict
            Must contain ``"sampling"`` key (bool).
        X_sel : pd.DataFrame
        y : pd.Series
        subsample_ratio : float
            Ratio to use when sampling is enabled.
        """
        if not method_cfg.get("sampling", False):
            return cls()
        return cls(
            subsample_ratio=float(subsample_ratio),
            X_np=X_sel.values.astype(np.float32),
            y_np=y.values.astype(np.float32).ravel(),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DAG FEATURE EXTRACTOR  (for surrogate modeling)
# ══════════════════════════════════════════════════════════════════════════════

OP_NAMES = [
    "Identity", "Negate", "Inverse", "SumFeatures",
    "SelectFeatures", "ConstantBrick", "Power", "Sqrt",
    "Ln", "Sin", "Cos", "Exp", "ExpAffine", "ChannelBoost",
    "Abs",
]


class DAGFeatureExtractor:
    """Convert a DAG to a fixed-size numerical vector for surrogate modeling.

    Features per DAG:
      * One-hot counts of each operation type across all nodes (|OP_NAMES|)
      * One-hot combiner type counts (add, mul)
      * Global graph stats: edge count, active-node count, max in-degree,
        mean in-degree, node count
    """

    def __init__(self, max_nodes: int = 15):
        self.max_nodes = max_nodes
        self.n_ops = len(OP_NAMES)
        self.feature_dim = self.n_ops + 2 + 5

    def extract(self, dag) -> np.ndarray:
        """Return shape ``(1, feature_dim)``."""
        features: list[float] = []

        op_counts = [0.0] * self.n_ops
        comb_counts = [0.0, 0.0]  # add, mul
        nodes = self._get_operation_list(dag)

        for node in nodes:
            name = self._get_op_name(node)
            if name in OP_NAMES:
                idx = OP_NAMES.index(name)
                op_counts[idx] += 1.0
            comb = getattr(node, "combiner", None)
            if comb == "add":
                comb_counts[0] += 1.0
            elif comb == "mul":
                comb_counts[1] += 1.0

        features.extend(op_counts)
        features.extend(comb_counts)

        adj = self._get_adj_matrix(dag)
        n_edges = float(np.sum(adj > 0))
        n_active = float(self._count_active_nodes(adj))
        in_deg = np.sum(adj, axis=0)
        max_in_deg = float(np.max(in_deg))
        mean_in_deg = float(np.mean(in_deg))
        n_nodes = float(len(nodes))

        features.extend([n_edges, n_active, max_in_deg, mean_in_deg, n_nodes])

        return np.array(features, dtype=np.float32).reshape(1, -1)

    def _get_operation_list(self, dag):
        """Return the list of operation nodes from the DAG, or an empty list."""
        ops = getattr(dag, "operations", None)
        if ops is not None and len(ops) > 0:
            return ops
        return []

    def _get_adj_matrix(self, dag):
        """Return the adjacency matrix as a float32 ndarray, or a zero matrix if unavailable."""
        m = getattr(dag, "matrix", None)
        if m is not None:
            return np.asarray(m, dtype=np.float32)
        return np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)

    def _get_op_name(self, node):
        """Return the class name of the node's operation, or an empty string."""
        op = getattr(node, "operation", None)
        if op is not None:
            return op.__class__.__name__
        return ""

    def _count_active_nodes(self, adj: np.ndarray) -> int:
        """Count nodes reachable from the output node via the adjacency matrix."""
        n = adj.shape[0]
        active = np.zeros(n, dtype=bool)
        if n > 0:
            active[-1] = True
        for _ in range(n):
            for j in range(n):
                if active[j]:
                    for i in range(n):
                        if adj[i, j] > 0:
                            active[i] = True
        return int(np.sum(active))


# ══════════════════════════════════════════════════════════════════════════════
#  LinGAM SURROGATE  for noisy-loss smoothing
# ══════════════════════════════════════════════════════════════════════════════

class LinGAMSurrogate:
    """LinearGAM-based surrogate that blends observed losses with predictions
    to smooth out noise during search.

    Parameters
    ----------
    max_nodes : int
        Maximum nodes in the DAG (passed to the feature extractor).
    min_samples : int
        Minimum number of evaluations before the surrogate is fitted and used.
    refit_every : int
        Refit the model every ``refit_every`` evaluations.
    alpha : float
        Blending weight: ``blended = alpha * true_loss + (1-alpha) * pred``.
    verbose : bool
        If True, print fit diagnostics.
    """

    def __init__(
        self,
        max_nodes: int = 15,
        min_samples: int = 10,
        refit_every: int = 5,
        alpha: float = 0.7,
        verbose: bool = False,
    ):
        self.extractor = DAGFeatureExtractor(max_nodes=max_nodes)
        self.min_samples = min_samples
        self.refit_every = refit_every
        self.alpha = alpha
        self.verbose = verbose
        self.X_train: list[np.ndarray] = []
        self.y_train: list[float] = []
        self.model = None
        self.n_evaluations = 0

    def update(self, dag, true_loss: float) -> float:
        """Register a new (dag, true_loss) observation and return the
        blended loss that the search algorithm should use."""
        features = self.extractor.extract(dag)
        self.X_train.append(features.flatten().copy())
        self.y_train.append(true_loss)
        self.n_evaluations += 1

        blended = self.blend(true_loss, dag)

        if (
            self.n_evaluations >= self.min_samples
            and self.n_evaluations % self.refit_every == 0
        ):
            self._refit()

        return blended

    def predict(self, dag) -> float | None:
        """Return surrogate-predicted loss for *dag*, or None if not ready."""
        if self.model is None:
            return None
        features = self.extractor.extract(dag)
        try:
            return float(self.model.predict(features)[0])
        except Exception:
            return None

    def blend(self, true_loss: float, dag) -> float:
        """Blend true loss with surrogate prediction."""
        pred = self.predict(dag)
        if pred is None:
            return true_loss
        return self.alpha * true_loss + (1.0 - self.alpha) * pred

    def _refit(self):
        """Re-fit the LinearGAM surrogate on all accumulated (X_train, y_train) data."""
        from pygam import LinearGAM

        X = np.array(self.X_train)
        y = np.array(self.y_train)
        try:
            self.model = LinearGAM().fit(X, y)
            if self.verbose:
                print(f"  [surrogate] refit on {len(self.y_train)} samples")
        except Exception as e:
            if self.verbose:
                print(f"  [surrogate] refit failed: {e}")
