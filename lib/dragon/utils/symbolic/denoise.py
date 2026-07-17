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

    def __init__(self, max_samples: int = 600, nu: float = 2.5, seed: int = 42):
        self.max_samples = max_samples
        self.nu = nu
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

        applied = bool(ratio < 1.15)
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
        gp = self._make_gp()
        if len(ytr) > self.max_samples:
            sub = np.random.default_rng(self.seed).choice(len(ytr), self.max_samples, replace=False)
            gp.fit(Xtr[sub], ytr[sub])
        else:
            gp.fit(Xtr, ytr)
        return gp.predict(Xte)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  LinGAM DENOISER
# ══════════════════════════════════════════════════════════════════════════════

class LinGAMDenoiser:
    """LinearGAM denoiser: fit y~X then predict on X."""

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
            preds = np.stack([net(Xt).squeeze(1).numpy() for _ in range(self.n_forward)])

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
