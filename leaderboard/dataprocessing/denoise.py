# denoise.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Config import Experiment as _CfgExp


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO DENOISER  (GPR-Matérn self-validating denoiser)
# ══════════════════════════════════════════════════════════════════════════════

class AutoDenoiser:
    """Self-validating GPR-Matérn denoiser.

    Parameters
    ----------
    max_samples : maximum number of samples used for GPR fitting
    nu          : Matérn kernel smoothness parameter
    """

    def __init__(self, max_samples: int = 600, nu: float = 2.5):
        self.max_samples = max_samples
        self.nu          = nu

    def denoise(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Return (y_denoised, info_dict)."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.model_selection import KFold

        max_samples = self.max_samples
        nu          = self.nu
        n     = len(y)
        std_y = float(y.std())
        if n < 8 or std_y < 1e-12:
            return y.copy(), {"method": "auto", "applied": False,
                              "reason": "too few samples / constant y", "noise_var": 0.0}

        Xs = _SS().fit_transform(X.values.astype(float))
        yv = y.values.astype(float)

        def _make_gp():
            kern = (ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=nu)
                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e2)))
            return GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=2,
                                            normalize_y=True,
                                            random_state=_CfgExp.RANDOM_SEED)

        def _fit_predict(Xtr, ytr, Xte):
            gp = _make_gp()
            if len(ytr) > max_samples:
                sub = np.random.default_rng(_CfgExp.RANDOM_SEED).choice(
                    len(ytr), max_samples, replace=False)
                gp.fit(Xtr[sub], ytr[sub])
            else:
                gp.fit(Xtr, ytr)
            return gp.predict(Xte)

        kf = KFold(n_splits=4, shuffle=True, random_state=_CfgExp.RANDOM_SEED)
        fold_err = []
        for tr, te in kf.split(Xs):
            try:
                yh_te = _fit_predict(Xs[tr], yv[tr], Xs[te])
                fold_err.append(np.mean((yh_te - yv[te]) ** 2))
            except Exception:
                fold_err.append(np.var(yv))
        rmse_holdout = float(np.sqrt(np.mean(fold_err)))

        gp_full = _make_gp()
        try:
            if n > max_samples:
                sub = np.random.default_rng(_CfgExp.RANDOM_SEED).choice(n, max_samples, replace=False)
                gp_full.fit(Xs[sub], yv[sub])
            else:
                gp_full.fit(Xs, yv)
            y_hat       = gp_full.predict(Xs)
            noise_level = float(gp_full.kernel_.k2.noise_level)
        except Exception as e:
            return y.copy(), {"method": "auto", "applied": False,
                              "reason": f"gpr failed ({e})", "noise_var": 0.0}

        sigma_hat = np.sqrt(max(noise_level, 0.0)) * std_y
        ratio     = rmse_holdout / max(sigma_hat, 1e-12)
        noise_var = float(sigma_hat ** 2)
        var_y     = float(np.var(yv))
        snr_db    = 10.0 * np.log10(max(var_y - noise_var, 1e-30) / max(noise_var, 1e-30))

        applied = bool(ratio < 1.15)
        info = {"method": "auto", "applied": applied, "ratio": float(ratio),
                "noise_var": noise_var, "noise_sigma": float(sigma_hat),
                "snr_db": float(snr_db), "nu": nu}
        if applied:
            return pd.Series(y_hat, index=y.index, name=y.name), info
        return y.copy(), info


# ══════════════════════════════════════════════════════════════════════════════
#  MC-DROPOUT WEIGHTER
# ══════════════════════════════════════════════════════════════════════════════

class MCDropoutWeighter:
    """Estimate per-sample confidence weights via MC-Dropout.

    Parameters
    ----------
    n_forward   : number of stochastic forward passes
    dropout_p   : dropout probability during training and inference
    n_epochs    : training epochs
    hidden      : hidden layer width
    random_seed : RNG seed (defaults to Experiment.RANDOM_SEED)
    """

    def __init__(
        self,
        n_forward:   int   = 50,
        dropout_p:   float = 0.15,
        n_epochs:    int   = 300,
        hidden:      int   = 64,
        random_seed: int   = None,
    ):
        self.n_forward   = n_forward
        self.dropout_p   = dropout_p
        self.n_epochs    = n_epochs
        self.hidden      = hidden
        self.random_seed = random_seed if random_seed is not None else _CfgExp.RANDOM_SEED

    def compute_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Return per-sample confidence weights (shape: (n,), dtype: float32)."""
        n, d = len(y), X.shape[1]
        if n < 8:
            return np.ones(n, dtype=np.float32)

        Xv    = X.values.astype(np.float32)
        yv    = y.values.astype(np.float32).ravel()
        xstd  = Xv.std(0) + 1e-8; xmean = Xv.mean(0)
        ystd  = float(yv.std()) + 1e-8; ymean = float(yv.mean())
        Xs = (Xv - xmean) / xstd; ys = (yv - ymean) / ystd

        Xt = torch.tensor(Xs); yt = torch.tensor(ys).unsqueeze(1)
        torch.manual_seed(self.random_seed)
        h   = self.hidden
        net = nn.Sequential(
            nn.Linear(d, h),      nn.ReLU(), nn.Dropout(p=self.dropout_p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p=self.dropout_p),
            nn.Linear(h // 2, 1),
        )
        opt   = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
        mse_f = nn.MSELoss()
        net.train()
        for _ in range(self.n_epochs):
            opt.zero_grad(); mse_f(net(Xt), yt).backward(); opt.step()

        net.train()
        with torch.no_grad():
            preds = np.stack([net(Xt).squeeze(1).numpy() for _ in range(self.n_forward)])

        variances  = preds.var(axis=0)
        mean_var   = float(variances.mean()) + 1e-30
        confidence = 1.0 / (1.0 + variances / mean_var)
        weights    = confidence * n / confidence.sum()
        return weights.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def apply_denoise(X: pd.DataFrame, y: pd.Series, method: str) -> tuple:
    """Dispatch to the requested denoising strategy.

    Parameters
    ----------
    method : 'auto' (GPR-Matérn self-validating) or 'stoch_sub' (in-loop subsampling, no-op here)
    """
    if method == "auto":
        return AutoDenoiser().denoise(X, y)
    elif method == "stoch_sub":
        return y.copy(), {"method": "stoch_sub", "note": "in-loop subsampling"}
    else:
        raise ValueError(f"Unknown denoise_method: {method!r}. "
                         f"Supported values: 'auto', 'stoch_sub'.")
