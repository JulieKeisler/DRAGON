# preprocessing.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from itertools import combinations

from Config import Experiment, Loss
from dataprocessing.Features import VarAugmentor, XGBoostSelector
from dataprocessing.Data import DatasetLoader

_dataset_loader = DatasetLoader()


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

class NoiseInjector:
    """Adds Gaussian noise to y (relative to std(y))."""

    def __init__(self, noise_std: float = Experiment.NOISE_STD, seed: int = Experiment.RANDOM_SEED):
        self.noise_std = noise_std
        self.seed      = seed

    def transform(self, y: pd.Series, run_id: int = 0) -> pd.Series:
        rng   = np.random.default_rng(self.seed + run_id + 9999)
        noise = rng.normal(0, self.noise_std * float(y.std()), len(y))
        return y + noise


# ══════════════════════════════════════════════════════════════════════════════
#  GPR DENOISER  (self-validating Matérn smoother)
# ══════════════════════════════════════════════════════════════════════════════

class GPRDenoiser:
    """Self-validating GPR-Matérn denoiser.

    Fits a GP with (Constant × Matérn) + WhiteKernel on a subsample,
    estimates the noise floor σ̂² by marginal likelihood, then applies
    a calibration guard: if held-out RMSE / σ̂ >= 1.15 the smoother is
    oversmoothing real structure and denoising is skipped.
    """

    def __init__(
        self,
        max_samples: int = 600,
        nu:          float = 2.5,
        seed:        int = Experiment.RANDOM_SEED,
    ):
        self.max_samples = max_samples
        self.nu          = nu
        self.seed        = seed

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, dict]:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        from sklearn.model_selection import KFold

        std_y = float(y.std())
        if len(y) < 8 or std_y < 1e-12:
            return y.copy(), {"applied": False, "reason": "too few samples / constant y"}

        Xs = self._scale(X.values.astype(float))
        yv = y.values.astype(float)

        # ── K-fold calibration guard ──────────────────────────────────────────
        fold_err = []
        for tr, te in KFold(n_splits=4, shuffle=True, random_state=self.seed).split(Xs):
            try:
                yh = self._fit_predict(Xs[tr], yv[tr], Xs[te])
                fold_err.append(np.mean((yh - yv[te]) ** 2))
            except Exception:
                fold_err.append(np.var(yv))
        rmse_holdout = float(np.sqrt(np.mean(fold_err)))

        # ── Full fit ──────────────────────────────────────────────────────────
        gp = self._make_gp()
        try:
            if len(y) > self.max_samples:
                sub = np.random.default_rng(self.seed).choice(len(y), self.max_samples, replace=False)
                gp.fit(Xs[sub], yv[sub])
            else:
                gp.fit(Xs, yv)
            y_hat        = gp.predict(Xs)
            noise_level  = float(gp.kernel_.k2.noise_level)
        except Exception as e:
            return y.copy(), {"applied": False, "reason": f"gpr failed: {e}"}

        sigma_hat = np.sqrt(max(noise_level, 0.0)) * std_y
        ratio     = rmse_holdout / max(sigma_hat, 1e-12)
        noise_var = float(sigma_hat ** 2)
        snr_db    = 10.0 * np.log10(max(np.var(yv) - noise_var, 1e-30) / max(noise_var, 1e-30))

        applied = bool(ratio < 1.15)
        info = {
            "applied":      applied,
            "ratio":        float(ratio),
            "noise_var":    noise_var,
            "noise_sigma":  float(sigma_hat),
            "snr_db":       float(snr_db),
        }
        if applied:
            return pd.Series(y_hat, index=y.index, name=y.name), info
        return y.copy(), info

    # ── Helpers ───────────────────────────────────────────────────────────────

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

class PreprocessingPipeline:
    """Orchestrates: noise injection → GPR denoise → var augmentation → XGBoost selection.

    Each step is optional and controlled by the method config. The pipeline returns
    a search-ready (X, y) pair plus preprocessing metadata.
    """

    def __init__(
        self,
        add_noise:          bool = False,
        pre_denoise:        bool = False,
        var_aug:            bool = True,
        use_lingam_denoiser: bool = False,
        is_synth_target:    bool = False,
        run_id:             int  = 0,
    ):
        self.add_noise       = add_noise
        self.pre_denoise     = pre_denoise
        self.var_aug         = var_aug
        self.use_lingam_denoiser = use_lingam_denoiser
        self.is_synth_target = is_synth_target
        self.run_id          = run_id

        self._noise_injector = NoiseInjector()
        self._gpr_denoiser   = GPRDenoiser()
        self._lingam_denoiser = LinGAMDenoiser()
        self._var_augmentor  = VarAugmentor()
        self._selector       = XGBoostSelector(n_top=Experiment.N_TOP_FEATURES)

    def transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
        """Returns (X_sel, y_clean, info)."""
        info = {}

        if self.add_noise:
            y = self._noise_injector.transform(y, run_id=self.run_id)
            info["noise_injected"] = True

        if self.pre_denoise:
            y, dn_info = self._gpr_denoiser.transform(X, y)
            info["gpr_denoise"] = dn_info

        if self.use_lingam_denoiser:
                y, dn_info = self._lingam_denoiser.transform(X, y)
                info["lingam_denoise"] = dn_info
        
        if self.var_aug:
            X = self._var_augmentor.transform(X, y)
            info["var_aug_features"] = X.shape[1]

        feature_names, feat_scores = self._selector.transform(X, y)
        X = X[feature_names]
        info["feature_names"] = feature_names
        info["feat_scores"] = feat_scores
        info["n_selected_features"] = len(feature_names)

        return X, y, info

# ── Helper: data preparation ──────────────────────────────────────────────────

def _prepare_data(method_cfg, target, run_id, *,
                  _y_predenoised,
                  _X_preprocessed=None, _y_preprocessed=None,
                  _feature_names_preloaded=None, _feat_scores_preloaded=None,
                  _X_preloaded=None, _y_preloaded=None):
    method_id = method_cfg["id"]
    seed = Experiment.RANDOM_SEED + run_id
    np.random.seed(seed); torch.manual_seed(seed)

    if (_X_preprocessed is not None and _y_preprocessed is not None
            and _feature_names_preloaded is not None
            and _feat_scores_preloaded is not None):
        return (_X_preprocessed,
                _y_preprocessed.copy(),
                _feature_names_preloaded,
                _feat_scores_preloaded,
                seed)

    if _X_preloaded is not None and _y_preloaded is not None:
        X_df = _X_preloaded; y = _y_preloaded.copy()
    else:
        X_df, y = _dataset_loader.load(target, run_id=run_id)

    if _y_predenoised is not None:
        y = _y_predenoised

    pre_dm = method_cfg.get("pre_denoise_method")
    pre_denoise = bool(pre_dm and _y_predenoised is None)
    pipeline = PreprocessingPipeline(
        add_noise=method_cfg.get("add_noise", False),
        pre_denoise=pre_denoise,
        var_aug=method_cfg.get("var_aug", True),
        use_lingam_denoiser=method_cfg.get("use_lingam_denoiser", False),
        run_id=run_id,
    )
    X_sel, y, info = pipeline.transform(X_df, y)
    if pre_denoise and "gpr_denoise" in info:
        print(f"[{method_id}/{target}/run{run_id}] pre_denoise({pre_dm}): {info['gpr_denoise']}")
    if "lingam_denoise" in info:
            print(f"[{method_id}/{target}/run{run_id}] pre_denoise({pre_dm}): {info['lingam_denoise']}")
    if method_cfg.get("denoise_method") == "stoch_sub":
        pass  # handled in _setup_searcher

    feature_names = info["feature_names"]
    feat_scores = info["feat_scores"]
    return X_sel, y, feature_names, feat_scores, seed
