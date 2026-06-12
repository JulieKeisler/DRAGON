# preprocessing.py
from __future__ import annotations

import numpy as np
import pandas as pd

from Config import Experiment
from dataprocessing.Denoise import GPRDenoiser, LinGAMDenoiser
from dataprocessing.Features import VarAugmentor, XGBoostSelector
from dataprocessing.Data import DatasetLoader



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


class PreprocessingPipeline:
    """Orchestrates: noise injection → denoise (GPR | LinearGAM) → var augmentation → XGBoost selection.

    Each step is optional and controlled by the method config. The pipeline returns
    a search-ready (X, y) pair plus preprocessing metadata.
    """

    def __init__(
        self,
        add_noise:          bool = False,
        denoiser:           str | None = None,
        var_aug:            bool = True,
        run_id:             int  = 0,
    ):
        self.add_noise       = add_noise
        self.denoiser        = denoiser
        self.var_aug         = var_aug
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

        if self.denoiser == "gpr":
            y, dn_info = self._gpr_denoiser.transform(X, y)
            info["gpr_denoise"] = dn_info

        if self.denoiser == "lingam":
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

    @classmethod
    def prepare_data(cls, method_cfg, target, run_id):
        method_id = method_cfg["id"]
        X_df, y = DatasetLoader().load(target, run_id=run_id)
        denoiser = method_cfg.get("denoiser")

        pipeline = cls(
            add_noise=method_cfg.get("add_noise", False),
            denoiser=denoiser,
            var_aug=method_cfg.get("var_aug", True),
            run_id=run_id,
        )

        X_sel, y, info = pipeline.transform(X_df, y)

        if "gpr_denoise" in info:
            print(f"[{method_id}/{target}/run{run_id}] denoiser({denoiser}): {info['gpr_denoise']}")
        if "lingam_denoise" in info:
            print(f"[{method_id}/{target}/run{run_id}] denoiser({denoiser}): {info['lingam_denoise']}")

        feature_names = info["feature_names"]
        feat_scores = info["feat_scores"]
        return X_sel, y, feature_names, feat_scores