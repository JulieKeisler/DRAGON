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

    def __init__(self, noise_std: float | None = None, seed: int = Experiment.RANDOM_SEED,
                 noise_mode: str | None = None):
        self.noise_std = Experiment.NOISE_STD if noise_std is None else noise_std
        self.seed      = seed
        self.noise_mode = (noise_mode
                           if noise_mode is not None
                           else getattr(Experiment, "NOISE_MODE", "absolute")).lower()

    def transform(self, y: pd.Series, run_id: int = 0) -> pd.Series:
        rng = np.random.default_rng(self.seed + run_id + 9999)
        if self.noise_mode == "relative":
            # y * (1 + N(0, eps)): each point is perturbed in proportion to its OWN
            # magnitude, so the local signal-to-noise ratio is 1/eps everywhere.
            # The default "absolute" mode scales by std(y) of the whole dataset, which
            # on wide-dynamic-range targets (newton spans 4e11) buries >99% of points
            # at eps=0.01 and makes eps incomparable across benchmarks.
            return y * (1.0 + rng.normal(0, self.noise_std, len(y)))
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
        # Guards fired by VarAugmentor on the features that were actually
        # SELECTED.  These change the function the feature name denotes
        # ("1/(x)" is 0 where |x| <= 1e-12; a product is clipped at the float32
        # ceiling), so they are reported rather than left silent.
        _gr = getattr(self._var_augmentor, "guard_report", None) or {}
        _sel_guards = {n: _gr[n] for n in feature_names if n in _gr}
        if _sel_guards:
            info["var_aug_guards"] = _sel_guards
            print(f"[var_aug] guards fired on selected features: {_sel_guards}")

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
        # Carry the VarAugmentor guard firings to the result writer.  method_cfg
        # reaches _build_result (and is copied into each SPAR sub_cfg), so the
        # counts land in results.json instead of only in the log.
        try:
            method_cfg["_var_aug_guards"] = info.get("var_aug_guards", {}) or {}
        except Exception:
            pass
        return X_sel, y, feature_names, feat_scores