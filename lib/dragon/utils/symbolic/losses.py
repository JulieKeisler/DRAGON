"""Search and alignment losses of DragonSR.

Ported verbatim from the frozen experiment code (``leaderboard/pipeline/
DragonOrchestrator.py``, CODE_VERSION AGGREGATE 2846c144); only the Huber
threshold, which the frozen code read from its global ``Config``, is a
module-level constant with the same value.

``SearchLoss`` scores a candidate's prediction during the search (the paper
uses ``corr`` = 1 - Pearson r^2, the normalised error of the best affine
rescaling).  ``AlignmentLoss`` scores the per-channel fit when the search loss
is the scale-free ``corr``.
"""
from __future__ import annotations

import numpy as np
import torch

# Fraction of std(y) used as the Huber threshold (frozen Config.Loss.HUBER_DELTA_FRAC).
HUBER_DELTA_FRAC = 0.20


class SearchLoss:
    """Registry for search loss functions used by DragonSearcher."""

    _LOSS_FNS = {}

    @staticmethod
    def correl(pred, true) -> float:
        if isinstance(pred, torch.Tensor): pred = pred.numpy()
        if isinstance(true, torch.Tensor): true = true.numpy()
        pred = pred.ravel(); true = true.ravel()
        pd_  = pred - pred.mean(); td_ = true - true.mean()
        sp   = np.sqrt(np.sum(pd_ ** 2)); st = np.sqrt(np.sum(td_ ** 2))
        if sp < 1e-12 or st < 1e-12:
            return 0.0
        return float(np.sum(pd_ * td_) / (sp * st + 1e-8))

    @classmethod
    def available(cls):
        return sorted(cls._LOSS_FNS.keys())

    @classmethod
    def compute(cls, loss_name, y_pred, y_true):
        if loss_name not in cls._LOSS_FNS:
            raise ValueError(
                f"Unknown search loss '{loss_name}'. Available: {cls.available()}"
            )
        return cls._LOSS_FNS[loss_name](cls._ensure_numpy(y_pred), cls._ensure_numpy(y_true))

    @classmethod
    def score(cls, loss_name, y_pred, y_true):
        """Safe scalar score for search losses.

        Returns +inf when the loss cannot be evaluated or is non-finite,
        otherwise clamps the value to be non-negative.
        """
        try:
            val = float(cls.compute(loss_name, y_pred, y_true))
        except Exception:
            return np.inf
        if not np.isfinite(val):
            return np.inf
        return max(0.0, val)

    @staticmethod
    def _ensure_numpy(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr).ravel()

    @classmethod
    def corr(cls, y_pred, y_true):
        r = cls.correl(y_pred, y_true)
        return float(1 - r ** 2) if np.isfinite(r) else 1.0

    @classmethod
    def mse(cls, y_pred, y_true):
        var_y = float(np.var(y_true))
        if var_y < 1e-30:
            return 1.0
        return float(np.mean((y_pred - y_true) ** 2) / var_y)

    @classmethod
    def raw_mse(cls, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    @classmethod
    def mae(cls, y_pred, y_true):
        var_y = float(np.var(y_true))
        if var_y < 1e-30:
            return 1.0
        return float(np.mean(np.abs(y_pred - y_true)) / np.sqrt(var_y))

    @classmethod
    def huber(cls, y_pred, y_true):
        var_y = float(np.var(y_true))
        if var_y < 1e-30:
            return 1.0
        delta = HUBER_DELTA_FRAC * float(np.sqrt(var_y))
        r = np.abs(y_pred - y_true)
        loss = np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))
        return float(np.mean(loss) / var_y)


SearchLoss._LOSS_FNS.update({
    "corr": SearchLoss.corr,
    "mse": SearchLoss.mse,
    "raw_mse": SearchLoss.raw_mse,
    "mae": SearchLoss.mae,
    "huber": SearchLoss.huber,
})


class AlignmentLoss:
    """Loss used to align channel predictions to target when correlation search loss is active."""

    _LOSS_FNS = {}

    @classmethod
    def available(cls):
        return sorted(cls._LOSS_FNS.keys())

    @classmethod
    def compute(cls, loss_name, pred, y, var_y, huber_delta_frac):
        if loss_name not in cls._LOSS_FNS:
            raise ValueError(
                f"Unknown alignment loss '{loss_name}'. Available: {cls.available()}"
            )
        pred_np = np.asarray(pred)
        y_np = np.asarray(y)
        return cls._LOSS_FNS[loss_name](pred_np, y_np, float(var_y), float(huber_delta_frac))

    @classmethod
    def channel_score(cls, search_loss, loss_kind, pred, y, var_y, huber_delta_frac):
        if search_loss == "corr":
            return cls.compute(loss_kind, pred, y, var_y, huber_delta_frac)
        return SearchLoss.score(search_loss, pred, y)

    @staticmethod
    def mse(pred, y, var_y, _huber_delta_frac):
        if var_y < 1e-30:
            return 1.0
        val = float(np.mean((y - pred) ** 2) / var_y)
        return max(0.0, val)

    @staticmethod
    def huber(pred, y, var_y, huber_delta_frac):
        if var_y < 1e-30:
            return 1.0
        delta = huber_delta_frac * float(np.sqrt(var_y))
        r = np.abs(y - pred)
        quad = np.minimum(r, delta)
        val = float(np.mean(0.5 * quad ** 2 + delta * (r - quad)) / (0.5 * var_y))
        return max(0.0, val)


AlignmentLoss._LOSS_FNS.update({
    "mse": AlignmentLoss.mse,
    "huber": AlignmentLoss.huber,
})
