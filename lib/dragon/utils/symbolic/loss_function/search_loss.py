"""Unified loss registry for DRAGON symbolic regression.

Provides ``SearchLoss`` for search-time scoring (correlation, MSE, MAE, Huber)
and ``AlignmentLoss`` for per-channel alignment scoring used in the OLS pipeline.
"""

from __future__ import annotations

import numpy as np
import torch


_DEFAULT_HUBER_DELTA_FRAC = 0.20


class SearchLoss:
    """Unified loss registry for search and alignment.

    Parameters
    ----------
    huber_delta_frac : float
        Fraction of ``sqrt(var_y)`` used as the Huber delta threshold.
        Default ``0.20``.
    """

    _LOSS_FNS = {}

    def __init__(self, huber_delta_frac: float = _DEFAULT_HUBER_DELTA_FRAC):
        self.huber_delta_frac = huber_delta_frac

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_numpy(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr).ravel()

    @staticmethod
    def correl(pred, true) -> float:
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        if isinstance(true, torch.Tensor):
            true = true.numpy()
        pred = pred.ravel()
        true = true.ravel()
        pd_ = pred - pred.mean()
        td_ = true - true.mean()
        sp = np.sqrt(np.sum(pd_ ** 2))
        st = np.sqrt(np.sum(td_ ** 2))
        if sp < 1e-12 or st < 1e-12:
            return 0.0
        return float(np.sum(pd_ * td_) / (sp * st + 1e-8))

    # ── search losses ──────────────────────────────────────────────────────

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
        delta = _DEFAULT_HUBER_DELTA_FRAC * float(np.sqrt(var_y))
        r = np.abs(y_pred - y_true)
        loss = np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))
        return float(np.mean(loss) / var_y)

    # ── alignment losses ───────────────────────────────────────────────────

    @classmethod
    def mse_align(cls, pred, y, var_y, _huber_delta_frac=None):
        if var_y < 1e-30:
            return 1.0
        val = float(np.mean((y - pred) ** 2) / var_y)
        return max(0.0, val)

    @classmethod
    def huber_align(cls, pred, y, var_y, huber_delta_frac=None):
        if var_y < 1e-30:
            return 1.0
        delta = (huber_delta_frac or _DEFAULT_HUBER_DELTA_FRAC) * float(np.sqrt(var_y))
        r = np.abs(y - pred)
        quad = np.minimum(r, delta)
        val = float(np.mean(0.5 * quad ** 2 + delta * (r - quad)) / (0.5 * var_y))
        return max(0.0, val)

    @classmethod
    def channel_score(cls, search_loss, loss_kind, pred, y, var_y, huber_delta_frac):
        """Score a single channel prediction.

        When ``search_loss == "corr"``, uses the alignment variant.
        Otherwise delegates to the standard search loss.
        """
        if search_loss == "corr":
            if loss_kind == "huber":
                return cls.huber_align(pred, y, var_y, huber_delta_frac)
            return cls.mse_align(pred, y, var_y)
        return cls.score(search_loss, pred, y)


SearchLoss._LOSS_FNS.update({
    "corr": SearchLoss.corr,
    "mse": SearchLoss.mse,
    "raw_mse": SearchLoss.raw_mse,
    "mae": SearchLoss.mae,
    "huber": SearchLoss.huber,
})


class AlignmentLoss:
    """Backward-compatible wrapper around ``SearchLoss`` alignment methods.

    Deprecated: use ``SearchLoss.mse_align`` / ``SearchLoss.huber_align`` directly.
    """

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
        return SearchLoss.channel_score(search_loss, loss_kind, pred, y, var_y, huber_delta_frac)

    @staticmethod
    def mse(pred, y, var_y, _huber_delta_frac):
        return SearchLoss.mse_align(pred, y, var_y)

    @staticmethod
    def huber(pred, y, var_y, huber_delta_frac):
        return SearchLoss.huber_align(pred, y, var_y, huber_delta_frac)


AlignmentLoss._LOSS_FNS.update({
    "mse": AlignmentLoss.mse,
    "huber": AlignmentLoss.huber,
})
