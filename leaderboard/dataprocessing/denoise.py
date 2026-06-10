from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Config import Experiment as _CfgExp




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