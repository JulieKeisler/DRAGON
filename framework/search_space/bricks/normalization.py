import torch
import torch.nn as nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = (X - torch.mean(X)) / torch.std(X)
        return X
