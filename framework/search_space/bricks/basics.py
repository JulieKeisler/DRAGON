import torch
import torch.nn as nn


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        return torch.zeros(*X.shape)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, X):
        if isinstance(X, list):
            return sum(X)
        else:
            return X


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.linear(X)
        return X