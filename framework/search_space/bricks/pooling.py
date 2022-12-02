import torch.nn as nn

from utils.tools import logger


class AVGPooling2D(nn.Module):
    def __init__(self, pool_size):
        super(AVGPooling2D, self).__init__()
        self.pool_size = pool_size
        self.pooling = nn.AvgPool2d((1, pool_size))

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.pooling(X)
        return X


class AVGPooling1D(nn.Module):
    def __init__(self, pool_size):
        super(AVGPooling1D, self).__init__()
        self.pool_size = pool_size
        self.pooling = nn.AvgPool1d(pool_size)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(1)
        X = self.pooling(X)
        if len(init_shape) < 3:
            X = X.squeeze(1)
        return X


class MaxPooling2D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPooling2D, self).__init__()
        self.pooling = nn.MaxPool2d((1, pool_size))

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.pooling(X)
        return X


class MaxPooling1D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPooling1D, self).__init__()
        self.pooling = nn.MaxPool1d(pool_size)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(1)
        X = self.pooling(X)
        if len(init_shape) < 3:
            X = X.squeeze(1)
        return X
