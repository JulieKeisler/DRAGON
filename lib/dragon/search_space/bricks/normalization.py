import torch.nn as nn

class BatchNorm1d(nn.Module):
    def __init__(self, d_in):
        super(BatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(d_in)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.norm(X)
        return X
    
class BatchNorm2d(nn.Module):
    def __init__(self, d_in):
        super(BatchNorm2d, self).__init__()
        self.d_in = d_in
        self.norm = nn.BatchNorm2d(d_in)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = X.permute(0, 3, 1, 2)
        X = self.norm(X)
        X = X.permute(0, 2, 3, 1)
        return X


class LayerNorm2d(nn.Module):
    def __init__(self, F, T, d_in):
        super().__init__()
        self.d_in = d_in
        self.F = F
        self.T = T
        self.norm = nn.LayerNorm((F, T, d_in))

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.norm(X)
        return X

class LayerNorm1d(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.d_in = d_in
        self.norm = nn.LayerNorm((d_in))

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.norm(X)
        return X