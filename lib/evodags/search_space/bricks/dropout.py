import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.dropout = torch.nn.Dropout(p=rate)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = self.dropout(X)
        return X
