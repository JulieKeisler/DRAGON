import torch.nn as nn
import numpy as np
from dragon.utils.tools import logger

class Simple_2DCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Simple_2DCNN, self).__init__()
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        X = X.permute(0, 3, 1, 2)
        init_shape = X.shape
        try:
            X = self.cnn(X)
        except RuntimeError as e:
            logger.error("RUNTIM ERROR: ", X.shape, " --> ", self.cnn)
            raise e
        pad = (int(np.floor((init_shape[3] - X.shape[3]) / 2)), int(np.ceil((init_shape[3] - X.shape[3]) / 2)),
               int(np.floor((init_shape[2] - X.shape[2]) / 2)), int(np.ceil((init_shape[2] - X.shape[2]) / 2)))
        X = nn.functional.pad(X, pad)
        X = X.permute(0, 2, 3, 1)
        return X


class Simple_1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Simple_1DCNN, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(-1)
            X = X.permute(0, 2, 1)
        X = self.cnn(X)
        if len(init_shape) < 3:
            X = X.permute(0, 2, 1)
            X = X.squeeze(-1)
        # logger.info(f'init shape: {init_shape}, cnn: {self.cnn}, output shape: {X.shape}')
        return X
