import torch
import torch.nn as nn

from dragon.search_space.cells import Brick

class Dropout(Brick):
    def __init__(self, input_shape, rate):
        super(Dropout, self).__init__(input_shape)
        self.dropout = torch.nn.Dropout(p=rate)

    def forward(self, X):
        X = self.dropout(X)
        return X
    
    def modify_operation(self, input_shape):
        pass
