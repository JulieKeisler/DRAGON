import torch
import torch.nn as nn
import numpy as np
from dragon.search_space.cells import Brick
from dragon.utils.tools import logger
from dragon.utils.exceptions import InvalidArgumentError



class BatchNorm1d(Brick):
    def __init__(self, input_shape, **args):
        super().__init__(input_shape)
        if len(input_shape) ==1:
            d_in = input_shape[0]
        else:
            d_in, l = input_shape
        self.d_in = d_in
        self.args = args
        self.norm = nn.BatchNorm1d(d_in)

    def forward(self, X):
        if X.shape[0] == 1:
            pass
        else:
            X = self.norm(X)
        return X
    
    def modify_operation(self, input_shape):
        d_in = input_shape[0]
        diff = d_in - self.d_in

        sign = diff / np.abs(diff) if diff !=0 else 1

        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        new_weight = nn.functional.pad(self.norm.weight, pad)
        new_bias = nn.functional.pad(self.norm.bias, pad)
        self.norm = nn.BatchNorm1d(d_in)
        self.norm.weight.data = new_weight
        self.norm.bias.data = new_bias
        self.d_in = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['norm.weight'].shape[0]
        self.modify_operation((input_shape,))
        super(BatchNorm1d, self).load_state_dict(state_dict, **kwargs)

class BatchNorm2d(Brick):
    def __init__(self, input_shape, **args):
        super().__init__(input_shape)
        d_in, h, w = input_shape
        self.d_in = d_in
        self.args = args
        self.norm = nn.BatchNorm2d(d_in)

    def forward(self, X):
        if X.shape[0] == 1:
            pass
        else:
            X = self.norm(X)
        return X
    
    def modify_operation(self, input_shape):
        d_in = input_shape[0]
        diff = d_in - self.d_in

        sign = diff / np.abs(diff) if diff !=0 else 1

        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        new_weight = nn.functional.pad(self.norm.weight, pad)
        new_bias = nn.functional.pad(self.norm.bias, pad)
        self.norm = nn.BatchNorm2d(d_in)
        self.norm.weight.data = new_weight
        self.norm.bias.data = new_bias
        self.d_in = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['norm.weight'].shape[0]
        self.modify_operation((input_shape,))
        super(BatchNorm2d, self).load_state_dict(state_dict, **kwargs)

class BatchNorm3d(Brick):
    def __init__(self, input_shape, **args):
        super().__init__(input_shape)
        d_in, h, w, _ = input_shape
        self.d_in = d_in
        self.args = args
        self.norm = nn.BatchNorm3d(d_in)

    def forward(self, X):
        if X.shape[0] == 1:
            pass
        else:
            X = self.norm(X)
        return X
    
    def modify_operation(self, input_shape):
        d_in = input_shape[0]
        diff = d_in - self.d_in

        sign = diff / np.abs(diff) if diff !=0 else 1

        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        new_weight = nn.functional.pad(self.norm.weight, pad)
        new_bias = nn.functional.pad(self.norm.bias, pad)
        self.norm = nn.BatchNorm3d(d_in)
        self.norm.weight.data = new_weight
        self.norm.bias.data = new_bias
        self.d_in = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['norm.weight'].shape[0]
        self.modify_operation((input_shape,))
        super(BatchNorm3d, self).load_state_dict(state_dict, **kwargs)

class LayerNorm3d(Brick):
    def __init__(self, input_shape, **args):
        super().__init__(input_shape)
        self.input_shape = input_shape
        self.norm = nn.LayerNorm(input_shape)

    def forward(self, X):
        X = self.norm(X)
        return X
    
    def modify_operation(self, input_shape):
        diff_0 = input_shape[0] - self.input_shape[0]
        diff_1 = input_shape[1] - self.input_shape[1]
        diff_2 = input_shape[2] - self.input_shape[2]
        diff_3 = input_shape[3] - self.input_shape[3]

        sign_0 = diff_0 / np.abs(diff_0) if diff_0 !=0 else 1
        sign_1 = diff_1 / np.abs(diff_1) if diff_1 !=0 else 1
        sign_2 = diff_2 / np.abs(diff_2) if diff_2 !=0 else 1
        sign_3 = diff_3 / np.abs(diff_3) if diff_3 !=0 else 1


        pad = (int(sign_3 * np.ceil(np.abs(diff_3)/2)), int(sign_3 * np.floor(np.abs(diff_3))/2),
               int(sign_2 * np.ceil(np.abs(diff_2)/2)), int(sign_2 * np.floor(np.abs(diff_2))/2),
               int(sign_1 * np.ceil(np.abs(diff_1)/2)), int(sign_1 * np.floor(np.abs(diff_1))/2),
               int(sign_0 * np.ceil(np.abs(diff_0)/2)), int(sign_0 * np.floor(np.abs(diff_0))/2))
        self.norm.weight.data = nn.functional.pad(self.norm.weight, pad)
        self.norm.bias.data = nn.functional.pad(self.norm.bias, pad)
        self.input_shape = input_shape
        self.norm.normalized_shape = self.input_shape
    
    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['norm.weight'].shape
        self.modify_operation(input_shape)
        super(LayerNorm3d, self).load_state_dict(state_dict, **kwargs)

class LayerNorm2d(Brick):
    def __init__(self, input_shape, **args):
        super().__init__(input_shape)
        self.input_shape = input_shape
        self.norm = nn.LayerNorm(input_shape)

    def forward(self, X):
        X = self.norm(X)
        return X
    
    def modify_operation(self, input_shape):
        diff_0 = input_shape[0] - self.input_shape[0]
        diff_1 = input_shape[1] - self.input_shape[1]
        diff_2 = input_shape[2] - self.input_shape[2]

        sign_0 = diff_0 / np.abs(diff_0) if diff_0 !=0 else 1
        sign_1 = diff_1 / np.abs(diff_1) if diff_1 !=0 else 1
        sign_2 = diff_2 / np.abs(diff_2) if diff_2 !=0 else 1

        pad = (int(sign_2 * np.ceil(np.abs(diff_2)/2)), int(sign_2 * np.floor(np.abs(diff_2))/2),
               int(sign_1 * np.ceil(np.abs(diff_1)/2)), int(sign_1 * np.floor(np.abs(diff_1))/2),
               int(sign_0 * np.ceil(np.abs(diff_0)/2)), int(sign_0 * np.floor(np.abs(diff_0))/2))
        self.norm.weight.data = nn.functional.pad(self.norm.weight, pad)
        self.norm.bias.data = nn.functional.pad(self.norm.bias, pad)
        self.input_shape = input_shape
        self.norm.normalized_shape = self.input_shape
    
    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['norm.weight'].shape
        self.modify_operation(input_shape)
        super(LayerNorm2d, self).load_state_dict(state_dict, **kwargs)



class LayerNorm1d(Brick):
    def __init__(self, input_shape, **args):
        super().__init__(input_shape)
        self.d_in = input_shape[-1]
        self.norm = nn.LayerNorm((input_shape[-1]))

    def forward(self, X):
        X = self.norm(X)
        return X
    
    def modify_operation(self, input_shape):
        d_in = input_shape[-1]
        diff = d_in - self.d_in

        sign = diff / np.abs(diff) if diff !=0 else 1

        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.norm.weight.data = nn.functional.pad(self.norm.weight, pad)
        self.norm.bias.data = nn.functional.pad(self.norm.bias, pad)
        self.norm.normalized_shape = (d_in,)
        self.d_in = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['norm.weight'].shape[0]
        self.modify_operation((input_shape,))
        super(LayerNorm1d, self).load_state_dict(state_dict, **kwargs)
