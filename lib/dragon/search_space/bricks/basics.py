import numpy as np
import torch
import torch.nn as nn
from dragon.search_space.cells import Brick
from dragon.utils.tools import logger


class Identity(Brick):
    def __init__(self, input_shape=None, **args):
        super(Identity, self).__init__(input_shape)

    def forward(self, X):
        return X
    def modify_operation(self, input_shape):
        pass


class MLP(Brick):
    def __init__(self, input_shape, out_channels):
        super(MLP, self).__init__(input_shape)
        self.in_channels = input_shape[-1]
        self.out_channels = out_channels
        self.linear = nn.Linear(self.in_channels, out_channels)

    def forward(self, X):
        try:
            X = self.linear(X)
        except Exception as e:
            logger.error(f'linear: {self.linear.weight.get_device()},\nx = {X.get_device()}')
            raise e
        return X
    
    def modify_operation(self, input_shape, hp=None):
        if hp is not None:
            d_out = hp["out_channels"]
        else:
            d_out = self.out_channels
        d_in = input_shape[-1]
        diff = d_in - self.in_channels
        diff_out = d_out - self.out_channels

        sign = diff / np.abs(diff) if diff !=0 else 1
        sign_out = diff_out / np.abs(diff_out) if diff_out !=0 else 1

        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2), 
               int(sign_out * np.ceil(np.abs(diff_out)/2)), int(sign_out * np.floor(np.abs(diff_out))/2))
        pad_bias = (int(sign_out * np.ceil(np.abs(diff_out)/2)), int(sign_out * np.floor(np.abs(diff_out))/2))
        self.in_channels = d_in
        self.out_channels = d_out
        self.linear.weight.data = nn.functional.pad(self.linear.weight, pad)
        self.linear.bias.data = nn.functional.pad(self.linear.bias, pad_bias)

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['linear.weight'].shape[1]
        output_shape = state_dict['linear.weight'].shape[0]
        self.modify_operation((input_shape,), hp={'out_channels': output_shape})
        super(MLP, self).load_state_dict(state_dict, **kwargs)

class ChannelsMLP(Brick):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.in_channels = input_shape[-1]
        self.out_channels = input_shape[-2]
        self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels))
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, X):
        X = torch.einsum('btcd,cd->btc', X, self.weight) + self.bias
        return X
    
    def modify_operation(self, input_shape):
        d_in = input_shape[-1]
        d_out = input_shape[-2]
        diff = d_in - self.in_channels
        diff_out = d_out - self.out_channels
        sign = diff / np.abs(diff) if diff !=0 else 1
        sign_out = diff_out / np.abs(diff_out) if diff_out !=0 else 1

        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2),
               int(sign_out * np.ceil(np.abs(diff_out)/2)), int(sign_out * np.floor(np.abs(diff_out))/2))
        self.weight.data = nn.functional.pad(self.weight, pad)
        pad = (int(sign_out * np.ceil(np.abs(diff_out)/2)), int(sign_out * np.floor(np.abs(diff_out))/2))
        self.bias.data = nn.functional.pad(self.bias.data, pad)
        self.in_channels = d_in
        self.out_channels = d_out

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['weight'].shape[1]
        self.modify_operation((input_shape,))
        super(ChannelsMLP, self).load_state_dict(state_dict, **kwargs)
