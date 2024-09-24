import numpy as np
import torch.nn as nn
from dragon.search_space import Brick
from dragon.utils.tools import logger

class Simple_1DRNN(Brick):
    def __init__(self, input_shape, num_layers, hidden_size):
        super(Simple_1DRNN, self).__init__(input_shape)
        self.input_shape = input_shape
        self.input_size = input_shape[-1]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_shape[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X, h=None):
        init_shape = X.shape
        try:
            if len(init_shape) < 3:
                X = X.unsqueeze(-1)
                X = X.permute(0, 2, 1)
            if h is None:
                X, h = self.rnn(X)
            else:
                if h.shape[1] >= X.shape[0]:
                    if h.shape[1] > X.shape[0]:
                        h = h[:, :X.shape[0]].contiguous()
                    X,h = self.rnn(X,h)
                else:
                    X, h = self.rnn(X)
            if len(init_shape) < 3:
                X = X.permute(0, 2, 1)
                X = X.squeeze(-1)
            return X, h.detach()
        except Exception as e:
            logger.error(f"{e}", exc_info=True)
            raise e
    
    def modify_operation(self, input_shape):
        T = input_shape[-1]
        diff = T - self.input_size
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.rnn.weight_ih_l0.data = nn.functional.pad(self.rnn.weight_ih_l0, pad)
        self.rnn.input_size = T
        self.input_size = T  
        self.input_shape = input_shape

    def load_state_dict(self, state_dict, **kwargs):
        T = state_dict['rnn.weight_ih_l0'].shape[-1]
        self.modify_operation((T,))
        super(Simple_1DRNN, self).load_state_dict(state_dict, **kwargs)      



class Simple_2DLSTM(Brick):
    def __init__(self, input_shape, hidden_size, num_layers):
        F, T, d_in = input_shape
        super(Simple_2DLSTM, self).__init__(input_shape)
        if d_in == 1:
            self.input_size = F
        else:
            self.input_size = d_in
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        bs, F, T, d_in = X.shape
        if d_in == 1:
            X_viewed = X.squeeze(-1)
            X_viewed = X_viewed.transpose(1,2)
        else:
            X_viewed = X.reshape(-1, T, d_in)
        X_lstm, _ = self.lstm(X_viewed)
        if d_in == 1:
            X_final = X_lstm.unsqueeze(-1)
            X_final = X_final.transpose(1, 2)
        else:
            X_final = X_lstm.reshape(bs, F, *X_lstm.shape[1:])
        return X_final
    
    def modify_operation(self, input_shape):
        F, T, d_in = input_shape
        if d_in == 1:
            d_in = F
        diff = d_in - self.input_size
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.lstm.weight_ih_l0.data = nn.functional.pad(self.lstm.weight_ih_l0, pad)
        self.lstm.input_size = d_in
        self.input_size = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['lstm.weight_ih_l0'].shape[1]
        self.modify_operation(input_shape)
        super(Simple_2DLSTM, self).load_state_dict(state_dict, **kwargs)


class Simple_1DLSTM(Brick):
    def __init__(self, input_shape, hidden_size, num_layers):
        super(Simple_1DLSTM, self).__init__(input_shape)
        self.input_shape = input_shape
        self.input_size = input_shape[-1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X, h=None):
        init_shape = X.shape
        try:
            if len(init_shape) < 3:
                X = X.unsqueeze(-1)
                X = X.permute(0, 2, 1)
            if h is None:
                X, (h,c) = self.lstm(X)
            else:
                h, c = h
                if h.shape[1] >= X.shape[0]:
                    if h.shape[1] > X.shape[0]:
                        h = h[:, :X.shape[0]].contiguous()
                        c = h[:, :X.shape[0]].contiguous()
                    X, (h,c) = self.lstm(X, (h,c))
                else:
                    X, (h,c) = self.lstm(X)
            
            h = h.detach()
            c = c.detach()
            if len(init_shape) < 3:
                X = X.permute(0, 2, 1)
                X = X.squeeze(-1)
            return X, (h,c)
        except Exception as e:
            logger.error(f"Input shape: {self.input_shape}, {self.input_size}, X shape: {X.shape}, init shape: {init_shape} lstm = {self.lstm} \n {e}", exc_info=True)
            if h is not None:
                logger.error(f"h: {h.shape}, c: {c.shape}")
            raise e

    def modify_operation(self, input_shape):
        T = input_shape[-1]
        diff = T - self.input_size
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.lstm.weight_ih_l0.data = nn.functional.pad(self.lstm.weight_ih_l0, pad)
        self.lstm.input_size = T
        self.input_size = T 
        self.input_shape = input_shape

    def load_state_dict(self, state_dict, **kwargs):
        T = state_dict['lstm.weight_ih_l0'].shape[-1]
        self.modify_operation((T,))
        super(Simple_1DLSTM, self).load_state_dict(state_dict, **kwargs)


class Simple_2DGRU(Brick):
    def __init__(self, input_shape, hidden_size, num_layers):
        F, T, d_in = input_shape
        super(Simple_2DGRU, self).__init__(input_shape)
        if d_in == 1:
            self.input_size = F
        else:
            self.input_size = d_in
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        bs, F, T, d_in = X.shape
        if d_in == 1:
            X_viewed = X.squeeze(-1)
            X_viewed = X_viewed.transpose(1,2)
        else:
            X_viewed = X.reshape(-1, T, d_in)
        X_gru, _ = self.gru(X_viewed)
        if d_in == 1:
            X_final = X_gru.unsqueeze(-1)
            X_final = X_final.transpose(1, 2)
        else:
            X_final = X_gru.reshape(bs, F, *X_gru.shape[1:])
        return X_final
    
    def modify_operation(self, input_shape):
        F, T, d_in = input_shape
        if d_in == 1:
            d_in = F
        diff = d_in - self.input_size
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.gru.weight_ih_l0.data = nn.functional.pad(self.gru.weight_ih_l0, pad)
        self.gru.input_size = d_in
        self.input_size = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['gru.weight_ih_l0'].shape[1]
        self.modify_operation(input_shape)
        super(Simple_2DGRU, self).load_state_dict(state_dict, **kwargs)


class Simple_1DGRU(Brick):
    def __init__(self, input_shape, num_layers, hidden_size):
        super(Simple_1DGRU, self).__init__(input_shape)
        self.input_shape = input_shape
        self.input_size = input_shape[-1]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_shape[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X, h=None):
        init_shape = X.shape
        try:
            if len(init_shape) < 3:
                X = X.unsqueeze(-1)
                X = X.permute(0, 2, 1)
            if h is None:
                X, h = self.gru(X)
            else:
                if h.shape[1] >= X.shape[0]:
                    if h.shape[1] > X.shape[0]:
                        h = h[:, :X.shape[0]].contiguous()
                    X, h = self.gru(X, h)
                else:
                    X, h = self.gru(X)
            if len(init_shape) < 3:
                X = X.permute(0, 2, 1)
                X = X.squeeze(-1)
            return X, h.detach()
        except Exception as e:
            logger.error(f"Input shape: {self.input_shape}, {self.input_size}, X shape: {X.shape}, init_shape: {init_shape}, gru = {self.gru}")
            if h is not None:
                logger.error(f"h: {h.shape}")
            raise e
    
    def modify_operation(self, input_shape):
        T = input_shape[-1]
        diff = T - self.input_size
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.gru.weight_ih_l0.data = nn.functional.pad(self.gru.weight_ih_l0, pad)
        self.gru.input_size = T
        self.input_size = T  
        self.input_shape = input_shape

    def load_state_dict(self, state_dict, **kwargs):
        T = state_dict['gru.weight_ih_l0'].shape[-1]
        self.modify_operation((T,))
        super(Simple_1DGRU, self).load_state_dict(state_dict, **kwargs)      