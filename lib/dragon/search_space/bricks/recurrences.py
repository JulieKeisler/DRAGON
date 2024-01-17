import torch.nn as nn


class Simple_2DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Simple_2DLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        bs, F, T, d_in = X.shape
        X_viewed = X.reshape(-1, T, d_in)
        X_lstm, _ = self.lstm(X_viewed)
        X_final = X_lstm.reshape(bs, F, *X_lstm.shape[1:])
        return X_final


class Simple_1DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Simple_1DLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(-1)
            X = X.permute(0, 2, 1)
        X, _ = self.lstm(X)
        if len(init_shape) < 3:
            X = X.permute(0, 2, 1)
            X = X.squeeze(-1)
        return X


class Simple_2DGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Simple_2DGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        bs, F, T, d_in = X.shape
        X_viewed = X.reshape(-1, T, d_in)
        X, _ = self.gru(X_viewed)
        X = X.reshape(bs, F, T, X.shape[-1])
        return X


class Simple_1DGRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(Simple_1DGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(-1)
            X = X.permute(0, 2, 1)
        X, _ = self.gru(X)

        if len(init_shape) < 3:
            X = X.permute(0, 2, 1)
            X = X.squeeze(-1)
        return X
