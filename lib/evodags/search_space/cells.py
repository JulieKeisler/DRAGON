import torch
import torch.nn as nn
import numpy as np

from lib.evodags.search_space.bricks.basics import Identity
from lib.evodags.search_space.sp_utils import get_layers, get_activation


class AdjCell(nn.Module):
    def __init__(self, adj_matrix, input_shape):
        super(AdjCell, self).__init__()
        self.matrix = adj_matrix.matrix
        self.nodes = [o.copy() for o in adj_matrix.operations]
        layers = [Identity()]
        self.nodes[0] = ["None", self.nodes[0], input_shape[-1]]
        for j in range(1, len(self.nodes)):
            parents_channels = [self.nodes[i][2] for i in range(j) if self.matrix[i, j] == 1]
            if self.nodes[j][0] == "concat":
                input_channel = sum(parents_channels)
            else:
                input_channel = max(parents_channels)
            if "Pooling" in self.nodes[j][1]:
                input_channel = max(self.nodes[j][2]+1, input_channel)

            if self.nodes[j][1] == "1DCNN":
                input_channel = max(self.nodes[j][2], input_channel)
            if self.nodes[j][1] == "2DCNN":
                input_channel = max(self.nodes[j][3], input_channel)
            try:
                layers.append(get_layers(self.nodes[j], input_shape, input_channel))
            except AssertionError as e:
                print("EXCEPTION : ", j, " ===> ", self.nodes[j], " ", input_shape, " ", input_channel, "\n ", adj_matrix.operations, "\n", self.nodes)
                raise e
            if "Pooling" in self.nodes[j][1]:
                self.nodes[j].insert(2, input_channel//self.nodes[j][2])
            if self.nodes[j][1] == "1DCNN":
                self.nodes[j][2] = input_channel - self.nodes[j][2] + 1
            if len(self.nodes[j]) < 3:
                self.nodes[j].append(input_channel)
            if self.nodes[j][1] == "Dropout":
                self.nodes[j].insert(2, input_channel)
        self.layers = nn.ModuleList(layers)
        self.output_shape = self.nodes[-1][2]

    def forward(self, X):
        device = X.get_device()
        N = len(self.layers)
        outputs = np.empty(N, dtype=object)
        outputs[0] = X
        for j in range(1, N):
            output = self.layers[j]([outputs[i] for i in range(j) if self.matrix[i, j] == 1])
            if device >= 0:
                output = output.to(device)
            outputs[j] = output
        return output


class CandidateOperation(nn.Module):
    def __init__(self, combiner, operation, input_channels, activation="id"):
        super(CandidateOperation, self).__init__()
        assert combiner in ['add', 'concat', 'mul'], f"Invalid combiner argument, got: {combiner}."
        self.combiner = combiner
        self.operation = operation
        self.input_channels = input_channels
        self.activation = get_activation(activation)

    def forward(self, X):
        X = self.combine(X)
        X = self.operation(X)
        X = self.activation(X)
        return X

    def combine(self, X):
        if isinstance(X, list):
            if self.combiner == "concat":
                X = torch.cat(X, dim=-1)
                return self.padding(X)
            elif self.combiner == "add":
                X = self.padding(X)
                return sum(X)
            elif self.combiner == "mul":
                X = self.padding(X)
                X_mul = X[0]
                for i in range(1, len(X)):
                    X_mul *= X[i]
                return X_mul
        else:
            return X

    def padding(self, X):
        if isinstance(X, list):
            pad_X = []
            for x in X:
                pad = (int(np.ceil((self.input_channels - x.shape[-1])/2)),
                       int(np.floor((self.input_channels - x.shape[-1]))/2))
                pad = nn.functional.pad(x, pad)
                pad_X.append(pad)
        else:

            pad = (int(np.ceil((self.input_channels - X.shape[-1])/2)),
                   int(np.floor((self.input_channels - X.shape[-1])/2)))
            pad_X = nn.functional.pad(X, pad)
        return pad_X