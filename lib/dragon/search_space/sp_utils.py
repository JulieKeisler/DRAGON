import torch.nn as nn

from dragon.search_space.bricks.attention import SpatialAttention, TemporalAttention, Attention1D
from dragon.search_space.bricks.basics import MLP, Identity, Zero
from dragon.search_space.bricks.convolutions import Simple_2DCNN, Simple_1DCNN
from dragon.search_space.bricks.dropout import Dropout
from dragon.search_space.bricks.pooling import MaxPooling1D, AVGPooling1D, MaxPooling2D, AVGPooling2D
from dragon.search_space.bricks.recurrences import Simple_1DLSTM, Simple_2DLSTM, Simple_1DGRU, Simple_2DGRU
from dragon.utils.exceptions import InvalidArgumentError
from dragon.utils.tools import logger



def get_dag_max_channels(dag, input_shape):
    max_channels = input_shape
    for node in dag.nodes:
        if len(node.operation) > 1:
            if node.operation[1] > max_channels:
                max_channels = node.operation[1]
    return max_channels


def get_matrix_max_channels(operations, input_shape):
    max_channels = input_shape
    for op in operations:
        if len(op) > 1:
            try:
                if op[1] > max_channels:
                    max_channels = op[1]
            except TypeError as e:
                logger.info(f"{e}\nop={op}")
    return max_channels


# noinspection PyBroadException,SpellCheckingInspection
def get_layers(args, input_shape, input_channels):
    from dragon.search_space.cells import CandidateOperation
    combiner = args[0]
    name = args[1]
    if len(input_shape) == 3:
        F, T, d_in = input_shape
    elif len(input_shape) == 2:
        F, T = input_shape
    elif len(input_shape) == 1:
        F = input_shape if isinstance(input_shape, int) else input_shape[0]
    if F == 0:
        logger.info(f'Input shape = {input_shape}')
    if name == "Zero":
        return CandidateOperation(combiner, Zero(), input_channels)
    elif name == "Identity":
        return CandidateOperation(combiner, Identity(), input_channels)
    elif name == "SpatialAttention":
        return CandidateOperation(combiner,
            SpatialAttention(T, F, input_channels, d_out=args[2], Nh=args[3], init=args[4]),
            input_channels, activation=args[5])
    elif name == "TemporalAttention":
        return CandidateOperation(combiner,
            TemporalAttention(T, F, input_channels, d_out=args[2], Nh=args[3], init=args[4]),
            input_channels, activation=args[5])
    elif name == "Attention1D":
        return CandidateOperation(combiner,
            Attention1D(input_channels, d_in=1, d_out=1, Nh=args[2], init=args[3]), input_channels, activation=args[4])
    elif name == "Attention1D2D":
        return CandidateOperation(combiner,
            Attention1D(T=F, d_in=input_channels, d_out=args[2], Nh=args[3], init=args[4]), input_channels,
            activation=args[5])
    elif name == "MaxPooling1D":
        return CandidateOperation(combiner, MaxPooling1D(args[2]), input_channels)
    elif name == "AvgPooling1D":
        return CandidateOperation(combiner, AVGPooling1D(args[2]), input_channels)
    elif name == "MaxPooling2D":
        return CandidateOperation(combiner, MaxPooling2D(args[2]), input_channels)
    elif name == "AvgPooling2D":
        return CandidateOperation(combiner, AVGPooling2D(args[2]), input_channels)
    elif name == "MLP":
        return CandidateOperation(combiner, MLP(input_channels, out_channels=args[2]), input_channels, activation=args[3])
    elif name == "2DCNN":
        kernel_size = (min(F, args[3]), min(T, args[3]))
        return CandidateOperation(combiner, Simple_2DCNN(in_channels=input_channels, out_channels=args[2],
                                                         kernel_size=kernel_size), input_channels, activation=args[4])
    elif name == "1DCNN":
        return CandidateOperation(combiner, Simple_1DCNN(in_channels=1, out_channels=1, kernel_size=args[2]),
                                  input_channels, activation=args[3])
    elif name == "1DCNN2D":
        return CandidateOperation(combiner, Simple_1DCNN(in_channels=F, out_channels=F, kernel_size=args[2]),
                                  input_channels, activation=args[3])
    elif name == "1DLSTM":
        return CandidateOperation(combiner, Simple_1DLSTM(input_size=input_channels, hidden_size=args[2], num_layers=args[3]),
                                  input_channels, activation=args[4])
    elif name == "2DLSTM":
        return CandidateOperation(combiner, Simple_2DLSTM(input_size=input_channels, hidden_size=args[2],
                                                          num_layers=args[3]), input_channels, activation=args[4])
    elif name == "1DGRU":
        return CandidateOperation(combiner, Simple_1DGRU(input_size=input_channels, hidden_size=args[2],
                                                         num_layers=args[3]), input_channels, activation=args[4])
    elif name == "2DGRU":
        return CandidateOperation(combiner, Simple_2DGRU(input_size=input_channels, hidden_size=args[2],
                                               num_layers=args[3]), input_channels, activation=args[4])
    elif name == "Dropout":
        return CandidateOperation(combiner, Dropout(rate=args[2]), input_channels)
    elif name == "Input":
        return Identity()
    else:
        raise InvalidArgumentError('Layer', name, args[2:])


def get_activation(activation):
    if activation == "swish":
        return nn.SiLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "id":
        return nn.Identity()
    else:
        raise InvalidArgumentError('Activation', activation, activation)
