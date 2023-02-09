from zellij.core.variables import CatVar, ArrayVar, DynamicBlock
from zellij.utils.neighborhoods import ArrayInterval, DynamicBlockInterval

from framework.search_algorithm.neighborhoods import LayersInterval, AdjMatrixHierarchicalInterval
from framework.search_space.dags import AdjMatrixVariable
from framework.search_space.variables import unitary_var, pooling_var, mlp_var, \
    recurrence_var, dropout_var, attention_var, convolution_var_1d, activation_var, create_int_var


def operations_var(label, shape, size):
    return DynamicBlock(
        label,
        CatVar(
            label + "Candidates",
            [
                unitary_var(label + " Unitary"),
                pooling_var(label + " Pooling", ["MaxPooling1D", "AvgPooling1D"]),
                mlp_var(label + " MLP"),
                convolution_var_1d(label + " Convolution", kernel=shape),
                attention_var(label + " Attention"),
                recurrence_var(label + " RNN", operations=["1DGRU", "1DLSTM"]),
                dropout_var(label + " Dropout")
            ],
            neighbor=LayersInterval([2, 1]),
        ),
        size,
        neighbor=DynamicBlockInterval(neighborhood=2),
    )


def NN_monash_var(label="Neural Network", shape=1000, size=10):
    NeuralNetwork = ArrayVar(
        AdjMatrixVariable(
            "Cell",
            operations_var("Feed Cell", shape, size),
            neighbor=AdjMatrixHierarchicalInterval()
        ),
        activation_var("NN Activation"),
        create_int_var("Seed", None, 0, 10000),
        label=label,
        neighbor=ArrayInterval(),
    )
    return NeuralNetwork


def monash_search_space(config):
    D = config["Lag"]
    size = config["SPSize"]
    return NN_monash_var(shape=D, size=size)
