import numpy as np
from zellij.core.variables import (
    ArrayVar,
    CatVar,
    IntVar,
    Constant,
    FloatVar
)

from zellij.utils.neighborhoods import (
    CatInterval,
    IntInterval,
    ArrayInterval,
    ConstantInterval,
    FloatInterval
)
from zellij.utils.converters import IntMinmax, CatMinmax, ArrayMinmax


def activation_var(label, activations=None):
    if activations is None:
        activations = [
            "swish",
            "relu",
            "leaky_relu",
            "sigmoid",
            "softmax",
            "gelu",
            "elu",
            "id",
        ]
    return CatVar(
        label,
        activations,
        neighbor=CatInterval(),
    )


def combiner_var(label, combiners=None):
    if combiners is not None:
        pass
    else:
        combiners = [
            "add",
            "mul",
            "concat"
        ]
    return CatVar(
        label,
        combiners,
        neighbor=CatInterval()
    )


def unitary_var(label, operations=None):
    if operations is None:
        operations = "Identity"
    return ArrayVar(
        combiner_var(label + " Combiner"),
        Constant(label + " Operation", operations, neighbor=ConstantInterval()),
        label=label,
        neighbor=ArrayInterval(),
    )


def int_neighborhood(b_min, b_max, scale=4):
    return np.ceil(max(int((b_max - b_min) / scale), 2))


def create_int_var(label, int_var, default_min, default_max):
    if int_var is None:
        default_neighborhood = int_neighborhood(default_min, default_max)
        int_var = IntVar(label, lower=default_min, upper=default_max, neighbor=IntInterval(default_neighborhood), to_continuous=IntMinmax())
    elif isinstance(int_var, int) or isinstance(int_var, np.int64) or (isinstance(int_var, list) and len(int_var) == 1):
        if isinstance(int_var, list):
            int_var = int_var[0]
        int_var = IntVar(label, lower=1, upper=int_var, neighbor=IntInterval(int_neighborhood(1, int_var)), to_continuous=IntMinmax())
    elif isinstance(int_var, list):
        if len(int_var) == 2:
            int_var = IntVar(label, lower=int_var[0], upper=int_var[1], neighbor=IntInterval(
                int_neighborhood(int_var[0], int_var[1])), to_continuous=IntMinmax())
        if len(int_var) == 3:
            int_var = IntVar(label, lower=int_var[0], upper=int_var[1], neighbor=IntInterval(int_var[2]), to_continuous=IntMinmax())
    return int_var


def attention_mts_var(label, operations=None, output=None, n_heads=None, initialization=None, activation=None, *kwargs):
    if operations is None:
        operations = ["SpatialAttention", "TemporalAttention"]
    operations = CatVar(label + " Operation", operations, neighbor=CatInterval())
    n_heads = create_int_var(label + " N_Heads", n_heads, 1, 32)
    output = create_int_var(label + " Output", output, 1, 30)
    if initialization is None:
        initialization = CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval())
    if activation is None:
        activation = activation_var(label + " Activation")
    assert isinstance(operations,
                      CatVar), f"{label} operations should be an instance of CatVar, got {type(operations)} instead."
    assert isinstance(output, IntVar), f"{label} output should be an instance of IntVar, got {type(output)} instead."
    assert isinstance(n_heads, IntVar), f"{label} n_heads should be an instance of IntVar, got {type(n_heads)} instead."
    assert isinstance(initialization, CatVar), f"{label} initialization should be an instance of CatVar, got " \
                                               f"{type(initialization)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(initialization)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operations, output, n_heads, initialization,
                    activation, *kwargs, label=label, neighbor=ArrayInterval())


def attention_var(label, operations=None, n_heads=None, initialization=None, activation=None, *kwargs):
    if operations is None:
        operations = "Attention1D"
    operations = Constant(label + " Operation", operations, neighbor=ConstantInterval())
    n_heads = create_int_var(label + " Nh", n_heads, 1, 10)
    if initialization is None:
        initialization = CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval())
    elif isinstance(initialization, list):
        initialization = CatVar(label + " Initialisation", initialization, neighbor=CatInterval())
    if activation is None:
        activation = activation_var(label + " Activation")
    elif isinstance(activation, list):
        activation = activation_var(label + " Activation", activations=activation)
    assert isinstance(operations,
                      Constant), f"{label} operations should be an instance of Constant, got {type(operations)} instead."
    assert isinstance(n_heads, IntVar), f"{label} n_heads should be an instance of IntVar, got {type(n_heads)} instead."
    assert isinstance(initialization, CatVar), f"{label} initialization should be an instance of CatVar, got " \
                                               f"{type(initialization)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(initialization)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operations, n_heads, initialization, activation, *kwargs,
                    label=label, neighbor=ArrayInterval())


def attention_var_2d(label, operations=None, output=None, n_heads=None, initialization=None, activation=None, *kwargs):
    if operations is None:
        operations = "Attention1D2D"
    operations = Constant(label + " Operation", operations, neighbor=ConstantInterval())
    output = create_int_var(label + " Output", output, 1, 512)
    n_heads = create_int_var(label + " Nh", n_heads, 1, 10)
    if initialization is None:
        initialization = CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval())
    elif isinstance(initialization, list):
        initialization = CatVar(label + " Initialisation", initialization, neighbor=CatInterval())
    if activation is None:
        activation = activation_var(label + " Activation")
    elif isinstance(activation, list):
        activation = activation_var(label + " Activation", activations=activation)
    assert isinstance(operations,
                      Constant), f"{label} operations should be an instance of Constant, got {type(operations)} instead."
    assert isinstance(n_heads, IntVar), f"{label} n_heads should be an instance of IntVar, got {type(n_heads)} instead."
    assert isinstance(output, IntVar), f"{label} output should be an instance of IntVar, got {type(output)} instead."
    assert isinstance(initialization, CatVar), f"{label} initialization should be an instance of CatVar, got " \
                                               f"{type(initialization)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(initialization)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operations, output, n_heads, initialization, activation, *kwargs, label=label, neighbor=ArrayInterval())


def pooling_var(label, operations, size=None):
    assert isinstance(operations, list), f"{label} operations should be an instance of list, got {type(operations)} " \
                                         f"instead."
    operations = CatVar(label + " Operation", operations, neighbor=CatInterval())
    size = create_int_var(label + " Size", size, 1, 32)
    assert isinstance(size, IntVar), f"{label} size should be an instance of IntVar, got {type(size)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operations, size, label=label, neighbor=ArrayInterval())


def dropout_var(label, operations=None, rate=None):
    if operations is None:
        operations = "Dropout"
    operations = Constant(label + " Operation", operations, neighbor=ConstantInterval())
    if rate is None:
        rate = 1
    rate = FloatVar(label + " Rate", 0, rate, neighbor=FloatInterval(0.01))
    assert isinstance(rate, FloatVar), f"{label} rate should be an instance of FloatVar, got {type(rate)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operations, rate, label=label, neighbor=ArrayInterval())


def mlp_var(label, output=None, activation=None):
    output = create_int_var(label + " Output", output, 1, 512)
    if activation is None:
        activation = activation_var(label + " Activation")
    assert isinstance(output, IntVar), f"{label} output should be an instance of IntVar, got {type(output)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(activation)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), Constant(label + " Operation", "MLP",
                    neighbor=ConstantInterval()), output, activation, label=label, neighbor=ArrayInterval())


def convolution_var_2d(label, kernel, operation=None, output=None, activation=None):
    if operation is None:
        operation = '2DCNN'
    if isinstance(operation, str) or (isinstance(operation, list) and len(operation == 1)):
        if isinstance(operation, list):
            operation = operation[0]
        operation = Constant(label + " Operation", operation, neighbor=ConstantInterval())
    if isinstance(operation, list):
        operation = CatVar(label + "Operation", operation, neighbor=CatInterval())
    output = create_int_var(label + " Output", output, 1, 100)
    kernel = create_int_var(label + " Kernel", kernel, None, None)
    if activation is None:
        activation = activation_var(label + " Activation")
    assert isinstance(output, IntVar), f"{label} output should be an instance of IntVar, got {type(output)} instead."
    assert isinstance(kernel, IntVar), f"{label} kernel should be an instance of IntVar, got {type(kernel)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(activation)} instead."
    assert (isinstance(operation, CatVar) or isinstance(operation,
                                                        Constant)), f"{label} operation should be an instance" \
                                                                    f" of CatVar or Constant, got {type(operation)} instead."

    return ArrayVar(combiner_var(label + " Combiner"), operation, output, kernel, activation, label=label, neighbor=ArrayInterval())


def convolution_var_1d(label, kernel, operation=None, activation=None):
    if operation is None:
        operation = '1DCNN'
    if isinstance(operation, str) or (isinstance(operation, list) and len(operation == 1)):
        if isinstance(operation, list):
            operation = operation[0]
        operation = Constant(label + " Operation", operation, neighbor=ConstantInterval())
    if isinstance(operation, list):
        operation = CatVar(label + "Operation", operation, neighbor=CatInterval())
    kernel = create_int_var(label + " Kernel", kernel, None, None)
    if activation is None:
        activation = activation_var(label + " Activation")
    assert isinstance(kernel, IntVar), f"{label} kernel should be an instance of IntVar, got {type(kernel)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(activation)} instead."
    assert (isinstance(operation, CatVar) or isinstance(operation,
                                                        Constant)), f"{label} operation should be an instance" \
                                                                    f" of CatVar or Constant, got {type(operation)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operation, kernel, activation, label=label, neighbor=ArrayInterval())


def convolution_var_1d_2d(label, kernel, operation=None, activation=None):
    if operation is None:
        operation = '1DCNN2D'
    if isinstance(operation, str) or (isinstance(operation, list) and len(operation == 1)):
        if isinstance(operation, list):
            operation = operation[0]
        operation = Constant(label + " Operation", operation, neighbor=ConstantInterval())
    if isinstance(operation, list):
        operation = CatVar(label + "Operation", operation, neighbor=CatInterval())
    kernel = create_int_var(label + " Kernel", kernel, None, None)

    if activation is None:
        activation = activation_var(label + " Activation")
    assert isinstance(kernel, IntVar), f"{label} kernel should be an instance of IntVar, got {type(kernel)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(activation)} instead."
    assert (isinstance(operation, CatVar) or isinstance(operation,
                                                        Constant)), f"{label} operation should be an instance" \
                                                                    f" of CatVar or Constant, got {type(operation)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operation, kernel, activation, label=label, neighbor=ArrayInterval())


def recurrence_var(label, operations, hidden_size=None, output=None, activation=None):
    operations = CatVar(label + " Operation", operations, neighbor=CatInterval())
    hidden_size = create_int_var(label + " Hidden Size", hidden_size, 1, 20)
    output = create_int_var(label + " Output", output, 1, 20)
    if activation is None:
        activation = activation_var(label + " Activation")
    assert isinstance(hidden_size,
                      IntVar), f"{label} hidden size should be an instance of IntVar, got {type(hidden_size)} instead."
    assert isinstance(output,
                      IntVar), f"{label} output size should be an instance of IntVar, got {type(output)} instead."
    assert isinstance(activation,
                      CatVar), f"{label} activation should be an instance of CatVar, got {type(activation)} instead."
    assert (isinstance(operations, CatVar) or isinstance(operations,
                                                         Constant)), f"{label} operations should be an instance" \
                                                                     f" of CatVar or Constant, got {type(operations)} instead."
    return ArrayVar(combiner_var(label + " Combiner"), operations, hidden_size, output, activation, label=label, neighbor=ArrayInterval())
