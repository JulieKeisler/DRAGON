import numpy as np
import torch.nn as nn
from dragon.search_algorithm.neighborhoods import CatHpInterval, EvoDagInterval, HpInterval, NodeInterval, int_neighborhood
from dragon.search_algorithm.zellij_neighborhoods import DynamicBlockInterval, FloatInterval, IntInterval, CatInterval, ConstantInterval
from dragon.search_space.bricks.attention import Attention1D, SpatialAttention, TemporalAttention
from dragon.search_space.bricks.basics import MLP, Identity
from dragon.search_space.bricks.convolutions import Conv1d, Conv2d
from dragon.search_space.bricks.dropout import Dropout
from dragon.search_space.bricks.normalization import BatchNorm1d, BatchNorm2d, LayerNorm1d, LayerNorm2d
from dragon.search_space.bricks.pooling import AVGPooling1D, AVGPooling2D, MaxPooling1D, MaxPooling2D
from dragon.search_space.bricks.recurrences import Simple_1DGRU, Simple_1DLSTM, Simple_1DRNN, Simple_2DGRU, Simple_2DLSTM
from dragon.search_space.dragon_variables import EvoDagVariable, HpVar, NodeVariable
from dragon.search_space.zellij_variables import DynamicBlock, FloatVar, IntVar, CatVar, Constant

def dag_var(label, operations, complexity=None):
    """dag_var(label, operations, complexity=None)

    Creates a DAG object, with some specified candidate operations.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    operations : `DynamicBlock`
        Candidate operations for the DAG.
    complexity : int, default = None
        Maximum number of nodes for the randomly created DAGs.

    Returns
    -------
        `EvoDagVariable`
    """
    return EvoDagVariable(
                    label=label,
                    operations = operations,
                    init_complexity=complexity,
                    neighbor=EvoDagInterval() 
                )

def node_var(label, operation, activation_function):
    """node_var(label, operations, activation_function)

    Creates a node outside a DAG, with some specified candidate operation and activation function.
    This function is used when a node is needed outside any DAG. 
    This situation often happens at the end of the network or between two DAGs.
    Usually, there will be only one input, therefore the combiner is useless and set to 'add'.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    operations : `HpVar`
        Candidate operation for the node.
    activation_function : `nn.Module`
        Activation function for the node.

    Returns
    -------
        `NodeVariable`
    """
    return NodeVariable(label=label, 
                combiner=Constant(label="out_combiner", value="add", neighbor=ConstantInterval()),
                operation=operation,
                activation_function=Constant(label="out_act", value=activation_function, neighbor=ConstantInterval()),
                neighbor=NodeInterval())

def activation_var(label, activations=None):
    """activation_var(label, activations=None)

    Creates a `CatVar` with multiple possible activation functions.
    If the list of possible activation functions is not given (`activations` set to None), then the default list will be used.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    activations : list, default=None
        List of the possble activation functions.
    
    Returns
    -------
        `CatVar`
    """
    if activations is None:
        # Activation functions by default
        activations = [
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.Identity(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.ELU(),
            nn.GELU(),
            nn.SiLU(),
        ]
    return CatVar(
        label,
        activations,
        neighbor=CatInterval(),
    )

def operations_var(label, size, candidates):
    """operations_var(label, size, candidates)

    Creates a `DynamicBlock` repeating `NodeVariable` objects corresponding to the candidates operations for a given DAG.
    The combiners and the activation functions used are the ones by default.
    This function provides a way of using the package without having to dig too deeply into the various objects. 
    The user is invited to directly manipulate the `DynamicBlock` class to customize the search space.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    size : int
        Maximum number of nodes that the DAG can have.
    candidates : list or ['2d', '1d']
        List of the candidate operations for the DAG. The values '1d' or '2d' give the default operations for a 1d DAG or a 2d one.

    Returns
    -------
        `DynamicBlock`
    """
    if isinstance(candidates, str):
        if candidates == "1d": # 1D operations by default
            candidates = [identity_var("Unitary"), attention_1d("Attention"), mlp_var("MLP"), conv_1d("Convolution", 10), 
                        pooling_1d('Pooling'), norm_1d("Norm")]
        elif candidates == "2d": # 2d operations by default
            candidates = [identity_var("Unitary"), attention_2d("Attention"), conv_2d("Convolution"), norm_2d('Norm'), 
                  pooling_2d('Pooling'), dropout('Dropout'), mlp_var('MLP')]
    return DynamicBlock(
                    label,
                    NodeVariable(
                        label = "Variable",
                        combiner=CatVar("Combiner", features=['add', 'mul', 'concat'], neighbor=CatInterval()), # Default combiners
                        operation=CatVar(
                                "Candidates",
                                candidates,
                                neighbor=CatHpInterval(neighborhood=0.7)
                            ),
                        activation_function=activation_var("Activation"), # Default activation functions
                        neighbor=NodeInterval()
                    ),
                    size,
                    neighbor=DynamicBlockInterval(neighborhood=2),
                    )

def identity_var(label):
    """identity_var(label)

    Creates a `HpVar` corresponding to the identity operation.
    
    Parameters
    ----------
    label : str
        Name of the variable.

    Returns
    -------
        `HpVar`
    """
    name = Constant(label=label, value=Identity, neighbor=ConstantInterval())
    return HpVar(label=label, operation=name, hyperparameters = {}, neighbor=HpInterval())

def mlp_var(label, max_int=512):
    """mlp_var(label, max_int=512)

    Creates a `HpVar` corresponding to the `MLP` (multi-layer perceptron) operation.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    max_int : int, default=512
        Maximum number of output channels.

    Returns
    -------
        `HpVar`
    """
    name = Constant(label=label, value=MLP, neighbor=ConstantInterval())
    hp = {
        "out_channels": IntVar(label + " Output", lower=1, upper=max_int, neighbor=IntInterval(int_neighborhood(1, max_int)))
    } 
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def mlp_const_var(label, out):
    """mlp_const_var(label)

    Creates a `HpVar` corresponding to the `MLP` (multi-layers perceptron) operation, but with a fix number of output channels.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    out : int
        Number of output channels.
    Returns
    -------
        `HpVar`
    """
    name = Constant(label=label, value=MLP, neighbor=ConstantInterval())
    hp = {
        "out_channels": Constant(label="out_constant", value=out, neighbor=ConstantInterval())
    } 
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def dropout(label):
    """dropout(label)

    Creates a `HpVar` corresponding to the dropout operation.
    The dropout rate goes from 0 to 1.
    
    Parameters
    ----------
    label : str
        Name of the variable.

    Returns
    -------
        `HpVar`
    """
    name = Constant(label=label, value=Dropout, neighbor=ConstantInterval())
    hp = {
        "rate": FloatVar(label=label + " rate", lower=0, upper=1, neighbor=FloatInterval(0.1))
    }
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def conv_1d(label, max_out, permute=True):
    """conv_1d(label)

    Creates a `HpVar` corresponding to the one-dimensional convolution operation.
    By default the input is of shape `(batch_size, L, C_in)`. Use `permute=False` if your input will be of size `(batch_size, C_in, L)`.
    By default `C_out` will go from 1 to 512. By default the padding is set to `same`.

    Parameters
    ----------
    label : str
        Name of the variable.
    max_out : 

    Returns
    -------
        `HpVar`
    """
    name = Constant(label=label, value=Conv1d, neighbor=ConstantInterval())
    hp = {
            "kernel_size": IntVar(label + " Ker", lower=1, upper=max_out, neighbor=IntInterval(int_neighborhood(1, max_out))),
            "out_channels": IntVar(label + " d_out", lower=1, upper=512, neighbor=IntInterval(int_neighborhood(1,512))),
            "padding": Constant(label="Padding", value="same", neighbor=ConstantInterval()), 
            "permute": Constant(label="Permute", value=permute, neighbor=ConstantInterval())
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def const_conv_1d(label, kernel, max_out, permute=True):
    name = Constant(label=label, value=Conv1d, neighbor=ConstantInterval())
    hp = {
            "kernel_size": Constant(label="kernel", value=kernel, neighbor=ConstantInterval()),
            "out_channels": IntVar(label + " d_out", lower=1, upper=max_out, neighbor=IntInterval(int_neighborhood(1, max_out))),
            "padding": Constant(label="Padding", value=0, neighbor=ConstantInterval()),
            "permute": Constant(label="Permute", value=permute, neighbor=ConstantInterval())
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def pooling_1d(label):
    name = CatVar(label + " Name", [AVGPooling1D, MaxPooling1D], neighbor=CatInterval())
    hp = {
            "pool_size": IntVar(label + " pooling", lower=1, upper=32, neighbor=IntInterval(int_neighborhood(1,32))),
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def attention_1d(label):
    name = Constant(label=label, value=Attention1D, neighbor=ConstantInterval())
    hp = {
            "Nh": IntVar(label + " Nh", lower=1, upper=32, neighbor=IntInterval(int_neighborhood(1,32))),
            "init": CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval()),
            "d_out": IntVar(label + " d_out", lower=1, upper=512, neighbor=IntInterval(int_neighborhood(1,512)))
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def norm_1d(label):
    name = CatVar(label, [BatchNorm1d, LayerNorm1d], neighbor=CatInterval())
    hp = {}
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def recurrence_1d(label, max_h=20):
    name = CatVar(label + " Name", [Simple_1DGRU, Simple_1DLSTM, Simple_1DRNN], neighbor=CatInterval())
    hp = {
            "hidden_size": IntVar(label + " hidden_size", lower=1, upper=max_h, neighbor=IntInterval(int_neighborhood(1,max_h))),
            "num_layers": IntVar(label + " num_layers", lower=1, upper=5, neighbor=IntInterval(int_neighborhood(1,5))),
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def conv_2d(label, max_out=10, permute=True):
    name = Constant(label=label, value=Conv2d, neighbor=ConstantInterval())
    hp = {
            "kernel_size": IntVar(label + " kernel", lower=1, upper=max_out, neighbor=IntInterval(int_neighborhood(1, max_out))),
            "out_channels": IntVar(label + " d_out", lower=1, upper=64, neighbor=IntInterval(int_neighborhood(1,64))),
            "padding": Constant(label="Padding", value="same", neighbor=ConstantInterval()),
            "permute": Constant(label="Permute", value=permute, neighbor=ConstantInterval())
        }
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def pooling_2d(label):
    name = CatVar(label + " Name", [AVGPooling2D, MaxPooling2D], neighbor=CatInterval())
    hp = {
            "pool": IntVar(label + " pooling", lower=1, upper=10, neighbor=IntInterval(int_neighborhood(1,10))),
            "stride": IntVar(label + " stride", lower=1, upper=5, neighbor=IntInterval(int_neighborhood(1,5))),
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def pooling_2d_const_var(label, pool=None):
    name = Constant(label=label, value=AVGPooling2D, neighbor=ConstantInterval())
    hp = {
        "pool": Constant(label="pool_constant", value=pool, neighbor=ConstantInterval())
    } 
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def norm_2d(label):
    name = CatVar(label, [BatchNorm2d, LayerNorm2d], neighbor=CatInterval())
    hp = {}
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def recurrence_2d(label):
    name = CatVar(label + " Name", [Simple_2DGRU, Simple_2DLSTM], neighbor=CatInterval())
    hp = {
            "hidden_size": IntVar(label + " hidden_size", lower=1, upper=20, neighbor=IntInterval(int_neighborhood(1,20))),
            "num_layers": IntVar(label + " num_layers", lower=1, upper=5, neighbor=IntInterval(int_neighborhood(1,5)))
        }
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

def attention_2d(label):
    name = CatVar(label + " Name", [SpatialAttention, TemporalAttention], neighbor=CatInterval())
    hp = {
            "Nh": IntVar(label + " Nh", lower=1, upper=32, neighbor=IntInterval(int_neighborhood(1,32))),
            "d_out": IntVar(label + " d_out", lower=1, upper=30, neighbor=IntInterval(int_neighborhood(1,30))),
            "init": CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval())
            
        }
    
    return HpVar(label=label, operation=name, hyperparameters=hp, neighbor=HpInterval())

