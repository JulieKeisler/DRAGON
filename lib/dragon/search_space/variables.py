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
from dragon.search_space.dags import EvoDagVariable, HpVar, NodeVariable
from dragon.search_space.zellij_variables import DynamicBlock, FloatVar, IntVar, CatVar, Constant


def create_int_var(label, int_var, default_min, default_max):
    if int_var is None:
        default_neighborhood = int_neighborhood(default_min, default_max)
        int_var = IntVar(label, lower=default_min, upper=default_max, neighbor=IntInterval(default_neighborhood))
    elif isinstance(int_var, int) or isinstance(int_var, np.int64) or (isinstance(int_var, list) and len(int_var) == 1):
        if isinstance(int_var, list):
            int_var = int_var[0]
        int_var = IntVar(label, lower=1, upper=int_var, neighbor=IntInterval(int_neighborhood(1, int_var)))
    elif isinstance(int_var, list):
        if len(int_var) == 2:
            int_var = IntVar(label, lower=int_var[0], upper=int_var[1], neighbor=IntInterval(
                int_neighborhood(int_var[0], int_var[1])))
        if len(int_var) == 3:
            int_var = IntVar(label, lower=int_var[0], upper=int_var[1], neighbor=IntInterval(int_var[2]))
    return int_var

def activation_var(label, activations=None):
    if activations is None:
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

def identity_var(label):
    name = Constant(label=label, value=Identity, neighbor=ConstantInterval())
    return HpVar(label=label, name=name, hyperparameters = {}, neighbor=HpInterval())

def mlp_var(label, max_int=512):
    name = Constant(label=label, value=MLP, neighbor=ConstantInterval())
    hp = {
        "out_channels": create_int_var(label + " Output", None, 1, max_int)
    } 
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def attention_2d(label):
    name = CatVar(label + " Name", [SpatialAttention, TemporalAttention], neighbor=CatInterval())
    hp = {
            "Nh": create_int_var(label + " Nh", None, 1, 32),
            "d_out": create_int_var(label + " d_out", None, 1, 30),
            "init": CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval())
            
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def attention_1d(label):
    name = Constant(label=label, value=Attention1D, neighbor=ConstantInterval())
    hp = {
            "Nh": create_int_var(label + " Nh", None, 1, 32),
            "init": CatVar(label + " Initialisation", ["random", "conv"], neighbor=CatInterval()),
            "d_out": create_int_var(label+" d_out", None, 1, 512)
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def conv_1d(label, max_out, permute=True):
    name = Constant(label=label, value=Conv1d, neighbor=ConstantInterval())
    hp = {
            "kernel_size": create_int_var(label + " Ker", None, 1, max_out),
            "out_channels": create_int_var(label + " Output", None, 1, 512),
            "padding": Constant(label="Padding", value="same", neighbor=ConstantInterval()), 
            "permute": Constant(label="Permute", value=permute, neighbor=ConstantInterval())
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def const_conv_1d(label, kernel, max_out, permute=True):
    name = Constant(label=label, value=Conv1d, neighbor=ConstantInterval())
    hp = {
            "kernel_size": Constant(label="kernel", value=kernel, neighbor=ConstantInterval()),
            "out_channels": create_int_var(label + " Output", None, 1, max_out),
            "padding": Constant(label="Padding", value=0, neighbor=ConstantInterval()),
            "permute": Constant(label="Permute", value=permute, neighbor=ConstantInterval())
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def conv_2d(label, max_out=10, permute=True):
    name = Constant(label=label, value=Conv2d, neighbor=ConstantInterval())
    hp = {
            "kernel_size": create_int_var(label + " Ker1", None, 1, max_out),
            "out_channels": create_int_var(label + " Out", None, 1, 64),
            "padding": Constant(label="Padding", value="same", neighbor=ConstantInterval()),
            "permute": Constant(label="Permute", value=permute, neighbor=ConstantInterval())
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def norm_2d(label):
    name = CatVar(label, [BatchNorm2d, LayerNorm2d], neighbor=CatInterval())
    hp = {}
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def norm_1d(label):
    name = CatVar(label, [BatchNorm1d, LayerNorm1d], neighbor=CatInterval())
    hp = {}
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def dropout(label):
    name = Constant(label=label, value=Dropout, neighbor=ConstantInterval())
    hp = {
        "rate": FloatVar(label=label + " rate", lower=0, upper=1, neighbor=FloatInterval(0.1))
    }
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())


def pooling_1d(label):
    name = CatVar(label + " Name", [AVGPooling1D, MaxPooling1D], neighbor=CatInterval())
    hp = {
            "pool_size": create_int_var(label + " pooling", None, 1, 32),
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def pooling_2d(label):
    name = CatVar(label + " Name", [AVGPooling2D, MaxPooling2D], neighbor=CatInterval())
    hp = {
            "pool": create_int_var(label + " pooling", None, 1, 10),
            "stride": create_int_var(label + " pooling", None, 1, 5),
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def recurrence_1d(label, max_h=20):
    name = CatVar(label + " Name", [Simple_1DGRU, Simple_1DLSTM, Simple_1DRNN], neighbor=CatInterval())
    hp = {
            "hidden_size": create_int_var(label + " rec", None, 1, max_h),
            "num_layers": create_int_var(label + " rec", None, 1, 5),
        }
    
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())


def recurrence_2d(label):
    name = CatVar(label + " Name", [Simple_2DGRU, Simple_2DLSTM], neighbor=CatInterval())
    hp = {
            "hidden_size": create_int_var(label + " rec", None, 1, 20),
            "num_layers": create_int_var(label + " rec", None, 1, 5),
        }
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())


def mlp_const_var(label, out=None, value=MLP):
    name = Constant(label=label, value=value, neighbor=ConstantInterval())
    hp = {
        "out_channels": Constant(label="out_constant", value=out, neighbor=ConstantInterval())
    } 
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())


def pooling_2d_const_var(label, pool=None):
    name = Constant(label=label, value=AVGPooling2D, neighbor=ConstantInterval())
    hp = {
        "pool": Constant(label="pool_constant", value=pool, neighbor=ConstantInterval())
    } 
    return HpVar(label=label, name=name, hyperparameters=hp, neighbor=HpInterval())

def operations_1d_var(label, size, candidates=None):
    if candidates is None:
        candidates = [identity_var("Unitary"), attention_1d("Attention"), mlp_var("MLP"), conv_1d("Convolution", max_out), 
                      pooling_1d('Pooling'), norm_1d("Norm")]
    return DynamicBlock(
                    label,
                    NodeVariable(
                        label = "Variable",
                        combiner=CatVar("Combiner", features=['add', 'mul', 'concat'], neighbor=CatInterval()),
                        operation=CatVar(
                                "Candidates",
                                candidates,
                                neighbor=CatHpInterval(neighborhood=0.7)
                            ),
                        activation_function=activation_var("Activation"),
                        neighbor=NodeInterval()
                    ),
                    size,
                    neighbor=DynamicBlockInterval(neighborhood=2),
                    )

def operations_2d_var(label, size, candidates=None):
    if candidates is None:
        candidates = [identity_var("Unitary"), attention_2d("Attention"), conv_2d("Convolution"), norm_2d('Norm'), 
                  pooling_2d('Pooling'), dropout('Dropout'), mlp_var('MLP')]
    return DynamicBlock(
                label,
                NodeVariable(
                    label = "Variable",
                    combiner=CatVar("Combiner", features=['add', 'mul', 'concat'], neighbor=CatInterval()),
                    operation=CatVar(
                            "Candidates",
                            candidates,
                            neighbor=CatHpInterval(neighborhood=0.7)
                        ),
                    activation_function=activation_var("Activation"),
                    neighbor=NodeInterval()
                ),
                size,
                neighbor=DynamicBlockInterval(neighborhood=2),
            )

def dag_var(label, operations, complexity=None):
    return EvoDagVariable(
                    label=label,
                    operations = operations,
                    init_complexity=complexity,
                    neighbor=EvoDagInterval() 
                )

def node_var(label, operation, activation_function):
    return NodeVariable(label=label, 
                combiner=Constant(label="out_combiner", value="add", neighbor=ConstantInterval()),
                operation=operation,
                activation_function=Constant(label="out_act", value=activation_function, neighbor=ConstantInterval()),
                neighbor=NodeInterval())