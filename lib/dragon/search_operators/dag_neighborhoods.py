import random
import numpy as np

import warnings
from dragon.search_operators.addons import VarNeighborhood
from dragon.search_space.base_variables import CatVar, Constant
from dragon.search_space.dag_encoding import Node
from dragon.utils.tools import logger

from dragon.search_space.dag_variables import EvoDagVariable, HpVar, NodeVariable

warnings.filterwarnings("ignore")

def int_neighborhood(b_min, b_max, scale=4):
    return int(np.ceil(max(int((b_max - b_min) / scale), 2)))

class HpInterval(VarNeighborhood):
    """HpInterval

    :ref:`varneigh`, used to determine the neighbor of an HpVar.
    Mutate the operation if it is not a constant and the hyperparameters.

    Parameters
    ----------
    variable : HpVar, default=None
        Targeted :ref:`var`.

    Examples
    --------
    >>> from dragon.search_space.bricks import MLP
    >>> from dragon.search_space.base_variables import Constant, IntVar
    >>> from dragon.search_space.dag_variables import HpVar
    >>> from dragon.search_operators.base_neighborhoods import ConstantInterval, IntInterval
    >>> from dragon.search_algorithmdag_neighborhoods import HpInterval
    >>> mlp = Constant("MLP operation", MLP, neighbor=ConstantInterval())
    >>> hp = {"out_channels": IntVar("out_channels", 1, 10, neighbor=IntInterval(2))}
    >>> mlp_var = HpVar("MLP var", mlp, hyperparameters=hp, neighbor=HpInterval())
    >>> print(mlp_var)
    HpVar(MLP var, 
    >>> test_mlp = mlp_var.random()
    >>> print(test_mlp)
    [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 2}]
    >>> mlp_var.neighbor(test_mlp[0], test_mlp[1])
    [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 1}]
    """
    def __init__(self, neighborhood=None, variable=None):
        super(HpInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, operation, hp, size=1, *kwargs):
        if size > 1:
            res = []
            for _ in range(size):
                new_hp = hp.copy()
                new_operation = operation
                hp_index = list(set(np.random.choice(range(len(hp.keys())+1), size=len(hp.keys())+1)))
                for idx in hp_index:
                    if idx >= len(hp.keys()):
                        if hasattr(self._target.operation, "neighbor"):
                            new_operation = self._target.operation.neighbor(operation)
                    else:
                        h = list(hp.keys())[idx]
                        new_hp[h] = self._target.hyperparameters[h].neighbor(hp[h])
                res.append([new_operation, new_hp])
            return res
        else:
            new_hp = hp.copy()
            new_operation = operation
            hp_index = list(set(np.random.choice(range(len(hp.keys())+1), size=len(hp.keys())+1)))
            for idx in hp_index:
                if idx >= len(hp.keys()):
                    if hasattr(self._target.operation, "neighbor"):
                        new_operation = self._target.operation.neighbor(operation)
                else:
                    h = list(hp.keys())[idx]
                    new_hp[h] = self._target.hyperparameters[h].neighbor(hp[h])
            return [new_operation, new_hp]

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, HpVar) or variable is None, logger.error(
            f"Target object must be a `HpVar` for {self.__class__.__operation__},\
             got {variable}"
        )
        self._target = variable  

class CatHpInterval(VarNeighborhood):
    """CatHpInterval

    :ref:`varneigh`, used to determine the neighbor of a CatVar of candidates operations.
    Given a probability `neighborhood`, draw a neighbor of the current operation, or draw a complete new operation

    Parameters
    ----------
    variable : CatVar, default=None
        Targeted :ref:`var`.
    neighborhood: float < 1, default=0.9
        Probability of drawing a neighbor instead of changing the whole operation.

    Examples
    --------
    >>> from dragon.search_space.bricks import MLP, LayerNorm1d, BatchNorm1d
    >>> from dragon.search_space.base_variables import Constant, IntVar, CatVar
    >>> from dragon.search_space.dag_variables import HpVar
    >>> from dragon.search_operators.base_neighborhoods import ConstantInterval, IntInterval, CatInterval
    >>> from dragon.search_algorithmdag_neighborhoods import HpInterval, CatHpInterval
    >>> mlp = Constant("MLP operation", MLP, neighbor=ConstantInterval())
    >>> hp = {"out_channels": IntVar("out_channels", 1, 10, neighbor=IntInterval(2))}
    >>> mlp_var = HpVar("MLP var", mlp, hyperparameters=hp, neighbor=HpInterval())
    >>> norm = CatVar("1d norm layers", features=[LayerNorm1d, BatchNorm1d], neighbor=CatInterval())
    >>> norm_var = HpVar("Norm var", norm, hyperparameters={}, neighbor=HpInterval())
    >>> candidates=CatVar("Candidates", features=[mlp_var, norm_var],neighbor=CatHpInterval(neighborhood=0.4))
    >>> print(candidates)
    CatVar(Candidates, [HpVar(MLP var, , HpVar(Norm var, ])
    >>> test_candidates = candidates.random()
    >>> print(test_candidates)
    [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 2}]
    >>> candidates.neighbor(test_candidates[0], test_candidates[1], size=10)
    [[<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 2}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 1}], [<class 'dragon.search_space.bricks.normalization.BatchNorm1d'>, {}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 1}], [<class 'dragon.search_space.bricks.normalization.LayerNorm1d'>, {}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 3}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 1}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 2}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 4}], [<class 'dragon.search_space.bricks.basics.MLP'>, {'out_channels': 3}]]
    """
    def __init__(self, neighborhood=None, variable=None):
        super(CatHpInterval, self).__init__(variable)
        if neighborhood is None:
            neighborhood = 0.9
        self._neighborhood = neighborhood

    def __call__(self, operation, hp, size=1, *kwargs):
        if size > 1:
            res = []
            for _ in range(size):
                p = np.random.uniform()
                if p>self._neighborhood:
                    # Draw completely new layer with a probability of 1-p
                    new_layer = self._target.random()
                else:
                    # Draw a neighbor of the layer
                    for f in self._target.features:
                        assert isinstance(f, HpVar), f"Target features should be of type HpVar but got {f} instead."
                        assert isinstance(f.operation, CatVar) or isinstance(f.operation, Constant), f"Target features should have operation argument of isntance Constant or CatVar but got {f.operation} instead."

                        if isinstance(f.operation, CatVar):
                            bool = operation in f.operation.features
                        elif isinstance(f.operation, Constant):
                            bool = operation == f.operation.value
                        
                        if bool:
                            new_layer = f.neighbor(operation, hp)
                            break
                res.append(new_layer)
            return res
        else:
            p = np.random.uniform()
            if p>self._neighborhood:
                # Draw completely new layer with a probability of 1-p
                new_layer = self._target.neighbor(operation, hp)
            else:
                # Neighbor of layer
                for f in self._target.features:
                    assert isinstance(f, HpVar), f"Target features should be of type HpVar but got {f} instead."
                    assert isinstance(f.operation, Constant) or isinstance(f.operation, CatVar), f"Target features should have operation argument of isntance Constant or CatVar but got {f.operation} instead."
                    if isinstance(f.operation, CatVar):
                        bool = operation in f.operation.features
                    elif isinstance(f.operation, Constant):
                        bool = operation == f.operation.value
                    if bool:
                        new_layer = f.neighbor(operation, hp)
                        break
            return new_layer

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        assert isinstance(neighborhood, list) or neighborhood is None, logger.error(
            f"Layers neighborhood must be a list of weights, got {neighborhood}"
        )
        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, CatVar) or variable is None, logger.error(
            f"Target object must be a `CatInterval` for {self.__class__.__operation__},\
             got {variable}"
        )
        self._target = variable

class NodeInterval(VarNeighborhood):
    """NodeInterval

    :ref:`varneigh`, used to determine the neighbor of a Node.
    Change the combiner and/or the operation and/or the hyperparameters and/or the activation function.

    Parameters
    ----------
    variable : CatVar, default=None
        Targeted :ref:`var`.

    Examples
    --------
    >>> from dragon.search_space.dag_variables import NodeVariable, HpVar
    >>> from dragon.search_space.bricks import MLP
    >>> from dragon.search_space.base_variables import Constant, IntVar, CatVar
    >>> from dragon.search_space.bricks_variables import activation_var
    >>> from dragon.search_operators.base_neighborhoods import ConstantInterval, IntInterval, CatInterval
    >>> from dragon.search_algorithmdag_neighborhoods import NodeInterval, HpInterval
    >>> combiner = CatVar("Combiner", features = ['add', 'mul'], neighbor=CatInterval())
    >>> operation = HpVar("Operation", Constant("MLP operation", MLP, neighbor=ConstantInterval()), 
    ...                   hyperparameters={"out_channels": IntVar("out_channels", 1, 10, neighbor=IntInterval(1))}, neighbor=HpInterval())
    >>> node = NodeVariable(label="Node variable", 
    ...                     combiner=combiner,
    ...                     operation=operation,
    ...                     activation_function=activation_var("Activation"), neighbor=NodeInterval())
    >>> print(node)
    Combiner: CatVar(Combiner, ['add', 'mul']) - Operation: HpVar(Operation,  - Act. Function: CatVar(Activation, [ReLU(), LeakyReLU(negative_slope=0.01), Identity(), Sigmoid(), Tanh(), ELU(alpha=1.0), GELU(approximate='none'), SiLU()])
    >>> test_node = node.random()
    >>> print(test_node)

    (combiner) mul -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 6} -- (activation) LeakyReLU(negative_slope=0.01) -- 
    >>> neighbor = node.neighbor(test_node)
    >>> print('Neighbor: ', neighbor)
    Neighbor:  
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 6} -- (activation) LeakyReLU(negative_slope=0.01) -- 
    >>> neighbor.set((3,))
    >>> print('Neighbor after setting: ', neighbor)
    Neighbor after setting:  
    (input shape) (3,) -- (combiner) add -- (op) MLP(
    (linear): Linear(in_features=3, out_features=6, bias=True)
    ) -- (activation) LeakyReLU(negative_slope=0.01) -- (output shape) (6,)
    >>> node.neighbor(neighbor)

    (input shape) (3,) -- (combiner) mul -- (op) MLP(
    (linear): Linear(in_features=3, out_features=5, bias=True)
    ) -- (activation) LeakyReLU(negative_slope=0.01) -- (output shape) (5,)
    """
    def __init__(self, neighborhood=None, variable=None):
        super(NodeInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, node, size=1, *kwargs):
        assert isinstance(node, Node), f"node should be of type Node but got {node} instead."
        if size > 1:
            res = []
            for _ in range(size):
                new_node = node.copy()
                changed = {}
                idx_list = list(set(np.random.choice(range(3), size=3)))
                if 0 in idx_list:
                    changed["combiner"] = self._target.combiner.neighbor(node.combiner)
                if 1 in idx_list:
                    op = self._target.operation.neighbor(node.name, node.hp, node.operation)
                    changed["operation"], changed["hp"] = op[0], op[1]
                new_node.modification(**changed)
                if 2 in idx_list:
                    new_node.activation = self._target.activation_function.neighbor(node.activation)
                res.append(new_node)
            return res
        else:
            node.copy()
            changed = {}
            idx_list = list(set(np.random.choice(range(3), size=3)))
            if 0 in idx_list:
                changed["combiner"] = self._target.combiner.neighbor(node.combiner)
            if 1 in idx_list:
                op = self._target.operation.neighbor(node.name, node.hp)
                changed["operation"], changed["hp"] = op[0], op[1]
            if hasattr(node, "input_shapes"):
                node.modification(**changed)
            else:
                if "combiner" in changed:
                    node.combiner = changed['combiner']
                if "operation" in changed:
                    node.name = changed["operation"]
                if "hp" in changed:
                    node.hp = changed['hp']
            if 2 in idx_list:
                node.activation = self._target.activation_function.neighbor(node.activation)
            return node

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        assert isinstance(neighborhood, list) or neighborhood is None, logger.error(
            f"Nodes neighborhood must be a list of weights, got {neighborhood}"
        )
        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, NodeVariable) or variable is None, logger.error(
            f"Target object must be a `NodeVariable` for {self.__class__.__operation__},\
             got {variable}"
        )
        self._target = variable

class EvoDagInterval(VarNeighborhood):
    """NodeInterval

    :ref:`varneigh`, used to determine the neighbor of an EvoDagVariable.
    May perform several modifications such as adding / deleting nodes, changing the nodes content, adding/removing connections.

    Parameters
    ----------
    variable : EvoDagVariable, default=None
        Targeted :ref:`var`.

    Examples
    --------
    >>> from dragon.search_space.dag_variables import HpVar, NodeVariable, EvoDagVariable
    >>> from dragon.search_space.bricks import MLP, MaxPooling1D, AVGPooling1D
    >>> from dragon.search_space.base_variables import Constant, IntVar, CatVar, DynamicBlock
    >>> from dragon.search_space.bricks_variables import activation_var
    >>> from dragon.search_algorithmdag_neighborhoods import CatHpInterval, EvoDagInterval, NodeInterval, HpInterval
    >>> from dragon.search_operators.base_neighborhoods import ConstantInterval, IntInterval, CatInterval, DynamicBlockInterval
    >>> mlp = HpVar("Operation", Constant("MLP operation", MLP, neighbor=ConstantInterval()), hyperparameters={"out_channels": IntVar("out_channels", 1, 10, neighbor=IntInterval(5))}, neighbor=HpInterval())
    >>> pooling = HpVar("Operation", CatVar("Pooling operation", [MaxPooling1D, AVGPooling1D], neighbor=CatInterval()), hyperparameters={"pool_size": IntVar("pool_size", 1, 5, neighbor=IntInterval(2))}, neighbor=HpInterval())
    >>> candidates = NodeVariable(label = "Candidates", 
    ...                           combiner=CatVar("Combiner", features=['add', 'concat'], neighbor=CatInterval()), 
    ...                           operation=CatVar("Candidates", [mlp, pooling], neighbor=CatHpInterval(0.4)),  
    ...                           activation_function=activation_var("Activation"), neighbor=NodeInterval())
    >>> operations = DynamicBlock("Operations", candidates, repeat=5, neighbor=DynamicBlockInterval(2))
    >>> dag = EvoDagVariable(label="DAG", operations=operations, neighbor=EvoDagInterval())
    >>> print(dag)
    EvoDagVariable(DAG,             - Operations:
    DynamicBlock(Operations, Combiner: CatVar(Combiner, ['add', 'concat']) - Operation: CatVar(Candidates, [HpVar(Operation, , HpVar(Operation, ]) - Act. Function: CatVar(Activation, [ReLU(), LeakyReLU(negative_slope=0.01), Identity(), Sigmoid(), Tanh(), ELU(alpha=1.0), GELU(approximate='none'), SiLU()])
    >>> test_dag = dag.random()
    >>> print(test_dag)
    NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) add -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 3} -- (activation) Sigmoid() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Identity() -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    >>> neighbor = dag.neighbor(test_dag, 3)
    >>> print(neighbor)
    [NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 6} -- (activation) ReLU() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 3} -- (activation) Sigmoid() -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]], NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 3} -- (activation) Sigmoid() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Identity() -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]], NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 3} -- (activation) Sigmoid() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Identity() -- ] | MATRIX:[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]
    >>> neighbor[0].set((3,))
    >>> print('First neighbor after setting: ', neighbor)
    First neighbor after setting:  [ModuleList(
    (0): 
    (input shape) (3,) -- (combiner) add -- (op) Identity() -- (activation) Identity() -- (output shape) (3,)
    (1): 
    (input shape) (3,) -- (combiner) add -- (op) MLP(
        (linear): Linear(in_features=3, out_features=6, bias=True)
    ) -- (activation) ReLU() -- (output shape) (6,)
    (2): 
    (input shape) (9,) -- (combiner) concat -- (op) AVGPooling1D(
        (pooling): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
    ) -- (activation) Sigmoid() -- (output shape) (3,)
    ), NODES: [
    (input shape) (3,) -- (combiner) add -- (op) Identity() -- (activation) Identity() -- (output shape) (3,), 
    (input shape) (9,) -- (combiner) concat -- (op) AVGPooling1D(
    (pooling): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
    ) -- (activation) Sigmoid() -- (output shape) (3,), 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Identity() -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]], NODES: [
    (input shape) (3,) -- (combiner) add -- (op) Identity() -- (activation) Identity() -- (output shape) (3,), 
    (input shape) (9,) -- (combiner) concat -- (op) AVGPooling1D(
    (pooling): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
    ) -- (activation) Sigmoid() -- (output shape) (3,), 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Identity() -- ] | MATRIX:[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]
    >>> dag.neighbor(neighbor[0])
    NODES: [
    (input shape) (3,) -- (combiner) add -- (op) Identity() -- (activation) Identity() -- (output shape) (3,), 
    (input shape) (3,) -- (combiner) concat -- (op) MLP(
    (linear): Linear(in_features=3, out_features=9, bias=True)
    ) -- (activation) ReLU() -- (output shape) (9,), 
    (input shape) (12,) -- (combiner) concat -- (op) AVGPooling1D(
    (pooling): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
    ) -- (activation) Sigmoid() -- (output shape) (4,)] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    """
    def __init__(self, neighborhood=None, variable=None, nb_mutations=None):
        super(EvoDagInterval, self).__init__(variable)
        self._neighborhood = neighborhood
        self.nb_mutations = nb_mutations

    def __call__(self, value, size=1):
        if size == 1:
            valid = False
            while not valid:
                inter = value.copy()
                # choose the nodes that will be modified
                if self.nb_mutations is None:
                    nb_mutations = size=len(inter.operations)
                else:
                    nb_mutations = self.nb_mutations
                variable_idx = list(set(np.random.choice(range(len(inter.operations)), nb_mutations)))
                modifications = []
                for i in range(len(variable_idx)):
                    idx = variable_idx[i]
                    if idx == 0:
                        choices = ['add', 'children']
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ["children"]
                    elif idx == len(inter.operations) - 1:
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ['delete', 'modify', 'parents']
                        elif inter.matrix.shape[0] == 2:
                            choices = ['add', 'modify', 'parents']
                        else:
                            choices = ['add', 'delete', 'modify', 'parents']
                    else:
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ['delete', 'modify', 'children', 'parents']
                        else:
                            choices = ['add', 'delete', 'modify', 'children', 'parents']
                    # choose the modification we are going to perform
                    modification = random.choice(choices)
                    if idx >= 0:
                        inter = self.modification(modification, idx, inter)
                        modifications.append(modification)
                        if modification == "add":
                            variable_idx[i+1:] = [j + 1 for j in variable_idx[i+1:]]
                        elif modification == "delete":
                            variable_idx[i+1:] = [j - 1 for j in variable_idx[i+1:]]
                    else:
                        logger.error(f'Idx: {idx}, modification: {modification}, modifications: {modifications}, variable idx: {variable_idx}')
                        pass
                try:
                    inter.assert_adj_matrix()
                    valid = True
                except AssertionError as e:
                    logger.error(f"Modifications = {modification}, idx={idx}, value=\n{value}\n{e}", exc_info=True)
            return inter
        else:
            res = []
            for _ in range(size):
                inter = value.copy()
                variable_idx = list(set(np.random.choice(range(len(inter.operations)), size=len(inter.operations))))
                for i in range(len(variable_idx)):
                    idx = variable_idx[i]
                    if idx == 0:
                        modification = random.choice(['add', 'children'])
                    elif idx == len(inter.operations) - 1:
                        modification = random.choice(['add', 'delete', 'modify', 'parents'])
                    else:
                        modification = random.choice(['add', 'delete', 'modify', 'children', 'parents'])
                    inter = self.modification(modification, idx, inter)
                    if modification == "add":
                        variable_idx[i + 1:] = [j + 1 for j in variable_idx[i + 1:]]
                    elif modification == "delete":
                        variable_idx[i + 1:] = [j - 1 for j in variable_idx[i + 1:]]
                inter.assert_adj_matrix()

                res.append(inter)
            return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if isinstance(neighborhood, list):
            self._neighborhood = neighborhood[0]
            self.target.operations.value.neighborhood = neighborhood
        else:
            self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, EvoDagVariable) or variable is None, logger.error(
            f"Target object must be a `EvoDagVariable` for {self.__class__.__operation__},\
                 got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self.target.operations.value, "neighbor"), logger.error(
                f"To use `EvoDagVariable`, value for operations for `EvoDagVariable` must have a `neighbor` method. "
                f"Use `neighbor` kwarg when defining a variable "
            )

    def modification(self, modif, idx, inter):
        assert modif in ['add', 'delete', 'modify', 'children', 'parents'], f"""Modification should be in ['add', 
        'delete', 'modify', 'children', 'parent'], got{modif} instead"""
        if modif == "add":  # Add new node after the one selected
            idxx = idx + 1
            new_node = self.target.operations.value.random(1)
            inter.operations.insert(idxx, new_node)
            N = len(inter.operations)
            parents = np.random.choice(2, idxx)
            while sum(parents) == 0:
                parents = np.random.choice(2, idxx)
            children = np.random.choice(2, N - idxx - 1)
            if N - idxx - 1 > 0:
                while sum(children) == 0:
                    children = np.random.choice(2, N - idxx - 1)
            inter.matrix = np.insert(inter.matrix, idxx, 0, axis=0)
            inter.matrix = np.insert(inter.matrix, idxx, 0, axis=1)
            inter.matrix[idxx, idxx+1:] = children
            inter.matrix[:idxx, idxx] = parents
            inter.matrix[-2, -1] = 1  # In case we add a node at the end
            if hasattr(inter.operations[idxx], "set_operation"):
                if hasattr(inter.operations[idx], "input_shapes"):
                    input_shapes = [inter.operations[i].output_shape for i in range(idxx) if parents[i] == 1]
                    inter.operations[idxx].set_operation(input_shapes)
        elif modif == "delete":  # Delete selected node
            inter.matrix = np.delete(inter.matrix, idx, axis=0)
            inter.matrix = np.delete(inter.matrix, idx, axis=1)
            for i in range(inter.matrix.shape[0] - 1):
                new_row = inter.matrix[i, i + 1:]
                while sum(new_row) == 0:
                    new_row = np.random.choice(2, inter.matrix.shape[0] - i - 1)
                inter.matrix[i, i + 1:] = new_row
            for j in range(1, inter.matrix.shape[1]):
                new_col = inter.matrix[:j, j]
                while sum(new_col) == 0:
                    new_col = np.random.choice(2, j)
                inter.matrix[:j, j] = new_col
            inter.operations.pop(idx)
        elif modif == "modify":  # Modify node operation
            inter.operations[idx] = self.target.operations.value.neighbor(inter.operations[idx])
        elif modif == "children":  # Modify node children
            new_row = np.zeros(inter.matrix.shape[0] - idx - 1)
            while sum(new_row) == 0:
                new_row = np.random.choice(2, new_row.shape[0])
            inter.matrix[idx, idx + 1:] = new_row
            for j in range(1, inter.matrix.shape[1]):
                new_col = inter.matrix[:j, j]
                while sum(new_col) == 0:
                    new_col = np.random.choice(2, j)
                inter.matrix[:j, j] = new_col
        elif modif == "parents":  # Modify node parents
            new_col = np.zeros(idx)
            while sum(new_col) == 0:
                new_col = np.random.choice(2, new_col.shape[0])
            inter.matrix[:idx, idx] = new_col
            for i in range(inter.matrix.shape[0] - 1):
                new_row = inter.matrix[i, i + 1:]
                while sum(new_row) == 0:
                    new_row = np.random.choice(2, inter.matrix.shape[0] - i - 1)
                inter.matrix[i, i + 1:] = new_row
            inter.matrix[-2, -1] = 1
        # Reconstruct nodes:
        for j in range(1, len(inter.operations)):
            if hasattr(inter.operations[j], "modification"):
                if hasattr(inter.operations[j], "input_shapes"):
                    input_shapes = [inter.operations[i].output_shape for i in range(j) if inter.matrix[i, j] == 1]
                    inter.operations[j].modification(input_shapes=input_shapes)
        return inter

