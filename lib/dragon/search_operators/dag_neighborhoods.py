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

    `Addon`, used to determine the neighbor of an HpVar.
    Mutate the operation if it is not a constant and the hyperparameters.

    Parameters
    ----------
    variable : HpVar, default=None
        Targeted `Variable`

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

    `Addon`, used to determine the neighbor of a CatVar of candidates operations.
    Given a probability `neighborhood`, draw a neighbor of the current operation, or draw a complete new operation

    Parameters
    ----------
    variable : CatVar, default=None
        Targeted `Variable`
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

    `Addon`, used to determine the neighbor of a Node.
    Change the combiner and/or the operation and/or the hyperparameters and/or the activation function.

    Parameters
    ----------
    variable : CatVar, default=None
        Targeted `Variable`

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

    `Addon`, used to determine the neighbor of an EvoDagVariable.
    May perform several modifications such as adding / deleting nodes, changing the nodes content, adding/removing connections.

    Parameters
    ----------
    variable : EvoDagVariable, default=None
        Targeted `Variable`.

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

    def _sample_variable_indices(self, n_operations):
        """Sample variable indices with bias towards fewer mutations.
        
        Parameters
        ----------
        n_operations : int
            Number of operations in the DAG
            
        Returns
        -------
        list
            Sampled indices with bias towards smaller samples
        """
        if self.nb_mutations is None:
            max_mutations = n_operations
        else:
            max_mutations = min(self.nb_mutations, n_operations)
        
        # Probabilités décroissantes exponentiellement pour le nombre de mutations
        probs = np.exp(-np.arange(1, max_mutations + 1) / 2)
        probs = probs / probs.sum()
        
        # Échantillonner le nombre de mutations
        n_mutations = np.random.choice(range(1, max_mutations + 1), p=probs)
        
        # Échantillonner les indices sans remplacement
        return list(set(np.random.choice(range(n_operations), n_mutations, replace=False)))

    def __call__(self, value, size=1):
        if size == 1:
            valid = False
            while not valid:
                inter = value.copy()
                # choose the nodes that will be modified with bias towards fewer mutations
                variable_idx = self._sample_variable_indices(len(inter.operations))
                modifications = []
                for i in range(len(variable_idx)):
                    idx = variable_idx[i]
                    # Skip invalid indices
                    if idx < 0 or idx >= len(inter.operations):
                        logger.warning(f'Skipping invalid idx: {idx}, current operations length: {len(inter.operations)}')
                        continue
                        
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
                    inter = self.modification(modification, idx, inter)
                    modifications.append(modification)
                    if modification == "add":
                        # Update all subsequent indices (they shift by +1)
                        for j in range(i + 1, len(variable_idx)):
                            if variable_idx[j] > idx:
                                variable_idx[j] += 1
                    elif modification == "delete":
                        # Update all subsequent indices (they shift by -1)
                        for j in range(i + 1, len(variable_idx)):
                            if variable_idx[j] > idx:
                                variable_idx[j] -= 1
                            elif variable_idx[j] == idx:
                                # Mark for deletion if it was the deleted node
                                variable_idx[j] = -1
                try:
                    inter.assert_adj_matrix()
                    valid = True
                except AssertionError as e:
                    logger.error(f"Modifications = {modifications}, value=\n{value}\n{e}", exc_info=True)
            return inter
        else:
            res = []
            for _ in range(size):
                inter = value.copy()
                variable_idx = self._sample_variable_indices(len(inter.operations))
                for i in range(len(variable_idx)):
                    idx = variable_idx[i]
                    # Skip invalid indices
                    if idx < 0 or idx >= len(inter.operations):
                        continue
                        
                    if idx == 0:
                        modification = random.choice(['add', 'children'])
                    elif idx == len(inter.operations) - 1:
                        modification = random.choice(['add', 'delete', 'modify', 'parents'])
                    else:
                        modification = random.choice(['add', 'delete', 'modify', 'children', 'parents'])
                    inter = self.modification(modification, idx, inter)
                    if modification == "add":
                        # Update all subsequent indices (they shift by +1)
                        for j in range(i + 1, len(variable_idx)):
                            if variable_idx[j] > idx:
                                variable_idx[j] += 1
                    elif modification == "delete":
                        # Update all subsequent indices (they shift by -1)
                        for j in range(i + 1, len(variable_idx)):
                            if variable_idx[j] > idx:
                                variable_idx[j] -= 1
                            elif variable_idx[j] == idx:
                                variable_idx[j] = -1
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

    def _flip_single_connection(self, matrix, idx, connection_type):
        """Flip a single connection (parent or child) for node idx.
        
        Parameters
        ----------
        matrix : np.ndarray
            Adjacency matrix
        idx : int
            Node index
        connection_type : str
            Either 'parent' or 'child'
        """
        if connection_type == 'child':
            # Modify children connections
            if idx < matrix.shape[0] - 1:
                available_children = list(range(idx + 1, matrix.shape[0]))
                child_idx = random.choice(available_children)
                matrix[idx, child_idx] = 1 - matrix[idx, child_idx]
        else:  # parent
            # Modify parent connections
            if idx > 0:
                available_parents = list(range(idx))
                parent_idx = random.choice(available_parents)
                matrix[parent_idx, idx] = 1 - matrix[parent_idx, idx]
        
        return matrix

    def _ensure_valid_connections(self, matrix, idx, connection_type):
        """Ensure node has at least one connection after modification."""
        if connection_type == 'child':
            # Check if node has at least one child
            if idx < matrix.shape[0] - 1 and np.sum(matrix[idx, idx+1:]) == 0:
                # Add one random child
                child_idx = np.random.choice(range(idx + 1, matrix.shape[0]))
                matrix[idx, child_idx] = 1
        else:  # parent
            # Check if node has at least one parent
            if idx > 0 and np.sum(matrix[:idx, idx]) == 0:
                # Add one random parent
                parent_idx = np.random.choice(range(idx))
                matrix[parent_idx, idx] = 1
        
        return matrix

    def modification(self, modif, idx, inter):
        assert modif in ['add', 'delete', 'modify', 'children', 'parents'], f"""Modification should be in ['add', 
        'delete', 'modify', 'children', 'parent'], got{modif} instead"""
        if modif == "add":  # Add new node after the one selected
            idxx = idx + 1
            new_node = self.target.operations.value.random(1)
            inter.operations.insert(idxx, new_node)
            N = len(inter.operations)
            
            # Créer exactement UNE connexion entrante
            parents = np.zeros(idxx, dtype=int)
            parent_idx = np.random.choice(idxx)
            parents[parent_idx] = 1
            
            # Créer exactement UNE connexion sortante (si possible)
            children = np.zeros(N - idxx - 1, dtype=int)
            if N - idxx - 1 > 0:
                child_idx = np.random.choice(N - idxx - 1)
                children[child_idx] = 1
            
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
            # Ensure all nodes have at least one parent and one child
            for i in range(inter.matrix.shape[0] - 1):
                if np.sum(inter.matrix[i, i+1:]) == 0:
                    child_idx = np.random.choice(range(i + 1, inter.matrix.shape[0]))
                    inter.matrix[i, child_idx] = 1
            for j in range(1, inter.matrix.shape[1]):
                if np.sum(inter.matrix[:j, j]) == 0:
                    parent_idx = np.random.choice(range(j))
                    inter.matrix[parent_idx, j] = 1
            inter.operations.pop(idx)
        elif modif == "modify":  # Modify node operation
            inter.operations[idx] = self.target.operations.value.neighbor(inter.operations[idx])
        elif modif == "children":  # Modify ONE child connection
            inter.matrix = self._flip_single_connection(inter.matrix, idx, 'child')
            inter.matrix = self._ensure_valid_connections(inter.matrix, idx, 'child')
            # Ensure only the modified child has at least one parent (if needed)
            # On cherche quel enfant a été modifié
            for j in range(idx + 1, inter.matrix.shape[1]):
                if np.sum(inter.matrix[:j, j]) == 0:
                    # Ce nœud n'a plus de parent, on lui en ajoute un
                    parent_idx = np.random.choice(range(j))
                    inter.matrix[parent_idx, j] = 1
                    break  # On ne vérifie que le nœud concerné
        elif modif == "parents":  # Modify ONE parent connection
            inter.matrix = self._flip_single_connection(inter.matrix, idx, 'parent')
            inter.matrix = self._ensure_valid_connections(inter.matrix, idx, 'parent')
            # Ensure only the modified parent has at least one child (if needed)
            # On cherche quel parent a été modifié
            for i in range(idx):
                if np.sum(inter.matrix[i, i+1:]) == 0:
                    # Ce nœud n'a plus d'enfant, on lui en ajoute un
                    child_idx = np.random.choice(range(i + 1, inter.matrix.shape[0]))
                    inter.matrix[i, child_idx] = 1
                    break  # On ne vérifie que le nœud concerné
            inter.matrix[-2, -1] = 1
        # Reconstruct nodes:
        for j in range(1, len(inter.operations)):
            if hasattr(inter.operations[j], "modification"):
                if hasattr(inter.operations[j], "input_shapes"):
                    input_shapes = [inter.operations[i].output_shape for i in range(j) if inter.matrix[i, j] == 1]
                    inter.operations[j].modification(input_shapes=input_shapes)
        return inter