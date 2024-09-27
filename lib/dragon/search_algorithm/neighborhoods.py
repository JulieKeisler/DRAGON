import copy
import random
import numpy as np

import warnings
from dragon.search_space.addons import VarNeighborhood
from dragon.search_space.zellij_variables import Block, CatVar, IntVar, Constant
from dragon.search_algorithm.zellij_neighborhoods import ConstantInterval
from dragon.search_space.cells import Node
from dragon.utils.tools import logger

from dragon.search_space.dragon_variables import EvoDagVariable, HpVar, NodeVariable
from dragon.utils.exceptions import InvalidArgumentError

warnings.filterwarnings("ignore")

def int_neighborhood(b_min, b_max, scale=4):
    return np.ceil(max(int((b_max - b_min) / scale), 2))

class EvoDagInterval(VarNeighborhood):
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
            f"Target object must be a `EvoDagVariable` for {self.__class__.__name__},\
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
                input_shapes = [inter.operations[i].output_shape for i in range(j) if inter.matrix[i, j] == 1]
                inter.operations[j].modification(input_shapes=input_shapes)
        return inter
    

class NodeInterval(VarNeighborhood):
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
            node.modification(**changed)
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
            f"Target object must be a `NodeVariable` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

class CatHpInterval(VarNeighborhood):
    def __init__(self, neighborhood=None, variable=None):
        super(CatHpInterval, self).__init__(variable)
        if neighborhood is None:
            neighborhood = 0.9
        self._neighborhood = neighborhood

    def __call__(self, name, hp, size=1, *kwargs):
        if size > 1:
            res = []
            for _ in range(size):
                p = random.uniform()
                if p>self._neighborhood:
                    # Draw completely new layer with a probability of p
                    new_layer = self._target.neighbor(name, hp)
                else:
                    # Neighbor of layer
                    for f in self._target.features:
                        assert isinstance(f, HpVar), f"Target features should be of type HpVar but got {f} instead."
                        assert isinstance(f.name, CatVar) or isinstance(f.name, Constant), f"Target features should have name argument of isntance Constant or CatVar but got {f.name} instead."

                        if isinstance(f.name, CatVar):
                            bool = name in f.name.features
                        elif isinstance(f, Constant):
                            bool = name == f.name.value
                        
                        if bool:
                            new_layer = f.neighbor(name, hp)
                            break
                res.append(new_layer)
            return res
        else:
            p = np.random.uniform()
            if p>self._neighborhood:
                # Draw completely new layer with a probability of p
                new_layer = self._target.neighbor(name, hp)
            else:
                # Neighbor of layer
                for f in self._target.features:
                    assert isinstance(f, HpVar), f"Target features should be of type HpVar but got {f} instead."
                    assert isinstance(f.name, Constant) or isinstance(f.name, CatVar), f"Target features should have name argument of isntance Constant or CatVar but got {f.name} instead."
                    if isinstance(f.name, CatVar):
                        bool = name in f.name.features
                    elif isinstance(f.name, Constant):
                        bool = name == f.name.value
                    if bool:
                        new_layer = f.neighbor(name, hp)
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
            f"Target object must be a `CatInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class HpInterval(VarNeighborhood):
    def __init__(self, neighborhood=None, variable=None):
        super(HpInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, name, hp, size=1, *kwargs):
        if size > 1:
            res = []
            for _ in range(size):
                new_hp = hp.copy()
                new_name = name
                hp_index = list(set(np.random.choice(range(len(hp.keys())+1), size=len(hp.keys())+1)))
                for idx in hp_index:
                    if idx >= len(hp.keys()):
                        if hasattr(self._target.name, "neighbor"):
                            new_name = self._target.name.neighbor(name)
                    else:
                        h = list(hp.keys())[idx]
                        new_hp[h] = self._target.hyperparameters[h].neighbor(hp[h])
                res.append([new_name, new_hp])
            return res
        else:
            new_hp = hp.copy()
            new_name = name
            hp_index = list(set(np.random.choice(range(len(hp.keys())+1), size=len(hp.keys())+1)))
            for idx in hp_index:
                if idx >= len(hp.keys()):
                    if hasattr(self._target.name, "neighbor"):
                        new_name = self._target.name.neighbor(name)
                else:
                    h = list(hp.keys())[idx]
                    new_hp[h] = self._target.hyperparameters[h].neighbor(hp[h])
            return [new_name, new_hp]

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        assert isinstance(neighborhood, list) or neighborhood is None, logger.error(
            f"Layers neighborhood must be a list of weights, got {neighborhood}"
        )
        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, HpVar) or variable is None, logger.error(
            f"Target object must be a `HpVar` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable  