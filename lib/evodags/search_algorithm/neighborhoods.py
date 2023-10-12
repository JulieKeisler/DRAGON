import random
import numpy as np

import warnings
from zellij.core.addons import VarNeighborhood, logger
from zellij.core.variables import CatVar, IntVar, Constant
from zellij.utils.neighborhoods import ConstantInterval

from evodags.search_space.dags import AdjMatrixVariable
from evodags.utils.exceptions import InvalidArgumentError

warnings.filterwarnings("ignore")


class AdjMatrixHierarchicalInterval(VarNeighborhood):
    def __init__(self, neighborhood=None, variable=None):
        super(AdjMatrixHierarchicalInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, value, size=1, neigh="local"):
        if size == 1:
            inter = value.copy()
            if neigh == "large":
                variable_idx = list(set(np.random.choice(range(len(inter.operations)), size=len(inter.operations))))
            elif neigh == "local":
                variable_idx = list(
                    set(np.random.choice(range(1, len(inter.operations)), size=len(inter.operations) - 1)))
            modifications = []
            variable_idx.sort()
            for i in range(len(variable_idx)):
                idx = variable_idx[i]
                if neigh == "large":
                    choices = ['add', 'delete', 'modify', 'children', 'parents']
                    if idx == 0:
                        choices = ['add', 'children']
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ["children"]
                    elif idx == len(inter.operations) - 1:
                        choices = ['add', 'delete', 'modify', 'parents']
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ['delete', 'modify', 'parents']
                        elif inter.matrix.shape[0] == 2:
                            choices = ['add', 'modify', 'parents']
                    else:
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ['delete', 'modify', 'children', 'parents']
                    modification = random.choice(choices)
                elif neigh == "local":
                    modification = "modify"
                inter = self.modification(modification, idx, inter, neigh)
                modifications.append(modification)
                if modification == "add":
                    variable_idx[i + 1:] = [j + 1 for j in variable_idx[i + 1:]]
                elif modification == "delete":
                    variable_idx[i + 1:] = [j - 1 for j in variable_idx[i + 1:]]
            inter.assert_adj_matrix()
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
                    inter = self.modification(modification, idx, inter, neigh)
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
        assert isinstance(variable, AdjMatrixVariable) or variable is None, logger.error(
            f"Target object must be a `AdjMatrixVariable` for {self.__class__.__name__},\
                 got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self.target.operations.value, "neighbor"), logger.error(
                f"To use `AdjMatrixVariable`, value for operations for `AdjMatrixVariable` must have a `neighbor` method. "
                f"Use `neighbor` kwarg when defining a variable "
            )

    def modification(self, modif, idx, inter, neigh):
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
            inter.matrix[idxx, idxx + 1:] = children
            inter.matrix[:idxx, idxx] = parents
            inter.matrix[-2, -1] = 1  # In case we add a node at the end
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
        return inter


class HierarchicalLayersInterval(VarNeighborhood):
    def __init__(self, neighborhood=None, variable=None):
        super(HierarchicalLayersInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, layer, size=1, neigh="local"):
        if size > 1:
            res = []
            if neigh == "large":
                for _ in range(size):
                    new_layer = self._target.random()
                    while new_layer == layer:
                        v = self._target.random()
                    res.append(v)
            elif neigh == "local":
                type = value_to_layer_type(layer, self._target.features)
                new_layer = type.neighbor(layer)
                res.append(new_layer)
            return res
        else:
            if neigh == 'local':
                type = value_to_layer_type(layer, self._target.features)
                new_layer = type.neighbor(layer)
            elif neigh == 'large':
                new_layer = self._target.random()
                while new_layer == layer:
                    new_layer = self._target.random()
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


def value_to_layer_type(value, layer_types):
    idx_layer = 0
    found = False
    if value[0] == "Input":
        return Constant('Input', "Input", neighbor=ConstantInterval())
    while idx_layer < len(layer_types) and not found:
        if len(value) == len(layer_types[idx_layer].values):
            layer_type = layer_types[idx_layer]
            is_type = True
            idx = 0
            while idx < len(value) and is_type:
                attribute = layer_type[idx]
                if isinstance(attribute, CatVar):
                    is_type = value[idx] in attribute.features
                elif isinstance(attribute, IntVar):
                    is_type = attribute.low_bound <= value[idx] <= attribute.up_bound
                elif isinstance(attribute, Constant):
                    is_type = (attribute.value == value[idx])
                idx += 1
            if is_type:
                return layer_types[idx_layer]
        idx_layer += 1
    raise InvalidArgumentError("Value to layer type", value, layer_types)


class AdjMatrixInterval(VarNeighborhood):
    def __init__(self, neighborhood=None, variable=None):
        super(AdjMatrixInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, value, size=1):
        if size == 1:
            valid = False
            while not valid:
                inter = value.copy()
                variable_idx = list(set(np.random.choice(range(len(inter.operations)), size=len(inter.operations))))
                modifications = []
            
                for i in range(len(variable_idx)):
                    idx = variable_idx[i]
                    choices = ['add', 'delete', 'modify', 'children', 'parents']
                    if idx == 0:
                        choices = ['add', 'children']
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ["children"]
                    elif idx == len(inter.operations) - 1:
                        choices = ['add', 'delete', 'modify', 'parents']
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ['delete', 'modify', 'parents']
                        elif inter.matrix.shape[0] == 2:
                            choices = ['add', 'modify', 'parents']
                    else:
                        if inter.matrix.shape[0] == self.target.max_size:
                            choices = ['delete', 'modify', 'children', 'parents']
                    modification = random.choice(choices)
                    inter = self.modification(modification, idx, inter)
                    modifications.append(modification)
                    if modification == "add":
                        variable_idx[i+1:] = [j + 1 for j in variable_idx[i+1:]]
                    elif modification == "delete":
                        variable_idx[i+1:] = [j - 1 for j in variable_idx[i+1:]]
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
        assert isinstance(variable, AdjMatrixVariable) or variable is None, logger.error(
            f"Target object must be a `AdjMatrixVariable` for {self.__class__.__name__},\
                 got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self.target.operations.value, "neighbor"), logger.error(
                f"To use `AdjMatrixVariable`, value for operations for `AdjMatrixVariable` must have a `neighbor` method. "
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
        return inter


class LayersInterval(VarNeighborhood):
    def __init__(self, neighborhood=None, variable=None):
        super(LayersInterval, self).__init__(variable)
        self._neighborhood = neighborhood

    def __call__(self, layer, size=1, *kwargs):
        if size > 1:
            res = []
            type = value_to_layer_type(layer, self._target.features)
            new_layer = type.neighbor(layer)
            res.append(new_layer)
            return res
        else:
            type = value_to_layer_type(layer, self._target.features)
            new_layer = type.neighbor(layer)
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
