import numpy as np
import torch.nn as nn
from dragon.search_space.zellij_variables import CatVar, Variable, DynamicBlock
from dragon.search_space.cells import Node

from dragon.utils.tools import logger

class AdjMatrix(object):
    def __init__(self, operations, matrix):
        self.matrix = matrix
        self.operations = operations
        self.assert_adj_matrix()

    def assert_adj_matrix(self):
        assert isinstance(self.operations, list), f"""Operations should be a list, got {self.operations} instead."""
        assert isinstance(self.matrix, np.ndarray) and (self.matrix.shape[0] == self.matrix.shape[1]), f"""Matrix should be a 
        squared array. got {self.matrix} instead."""
        assert self.matrix.shape[0] == len(
            self.operations), f"""Matrix and operations should have the same dimension got {self.matrix.shape[0]} 
                and {len(self.operations)} instead. """
        assert np.sum(np.triu(self.matrix, k=1) != self.matrix) == 0, f"""The adjacency matrix should be upper-triangular with 0s on the
        diagonal. Got {self.matrix}. """
        for i in range(self.matrix.shape[0] - 1):
            assert sum(self.matrix[i]) > 0, f"""Node {i} does not have any child."""
        for j in range(1, self.matrix.shape[1]):
            assert sum(self.matrix[:, j]) > 0, f"""Node {j} does not have any parent."""
        try:
            if self.operations[0][0] != "Input":
                logger.error(self.operations)
        except TypeError as e:
            # logger.error(e)
            pass

    def __eq__(self, other):
        if len(self.operations) == len(other.operations):
            return (other.matrix == self.matrix).all() and (
                    sum([other.operations[i] != self.operations[i] for i in range(len(self.operations))]) == 0)
        else:
            return False

    def copy(self):
        new_op = self.operations.copy()
        new_matrix = self.matrix.copy()
        return AdjMatrix(new_op, new_matrix)

    def __str__(self):
        matrix_str = f"NODES: {self.operations.__str__()} | MATRIX:{self.matrix.tolist().__str__()}"
        return matrix_str

    def __repr__(self):
        matrix_repr = f"NODES: {self.operations.__repr__()} | MATRIX:{self.matrix.tolist().__repr__()}"
        return matrix_repr


def fill_adj_matrix(matrix):
    for i in range(matrix.shape[0] - 1):
        new_row = matrix[i, i + 1:]
        while sum(new_row) == 0:
            new_row = np.random.choice(2, new_row.shape[0])
        matrix[i, i + 1:] = new_row
    for j in range(1, matrix.shape[1]):
        new_col = matrix[:j, j]
        while sum(new_col) == 0:
            new_col = np.random.choice(2, new_col.shape[0])
        matrix[:j, j] = new_col
    return matrix


# Directed Acyclic Graph represented by adjacency matrix for NAS
class AdjMatrixVariable(Variable):
    def __init__(self, label, operations, init_complexity=None, **kwargs):
        assert isinstance(operations, DynamicBlock), f"""
        Operations must inherit from `DynamicBlock`, got {operations}
        """
        self.operations = operations
        self.max_size = operations.repeat
        self.complexity = init_complexity
        super(AdjMatrixVariable, self).__init__(label, **kwargs)

    def random(self, size=1):
        """random(size=1)
            Parameters
            ----------
            size : int, default=1
                Number of draws.
            Returns AdjMatrix
            ---------
        """
        operations = self.operations.random()
        if self.complexity is not None:
            operations = operations[: min(self.complexity, len(operations))]
        operations = [['Input']] + operations
        matrix = np.random.randint(0, 2, (len(operations), len(operations)))
        matrix = np.triu(matrix, k=1)
        matrix = fill_adj_matrix(matrix)
        adj_matrix = AdjMatrix(operations, matrix)
        return adj_matrix

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: False
            Return False, a dynamic block cannot be constant. (It is a binary)

        """
        return False

    def subset(self, lower, upper):
        new_values = self.operations.subset(lower, upper)
        return AdjMatrixVariable(self.label, new_values)

    def __repr__(self):
        return (
                super(AdjMatrixVariable, self).__repr__()
                + f"\
        \t- Operations:\n"
                + self.operations.__repr__()
        )


class Operation(Variable):
    def __init__(self, label, combiner, module, activation_function, **kwargs):
        super().__init__(label, **kwargs)
        assert isinstance(combiner, CatVar), f"The combiner should be of type CatVar but got {combiner} instead."
        assert isinstance(module, nn.Module), f"The module should be of type nn.Module but got {module} instead."
        assert isinstance(activation_function, CatVar), f"The activation function should be of type CatVar but got {activation_function} instead."

        self.combiner = combiner
        self.module = module
        self.activation_function = activation_function
        self.hyperparameters = kwargs
        for p in self.hyperparameres:
            assert isinstance(p, Variable), f"The variables should be of Vairables but got {p} instead."
        
    def random(self, size=1):
        c = self.combiner.random()
        f = self.activation_function.random()
        hp = {}
        for p in self.hyperparameters.keys():
            hp[p] = self.hyperparameters[p].random()
        return self.module(**hp)
    

class EvoDagVariable(Variable):
    def __init__(self, label, operations, init_complexity=None, **kwargs):
        assert isinstance(operations, DynamicBlock), f"""
        Operations must inherit from `DynamicBlock`, got {operations}
        """
        self.operations = operations
        self.max_size = operations.repeat
        self.complexity = init_complexity
        super(EvoDagVariable, self).__init__(label, **kwargs)

    def random(self, size=1):
        """random(size=1)
            Parameters
            ----------
            size : int, default=1
                Number of draws.
            Returns AdjMatrix
            ---------
        """
        operations = self.operations.random()
        if self.complexity is not None:
            operations = operations[: min(self.complexity, len(operations))]
        from dragon.search_space.bricks import Identity
        operations = [Node(combiner="add", name=Identity, hp={})] + operations
        matrix = np.random.randint(0, 2, (len(operations), len(operations)))
        matrix = np.triu(matrix, k=1)
        matrix = fill_adj_matrix(matrix)
        adj_matrix = AdjMatrix(operations, matrix)
        return adj_matrix

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: False
            Return False, a dynamic block cannot be constant. (It is a binary)

        """
        return False


    def __repr__(self):
        return (
                super(EvoDagVariable, self).__repr__()
                + f"\
        \t- Operations:\n"
                + self.operations.__repr__()
        )
    
    def subset(self):
        return super().subset()
    

class NodeVariable(Variable):
    def __init__(self, label, combiner, operation, activation_function, **kwargs):
        super().__init__(label, **kwargs)
        assert isinstance(combiner, Variable), f"The combiner should be of type Variable but got {combiner} instead."
        assert isinstance(operation, Variable), f"The operation should be of type Variable but got {operation} instead."
        assert isinstance(activation_function, Variable), f"The activation function should be of type Variable but got {activation_function} instead."

        self.combiner = combiner
        self.operation = operation
        self.activation_function = activation_function
        
    def random(self, size=1):
        if size == 1:
            c = self.combiner.random()
            op = self.operation.random()
            name, hp = op[0], op[1]
            f = self.activation_function.random()
            return Node(combiner=c, name=name, hp=hp, activation=f)
        else:
            res = []
            for _ in range(size):
                c = self.combiner.random()
                op = self.operation.random()
                name, hp = op[0], op[1]
                f = self.activation_function.random()
                res.append(Node(c, name, hp, f))
            return res
    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: False
            Return False, a dynamic block cannot be constant. (It is a binary)

        """
        return self.combiner.isconstant() and self.operation.isconstant() and self.activation_function.isconstant()
    
    def subset(self):
        return super().subset()
    
    def __repr__(self):
        return f"Combiner: {self.combiner.__repr__()} - Operation: {self.operation.__repr__()} - Act. Function: {self.activation_function.__repr__()}"
    
class HpVar(Variable):
    def __init__(self, label, name, hyperparameters, **kwargs):
        super().__init__(label, **kwargs)
        for h in hyperparameters:
            assert isinstance(hyperparameters[h], Variable), f"The hyperparameters should be instances of Variable but got {h} instead."
        self.name = name
        self.label = label
        self.hyperparameters = hyperparameters

    def random(self, size = 1):
        if size == 1:
            if isinstance(self.name, Variable):
                name = self.name.random()
            else:
                name = self.name
            hp = {}
            for h in self.hyperparameters:
                hp[h] = self.hyperparameters[h].random()
            return [name, hp]
        else:
            res = []
            for _ in range(size):
                if isinstance(self.name, Variable):
                    name = self.name.random()
                else:
                    name = self.name
                hp = {}
                for h in self.hyperparameters:
                    hp[h] = h.random(1)
                res.append([name, hp])

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: False

        """
        isconstant = self.name.isconstant()
        for h in self.hyperparameters:
            if not self.hyperparameters[h].isconstant():
                isconstant = False
        return isconstant

    def subset(self, lower, upper):
        pass

