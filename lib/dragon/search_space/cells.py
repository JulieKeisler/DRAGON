import os
import torch
import torch.nn as nn
import numpy as np
from dragon.utils.tools import logger
from dragon.utils.exceptions import InvalidArgumentError

class AdjMatrix(nn.Module):
    """AdjMatrix(nn.Module)

    The class `AdjMatrix` is the implementation of the Directed Acyclic Graphs as an adjacency matrix combined with a list of nodes.

    Parameters
    ----------
    operations : list
        List of nodes, ie: the operations that would be performed within the graph.
    matrix: np.array
        Adjacency matrix. The order of the operations and adjacency matrix's entries should be the same.
    """
    def __init__(self, operations, matrix):
        super(AdjMatrix, self).__init__()
        self.matrix = matrix
        self.operations = operations
        self.assert_adj_matrix()

    def assert_adj_matrix(self):
        assert isinstance(self.operations, list), f"""Operations should be a list, got {self.operations} instead."""
        assert isinstance(self.matrix, np.ndarray) and (self.matrix.shape[0] == self.matrix.shape[1]), f"""Matrix should be a 
        squared array. Got {self.matrix} instead."""
        assert self.matrix.shape[0] == len(
            self.operations), f"""Matrix and operations should have the same dimension got {self.matrix.shape[0]} 
                and {len(self.operations)} instead. """
        assert np.sum(np.triu(self.matrix, k=1) != self.matrix) == 0, f"""The adjacency matrix should be upper-triangular with 0s on the
        diagonal. Got {self.matrix}. """
        for i in range(self.matrix.shape[0] - 1):
            assert sum(self.matrix[i]) > 0, f"""Node {i} does not have any child."""
        for j in range(1, self.matrix.shape[1]):
            assert sum(self.matrix[:, j]) > 0, f"""Node {j} does not have any parent."""

    def copy(self):
        new_op = self.operations.copy()
        new_matrix = self.matrix.copy()
        return AdjMatrix(new_op, new_matrix)
    
    def set(self, input_shape):
        # Set the first layer of the DAG
        self.operations[0].set(input_shape)

        # Set the other layers of the DAG
        for j in range(1, len(self.operations)):
            input_shapes = [self.operations[i].output_shape for i in range(j) if self.matrix[i, j] == 1]
            self.operations[j].set(input_shapes)
        
        self.layers = nn.ModuleList(self.operations)
        self.output_shape = self.operations[-1].output_shape

    def forward(self, X):
        device = X.get_device()
        N = len(self.layers)
        outputs = np.empty(N, dtype=object)
        outputs[0] = X
        for j in range(1, N):
            inputs = [outputs[i] for i in range(j) if self.matrix[i, j] == 1]
            output = self.layers[j](inputs)
            if device >= 0:
                try:
                    output = output.to(device)
                except Exception as e:
                    logger.error(f'j={j}, layer: {self.layers[j]}, Output: {output}')
            outputs[j] = output
        return output


    def __str__(self):
        if hasattr(self, "layers"):
            return self.layers.__str__()
        else:
            matrix_str = f"NODES: {self.operations.__str__()} | MATRIX:{self.matrix.tolist().__str__()}"
            return matrix_str

    def __repr__(self):
        if hasattr(self, "layers"):
            return self.layers.__repr__()
        else:
            matrix_repr = f"NODES: {self.operations.__repr__()} | MATRIX:{self.matrix.tolist().__repr__()}"
            return matrix_repr
    

class Node(nn.Module):
    def __init__(self, combiner, name, hp, activation=nn.Identity(), input_comp="Pad"):
        super(Node, self).__init__()
        assert combiner in ['add', 'concat', 'mul'], f"Invalid combiner argument, got: {combiner}."
        self.combiner = combiner
        self.name = name
        self.hp = hp
        self.input_comp = input_comp
        
        self.activation = activation

    def copy(self):
        args = {"combiner": self.combiner, "name": self.name, "hp": self.hp, "activation": self.activation}
        new_node = Node(**args)
        if hasattr(self, "input_shape"):
            new_node.set_operation([self.input_shape], self.device)
        return new_node

    def set_operation(self, input_shapes, device=None):
        assert isinstance(input_shapes, list), f'Input shapes should be a list.'
        self.input_shapes = input_shapes
        self.input_shape = self.compute_input_shape(input_shapes)
        try:
            self.operation = self.name(self.input_shape, **self.hp)
        except Exception as e:
            logger.error(f'Name: {self.name}, hp: {self.hp}, input_shape: {self.input_shape}')
            raise e
        for n, p in self.operation.named_parameters():
            if "theta" not in n:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if device is not None:
            self.operation = self.operation.to(device)
            self.device = device
        else:
            self.device = "cpu"
        try:
            self.output_shape = self.compute_output_shape()
        except Exception as e:
            raise e
        
    def set(self, input_shapes):   
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        if hasattr(self, "operation"):# The layer has already been initialized and trained
            self.modification(input_shapes=input_shapes) # We only update the input shape
        else:
            self.set_operation(input_shapes=input_shapes)# We set the layer with the input shape


    def forward(self, X, h=None):
        X = self.combine(X)
        try:
            X = self.operation(X, h)
        except Exception as e:
            X = self.operation(X)
        if isinstance(X, tuple):
            X, h = X
            X = self.activation(X)
            return X, h
        else:
            X = self.activation(X)
            return X
    
    def compute_input_shape(self, input_shapes):
        if self.combiner in ["add", "mul"]:
            if self.input_comp == "Crop":
                return tuple(np.mean(input_shapes, axis=0).astype(int))
            elif self.input_comp == "Pad":
                return tuple(np.max(input_shapes, axis=0).astype(int))
        elif self.combiner=="concat":
            channels = [shape[-1] for shape in input_shapes]
            others = [shape[:-1] for shape in input_shapes]
            if self.input_comp == "Crop":
                return tuple(np.mean(others, axis=0).astype(int)) + (np.sum(channels),)
            elif self.input_comp == "Pad":
                return tuple(np.max(others, axis=0).astype(int)) + (np.sum(channels),)
        else:
            raise InvalidArgumentError('Combiner', self.combiner, input_shapes)
            
    def combine(self, X):
        if isinstance(X, list):
            assert self.combiner in ['concat', 'add', 'mul'], f"Invalid combiner: {self.combiner}"
            if self.combiner == "concat":
                X_pad = self.padding(X, start=-2, pad_start=(0,0))
                X = torch.cat(X_pad, dim=-1)
                return X
            elif self.combiner == "add":
                X = self.padding(X)
                return sum(X)
            elif self.combiner == "mul":
                X = self.padding(X)
                X_mul = X[0]
                for i in range(1, len(X)):
                     X_mul *= X[i]
                return X_mul
        else:
            return self.padding(X)

    def padding(self, X, start=-1, pad_start=()):
        if isinstance(X, list):
            pad_X = []
            for x in X:
                pad = pad_start
                for i in range(start, -(len(self.input_shape)+1), -1):
                    diff = self.input_shape[i] - x.shape[i]
                    sign = diff / np.abs(diff) if diff !=0 else 1
                    pad = pad + (int(sign * np.ceil(np.abs(diff)/2)),
                            int(sign * np.floor(np.abs(diff))/2))
                pad = nn.functional.pad(x, pad)
                pad_X.append(pad)
        else:
            pad=pad_start
            for i in range(start, -(len(self.input_shape)+1), -1):
                diff = self.input_shape[i] - X.shape[i]
                sign = diff / np.abs(diff) if diff !=0 else 1
                pad = pad + (int(sign * np.ceil(np.abs(diff)/2)),
                        int(sign * np.floor(np.abs(diff))/2))
            pad_X = nn.functional.pad(X, pad)
        return pad_X
    
    def compute_output_shape(self):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Set CuBLAS_WORKSPACE_CONFIG
        X = torch.zeros((2, ) + self.input_shape)
        if torch.cuda.is_available():
            model = self.to('cuda')
            X = X.to('cuda')
        else:
            model = self.to('cpu')
            X = X.to('cpu')
        out = model.forward(X)
        if isinstance(out, tuple):
            out, h = out
        shape = tuple(out.shape[1:])
        return shape
    
    def modification(self, combiner=None, name=None, hp=None, input_shapes=None, device=None):

        if device is not None:
            self.device = device
        if combiner is not None:
            self.combiner = combiner
        if input_shapes is None:
            input_shapes = self.input_shapes
        assert isinstance(input_shapes, list), f'Input shapes should be a list.'
        self.input_shapes = input_shapes
        self.input_shape = self.compute_input_shape(input_shapes)
        if ((name is None) or (name == self.name)) and ((hp is None) or (hp == self.hp)):
            self.modify_operation(self.input_shape)
        else:
            if name is not None:
                self.name = name
            if hp is not None: 
                self.hp = hp
            self.operation = self.name(self.input_shape, **self.hp)
            for n, p in self.operation.named_parameters():
                if "theta" not in n:
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            self.operation = self.operation.to(self.device)
        self.output_shape = self.compute_output_shape()


    def modify_operation(self, input_shape):
        self.operation.modify_operation(input_shape)

    def __repr__(self):
        line = "\n"
        if hasattr(self, "input_shape"):
            line += f"(input shape) {self.input_shape} -- "
        line += f"(combiner) {self.combiner} -- "
        if hasattr(self, "operation"):
            line += f"(op) {self.operation.__repr__()} -- "
        else:
            line += f"(name) {self.name} -- "
            line += f"(hp) {self.hp} -- "
        line += f"(activation) {self.activation} -- "
        if hasattr(self, "output_shape"):
            line += f"(output shape) {self.output_shape}"
        return line
                
    def load_state_dict(self, state_dict, **kwargs):
        try:
            super(Node, self).load_state_dict(state_dict, **kwargs)
        except Exception as e:
            new_dict = {}
            prefix = "operation."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]  # Retirer le préfixe
                    new_dict[new_key] = value
            self.operation.load_state_dict(new_dict, **kwargs)


def fill_adj_matrix(matrix):
    """fill_adj_matrix(matrix)
    Add random edges into an adjacency matrix in case it contains orphan nodes (no incoming connection) or nodes having no outgoing connection.
    Except from the first node, all nodes should have at least one incoming connection, meaning the corresponding column should not sum to zero.
    Except from the last node, all nodes should have at least one outgoing connection, meaning the corresponding row should not sum to zero.

    Parameters
    ----------
    matrix : np.array
        Adjacency matrix from a directed acyclic graph that may contain orphan nodes.
    
    Returns
    -------
        matrix: adjacency matrix from a directed acyclic graph that does not contain orphan nodes.
    """
    # Add outgoing connections if needed.
    for i in range(matrix.shape[0] - 1):
        new_row = matrix[i, i + 1:]
        while sum(new_row) == 0:
            new_row = np.random.choice(2, new_row.shape[0])
        matrix[i, i + 1:] = new_row
    # Add incoming connections if needed.
    for j in range(1, matrix.shape[1]):
        new_col = matrix[:j, j]
        while sum(new_col) == 0:
            new_col = np.random.choice(2, new_col.shape[0])
        matrix[:j, j] = new_col
    return matrix

class Brick(nn.Module):
    def __init__(self, input_shape, **args):
        super().__init__()
        self.input_shape = input_shape

    def foward(self, X):
        raise NotImplementedError
    
    def modify_operation(self, input_shape):
        raise NotImplementedError

    def pad(self, X): 
        raise NotImplementedError
    