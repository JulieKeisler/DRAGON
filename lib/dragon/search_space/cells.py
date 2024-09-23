import os
import torch
import torch.nn as nn
import numpy as np
from dragon.utils.tools import logger
from dragon.utils.exceptions import InvalidArgumentError


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

class WeightsAdjCell(nn.Module):
    def __init__(self, adj_matrix):
        super(WeightsAdjCell, self).__init__()
        self.matrix = adj_matrix.matrix
        self.layers = nn.ModuleList(adj_matrix.operations)
        self.output_shape = adj_matrix.operations[-1].output_shape

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

def set_node(node, input_shapes):
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]
    if hasattr(node, "operation"):# The layer has already been initialized and trained
        node.modification(input_shapes=input_shapes) # We only update the input shape
    else:
        node.set_operation(input_shapes=input_shapes)# We set the layer with the input shape

    
def set_cell(cell, input_shape):
    # Set the first layer of the DAG
    set_node(cell.operations[0], input_shape)
         
    # Set the other layers of the DAG
    for j in range(1, len(cell.operations)):
        input_shapes = [cell.operations[i].output_shape for i in range(j) if cell.matrix[i, j] == 1]
        set_node(cell.operations[j], input_shapes)