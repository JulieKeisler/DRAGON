import os
import torch
import torch.nn as nn
import numpy as np
from dragon.utils.tools import logger
from dragon.utils.exceptions import InvalidArgumentError


class Brick(nn.Module):
    """Brick(nn.Module)

    The Meta class `Brick` serves as a basis to incorporate the `nn.Module` layers from PyTorch into DRAGON.
    In addition to the `__init__` and `forward` functions, they should have a method to modify the layer given an input shape.
    The `**args` correspond to the layer hyperparameters.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor.
    """
    def __init__(self, input_shape, **args):
        super().__init__()
        self.input_shape = input_shape

    def foward(self, X):
        """forward(X)

        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.
        """
        raise NotImplementedError
    
    def modify_operation(self, input_shape):
        """modify_operation(input_shape)

        Modify the operation so it can take a tensor of shape `input_shape` as input.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
        raise NotImplementedError

class Node(nn.Module):
    """Node(nn.Module)

    The class `Node` is the implementation of a DAG node. Each node is made of a combiner, an operation and an activation function.
    The operation is parametrized by a set of hyperparameters.

    Parameters
    ----------
    combiner: str
        Name of the combiner. The only combiner implemented for now within DRAGON are: 'add', 'concat' and 'mul'
    operation: `Brick`
        Operation that will be performed within the node.
    hp: dict
        Dictionary containing the hyperparameters. The keys name should match the arguments of the `operation` class `__init__` function.
    activation: nn.Module, default=nn.Identity()
        Activation function.
    input_comp: ['Pad', 'Crop'], default='Pad'
        Defines how the combiner will compute the input shape.
        When set to 'Pad', the maximum input shape from all the incoming tensors will be taken.
        When set to 'Crop' the mean input shape will be taken.

    Examples
    ---------

    >>> import torch.nn as nn
    >>> from dragon.search_space.bricks import MLP
    >>> from dragon.search_space.dragon_variables import Node
    >>> print(Node(combiner="add", operation=MLP, hp={"out_channels": 10}, activation=nn.ReLU()))
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) ReLU() -- 
    """
    def __init__(self, combiner, operation, hp, activation=nn.Identity(), input_comp="Pad"):
        super(Node, self).__init__()
        assert combiner in ['add', 'concat', 'mul'], f"Invalid combiner argument, got: {combiner}."
        self.combiner = combiner
        self.name = operation
        self.hp = hp
        self.activation = activation
        self.input_comp = input_comp
    

    def copy(self):
        """copy()
        Creates an new `Node` variable which is a copy of this one.

        Returns
        -------
        new_node: `Node`
            Copy of the actual Node.
        """
        args = {"combiner": self.combiner, "operation": self.name, "hp": self.hp, "activation": self.activation}
        new_node = Node(**args)
        if hasattr(self, "input_shape"):
            new_node.set_operation([self.input_shape], self.device)
        return new_node

    def set_operation(self, input_shapes, device=None):
        """set_operation(input_shapes, device=None)

        Initialize the operation using the new input shapes.
        First, the global input shape of the operation is computed, using the `combiner` type and the `input_comp` attribute
        Then, the operation is initialized with the global input shape and the hyperparameters. The operation parameters are modified with the xavier_uniform initialization.
        Finally, the node ouput shape is computed.

        Parameters
        ----------
        input_shapes: list, tuple or int
            Input shapes of the multiple (or single) input vectors of the node.
        device: str, default=None
            Device on which the node operation should be computed.
        """
        # Convert the input_shapes to a list
        if isinstance(input_shapes, tuple):
            input_shapes = [input_shapes]
        elif isinstance(input_shapes, int):
            input_shapes = [(input_shapes,)]

        self.input_shapes = input_shapes
        # Compute global input shape with the combiner and self.input_comp
        self.input_shape = self.compute_input_shape(input_shapes)
        # Initialize the operation
        self.operation = self.name(self.input_shape, **self.hp)
        # Initialize the weights
        for n, p in self.operation.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        # Pass the operation on the device
        if device is not None:
            self.operation = self.operation.to(device)
            self.device = device
        else:
            self.device = "cpu"
        # Compute the layer output shape
        self.output_shape = self.compute_output_shape()

    def compute_input_shape(self, input_shapes):
        """compute_input_shape(X, h=None)

        Compute the global input shape for the operation, given the (possibly) multiple input shapes.
        The global shape depends on the combiner type and the value of `self.input_comb`.

        Parameters
        ----------
        input_shapes: list
            List containing the input shapes of the different input tensors.
        Returns
        -------
            tuple
        """
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
        """combine(X)

        Use the combiner to combine the input vectors. First the vectors are modified to have the global input shape using the `self.padding` function.
        Then they are combined by addition, multiplication or concatenation.

        Parameters
        ----------
        X: list
            List containing the input tensors.
        Returns
        -------
            `torch.Tensor`
        """
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
        """padding(X, start=-1, pad_start=())

        Modify the input tensors gathered in X so they all have the global input shape.
        The padding is performed over all dimensions for the 'add' and 'mul' combiners, but not on the last one for the 'concat' combiner.

        Parameters
        ----------
        X: list or torch.Tensor
            List containing the input tensors.
        start: int, default=-1
            Dimension where to start the padding. It depends on the combiner.
        pad_start: tuple, default=()
            Default padding over the last dimension. It depends on the combiner.
        Returns
        -------
            pad_X: list
                List containing the tensors with the right shape.
        """
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
        """compute_output_shape()

        Compute the output shape of the node. A fake vector is created with a shape equals to the global input shape.
        This fake vector is processed by the operation. The output vector shape will be the node output shape.

        Returns
        -------
            shape: tuple
                The node output shape.
        """
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # We create an fake vector with the global input shape, and a batch of size two (a batch size > 1 is required by some operations)
        X = torch.zeros((2, ) + self.input_shape)
        if torch.cuda.is_available():
            model = self.to('cuda')
            X = X.to('cuda')
        else:
            model = self.to('cpu')
            X = X.to('cpu')
        # We pass this fake vector to the operation.
        out = model.forward(X)
        # We return the ouput shape
        if isinstance(out, tuple):
            out, h = out
        shape = tuple(out.shape[1:])
        return shape
    
    def modification(self, combiner=None, operation=None, hp=None, input_shapes=None, device=None):
        """modification(combiner=None, name=None, hp=None, input_shapes=None, device=None)

        Modify the node. The modifications can be applied to the combiner, the operation, the operation's hyperparameters or the input shapes.
        The values set to None will stay unchanged.
        If the operation and the hyperparameters do not change, the operation is just modified. Otherwise a new one will be created.

        Parameters
        ----------
        combiner: str, default=None
            Name of the new combiner.
        operation: `Brick`, default=None
            New operation that will be performed within the node.
        hp: dict, default=None
            Dictionary containing the new hyperparameters. The keys name should match the arguments of the `operation` class `__init__` function.
        input_shape: list, tuple or int, default=None
            List of the new input shapes.
        device: str, default=None
            Name of the device where the node is computed.
        """
        if device is not None:
            self.device = device
        if combiner is not None:
            self.combiner = combiner
        if input_shapes is None:
            input_shapes = self.input_shapes
        # Convert the input_shapes to a list
        if isinstance(input_shapes, tuple):
            input_shapes = [input_shapes]
        elif isinstance(input_shapes, int):
            input_shapes = [(input_shapes,)]
        self.input_shapes = input_shapes

        # Compute the new input shapes
        self.input_shape = self.compute_input_shape(input_shapes)

        # If the operation and the hyperparameters do not change, they are just modified.
        if ((operation is None) or (operation == self.name)) and ((hp is None) or (hp == self.hp)):
            self.modify_operation(self.input_shape)
        
        # Otherwise, a new one is created.
        else:
            if operation is not None:
                self.name = operation
            if hp is not None: 
                self.hp = hp
            self.operation = self.name(self.input_shape, **self.hp)
            for n, p in self.operation.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.operation = self.operation.to(self.device)
        
        # We compute the new output shape.
        self.output_shape = self.compute_output_shape()


    def modify_operation(self, input_shape):
        """modify_operation(input_shape)

        Modify the operation so it can take as input a tensor of shape `input_shape`.

        Parameters
        ----------
        input_shape: tuple
            New input shape.
        """
        self.operation.modify_operation(input_shape)

    def set(self, input_shapes):
        """set(input_shapes)

        Initialize or modify the node with the incoming shapes `input_shape`s .
        
        Parameters
        ----------
        input_shapes: list, tuple or int
            Input shapes of the multiple (or single) input vectors of the node.
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        if hasattr(self, "operation"): # The layer has already been initialized and trained
            self.modification(input_shapes=input_shapes) # We only update the input shape
        else:
            self.set_operation(input_shapes=input_shapes)# We set the layer with the input shape



    def forward(self, X, h=None):
        """forward(X, h=None)

        Forward pass of the layer. The inputs are first combined by the combiner, then processed by the operation and the activation function.

        Parameters
        ----------
        X: torch.Tensor or list
            Input tensor or list of input tensors.
        h: torch.Tensor, default=None
            Hidden state, used in the case of recurrent layer.
        Returns
        -------
        X or (X,h): `torch.Tensor`
            Processed tensor(s).
        """
        # Combiner
        X = self.combine(X)
        # Operation: we first test if the operation is recurrent, if not an exception is catched.
        try:
            X = self.operation(X, h)
        except Exception as e:
            X = self.operation(X)
        # The output tensor is processed by an activation function.
        if isinstance(X, tuple):
            X, h = X
            X = self.activation(X)
            return X, h
        else:
            X = self.activation(X)
            return X

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


class AdjMatrix(nn.Module):
    """AdjMatrix(nn.Module)

    The class `AdjMatrix` is the implementation of an Directed Acyclic Graph (DAG) using its adjacency matrix combined with the nodes list.

    Parameters
    ----------
    operations : list
        List of nodes, ie: the operations that would be performed within the graph.
    matrix: np.array
        Adjacency matrix. The order of the operations and adjacency matrix's entries should be the same.
    
    Examples
    --------
    >>> import numpy as np
    >>> from dragon.search_space.dragon_variables import AdjMatrix
    >>> import torch.nn as nn
    >>> from dragon.search_space.bricks import MLP, Identity
    >>> from dragon.search_space.dragon_variables import Node
    >>> node_1 = Node(combiner="add", operation=MLP, hp={"out_channels": 10}, activation=nn.ReLU())
    >>> node_2 = Node(combiner="add", operation=MLP, hp={"out_channels": 5}, activation=nn.ReLU())
    >>> node_3 = Node(combiner="concat", operation=Identity, hp={}, activation=nn.Softmax())
    >>> operations = [node_1, node_2, node_3]
    >>> matrix = np.array([[0, 1, 1],
                           [0, 0, 1],
                           [0, 0, 0]])
    >>> print(AdjMatrix(operations, matrix))
    NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) ReLU() -- , 
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 5} -- (activation) ReLU() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Softmax(dim=None) -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    """

    def __init__(self, operations, matrix):
        super(AdjMatrix, self).__init__()
        self.matrix = matrix
        self.operations = operations
        self.assert_adj_matrix()

    def assert_adj_matrix(self):
        """ assert_adj_matrix()
        The `operations` and `matrix` variables should verify some properties such as:
            - The `operations` variable should be a list.
            - The `matrix` variable should be a squared upper-triangular numpy array filled with 0s on the diagonal.
            - The `matrix` variable should not contain empty rows beside the last one and empty columns beside the first one. It would indeed emply nodes without incoming or outgoing connections.
            - The `matrix` variable and the :node: operations variable should have the same dimension.
        """
        assert isinstance(self.operations, list), f"""Operations should be a list, got {self.operations} instead."""
        assert isinstance(self.matrix, np.ndarray) and (self.matrix.shape[0] == self.matrix.shape[1]), f"""Matrix should be a 
        squared array. Got {self.matrix} instead."""
    
        assert np.sum(np.triu(self.matrix, k=1) != self.matrix) == 0, f"""The adjacency matrix should be upper-triangular with 0s on the
        diagonal. Got {self.matrix}. """
        for i in range(self.matrix.shape[0] - 1):
            assert sum(self.matrix[i]) > 0, f"""Node {i} does not have any outgoing connections."""
        for j in range(1, self.matrix.shape[1]):
            assert sum(self.matrix[:, j]) > 0, f"""Node {j} does not have any incoming connections."""
        assert self.matrix.shape[0] == len(
            self.operations), f"""Matrix and operations should have the same dimension got {self.matrix.shape[0]} 
                and {len(self.operations)} instead. """

    def copy(self):
        """copy()
        Creates an new `AdjMatrix` variable which is a copy of this one.

        Returns
        -------
        adj_matrix: `AdjMatrix`
            Copy of the actual variable.

        """
        new_op = self.operations.copy()
        new_matrix = self.matrix.copy()
        adj_matrix = AdjMatrix(new_op, new_matrix)
        return adj_matrix
    
    def set(self, input_shape):
        """set(input_shape)

        Initialize the `nn.Module` within the `operations` list, with the new input shape.
        If the layers have already been initialized, they may be modified if the `input_shape` has changed since their initialization.
        The layers are initialized or modified one after the other, in the :node: operations list order.

        Parameters
        ----------
        input_shape : int or tuple
            Shape of the DAG's input tensor.
        """
        # Set the first layer of the DAG
        self.operations[0].set(input_shape)

        # Set the other layers of the DAG
        for j in range(1, len(self.operations)):
            input_shapes = [self.operations[i].output_shape for i in range(j) if self.matrix[i, j] == 1]
            self.operations[j].set(input_shapes)
        
        self.layers = nn.ModuleList(self.operations)
        self.output_shape = self.operations[-1].output_shape

    def forward(self, X):
        """forward(X)

        Forward pass through the DAG. The latent vectors are processed layer by layer, following the :node: operations list order.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        output: `torch.Tensor`
            Network output tensor.
        """
        device = X.get_device()
        N = len(self.layers)
        # Store the outputs of each layer.
        outputs = np.empty(N, dtype=object)
        outputs[0] = X
        for j in range(1, N):
            # Get the inputs from the different incoming connections of layer j
            inputs = [outputs[i] for i in range(j) if self.matrix[i, j] == 1]
            # Compute the layer output
            output = self.layers[j](inputs)
            if device >= 0:
                # Make sure the output is on the right device
                output = output.to(device)
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
    
def fill_adj_matrix(matrix):
    """fill_adj_matrix(matrix)
    Add random edges into an adjacency matrix in case it contains orphan nodes (no incoming connection) or nodes having no outgoing connection.
    Except from the first node, all nodes should have at least one incoming connection, meaning the corresponding column should not sum to zero.
    Except from the last node, all nodes should have at least one outgoing connection, meaning the corresponding row should not sum to zero.

    Parameters
    ----------
    matrix : np.array
        Adjacency matrix from a DAG that may contain orphan nodes.
    
    Returns
    -------
        matrix: `np.array`
            Adjacency matrix from a DAG that does not contain orphan nodes.
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
