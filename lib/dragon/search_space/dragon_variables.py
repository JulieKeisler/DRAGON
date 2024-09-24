import numpy as np
from dragon.search_space.zellij_variables import Variable, DynamicBlock
from dragon.search_space.cells import AdjMatrix, Node, fill_adj_matrix

# Directed Acyclic Graph represented by adjacency matrix for NAS
class EvoDagVariable(Variable):
    """EvoDagVariable(Variable)

    The class `EvoDagVariable` defines :ref:`var` which represent Directed Acyclic Graph by creating objects from the `AdjMatrix` class.
    The candidate operations should be gathered within a `DynamicBlock`. The maximum size of this `DynamicBlock` will set the graph maximum number of nodes.

    Parameters
    ----------
    label : str
        Name of the variable.
    operations : `DynamicBlock`
        `DynamicBlock` containing :ref:`var` corresponding to the candidate operations.
    init_complexity : int
        Maximum number of nodes that the randomly created DAGs should have.
    """
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

            Create random DAGs. First, a list of random nodes is creating, with a size lower than the :code: complexity attribute. 
            The first element of this list will always be an Identity layer.
            Then, the adjacency matrix is created as an upper-triangle matrix, with the same size as the list.
            This adjacency matrix is corrected using the :code: fill_adj_matrix function to prevent nodes from having no incoming or outgoing connections.
            
            Parameters
            ----------
            size : int, default=1
                Number of draws.

            Returns
            -------
            matrices: list or AdjMatrix
                List containing the randomly created DAGs, or a single DAG if size=1.
        """
        matrices = []
        for _ in range(size):
            operations = self.operations.random()
            if self.complexity is not None:
                operations = operations[: min(self.complexity, len(operations))]
            from dragon.search_space.bricks import Identity
            operations = [Node(combiner="add", name=Identity, hp={})] + operations
            matrix = np.random.randint(0, 2, (len(operations), len(operations)))
            matrix = np.triu(matrix, k=1)
            matrix = fill_adj_matrix(matrix)
            adj_matrix = AdjMatrix(operations, matrix)
            matrices.append(adj_matrix)
        if size == 1:
            return matrices[0]
        return matrices

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
    

class NodeVariable(Variable):
    """NodeVariable(Variable)

    The class `NodeVariable` defines :ref:`var` which represent DAGs nodes by creating objects from the `Node` class.
    
    Parameters
    ----------
    label : str
        Name of the variable.
    operations : `DynamicBlock`
        `DynamicBlock` containing :ref:`var` corresponding to the candidate operations.
    init_complexity : int
        Maximum number of nodes that the randomly created DAGs should have.
    """
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
    
    def __repr__(self):
        return f"Combiner: {self.combiner.__repr__()} - Operation: {self.operation.__repr__()} - Act. Function: {self.activation_function.__repr__()}"
    
class HpVar(Variable):
    """EvoDagVariable(Variable)

    The class `EvoDagVariable` defines :ref:`var` which represent Directed Acyclic Graph by creating objects from the `AdjMatrix` class.
    The candidate operations should be gathered within a `DynamicBlock`. The maximum size of this `DynamicBlock` will set the graph maximum number of nodes.

    Parameters
    ----------
    label : str
        Name of the variable.
    operations : `DynamicBlock`
        `DynamicBlock` containing :ref:`var` corresponding to the candidate operations.
    init_complexity : int
        Maximum number of nodes that the randomly created DAGs should have.
    """
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
