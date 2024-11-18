import random
import numpy as np

from dragon.search_space.addons import Mutator, Crossover
from dragon.search_space.cells import AdjMatrix, fill_adj_matrix


class DAGTwoPoint(Crossover):
    """DAGTwoPoint

    Implementation of a two-point crossover on an array.
    A DAG-based crossover is used if some of the exchanged values are DAGs.

    Parameters
    ----------
    search_space: Variable, array-type
        Variable containing the search space considered.
    
    Examples
    ----------
    >>> from dragon.search_algorithm.operators import DAGTwoPoint
    >>> from dragon.search_space.zellij_variables import Constant, FloatVar, IntVar, CatVar,  ArrayVar
    >>> arr = ArrayVar(IntVar("Number of features", 1, 10), CatVar("Optimizer", features=['Adam', 'SGD', 'AdamMax']), FloatVar('learning rate', 0.001, 0.5), Constant('Seed', value=0))
    >>> parent_1 = arr.random()
    >>> parent_2 = arr.random()
    >>> crossover = DAGTwoPoint(arr)
    >>> print(parent_1)
    [5, 'AdamMax', 0.16718361674068502, 0]
    >>> print(parent_2)
    [9, 'SGD', 0.28364322926906005, 0]
    >>> print(crossover(parent_1, parent_2))
    ([5, 'AdamMax', 0.28364322926906005, 0], [9, 'SGD', 0.16718361674068502, 0])

    >>> from dragon.search_space.dragon_variables import HpVar, NodeVariable, EvoDagVariable
    >>> from dragon.search_space.bricks import MLP, MaxPooling1D, AVGPooling1D
    >>> from dragon.search_space.zellij_variables import DynamicBlock
    >>> from dragon.search_space.bricks_variables import activation_var
    >>> mlp = HpVar("Operation", Constant("MLP operation", MLP), hyperparameters={"out_channels": IntVar("out_channels", 1, 10)})
    >>> pooling = HpVar("Operation", CatVar("Pooling operation", [MaxPooling1D, AVGPooling1D]), hyperparameters={"pool_size": IntVar("pool_size", 1, 5)})
    >>> candidates = NodeVariable(label = "Candidates", 
    ...                           combiner=CatVar("Combiner", features=['add', 'concat']),
    ...                           operation=CatVar("Candidates", [mlp, pooling]),
    ...                           activation_function=activation_var("Activation"))
    >>> operations = DynamicBlock("Operations", candidates, repeat=3)
    >>> dag = EvoDagVariable(label="DAG", operations=operations)
    >>> arr = ArrayVar(dag, dag, Constant('Seed', value=0))
    >>> parent_1 = arr.random()
    >>> parent_2 = arr.random()
    >>> crossover = DAGTwoPoint(arr)
    >>> print(parent_1)
    [NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 3} -- (activation) ELU(alpha=1.0) -- ] | MATRIX:[[0, 1], [0, 0]], NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) SiLU() -- , 
    (combiner) add -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Sigmoid() -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]], 0]
    >>> print(parent_2)
    [NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 1} -- (activation) LeakyReLU(negative_slope=0.01) -- ] | MATRIX:[[0, 1], [0, 0]], NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 9} -- (activation) ELU(alpha=1.0) -- ] | MATRIX:[[0, 1], [0, 0]], 0]
    >>> print(crossover(parent_1, parent_2))
    ([NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 3} -- (activation) ELU(alpha=1.0) -- ] | MATRIX:[[0, 1], [0, 0]], NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) SiLU() -- , 
    (combiner) add -- (name) <class 'dragon.search_space.bricks.pooling.MaxPooling1D'> -- (hp) {'pool_size': 4} -- (activation) Sigmoid() -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]], 0], [NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.pooling.AVGPooling1D'> -- (hp) {'pool_size': 1} -- (activation) LeakyReLU(negative_slope=0.01) -- ] | MATRIX:[[0, 1], [0, 0]], NODES: [
    (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Identity() -- , 
    (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 9} -- (activation) ELU(alpha=1.0) -- ] | MATRIX:[[0, 1], [0, 0]], 0])
    """

    def __init__(self, search_space=None, size=10):
        self.size = size
        super(DAGTwoPoint, self).__init__(search_space)

    def __call__(self, ind1, ind2):
        """
        Crossover implementation:
            - Two crossover points `cxpoint1` and `cxpoint2` are picked randomly from the parent
            - Exchange the parts between `cxpoint1` and `cxpoint2`
            - For each element between those points, if one of them is an `AdjMatrix`: use the `adj_matrix_crossover` function to mix the graphs.
        """
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        for i in range(cxpoint1, cxpoint2):
            if isinstance(ind1[i], AdjMatrix):
                ind1[i], ind2[i] = adj_matrix_crossover(ind1[i], ind2[i], self.size)

        return ind1, ind2

    @Mutator.target.setter
    def target(self, search_space):
        self._target = search_space

def adj_matrix_crossover(p1, p2, size=10):
        """
        adj_matrix_crossover(p1, p2)
        DAG-based crossover
            - Select the index of the operations that would be exchange in each graphs.
            - Remove the corresponding lines and columns from the adjacency matrices
            - Compute the index where the new nodes will be inserted
            - Insert the new rows and columns within the adjacency matrices
            - Make sure no nodes without incoming or outgoing connections are remaining within the matrices
            - Make sure the new matrices are upper-triangular
            - Recreate the list of nodes
            - Create new `AdjMatrix` variables with the new nodes and matrices

        Parameters
        ----------
        p1: `AdjMatrix`
            First parent
        p2: `AdjMatrix
            Second parent
        size: int, default=10
            Maximum number of nodes that an offspring graph could have

        Returns
        ----------
        f1: `AdjMatrix`
            First offspring
        f2: `AdjMatrix`
            Second offspring

        Examples
        ----------
        >>> import numpy as np
        >>> from dragon.search_space.dragon_variables import AdjMatrix
        >>> import torch.nn as nn
        >>> from dragon.search_space.bricks import MLP, Identity
        >>> from dragon.search_space.dragon_variables import Node
        >>> node_1 = Node(combiner="add", operation=MLP, hp={"out_channels": 10}, activation=nn.ReLU())
        >>> node_2 = Node(combiner="add", operation=MLP, hp={"out_channels": 5}, activation=nn.ReLU())
        >>> node_3 = Node(combiner="concat", operation=Identity, hp={}, activation=nn.Softmax())
        >>> op1 = [node_1, node_2, node_3]
        >>> op2 = [node_1, node_3]
        >>> m1 = np.array([[0, 1, 1],
        ...                 [0, 0, 1],
        ...                 [0, 0, 0]])
        >>> m2 = np.array([[0, 1],
        ...                 [0, 0]])
        >>> print(AdjMatrix(op1, m1))
        NODES: [
        (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) ReLU() -- , 
        (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 5} -- (activation) ReLU() -- , 
        (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Softmax(dim=None) -- ] | MATRIX:[[0, 1, 1], [0, 0, 1], [0, 0, 0]]
        >>> print(AdjMatrix(op2, m2))
        NODES: [
        (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) ReLU() -- , 
        (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Softmax(dim=None) -- ] | MATRIX:[[0, 1], [0, 0]]
        >>> f1, f2 = adj_matrix_crossover(AdjMatrix(op1, m1), AdjMatrix(op2, m2))
        >>> print(f1)
        NODES: [
        (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) ReLU() -- , 
        (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 5} -- (activation) ReLU() -- , 
        (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Softmax(dim=None) -- ] | MATRIX:[[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        >>> print(f2)
        NODES: [
        (combiner) add -- (name) <class 'dragon.search_space.bricks.basics.MLP'> -- (hp) {'out_channels': 10} -- (activation) ReLU() -- , 
        (combiner) concat -- (name) <class 'dragon.search_space.bricks.basics.Identity'> -- (hp) {} -- (activation) Softmax(dim=None) -- ] | MATRIX:[[0, 1], [0, 0]]
        """
        crossed = False
        while not crossed:
            op1 = p1.operations
            op2 = p2.operations
            m1 = p1.matrix
            m2 = p2.matrix

            # Randomly select the points that would be exchange from each graph
            s1 = list(set(np.random.choice(range(1, len(op1)), size=len(op1) - 1)))
            s2 = list(set(np.random.choice(range(1, len(op2)), size=len(op2) - 1)))
            s1.sort()
            s2.sort()

            # Remove the corresponding points from the adjacency matrices
            it = 0
            for i1 in s1:
                m1 = np.delete(m1, i1 - it, axis=0)
                m1 = np.delete(m1, i1 - it, axis=1)
                it+=1
            it = 0
            for i2 in s2:
                m2 = np.delete(m2, i2 - it, axis=0)
                m2 = np.delete(m2, i2 - it, axis=1)
                it+=1

            # Compute the index where the new nodes will be inserted
            old_s1 = np.array(list(set(range(len(op1))) - set(s1)))
            old_s2 = np.array(list(set(range(len(op2))) - set(s2)))
            new_s1 = [np.argmin(np.abs(old_s2 - s1[0]))]
            if new_s1[0] == old_s2[new_s1[0]]:
                new_s1[0] += 1
            for i1 in range(1, len(s1)):
                new_s1.append(min(s1[i1] - s1[i1-1] + new_s1[i1-1], len(old_s2) + len(new_s1)))
            new_s2 = [np.argmin(np.abs(old_s1 - s2[0]))]
            if new_s2[0] == old_s1[new_s2[0]]:
                new_s2[0] += 1
            for i2 in range(1, len(s2)):
                new_s2.append(min(s2[i2] - s2[i2 - 1] + new_s2[i2-1], len(old_s1) + len(new_s2)))
            
            # Insert the new nodes
            m1 = np.insert(m1, np.clip(new_s2, 0, m1.shape[0]), 0, axis=0)
            m1 = np.insert(m1, np.clip(new_s2, 0, m1.shape[1]), 0, axis=1)
            m2 = np.insert(m2, np.clip(new_s1, 0, m2.shape[0]), 0, axis=0)
            m2 = np.insert(m2, np.clip(new_s1, 0, m2.shape[1]), 0, axis=1)
            # Make sure no nodes without incoming or outgoing connections are remaining within the matrices
            for i in range(len(s1)):
                diff = new_s1[i] - s1[i]
                if diff >= 0:
                    length = min(m2.shape[0] - diff, p1.matrix.shape[0])
                    m2[diff:diff+length, new_s1[i]] = p1.matrix[:length, s1[i]]
                    m2[new_s1[i], diff:diff+length] = p1.matrix[s1[i], :length]
                if diff < 0:
                    length = min(m2.shape[0], p1.matrix.shape[0]+diff)
                    m2[:length, new_s1[i]] = p1.matrix[-diff:-diff+length, s1[i]]
                    m2[new_s1[i], :length] = p1.matrix[s1[i], -diff:-diff+length]
            for i in range(len(s2)):
                diff = new_s2[i] - s2[i]
                if diff >= 0:
                    length = min(m1.shape[0] - diff, p2.matrix.shape[0])
                    m1[diff:diff+length, new_s2[i]] = p2.matrix[:length, s2[i]]
                    m1[new_s2[i], diff:diff+length] = p2.matrix[s2[i], :length]
                if diff < 0:
                    length = min(m1.shape[0], p2.matrix.shape[0]+diff)
                    m1[:length, new_s2[i]] = p2.matrix[-diff:-diff + length, s2[i]]
                    m1[new_s2[i], :length] = p2.matrix[s2[i], -diff:-diff + length]
            # Make sure the new matrices are upper-triangular
            m1 = np.triu(m1, k=1)
            m1 = fill_adj_matrix(m1)
            m2 = np.triu(m2, k=1)
            m2 = fill_adj_matrix(m2)
            # Recreate the list of nodes
            op1 = [op1[i] for i in range(len(op1)) if i not in s1]
            op2 = [op2[i] for i in range(len(op2)) if i not in s2]
            for i in range(len(new_s1)):
                op2 = op2[:new_s1[i]] + [p1.operations[s1[i]]] + op2[new_s1[i]:]
            for i in range(len(new_s2)):
                op1 = op1[:new_s2[i]] + [p2.operations[s2[i]]] + op1[new_s2[i]:]
            if max(len(op1), len(op2)) <= size:
                crossed = True
        for j in range(1, len(op1)):
            if hasattr(op1[j], "modification"):
                if hasattr(op1[j], "input_shape"):
                    input_shapes = [op1[i].output_shape for i in range(j) if m1[i, j] == 1]
                    op1[j].modification(input_shapes=input_shapes)
        for j in range(1, len(op2)):
            if hasattr(op2[j], "modification"):
                if hasattr(op2[j], "input_shape"):
                    input_shapes = [op2[i].output_shape for i in range(j) if m2[i, j] == 1]
                    op2[j].modification(input_shapes=input_shapes)
        # Create new `AdjMatrix` variables with the new nodes and matrices
        return AdjMatrix(op1, m1), AdjMatrix(op2, m2)