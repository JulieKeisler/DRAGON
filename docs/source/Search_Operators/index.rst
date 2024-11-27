.. _search_operators:

=============================
Presentation
=============================


Once the search space is defined, a simple random search can be used to find good configurations.
However, training a deep neural network is a long and resource-intensive process.
Therefore, the search algorithm must train neural networks as little as possible.
It is therefore essential to use information from previous training and evaluation.
By identifying good solutions, they can be modified to optimize their performance.
To make these modifications, a `neighbor` attribute can be associated with each of the variables defined in the `Search Space section <../Search_Space/index.rst>`_.
They can be thought of as neighborhood or mutation operators.
These attributes are added using `Addons`. A `Addon` is an object that is linked to another.
It allows extending functionalities of the other object (in this case a `Variable`) without modifying its implementation.

DRAGON provides an implementation of one neighborhood per variable, but users can implement their own.  
A neighborhood is a class inheriting from the abstract class `VarNeighborhood`, which is an `Addon`.  
It should have the following structure:
The `addons` implementations are described in detail here:

.. toctree::
   :maxdepth: 1

   addons


**DRAGON** provides an implementation of one neighborhood per variable, but users can implement their own.  
A neighborhood is a class inheriting from the abstract class `VarNeighborhood`, which is an `Addon`.  
It should have the following structure:

.. code-block:: python

   class CustomInterval(VarNeighborhood):
      """CustomInterval
      Addon used to determine the neighbor function of a Variable.

      Parameters
      ----------
      variable : Variable, default=None
         Targeted Variable.
      neighborhood, default=None
         Parameter of the neighborhood
      """
      def __call__(self, value, size=1): 
            """
            Function defining how to choose `size` neighbors surrounding the variable `value`
            """

      @VarNeighborhood.neighborhood.setter
      def neighborhood(self, neighborhood):
         """
         Set the neighborhood parameter 
         (it might be for example the probability to mutate each ArrayVar values).
         """
         self._neighborhood = neighborhood

      @VarNeighborhood.target.setter
      def target(self, variable):
            """
            Give to the CustomInterval object the information of the related `Variable`.
            """
         self._target = variable

Here is an example of a neighborhood defined for the `IntVar` variable.  
The neighborhood selects new values within an interval surrounding the current one, parameterized by an interval size:

.. code-block:: python

   from dragon.search_space.addons import VarNeighborhood

      class IntInterval(VarNeighborhood): 
      def __call__(self, value, size=1):
         """Get the upper bound for the interval:
         the minum value between [current value + the neighborhood] and 
         [the maximum value the variable can take]"""
         upper = np.min([value + self.neighborhood + 1, self.target.up_bound])
         """Get the lower bound for the interval: 
         the maximum value between [current value - the neighborhood] and 
         [the minimum value the variable can take]"""
         lower = np.max([value - self.neighborhood, self.target.low_bound])

         res = []
         for _ in range(size):
               v = np.random.randint(lower, upper)
               while v == value:
                  v = np.random.randint(lower, upper)
               res.append(int(v))
         return res
         if size == 1:
            return v[0]
         else
            return v

      @VarNeighborhood.neighborhood.setter
      def neighborhood(self, neighborhood):
         self._neighborhood = neighborhood

      @VarNeighborhood.target.setter
      def target(self, variable):
         self._target = variable


This `IntInterval` is assigned to the `Variable` while we define it:

.. code-block:: python

   from dragon.search_space.base_variables import IntVar

   v = IntVar("An integer variable", 0, 5, neighbor=IntInterval(neighborhood=1))
   v.neighbor(4) # Example usage
   3

**DRAGON** provides implementation of neighborhoods operators for each variable from `Search Space <../Search_Space/index.rst>`_.

Base and composed variables
------------

The neighborhoods operators available for base and composed variables within **DRAGON** are listed below.

.. list-table:: Base and composed neighborhoods
   :widths: 25 20 25 30
   :header-rows: 1

   * - Type
     - Variable Name
     - Neighbor Name
     - Main parameters
   * - Integer
     - `IntVar`
     - `IntInterval`
     - Interval size
   * - Float
     - `FloatVar`
     - `FloatInterval`
     - Interval size
   * - Categorical (string, etc)
     - `CatVar`
     - `CatInterval`
     - 
   * - Constant (any object)
     - `Constant`
     - `ConstantInterval`
     - 
   * - Array of Variables   
     - `ArrayVar`
     - `ArrayInterval`
     - 
   * - Fix number of repeats
     - `Block`
     - `BlockInterval`
     - 
   * - Random number of repeats
     - `DynamicBlock`
     - `DynamicBlockInterval`
     - Neighborhood of the `DynamicBlock` size.
     
If a composed variable has a `neighbor` addon, then all the values composing this variable should have a `neighbor` addon.
For example with a `Block`:

.. code-block:: python
   from dragon.search_space.base_variables import Block, FloatVar
   from dragon.search_operators.base_neighborhoods import BlockInterval, FloatInterval
   content = FloatVar("Float example", 0, 10, neighbor=FloatInterval(2))
   a = Block("Block example", content, 3, neighbor=BlockInterval())
   a.neighbor([2, 1, 6])
   [2, 1.3432682541165653, 7.886611679292923]

In this example, the `Block`'s value is a `FloatVar` variable. `Neighbor` addons are given to both `Variables`.
The addon `BlockInterval` makes use of the `FloatInterval` to create the new value.

The detailed implementation can be found here:

.. toctree::
   :maxdepth: 1

   search_operators

DAG encoding neighborhoods
------------


Besides the base and composed variables, the ones used for DAG encoding, namely `HpVar`, `NodeVariable` and `EvoDagVariable` also have implemented neighborhoods.

Operation neighborhood
~~~~~~~~~~~~~~~~~~~~

The `HpVar` neighborhood is called `HpInterval`.
Its arguments are an operation and a set of hyperparameters.
It selects among the operation and the various hyperparameters the ones that will be mutated.
The mutation applied to the operation is not more likely to be called than the hyperparameters one.
It does not have any effect if the operation is a `Constant`.
The chosen hyperparameters are mutated according to their `neighbor` addon.
The `HpInterval` object returns the new operation and hyperparameters.
It is possible to modify this operator to increase the probability of modifying the operation or to prevent hyperparameter mutations before a certain iteration of the search algorithm.

Node neighborhood
~~~~~~~~~~~~~~~~~~~~

If the operation within a `NodeVariable` is encoded as a `HpVar`, then its neighborhood will be the `HpInterval`. 
But, in the case of `CatVar` of `HpVar`, when dealing with candidate operations implemented in various `HpVar`, the neighborhood for the operation is called `CatHpInterval`.
This neighborhood chooses between modifying the current operation or drawing a completely new one.
It takes as argument a probability $p \in [0,1]$ of only modifying the current operation (by default equal to $0.9$).
With a probability $p$, the function will look for the `HpVar` corresponding to the current value and call the `HpInterval` of this variable.
The matching is done by looking at the `features` attribute if the `HpVar` operation is a `CatVar` or the `value` attribute if the operation is a `Constant`.
With a probability $1-p$, a new layer is drawn (with a new operation and new hyperparameters), by calling the `random` function of the `CatVar`.

The neighborhood class associated with a `NodeVariable` is called `NodeInterval`.
It selects among the combiner, the operation and the activation function what is to be modified.
For the selected elements, their `neighbor` attributes are invoked. In the current implementation, the chances for modifying any of these three elements are the same, which may be changed.

DAG neighborhood
~~~~~~~~~~~~~~~~~~~~

The `EvoDagVariable` neighborhood class is called `EvoDagInterval`.
This neighborhood may perform five types of mutations:
* Adding a node
* Deleting a node
* Modifying a node
* Modifying the input connections of a node
* Modifying the output connections of a node
First, it randomly selects the nodes to be modified.
A parameter `nb\_mutations` can be set to limit the number of nodes modified.
For each selected node, the allowed mutations can be different.
For example, if the selected node is the last one, its outgoing connections cannot be changed.
If the maximum number of nodes is reached, the `add` mutation cannot be used.
For each node, once the set of allowed mutations has been defined, one value from that set is drawn and performed.
After a mutation, some tests are performed to ensure a correct adjacency matrix structure and to adjust connections if necessary.
This prevents nodes from having no incoming or outgoing connections.
By modifying the nodes and edges, the input tensors of each node may have changed shape. 
In this case, the node's operation is modified by calling its `modification` to adjust the weights.

The detailed implementation of `HpInterval`, `CatHpInterval`, `NodeInterval` and `EvoDagInterval` can be found here:

.. toctree::
   :maxdepth: 1

   dragon_operators

Crossover
~~~~~~~~~~~~~~~~~~~~

Besides the neighborhood operators, a crossover has been implemented to use DRAGON with an evolutionary algorithm.
The crossover is not an `Addon`, it is a simple class implementing a two-point crossover.
The crossover `\_\_call\_\_` method takes as input two individuals `ind1` and `ind2` which should be array-like variables, with the same types of `Variables` at each position.
Two index points from arrays are picked randomly.
The segment between those two index points is swapped between the parents.
For each element of this segment, if one of them is an `AdjMatrix` variable, then the DAG-based crossover is used.

The DAG-based crossover takes as input two `AdjMatrix` elements and perform the following operations:
* Selects the indexes of the operations that would be exchanged in each graph.
* Removes the corresponding lines and columns from both adjacency matrices.
* Computes for each exchanging node, the index in the other graph where it will be inserted
* Inserts the new rows and columns within both adjacency matrices.
* Asserts no nodes without incoming or outgoing connections are remaining within the matrices.
* Asserts the new matrices are upper-triangular.
* Creates new `AdjMatrix` variables with the new nodes and matrices.

For a detailed implementation, see:

.. toctree::
   :maxdepth: 1

   other_operators