.. _search_operators:

=============================
Search operators presentation
=============================


Once the search space is defined, it is possible to use a simple Random Search to look for good configurations.
However, training a deep neural network is a long and resource-consuming process. 
Therefore, it is important that the search algorithm requires as little neural network training as possible. 
To do this, it is essential to use information from previous training and evaluations. 
By identifying good solutions, they can be modified to optimize their performance. 
To make these modifications, a `neighbor` attribute can be associated with each of the variables defined in `Search Space <../Search_Space/index.rst>`_. 
They can be seen as a neighborhood or a mutation operator. 
They 
These attributes are added using `addons`. An addon, is an object that is linked to another one. 
It allows to extend some functionnalities of the target without modifying its implementation. 
The `addons` implementations are described in detail here:

.. toctree::
   :maxdepth: 1

   addons


**DRAGON** provides an implementation of one typical neighborhood by variable, but the user may implement its own one.
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

Here is the example of a neighborhood defined for the `IntVar` variable. 
The neighborhood picks the new values within an interval surrounding the current one.
It is parameterized by an argument specifying the interval size.

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


This `IntInterval` is given to the `Variable` during its definition:

.. code-block:: python

   from dragon.search_space.zellij_variables import IntVar

   v = IntVar("An integer variable", 0, 5, neighbor=IntInterval(neighborhood=1))
   v.neighbor(4)
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
   from dragon.search_space.zellij_variables import Block, FloatVar
   from dragon.search_algorithm.zellij_neighborhoods import BlockInterval, FloatInterval
   content = FloatVar("Float example", 0, 10, neighbor=FloatInterval(2))
   a = Block("Block example", content, 3, neighbor=BlockInterval())
   a.neighbor([2, 1, 6])
   [2, 1.3432682541165653, 7.886611679292923]

In this example, the `Block` content: the `FloatVar` variable is given a `neighbor` addon.
The addon `BlockInterval` make use of the `FloatInterval` to create the new value.

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
It takes as input an operation and a set of hyperparameters.
It selects among the operation and the various hyperparameters the ones that will be mutated.
The mutation operation does not have more chance to be changed than the hyperparameters.
It is applied only if the operation is not a `Constant`.
The chosen hyperparameters are mutated according to their `neighbor` addon.
The `HpInterval` object returns the new operation and hyperparameters.

Node neighborhood
~~~~~~~~~~~~~~~~~~~~

The `NodeVariable` might take the operation as a `HpVar` or a `CatVar` of `HpVar` when dealing with candidate operations implemented in various `HpVar`.
In this case, the neighborhood for the `CatVar` of `HpVar` is called `CatHpInterval`.
This neighborhood chooses between modifying the current operation or draw a completely new one.
It takes as parameter the probability of only modifying the current operation (be default equals to 0.9).
With a probability p, the function will look for the `HpVar` corresponding to the current value and call the `HpInterval` of this variable.
The matching is done by looking at the `features` attibute if the `HpVar` operation is a `CatVar` or the `value` attribute if the operation is a `Constant`.
With a probability 1-p, a new layer is drawn (with a new operation a new hyperparameters), by calling the `random` function of the `CatVar`.

The neighborhood class associated to a `NodeVariable` is called `NodeInterval`.
It selects among the combiner, the operation and the activation function what will be modified.
For the selected elements, their `neighbor` attributes are called.

DAG neighborhood
~~~~~~~~~~~~~~~~~~~~

The `EvoDagVariable` neighborhood class is called `EvoDagInterval`.
This neighborhood may do five types of mutations:
* Adding a node
* Deleting a node
* Modifying a node
* Modifying the input connections of a node
* Modifying the output connections of a node
It first randomly choose the nodes that will be modified.
A parameter `nb_mutations` can be set to constrain the number of modified nodes.
For each node selected, the mutations that can be applied may vary.
For example, if the last node has been selected, its outgoing connections cannot be changed.
If the maximum number of node has been reached, the `add` mutation cannot be used.
A possible mutation is then associated to each selected node and performed.
After the mutation on a node has been performed, some tests are ran to insure that the matrix structure is correct.
It prevents nodes from having no incoming or outgoing connections.
The nodes operations are also modified by calling their `modification` function in case some input shapes have been modified.

The detailed implementation of `HpInterval`, `CatHpInterval`, `NodeInterval` and `EvoDagInterval` can be found here:

.. toctree::
   :maxdepth: 1

   dragon_operators


Besides the mutation operators, other search operators have been implemented to use **DRAGON** with an evolutionary algorithm.

.. toctree::
   :maxdepth: 1

   other_operators