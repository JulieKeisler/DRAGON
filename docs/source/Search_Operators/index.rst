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

Another neighborhood called `CatHpInterval` 

Node neighborhood
~~~~~~~~~~~~~~~~~~~~



.. toctree::
   :maxdepth: 1

   dragon_operators


Besides the mutation operators, other search operators have been implemented to use **DRAGON** with an evolutionary algorithm.

.. toctree::
   :maxdepth: 1

   other_operators