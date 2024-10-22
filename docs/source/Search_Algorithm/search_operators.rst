.. _zellij_neigh:

============
Zellij Neighborhoods
============

Once the search space is defined, each :ref:`var` should be changeable (ie: be replaced by a value from its neighborhood, mutated). Those modifications (mutations) are used by the search algorithms. 
The mutations are designed specifically for each :ref:`var`, and comes as an argument. They are all subclasses of :ref:`varneigh`.


.. _intervals:

**************
Base variables
**************

.. automodule:: dragon.search_algorithm.zellij_neighborhoods
   :members: IntInterval, FloatInterval, CatInterval, ConstantInterval
   :undoc-members:
   :show-inheritance:
   :noindex:


******************
Composed variables
******************

.. automodule:: dragon.search_algorithm.zellij_neighborhoods
   :members: ArrayInterval, BlockInterval, DynamicBlockInterval
   :undoc-members:
   :show-inheritance:
   :noindex: