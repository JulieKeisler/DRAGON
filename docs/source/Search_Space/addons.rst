.. _addons:

======
Addons
======
The addons have been originally implemented in the `zellij <https://zellij.readthedocs.io/en/latest/>`_ package. An addons, is an object that is linked to another one. It allows to extend
some functionnalities of the target without modifying its implementation.
The user can graft addons to :ref:`var` by using the kwargs
in the init function.

Known kwargs is:

  * :code:`neighbor`: Defines what is a neighbor for a given :ref:`var`. It uses the neighbor :ref:`spadd`.


.. automodule:: dragon.search_space.addons
   :members: Addon
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _varadd:

##############
Variable Addon
##############

.. automodule:: dragon.search_space.addons
   :members: VarAddon
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _varneigh:

**********
Subclasses
**********
.. automodule:: dragon.search_space.addons
  :members: VarNeighborhood
  :undoc-members:
  :show-inheritance:
  :noindex:

.. _spadd:

##################
Search space Addon
##################

.. automodule:: dragon.search_space.addons
   :members: SearchspaceAddon
   :undoc-members:
   :show-inheritance:
   :noindex:

**********
Subclasses
**********

.. automodule:: dragon.search_space.addons
   :members: Neighborhood, Operator, Mutator, Crossover, Selector
   :undoc-members:
   :show-inheritance:
   :noindex: