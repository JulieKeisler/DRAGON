Addons
++++++++
The addons have been originally implemented in the `zellij <https://zellij.readthedocs.io/en/latest/>`_ package. 
An addon is an object that is linked to another one. 
It allows to extend some functionnalities of the target without modifying its implementation.
The user can graft addons to `Variable` by using the kwargs in the init function.

A known addon is:

  * :code:`neighbor`: Defines what is a neighbor for a given `Variable`.


.. automodule:: dragon.search_space.addons
   :members: Addon, VarAddon, VarNeighborhood, SearchspaceAddon
   :undoc-members:
   :show-inheritance:
   :noindex:

**********
Subclasses
**********

.. automodule:: dragon.search_space.addons
   :members: Neighborhood, Mutator, Crossover
   :undoc-members:
   :show-inheritance:
   :noindex: