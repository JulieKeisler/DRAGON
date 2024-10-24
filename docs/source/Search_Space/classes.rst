.. _nas_classes:

============
DAGs Encoding
============

******************
Main Objects
******************

**DRAGON** search space is base on Directed Acyclic Graphs (DAGs) implemented using an adjacency matrix combined with a set of node.
The nodes represent the network layers and the edges the connexion between them.

.. tomodule:: dragon.search_space.cells
   :members: Brick, Node, EvoDagVariable, fill_adj_matrix
   :undoc-members:
   :show-inheritance:
   :noindex:au

******************
DRAGON variables
******************

Like the int, float, categorial and array based elements are embedded using :ref:`var` based classes, variables have also been implemented for the new objects: `AdjMatrix`, `Node` and `Brick`.

.. automodule:: dragon.search_space.dragon_variables
   :members: HpVar, NodeVariable, EvoDagVariable
   :undoc-members:
   :show-inheritance:
   :noindex: