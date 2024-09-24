.. _nas_classes:

============
DAGs Encoding
============

**DRAGON** search space is base on Directed Acyclic Graphs (DAGs) implemented using an adjacency matrix combined with a set of node.
The nodes represent the network layers and the edges the connexion between them.

******************
Main Objects
******************

.. automodule:: dragon.search_space.cells
   :members: AdjMatrix, fill_adj_matrix, Node, Brick, 
   :undoc-members:
   :show-inheritance:
   :noindex: