.. _search_space:

=============================
Search space documentation
=============================

The search space design is based on fundamental elements originally proposed within an hyper-parameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_.
These fundamental elements have been leveraged within the **DRAGON** package to generate new tools for optimizing both the architecture and the hyperparameters of deep neural networks. These tools are very generic and allow the user to use any :code: nn.module operation within the optimized architectures. Some basic operations are already implemented and ready to use to facilitate the use of the package.
Finally, we show in the `cell-based <cell-based.ipynb>`_ tutorial that this search space is defined as very generic, but can be constrained to represent a cell-based search space.

.. toctree::
   :caption: Search space fundamental elements

   zellij_variables
   addons

..

.. toctree::
   :caption: DRAGON search space elements

   classes
   dragon_variables

..

.. toctree::
   :caption: Candidate Operations

   bricks
   bricks_variables

.. toctree::

   cell-based
