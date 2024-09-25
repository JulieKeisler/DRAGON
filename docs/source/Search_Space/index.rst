.. _search_space:

=============================
Search space documentation
=============================

The search space design is based on fundamental elements originally proposed within an hyper-parameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_.

.. toctree::
   :caption: Search space fundamental elements
   :maxdepth: 1

   zellij_variables
   addons

..

These fundamental elements have been leveraged within the **DRAGON** package to generate new tools for optimizing both the architecture and the hyperparameters of deep neural networks. These tools are very generic and allow the user to use any `nn.Module` object within the optimized architectures. Some basic operations are already implemented and ready to use to facilitate the use of the package.

.. toctree::
   :caption: DRAGON search space elements
   :maxdepth: 1

   classes

..

.. toctree::
   :caption: Candidate Operations
   :maxdepth: 1

   bricks
..

The `cell-based <cell-based.ipynb>`_ tutorial shows that **DRAGON** can be constrained to represent a cell-based search space.

.. toctree::
   :maxdepth: 1

   cell-based
