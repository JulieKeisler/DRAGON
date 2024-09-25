.. _search_space:

=============================
Search space documentation
=============================

The search space design is based on fundamental elements originally proposed within an hyper-parameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_.
The search space is made of mix-variables such as integers, floats, categorical variables and graphs. Each of this object is associated with a *Variable*, which defines what values an object can take.

For example, an integer object will be associated with the *variable* `IntVar`, that will take as arguments the lower and upper bounds, defining where the integer is defined.

.. code-block:: python

   from dragon.search_space.zellij_variables import IntVar

   v = IntVar("An integer variable", 0, 5)

In this example, the variable `v` defines an integer which can take values from 0 to 5.

.. toctree::
   :maxdepth: 1

   zellij_variables
   addons

..

These fundamental elements have been leveraged within the **DRAGON** package to generate new tools for optimizing both the architecture and the hyperparameters of deep neural networks. These tools are very generic and allow the user to use any `nn.Module` object within the optimized architectures. Some basic operations are already implemented and ready to use to facilitate the use of the package.

.. toctree::
   :maxdepth: 1

   classes

..

.. toctree::
   :maxdepth: 1

   bricks
..

The `cell-based <cell-based.ipynb>`_ tutorial shows that **DRAGON** can be constrained to represent a cell-based search space.

.. toctree::
   :maxdepth: 1

   cell-based
