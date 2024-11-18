.. _search_space:

=============================
Presentation
=============================

The search space design is based on an abstract class called *Variable*, originally proposed within an hyper-parameters optimization package called `zellij <https://zellij.readthedocs.io/en/latest/>`_.
A variable should implements a *random* method detailing how to create a random value and an *isconstant* method specifying if the variable is a constant or not.
A variable can take *Addons* to implement additional features such as the `Search Operators <../Search_Operators/index.rst>`_.
The search space is made of base and composed variables to create more or less complex search spaces.
Among the composed variables, some have been created specifically for the DAG-encodings.

Base variables
------------

The base variables implements basic objects such as integers, floats or categorical variables. Each of this object is associated with a *Variable*, which defines what values an object can take.
For example, an integer object will be associated with the *variable* `IntVar`, that will take as arguments the lower and upper bounds, defining where the integer is defined.

.. code-block:: python

   from dragon.search_space.zellij_variables import IntVar

   v = IntVar("An integer variable", 0, 5)
   v.random()
::
   3

In this example, the variable `v` defines an integer which can take values from 0 to 5. When calling `v.random()`, the script returns an integer from this interval, here `3`.
All the base variables available within **DRAGON** are detailed in the `Base Variables <_base_variables>`_ section.


Composed variables
------------

The base variables can be composed to create more complex objects such as arrays of variables.

These fundamental elements have been leveraged within the **DRAGON** package to generate new tools for optimizing both the architecture and the hyperparameters of deep neural networks. These tools are very generic and allow the user to use any `nn.Module` object within the optimized architectures. Some basic operations are already implemented and ready to use to facilitate the use of the package.

DAG Encoding
------------

.. tikz::

   \draw[thick,->] (0,0) -- (2,0) node[anchor=north] {x};
   \draw[thick,->] (0,0) -- (0,2) node[anchor=east] {y};
   \draw[red,thick] (0,0) -- (1,1) node[anchor=south] {z};