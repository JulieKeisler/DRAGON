.. _search_operators:

=============================
Search Operators
=============================



Once the search space is defined, it is possible to use a simple Random Search to look for good configuration.
However, the Random Search is usually not very efficient. **DRAGON** gives access to several search algorithms based on the evolution of the representations.
Each :ref:`var` from the Search Space should come with a *Mutation*, also called *Neighborhood* attribute.
Those mutations will draw values in the neighborhood of the current configuration by performing small modifications.
They can be used by the Search Algorithms as mutation operators for the Evolutionary Algorithm or as neighborhoods for a Local Search for example.
The mutations are designed specifically for each :ref:`var`, and are implemented as an argument. They are all subclasses of :ref:`varneigh`.

.. toctree::
   :maxdepth: 1

   search_operators
   dragon_operators

Besides the mutation operators, other search operators have been implemented to use **DRAGON** with an evolutionary algorithm.

.. toctree::
   :maxdepth: 1

   other_operators
