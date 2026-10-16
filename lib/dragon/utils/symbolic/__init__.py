"""Symbolic regression components of DRAGON, used by DragonSR.

Submodules
----------
- ``preprocessing`` -- variable augmentation and XGBoost feature selection
- ``evaluation``    -- ``DragonSearcher``, the loss Mutant-UCB evaluates: builds a
  candidate graph, fits its constants and scores its channels
- ``combination``   -- the four model classes that combine a graph's channels
- ``losses``        -- search and alignment losses
- ``dag_to_formula`` -- the printer: a graph's channels as expressions, with every
  guard of the evaluated operators printed
- ``dag_inspector`` -- structural summary of a graph

The search space is ``dragon.search_space.symbolic`` and the estimator
``dragon.search_algorithm.dragonsr.DragonSR``.
"""
