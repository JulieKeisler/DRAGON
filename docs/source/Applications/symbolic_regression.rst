Symbolic regression
=============================

DragonSR applies DRAGON to symbolic regression: the search space is a DAG of
symbolic operators (`Symbolic search space <../Search_Space/symbolic.rst>`_)
and the search is Mutant-UCB run as parallel streams
(`DragonSR <../Search_Algorithm/dragonsr.rst>`_). The performance evaluation
function and its components live in ``dragon.utils.symbolic``:

=====================  ==========================================================
Module                 Role
=====================  ==========================================================
``preprocessing``      adds the engineered feature(s) most correlated with the
                       target (``VarAugmentor``) and keeps the most important
                       features by XGBoost gain (``XGBoostSelector``)
``evaluation``         ``DragonSearcher``: the loss Mutant-UCB calls -- builds the
                       graph, fits its constants, scores its channels
``combination``        ``OLSPostProcessor``: the four model classes over the
                       channels, and the parsimony rule between them
``losses``             search losses (``corr`` = 1 - Pearson r^2) and alignment
                       losses
``dag_to_formula``     prints a graph's channels as expressions, guards included
=====================  ==========================================================

The experiment code of the paper is kept unchanged under ``leaderboard/``;
``tests/test_dragonsr_equivalence.py`` checks that each library component
returns exactly what the frozen component it was ported from returns, and
``tests/test_dragonsr_smoke.py`` runs the estimator end to end.

.. automodule:: dragon.utils.symbolic.evaluation
   :members: DragonSearcher
   :noindex:

.. automodule:: dragon.utils.symbolic.combination
   :members: OLSPostProcessor
   :noindex:

.. automodule:: dragon.utils.symbolic.preprocessing
   :members: VarAugmentor, XGBoostSelector
   :noindex:
