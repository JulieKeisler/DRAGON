DragonSR
+++++++++++++++++++++++

``DragonSR`` is a symbolic regression estimator built on Mutant-UCB and the
symbolic search space. A run executes one stream per operator family and
objective, as threads sharing the evaluation budget:

* the **single-channel** objective scores a graph by its best channel, affinely
  rescaled;
* the **full** objective scores it by the best of four model classes fitted by
  least squares on its channels -- one channel rescaled, a linear combination,
  a transformed linear combination (which contains products of channel powers)
  and a ratio of polynomials of degree at most two.

Each stream draws ``k_init`` random single-node graphs, then spends its budget
in levels of ``t_per_level`` Mutant-UCB evaluations. Every candidate's
constants are fitted by a scale-free line search before it is scored. Streams
whose final loss lies within 2% of the lowest are tied; among them the run
reports the expression that contains a variable, of the simplest class, and
the shortest.

.. code-block:: python

   import numpy as np, pandas as pd
   from dragon.search_algorithm.dragonsr import DragonSR

   x = np.random.default_rng(0).uniform(-1, 1, 2000)
   X, y = pd.DataFrame({"x": x}), x**6 + x**5 + x**4 + x**3 + x**2 + x

   model = DragonSR(n_iterations=2000).fit(X, y)
   model.formula_        # the reported expression, e.g. (x - x**7)/(1 - x)
   model.winner_type_    # its model class: channel, ols, nested or rational
   model.predict(X)      # evaluates the printed expression on the raw inputs

The defaults are the configuration of the DragonSR paper: 20,000 evaluations
over 12 streams, 30 initial graphs and levels of 30 evaluations, graphs of at
most 15 nodes, the 14 most important features, the correlation loss with a
composition penalty of 0.01. On noisy targets the paper scored each candidate
on a random 10% of the observations (``subsample=0.1``).

``n_iterations`` is a ceiling shared by the streams: each stream spends at
most about ``k_init + max_complexity * t_per_level`` evaluations, which the
paper's settings (30 + 55 x 30) match to its 1,666 per stream. For a larger
budget, raise ``t_per_level`` as well; ``fit`` warns when the budget cannot be
spent.

Every guard of the evaluated operators (the exponential clamps its argument at
20, the logarithm passes non-positive values through, powers act on
:math:`|x|`) is printed with its operation, so ``formula_`` is the function the
search evaluated and ``predict`` reproduces it exactly. Feature names must
therefore be Python identifiers.

.. automodule:: dragon.search_algorithm.dragonsr
   :members: DragonSR, select_stream
   :undoc-members:
   :noindex:
