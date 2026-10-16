Symbolic search space
+++++++++++++++++++++++

DragonSR searches DAGs whose nodes are symbolic operators instead of neural
layers. A node combines the outputs of its predecessors by element-wise
addition or multiplication and applies one operator. The operators act
element-wise on every channel of their input -- inverse, negation, signed
powers :math:`\operatorname{sign}(x)|x|^p`, :math:`\ln`, :math:`\exp`,
:math:`\exp(ax)` with a fitted rate, :math:`\sin`, :math:`\cos`, a fitted
constant -- except three that act on the channel dimension:
``SelectFeatures`` returns a subset of the input features, drawn with a
probability that grows with their XGBoost importance; ``SumFeatures`` sums the
channels into one; ``ChannelBoost`` appends a channel built from the first and
last ones by one of the four arithmetic operations. A graph therefore outputs a
variable number of channels.

The operators are grouped into six families that share an algebraic core
(``select``, ``unary``, ``power``, ``sum``, ``boost``, ``const``); DragonSR
searches each family in its own stream.

================  ==========================================
Family            Added to the core
================  ==========================================
rational          ---
trig              :math:`\sin`, :math:`\cos`
log               :math:`\ln`
sqrt              powers with :math:`p \in \{0.5, 1, 2, 3\}`
exp               :math:`\exp`, :math:`\exp(ax)`
general_power     :math:`\ln`, :math:`\exp`, :math:`\exp(ax)`
================  ==========================================

.. code-block:: python

   from dragon.search_space.symbolic import SymbolicSearchSpace, OPERATOR_FAMILIES

   search_space, dag = SymbolicSearchSpace(
       feature_names, feature_scores, OPERATOR_FAMILIES["trig"], max_nodes=15
   ).build()
   graph = search_space.random()

.. automodule:: dragon.search_space.symbolic
   :members: OPERATOR_FAMILIES, operator_streams, CombinationBuilder, SymbolicSearchSpace
   :undoc-members:
   :noindex:
