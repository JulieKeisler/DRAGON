
Bricks implemention
--------------------


Basic operations
~~~~~~~~~~~~~~~~~
.. automodule:: dragon.search_space.bricks
   :members: Identity, MLP, Dropout
   :undoc-members:
   :show-inheritance:
   :noindex:

One-dimensional Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dragon.search_space.bricks
   :members: Conv1d, MaxPooling1D, AVGPooling1D, Simple_1DGRU, Simple_1DLSTM, Simple_1DRNN, LayerNorm1d, BatchNorm1d, Attention1D
   :undoc-members:
   :show-inheritance:
   :noindex:

Two-dimensional Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dragon.search_space.bricks
   :members: Conv2d, MaxPooling2D, AVGPooling2D, Simple_2DGRU, Simple_2DLSTM, LayerNorm2d, BatchNorm2d, SpatialAttention, TemporalAttention
   :undoc-members:
   :show-inheritance:
   :noindex:

Symbolic Regression Bricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following bricks are designed for symbolic regression. They are defined in
``dragon.search_space.bricks.symbolic_regression`` and can be composed into
DAGs just like any other brick.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Brick
     - Signature
     - Description
   * - ``SelectFeatures``
     - ``(input_shape, feature_indices, combination_weights)``
     - Slices input to selected feature indices; supports importance-weighted combination sampling
   * - ``SplitFeatures``
     - ``(input_shape, feature_index)``
     - Extracts a single feature by index
   * - ``SwitchFeatures``
     - ``(input_shape, index_a, index_b)``
     - Swaps two feature columns
   * - ``SumFeatures``
     - ``(input_shape,)``
     - Sums all input features into a single scalar per sample
   * - ``Negate``
     - ``(input_shape,)``
     - Negates the input: ``-x``
   * - ``Inverse``
     - ``(input_shape,)``
     - Reciprocal: ``1/x``
   * - ``Divide``
     - ``(input_shape_a, input_shape_b)``
     - Element-wise division of two parent outputs
   * - ``Substract``
     - ``(input_shape_a, input_shape_b)``
     - Element-wise subtraction of two parent outputs
   * - ``ChannelBoost``
     - ``(input_shape, mode)``
     - Appends an augmented channel via ``add`` / ``sub`` / ``mul`` / ``div``
   * - ``Power``
     - ``(input_shape, exponent)``
     - Learnable exponent: ``x^a`` where ``a`` is optimizable
   * - ``ConstantBrick``
     - ``(input_shape, value)``
     - Learnable scalar constant accessible via dichotomy or gradient descent
   * - ``Ln``
     - ``(input_shape,)``
     - Natural logarithm: ``ln(|x|)``
   * - ``Sin``
     - ``(input_shape,)``
     - Sine: ``sin(x)``
   * - ``Cos``
     - ``(input_shape,)``
     - Cosine: ``cos(x)``
   * - ``Exp``
     - ``(input_shape,)``
     - Exponential: ``exp(x)``
   * - ``ExpAffine``
     - ``(input_shape, a)``
     - Learnable affine exponential: ``exp(a * x)`` where ``a`` is optimizable
   * - ``Sqrt``
     - ``(input_shape,)``
     - Square root: ``sqrt(|x|)``

Constant optimization for ``ConstantBrick.value`` and ``ExpAffine.a`` is
provided by ``sweep_constants`` and ``scalefree_line_search`` in the same
module.  See the :doc:`Symbolic Regression <../SymbolicRegression/index>`
section for notebooks and the OLS pipeline documentation.

Bricks variables
------------------

Meta Variables
~~~~~~~~~~~~~~~

.. automodule:: dragon.search_space.bricks_variables
   :members: dag_var, node_var, activation_var, operations_var
   :undoc-members:
   :show-inheritance:
   :noindex:

Basic Variables
~~~~~~~~~~~~~~~~

.. automodule:: dragon.search_space.bricks_variables
   :members: identity_var, mlp_var, mlp_const_var, dropout
   :undoc-members:
   :show-inheritance:
   :noindex:

One-dimensional Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dragon.search_space.bricks_variables
   :members: conv_1d, const_conv_1d, pooling_1d, attention_1d, norm_1d, recurrence_1d
   :undoc-members:
   :show-inheritance:
   :noindex:

Two-dimensional Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dragon.search_space.bricks_variables
   :members: conv_2d, pooling_2d, pooling_2d_const_var, norm_2d, recurrence_2d, attention_2d
   :undoc-members:
   :show-inheritance:
   :noindex: