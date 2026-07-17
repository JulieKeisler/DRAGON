==========================
Symbolic Regression
==========================

**DRAGON** includes built-in support for symbolic regression (SR) through a dedicated set of bricks,
loss functions, and post-processing tools. This section documents the SR-specific components and
provides runnable notebooks.

SR Bricks
---------

The following bricks are available in ``dragon.search_space.bricks.symbolic_regression``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Brick
     - Description
   * - ``SumFeatures``
     - Sums all input features into a single scalar per sample
   * - ``SplitFeatures``
     - Extracts a single feature by index
   * - ``SelectFeatures``
     - Slices input to selected feature indices; supports importance-weighted combination sampling
   * - ``SwitchFeatures``
     - Swaps two feature columns
   * - ``Negate``
     - Negates the input: ``-x``
   * - ``Inverse``
     - Reciprocal: ``1/x``
   * - ``Divide``
     - Element-wise division of two parent outputs
   * - ``Substract``
     - Element-wise subtraction of two parent outputs
   * - ``ChannelBoost``
     - Appends an augmented (derived) channel via add/sub/mul/div
   * - ``Power``
     - Learnable exponent: ``x^a`` where ``a`` is optimizable
   * - ``ConstantBrick``
     - Learnable scalar constant accessible via dichotomy or gradient descent
   * - ``Ln``
     - Natural logarithm: ``ln(|x|)``
   * - ``Sin``
     - Sine: ``sin(x)``
   * - ``Cos``
     - Cosine: ``cos(x)``
   * - ``Exp``
     - Exponential: ``exp(x)``
   * - ``ExpAffine``
     - Learnable affine exponential: ``exp(a * x)`` where ``a`` is optimizable
   * - ``Sqrt``
     - Square root: ``sqrt(|x|)``

Loss Functions
--------------

SR loss functions live in ``dragon.utils.symbolic.loss_function``:

- ``SearchLoss``: Unified loss registry with search and alignment modes
  - ``corr``: 1 - R² (correlation-based)
  - ``mse``: Normalized MSE (MSE / var(y))
  - ``raw_mse``: Unnormalized MSE
  - ``mae``: Normalized MAE
  - ``huber``: Normalized Huber loss

- OLS post-processing: sparse OLS, nested OLS (with link functions), poly-rational OLS
- Parsimony-aware model selection via relative tolerance

Formula Extraction
------------------

``dragon.utils.symbolic`` provides:

- ``graph_to_formula()``: Extract a symbolic formula from a Dragon DAG
- ``graph_to_all_formulas()``: Extract formulas for all output channels
- ``expr_to_mini_dag()``: Compile a SymPy expression back into a minimal Dragon DAG
- ``format_formula_string()``: Precedence-aware parenthesis cleanup
- ``format_nested()``: Format nested-OLS result as a formula string
- ``format_rational()``: Format poly-rational OLS result as a formula string

Feature Selection
-----------------

``dragon.utils.symbolic.features_selection`` provides:

- ``GradientBoostingSelector``: XGBoost/LightGBM/CatBoost importance-based selection
- ``TreeSelector``: RandomForest/ExtraTrees importance-based selection
- ``VarianceSelector``: Variance-threshold fallback
- ``CombinationBuilder``: Weighted index combinations for ``SelectFeatures``

Constant Optimization
---------------------

SR-specific constant optimization (``ConstantBrick.value``, ``ExpAffine.a``):

- **Dichotomy line search**: Scale-free 1-D minimizer with golden-section refinement
- **Gradient optimization**: Adam/AdamW on learnable scalar parameters

Composition Penalty
-------------------

A complexity penalty for deeply nested same-family functions (e.g., ``sin(cos(sin(x)))``).
Configured via ``Config.Loss.COMPOSITION_PENALTY``, ``COMPOSITION_BASE``, and ``COMPOSITION_GROUPS``.

Notebooks
---------

.. toctree::
   :maxdepth: 1

   quickstart
   ols_version
   polyrat
   multivariate
   noisy
   constant_optimization

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Notebook
     - Description
   * - ``quickstart``
     - Simplest workflow: run the search and inspect the best formula. Ideal gas law recovery.
   * - ``ols_version``
     - OLS-combined prediction: all channels are combined via sparse OLS weights for a single formula.
   * - ``polyrat``
     - Polynomial rational prediction: fits a rational function over channel predictions for compact formulas.
   * - ``multivariate``
     - Higher-dimensional target (10 features, 5 used) with feature selection to identify relevant inputs.
   * - ``noisy``
     - Noisy ideal gas data (10% Gaussian noise) demonstrating robustness to measurement noise.
   * - ``constant_optimization``
     - Adds dichotomy line search to refine learnable constants after the evolutionary search.
