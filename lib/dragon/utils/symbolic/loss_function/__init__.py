"""Loss functions and OLS pipeline for DRAGON symbolic regression.

This package provides:

- ``SearchLoss`` / ``AlignmentLoss`` — unified loss registries for correlation,
  MSE, MAE, Huber, and alignment scoring
- ``ols_pipeline.evaluate`` — end-to-end OLS evaluation: channel selection,
  sparse OLS, nested link-function OLS, polynomial-rational OLS, and
  parsimony-aware model selection
- ``solvers`` — robust least-squares solvers (``safe_lstsq``, ``sparse_lstsq``)
- ``fitting`` — nested OLS and polynomial-rational OLS fitting routines
- ``selection`` — effective-parameter counting and model selection with
  parsimony tolerance
"""

from dragon.utils.symbolic.loss_function.search_loss import SearchLoss, AlignmentLoss
from dragon.utils.symbolic.loss_function.solvers import safe_lstsq, sparse_lstsq
from dragon.utils.symbolic.loss_function.fitting import nested_ols, poly_rational_ols, build_poly_features
from dragon.utils.symbolic.loss_function.selection import eff_params, select_model

__all__ = [
    "SearchLoss",
    "AlignmentLoss",
    "safe_lstsq",
    "sparse_lstsq",
    "nested_ols",
    "poly_rational_ols",
    "build_poly_features",
    "eff_params",
    "select_model",
]
