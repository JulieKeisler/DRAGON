from dragon.search_space import Brick
import torch
from dragon.utils.tools import logger
import torch.nn as nn

class SumFeatures(Brick):
    def __init__(self, input_shape=None, **args):
        super(SumFeatures, self).__init__(input_shape)

    def forward(self, X):
        return X.sum(dim=-1, keepdim=True)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "SumFeatures()"
    
class SplitFeatures(Brick):
    def __init__(self, input_shape, feature_index=0, **args):
        super(SplitFeatures, self).__init__(input_shape)
        self.feature_index = feature_index

    def forward(self, X):
        # Split along the feature dimension
        split_tensors = torch.split(X, 1, dim=-1)
        # Select the feature at self.feature_index
        return split_tensors[self.feature_index]

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"SplitFeatures(feature_index={self.feature_index})"
    
class Negate(Brick):
    def __init__(self, input_shape, **args):
        super(Negate, self).__init__(input_shape)

    def forward(self, X):
        return -X

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Negate()"

# class SelectFeatures(Brick):
#     def __init__(self, input_shape, feature_indices=None, **args):
#         super(SelectFeatures, self).__init__(input_shape)
#         self.feature_indices = feature_indices

#     def forward(self, X, h=None):
#         if self.feature_indices is None:
#             return X

#         n_features = X.shape[-1]
#         idx = torch.as_tensor(self.feature_indices, device=X.device)

#         if idx.max() >=n_features:
#             #logger.warning(f'Index {idx}>X shaepe: {X.shape}, returning X.')
#             return X

#         return X[..., idx]


#     def modify_operation(self, input_shape):
#         self.input_shape = input_shape

#     def __repr__(self):
#         return f"SelectFeatures(feature_indices={self.feature_indices})"
    

class SelectFeatures(Brick):
    """Select (or sample) a subset of input features.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    feature_indices : list[int] or None
        Explicit indices to select.  When *None* and *feature_probs* is
        provided, a single index is sampled according to *feature_probs*
        at construction time.
    feature_probs : list[float] or None
        Per-feature probability / importance distribution (one value per
        column of X).  Used in two ways:

        1. **At init** – if *feature_indices* is ``None``, one index is
           drawn from this distribution so the node starts with the most
           important features.
        2. **At search-config time** – pass the array to the static helper
           :meth:`combination_weights` to obtain ``CatVar`` weights that
           bias the evolutionary search toward combinations of important
           features.
    """

    def __init__(self, input_shape, feature_indices=None, feature_probs=None, **args):
        super(SelectFeatures, self).__init__(input_shape)
        self.feature_probs = feature_probs

        if feature_indices is not None:
            self.feature_indices = feature_indices
        elif feature_probs is not None:
            # Sample one feature index according to the importance distribution
            import random as _rnd
            population = list(range(len(feature_probs)))
            self.feature_indices = _rnd.choices(population, weights=feature_probs, k=1)
        else:
            self.feature_indices = None

    def forward(self, X, h=None):
        if self.feature_indices is None:
            return X

        n_features = X.shape[-1]
        idx = torch.as_tensor(self.feature_indices, device=X.device)

        if idx.max() >= n_features:
            return X

        return X[..., idx]

    # ------------------------------------------------------------------
    # Utility for search-space configuration
    # ------------------------------------------------------------------
    @staticmethod
    def combination_weights(feature_probs, combinations):
        """Convert per-feature importances into per-combination ``CatVar`` weights.

        Parameters
        ----------
        feature_probs : list[float] or array-like
            Importance score for every feature (e.g. from XGBoost).
            Does **not** need to sum to 1 – it will be normalised internally.
        combinations : list[list[int]]
            The list of index-lists that will be passed as ``features``
            to ``CatVar`` (e.g. ``[[0], [1], [0,1], …]``).

        Returns
        -------
        weights : list[float]
            Normalised weights (sum ≈ 1) ready for ``CatVar(…, weights=…)``.
        """
        import numpy as np
        probs = np.asarray(feature_probs, dtype=float)
        probs = probs / probs.sum()# normalise to probabilities
        raw = []
        for combo in combinations:
            # Weight of a combination = sum of its members' importances
            raw.append(float(sum(probs[i] for i in combo)))
        total = sum(raw)
        return [w / total for w in raw]

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"SelectFeatures(feature_indices={self.feature_indices})"

class Inverse(Brick):
    def __init__(self, input_shape, **args):
        super(Inverse, self).__init__(input_shape)

    def forward(self, X):
        return 1 / (X + 1e-8)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Inverse()"
    
class SwitchFeatures(Brick):
    def __init__(self, input_shape, index1=0, index2=1, **args):
        super(SwitchFeatures).__init__(input_shape)
        self.index1 = index1
        self.index2 = index2

    def forward(self, X):
        # X shape: (batch, features)
        assert X.shape[-1] > max(self.index1, self.index2), \
            f"Invalid indices {self.index1}, {self.index2} for input shape {X.shape}"

        X = X.clone()

        tmp = X[..., self.index1].clone()
        X[..., self.index1] = X[..., self.index2]
        X[..., self.index2] = tmp

        return X

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"SwitchFeatures(index1={self.index1}, index2={self.index2})"


class Divide(Brick):
    def __init__(self, input_shape=None, eps=1e-6, **args):
        super(Divide, self).__init__(input_shape)
        self.input_shape = input_shape

    def forward(self, X):
        """
        X shape: (..., 2)
        X[..., 0] = numerator
        X[..., 1] = denominator
        """
        if X.shape[-1] != 2:
            #logger.warning(f"Divide expects 2 features, got {X.shape[-1]}")
            return X
        else:
            num = X[..., 0]
            den = X[..., 1]
            return num / (den + 1e-8)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"Divide"
    
class Substract(Brick):
    def __init__(self, input_shape, eps=1e-8, **args):
        super(Substract, self).__init__(input_shape)
        self.eps = eps

    def forward(self, X):
        """
        X shape: (batch, 2)
        returns: (batch, 1)
        """
        if X.shape[-1] != 2:
            #logger.warning("Subtract expects exactly 2 inputs")
            return X
        return X[..., 0:1] - X[..., 1:2]

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Substract()"


# class ConstantBrick(Brick):
#     def __init__(self, input_shape=None, value=0.0, **args):
#         super().__init__(input_shape)

#         # Paramètre apprenable
#         self.value = nn.Parameter(torch.tensor([[value]], dtype=torch.float32))

#     def forward(self, X=None):
#         if X is not None:
#             batch_size = X.shape[0]
#             device = X.device
#         else:
#             batch_size = 1
#             device = self.value.device

#         return self.value.to(device).expand(batch_size, 1)

#     def modify_operation(self, input_shape):
#         self.input_shape = input_shape

#     def __repr__(self):
#         return f"Constant(value={self.value.item():.4f})"
