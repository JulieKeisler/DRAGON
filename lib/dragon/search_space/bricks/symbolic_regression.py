from dragon.search_space import Brick
import torch
from dragon.utils.tools import logger

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

class SelectFeatures(Brick):
    def __init__(self, input_shape, feature_indices=None, **args):
        super(SelectFeatures, self).__init__(input_shape)
        self.feature_indices = feature_indices

    def forward(self, X, h=None):
        if self.feature_indices is None:
            return X

        n_features = X.shape[-1]
        idx = torch.as_tensor(self.feature_indices, device=X.device)

        if idx.max() >=n_features:
            #logger.warning(f'Index {idx}>X shaepe: {X.shape}, returning X.')
            return X

        return X[..., idx]


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


class ConstantBrick(Brick):
    def __init__(self, input_shape=None, value=0.0, **args):
        """
        Brick qui renvoie une constante pour tous les échantillons du batch.
        """
        super(ConstantBrick, self).__init__(input_shape)
        self.value = float(value)

    def forward(self, X=None):
        """
        Renvoie la constante avec la bonne shape.
        X est optionnel : si fourni, renvoie batch_size x 1
        """
        if X is not None:
            batch_size = X.shape[0]
            device = X.device
        else:
            batch_size = 1
            device = torch.device('cpu')
        return torch.full((batch_size, 1), self.value, device=device)

    def modify_operation(self, input_shape):
        """
        Conserve la compatibilité avec l'interface Dragon.
        """
        self.input_shape = input_shape

    def __repr__(self):
        return f"Constant(value={self.value})"
