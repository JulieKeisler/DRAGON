from dragon.search_space import Brick
import torch
from dragon.utils.tools import logger
import torch.nn as nn
import numpy as np

class SumFeatures(Brick):
    """Sum all input features into a single scalar per sample.

    Reduces the last dimension by summation, returning a tensor of
    shape ``(batch, 1)``.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    """

    def __init__(self, input_shape=None, **args):
        super(SumFeatures, self).__init__(input_shape)

    def forward(self, X):
        return X.sum(dim=-1, keepdim=True)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "SumFeatures()"


class SplitFeatures(Brick):
    """Extract a single feature by index from the input tensor.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    feature_index : int, default=0
        Index of the feature column to extract.
    """

    def __init__(self, input_shape, feature_index=0, **args):
        super(SplitFeatures, self).__init__(input_shape)
        self.feature_index = feature_index

    def forward(self, X):
        split_tensors = torch.split(X, 1, dim=-1)
        return split_tensors[self.feature_index]

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"SplitFeatures(feature_index={self.feature_index})"


class Negate(Brick):
    """Element-wise negation of the input tensor.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    """

    def __init__(self, input_shape, **args):
        super(Negate, self).__init__(input_shape)

    def forward(self, X):
        return -X

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Negate()"

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
    """Swap two feature columns in the input tensor.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    index1 : int, default=0
        Index of the first feature to swap.
    index2 : int, default=1
        Index of the second feature to swap.
    """

    def __init__(self, input_shape, index1=0, index2=1, **args):
        super(SwitchFeatures, self).__init__(input_shape)
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
    """Element-wise division of the first feature by the second.

    Expects the input tensor to have exactly 2 features in the last
    dimension: ``X[..., 0]`` is the numerator, ``X[..., 1]`` is the
    denominator.  A small epsilon is added to avoid division by zero.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    eps : float, default=1e-6
        Small constant added to the denominator for numerical stability.
    """

    def __init__(self, input_shape=None, eps=1e-6, **args):
        super(Divide, self).__init__(input_shape)
        self.input_shape = input_shape

    def forward(self, X):
        if X.shape[-1] != 2:
            return X
        else:
            num = X[..., 0]
            den = X[..., 1]
            return num / (den + 1e-8)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Divide()"


class Substract(Brick):
    """Element-wise subtraction: first feature minus second feature.

    Expects the input tensor to have exactly 2 features in the last
    dimension.  Returns a tensor with one feature.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    eps : float, default=1e-8
        Unused Kept for API consistency.
    """

    def __init__(self, input_shape, eps=1e-8, **args):
        super(Substract, self).__init__(input_shape)
        self.eps = eps

    def forward(self, X):
        if X.shape[-1] != 2:
            return X
        return X[..., 0:1] - X[..., 1:2]

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Substract()"


class ChannelBoost(Brick):
    """Augments the input by appending one boosted channel.

    The input tensor is preserved, and a new channel is created from either
    two selected channels or a simple pair derived from the available inputs.
    Supported modes are: add, sub, mul, div.
    """

    def __init__(self, input_shape=None, mode="add", eps=1e-8, **args):
        super(ChannelBoost, self).__init__(input_shape)
        self.mode = mode
        self.eps = eps

    def forward(self, X):
        n = X.shape[-1]
        if n == 0:
            return X
        if n == 1:
            derived = X[..., 0:1]
        else:
            a = X[..., 0:1]
            b = X[..., -1:]
            if self.mode == "add":
                derived = a + b
            elif self.mode == "sub":
                derived = a - b
            elif self.mode == "mul":
                derived = a * b
            elif self.mode == "div":
                derived = a / (b + self.eps)
            else:
                derived = a + b
        return torch.cat([X, derived], dim=-1)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"ChannelBoost(mode={self.mode})"


class Power(Brick):
    """Raise the input tensor to a learnable exponent.

    The entire input (all features) is raised elementwise to a power
    that is initialised from *exponent* and then optimised via gradient
    descent together with the rest of the model.
    """

    def __init__(self, input_shape=None, exponent=2.0, **args):
        super(Power, self).__init__(input_shape)
        #self.exponent = nn.Parameter(torch.tensor(float(exponent)))
        self.register_buffer('exponent', torch.tensor(float(exponent)))

    def forward(self, X):
        # Safe power: |X|^exp * sign(X) avoids NaN for negative inputs
        # when the exponent is fractional.
        return torch.pow(torch.abs(X) + 1e-30, self.exponent) * torch.sign(X)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"Power(exponent={self.exponent.item():.4f})"


class ConstantBrick(Brick):
    """A learnable scalar constant that broadcasts to the batch dimension.

    The constant value is stored as a learnable ``nn.Parameter`` and can
    be optimised via gradient descent or the dichotomy line-search used
    by :class:`DragonSearcher`.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    value : float, default=1.0
        Initial value of the learnable constant.
    """

    def __init__(self, input_shape=None, value=1.0, **args):
        super().__init__(input_shape)

        self.value = nn.Parameter(torch.tensor([[value]], dtype=torch.float32))

    def forward(self, X=None):
        if X is not None:
            batch_size = X.shape[0]
            device = X.device
        else:
            batch_size = 1
            device = self.value.device

        return self.value.to(device).expand(batch_size, 1)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"Constant(value={self.value.item():.4f})"


class Ln(Brick):
    """Element-wise natural log. Returns X unchanged where X <= 0."""

    def __init__(self, input_shape=None, **args):
        super(Ln, self).__init__(input_shape)

    def forward(self, X):
        mask = X > 0
        return torch.where(mask, torch.log(X + 1e-30), X)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Ln()"
    
class Sin(Brick):
    """Element-wise sin."""

    def __init__(self, input_shape=None, **args):
        super(Sin, self).__init__(input_shape)

    def forward(self, X):
        return torch.sin(X)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Sin()"

class Cos(Brick):
    """Element-wise cos."""

    def __init__(self, input_shape=None, **args):
        super(Cos, self).__init__(input_shape)

    def forward(self, X):
        return torch.cos(X)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Cos()"

class Exp(Brick):
    """Element-wise exp. Returns X unchanged where exp(X) would overflow."""

    def __init__(self, input_shape=None, **args):
        super(Exp, self).__init__(input_shape)

    def forward(self, X):
        # Avoid overflow: cap the exponent at a reasonable value (e.g. 20)
        capped_X = torch.clamp(X, max=20)
        return torch.exp(capped_X)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Exp()"


class ExpAffine(Brick):
    """Element-wise exp(a * X) with learnable scalar a.

    The parameter a is initialized to 1.0 by default and optimized during
    candidate constant fitting, similarly to ConstantBrick parameters.
    """

    def __init__(self, input_shape=None, a=1.0, **args):
        super(ExpAffine, self).__init__(input_shape)
        self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float32))

    def forward(self, X):
        return torch.exp(self.a * X)

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return f"ExpAffine(a={self.a.item():.4f})"

class Sqrt(Brick):
    """Element-wise square root. Negative inputs are clamped to zero.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (excluding batch dimension).
    """

    def __init__(self, input_shape=None, **args):
        super(Sqrt, self).__init__(input_shape)

    def forward(self, X):
        return torch.sqrt(torch.clamp(X, min=0))

    def modify_operation(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "Sqrt()"


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANT OPTIMIZATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def scalefree_line_search(f, current, max_mag=1e12, iters=40, flat_rtol=1e-6):
    """Golden-section search over a log-spaced candidate grid.

    Parameters
    ----------
    f : callable
        Objective function ``f(x) -> float`` to minimise.
    current : float
        Current value (used as a candidate).
    max_mag : float
        Largest absolute candidate value.
    iters : int
        Golden-section refinement iterations.
    flat_rtol : float
        Relative tolerance below which the function is considered flat.

    Returns
    -------
    float
        The best ``x`` found.
    """
    cands = [0.0, float(current)]
    m = 1e-3
    while m <= max_mag * (1.0 + 1e-9):
        cands.append(m)
        cands.append(-m)
        m *= 10.0
    cands = sorted(set(cands))
    vals = [f(x) for x in cands]
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin <= flat_rtol * (abs(vmin) + 1e-12):
        return float(current)
    i = int(np.argmin(vals))
    lo = cands[max(0, i - 1)]
    hi = cands[min(len(cands) - 1, i + 1)]
    best_x, best_f = cands[i], vals[i]
    if hi > lo:
        gr = (5.0 ** 0.5 - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = f(c)
        fd = f(d)
        for _ in range(iters):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = f(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = f(d)
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if fm <= best_f:
            best_x, best_f = mid, fm
    return best_x


def sweep_constants(model, probe_fn, iters=40, max_mag=1e12, sweeps=1):
    """Optimise scalar constants in a model via dichotomy line search.

    Scans all ``ExpAffine`` (attribute ``a``) and ``ConstantBrick``
    (attribute ``value``) modules and optimises each one independently.

    Parameters
    ----------
    model : nn.Module
        The model containing constant parameters.
    probe_fn : callable
        ``probe_fn(model) -> float`` that evaluates the model's loss.
        The function should call ``model(Xb)`` internally; it receives
        the model with the candidate constant already set.
    iters : int
        Golden-section iterations per parameter.
    max_mag : float
        Maximum absolute value for the line search grid.
    sweeps : int
        Number of full sweeps over all parameters.
    """
    targets = []
    for m in model.modules():
        cls = m.__class__.__name__
        if cls == "ExpAffine" and hasattr(m, "a"):
            targets.append((m, "a"))
        elif cls == "ConstantBrick" and hasattr(m, "value"):
            targets.append((m, "value"))
    if not targets:
        return

    for _ in range(max(1, sweeps)):
        for mod, attr in targets:
            param = getattr(mod, attr)
            cur = float(param.detach().reshape(-1)[0])

            def obj(val, _p=param):
                old = _p.detach().clone()
                with torch.no_grad():
                    _p.fill_(float(val))
                    v = probe_fn(model)
                _p.copy_(old)
                return v

            best = scalefree_line_search(obj, cur, max_mag=max_mag, iters=iters)
            with torch.no_grad():
                param.fill_(float(best))