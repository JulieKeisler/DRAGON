# features.py
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from Config import Experiment


# ══════════════════════════════════════════════════════════════════════════════
#  XGBOOST FEATURE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostSelector:
    """Ranks features by XGBoost importance and returns the top-k subset.

    Parameters
    ----------
    n_top         : number of features to keep (default: Experiment.N_TOP_FEATURES)
    n_estimators  : XGBoost trees
    max_depth     : XGBoost tree depth
    learning_rate : XGBoost learning rate
    seed          : random seed (default: Experiment.RANDOM_SEED)
    """

    def __init__(
        self,
        n_top:         int   = Experiment.N_TOP_FEATURES,
        n_estimators:  int   = 100,
        max_depth:     int   = 6,
        learning_rate: float = 0.1,
        seed:          int   = Experiment.RANDOM_SEED,
    ):
        self.n_top         = n_top
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.seed          = seed

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[list[str], dict[str, float]]:
        """Return (top_feature_names, full_score_dict) ranked by XGBoost importance."""
        X_sc = StandardScaler().fit_transform(X)
        z_sc = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()

        mdl = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbosity=0,
        )
        mdl.fit(X_sc, z_sc)

        fi        = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_feats = fi.head(self.n_top).index.tolist()
        scores    = fi.to_dict()
        return top_feats, scores


# ══════════════════════════════════════════════════════════════════════════════
#  VARIABLE AUGMENTER
# ══════════════════════════════════════════════════════════════════════════════

class VarAugmentor:
    """Adds the single most predictive engineered feature to X."""

    _R2_THRESHOLD = 0.999
    _F32_MAX      = np.finfo(np.float32).max

    def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        raw_cols = set(X.columns)
        cands    = self._build_candidates(X)
        cands    = self._clean(cands, raw_cols)

        names  = list(cands.keys())
        values = np.array(list(cands.values()), dtype=np.float64)

        r2_single               = self._batch_r2(values, y.values.ravel())
        best_single_idx         = int(np.argmax(r2_single))
        best_single_name        = names[best_single_idx]
        best_single_r2          = float(r2_single[best_single_idx])

        best_pair_name, best_pair_r2 = self._best_pair(names, values, y.values.ravel())

        single_is_new = best_single_name not in raw_cols
        pair_new_cols = [n for n in best_pair_name if n not in raw_cols]

        if best_single_r2 >= self._R2_THRESHOLD and len(pair_new_cols) >= (1 if single_is_new else 2):
            to_add = [best_single_name] if single_is_new else []
            print(f"[var_aug] single: {best_single_name}  r²={best_single_r2:.4f}")
        else:
            to_add = pair_new_cols
            print(f"[var_aug] pair: {best_pair_name}  r²={best_pair_r2:.4f}")

        if to_add:
            X = pd.concat([X, pd.DataFrame({n: cands[n] for n in to_add}, index=X.index)], axis=1)
        return X

    def _build_candidates(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        cands = {c: X[c].values.astype(np.float64) for c in X.columns}
        for c in list(X.columns):
            v = cands[c]
            cands[f"({c})**2"]            = v ** 2
            cands[f"({c})**3"]            = v ** 3
            cands[f"1/({c})"]             = np.where(np.abs(v) > 1e-12, 1.0 / v, 0.0)
            cands[f"sqrt(abs({c}))"]      = np.sqrt(np.abs(v))
            cands[f"-({c})"]              = -v
            cands[f"log(abs({c})+1e-12)"] = np.log(np.abs(v) + 1e-12)
        for ci, cj in combinations(X.columns, 2):
            vi, vj = X[ci].values.astype(np.float64), X[cj].values.astype(np.float64)
            cands[f"({ci}+{cj})"]  = vi + vj
            cands[f"({ci}-{cj})"]  = vi - vj
            cands[f"({cj}-{ci})"]  = vj - vi
            cands[f"({ci}*{cj})"]  = np.clip(vi * vj, -self._F32_MAX, self._F32_MAX)
            cands[f"({ci}/{cj})"]  = np.where(np.abs(vj) > 1e-12, vi / vj, 0.0)
            cands[f"({cj}/{ci})"]  = np.where(np.abs(vi) > 1e-12, vj / vi, 0.0)
        return cands

    def _clean(self, cands: dict, raw_cols: set) -> dict:
        return {
            name: np.clip(np.where(np.isfinite(v), v, 0.0), -self._F32_MAX, self._F32_MAX)
            for name, v in cands.items()
            if name in raw_cols or np.std(v) > 1e-15
        }

    def _batch_r2(self, T: np.ndarray, y: np.ndarray) -> np.ndarray:
        T   = np.where(np.isfinite(T), T, 0.0)
        Tc  = T - T.mean(axis=1, keepdims=True)
        Ts  = np.sqrt((Tc ** 2).mean(axis=1))
        y_c = y - y.mean()
        y_s = y.std()
        d   = Tc @ y_c / len(y)
        return np.divide(d, Ts * y_s, where=Ts > 1e-15, out=np.zeros(T.shape[0])) ** 2

    def _best_pair(
        self, names: list[str], values: np.ndarray, y: np.ndarray
    ) -> tuple[tuple[str, str], float]:
        best_r2, best_pair = 0.0, (names[0], names[1])
        for i in range(len(names)):
            vi   = values[i]
            rest = values[i + 1:]
            if not rest.shape[0]:
                break
            templates = [
                vi + rest,
                vi - rest,
                rest - vi,
                np.clip(vi * rest, -self._F32_MAX, self._F32_MAX),
                np.where(np.abs(rest) > 1e-12, vi / rest, 0.0),
                np.where(np.abs(vi)   > 1e-12, rest / vi, 0.0),
            ]
            br = np.zeros(rest.shape[0])
            for T in templates:
                r2     = self._batch_r2(np.atleast_2d(T), y)
                better = r2 > br
                br[better] = r2[better]
            for k in range(rest.shape[0]):
                if br[k] > best_r2:
                    best_r2   = br[k]
                    best_pair = (names[i], names[i + 1 + k])
        return best_pair, best_r2


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class CombinationBuilder:
    """Builds weighted feature-index combinations for Dragon's SelectFeatures node.

    Strategy
    --------
    For sizes 1 and 2 every possible combination is generated.
    For sizes 3+, only the top-M features (by importance) are considered,
    then combinations are ranked by sum of log-probabilities and the top
    `budget` are kept.

    Parameters
    ----------
    max_combo       : maximum subset size (clipped to n features)
    budget_per_size : optional dict overriding the per-size sample budget
    """

    _TOP_M = {3: 30, 4: 20, 5: 14, 6: 12}   # candidate pool size for sizes >= 3

    def __init__(self, max_combo: int = 6, budget_per_size: dict = None):
        self.max_combo       = max_combo
        self.budget_per_size = budget_per_size or {}

    def build(self, feature_names: list[str], feature_scores: dict[str, float]) -> list[list[int]]:
        """Return list of index lists, each representing a candidate feature subset."""
        n = len(feature_names)
        if n == 1:
            return [[0]]

        probs    = np.array([feature_scores.get(c, 1.0 / n) for c in feature_names])
        probs    = probs / probs.sum()
        log_prob = np.log(probs + 1e-300)

        max_combo = min(self.max_combo, n)
        budget    = {1: n, 2: n * (n - 1) // 2}
        for s in range(3, max_combo + 1):
            budget[s] = max(50, 200 // (s - 1))
        budget.update(self.budget_per_size)

        all_combos = []
        for size in range(1, max_combo + 1):
            cap = budget[size]
            if size <= 2:
                pool = list(combinations(range(n), size))
            else:
                M       = min(self._TOP_M.get(size, 12), n)
                top_idx = np.argsort(probs)[-M:]
                pool    = list(combinations(sorted(top_idx.tolist()), size))

            scored = [(sum(log_prob[i] for i in combo), list(combo)) for combo in pool]
            scored.sort(key=lambda x: -x[0])
            all_combos.extend(combo for _, combo in scored[:cap])

        return all_combos
