# ols.py
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge

from Config import OLS as _CfgOLS, Loss as _CfgLoss


# ══════════════════════════════════════════════════════════════════════════════
#  PEARSON CORRELATION  (module-level utility)
# ══════════════════════════════════════════════════════════════════════════════

def correl(pred, true) -> float:
    if isinstance(pred, torch.Tensor): pred = pred.numpy()
    if isinstance(true, torch.Tensor): true = true.numpy()
    pred = pred.ravel(); true = true.ravel()
    pd_  = pred - pred.mean(); td_ = true - true.mean()
    sp   = np.sqrt(np.sum(pd_ ** 2)); st = np.sqrt(np.sum(td_ ** 2))
    if sp < 1e-12 or st < 1e-12:
        return 0.0
    return float(np.sum(pd_ * td_) / (sp * st + 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
#  OLS POST-PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class OLSPostProcessor:
    """Post-processing OLS suite: sparse, nested, and poly-rational.

    Evaluation modes
    ----------------
    'full'    — best channel + sparse OLS + nested OLS + poly-rational OLS
    'ols'     — best channel + sparse OLS only
    'channel' — best channel only (no linear fitting)

    Parameters
    ----------
    rat_max_degree     : max polynomial degree for rational OLS
    max_basis_channels : number of top channels to use as rational basis
    max_features       : max monomial features in rational model
    parsimony_rel_tol  : relative tolerance for parsimony-aware model selection
    complexity_max     : hard cap on number of fitted terms (None = no cap)
    loss_kind          : 'mse' or 'huber'
    huber_delta_frac   : fraction of std(y) used as Huber δ
    """

    _EPS = 1e-8

    _LINKS = {
        'id'  : (lambda y: y,                                    lambda p: p),
        'log' : (lambda y: np.log(np.abs(y) + 1e-8),            lambda p: np.exp(np.clip(p, -50, 50))),
        'sqrt': (lambda y: np.sqrt(np.abs(y)),                   lambda p: p ** 2),
        'sq'  : (lambda y: y ** 2,                               lambda p: np.sqrt(np.abs(p))),
        'inv' : (lambda y: 1.0 / (np.abs(y) + 1e-8),            lambda p: 1.0 / (np.abs(p) + 1e-8)),
        'cbrt': (lambda y: np.sign(y) * np.abs(y) ** (1/3),     lambda p: p ** 3),
    }
    _UNARIES = {
        'id'  : lambda x: x,
        'sq'  : lambda x: x ** 2,
        'sqrt': lambda x: np.sqrt(np.abs(x)),
        'inv' : lambda x: 1.0 / (np.abs(x) + 1e-8),
        'neg' : lambda x: -x,
        'log' : lambda x: np.log(np.abs(x) + 1e-8),
    }
    _U_WRAP = {
        'id'  : lambda f: f,
        'sq'  : lambda f: f"({f})**2",
        'sqrt': lambda f: f"sqrt(abs({f}))",
        'inv' : lambda f: f"1/({f})",
        'neg' : lambda f: f"-({f})",
        'log' : lambda f: f"log(abs({f}))",
    }
    _LINK_WRAP = {
        'id'  : lambda s: s,
        'log' : lambda s: f"exp({s})",
        'sqrt': lambda s: f"({s})**2",
        'sq'  : lambda s: f"sqrt(abs({s}))",
        'inv' : lambda s: f"1/({s})",
        'cbrt': lambda s: f"({s})**3",
    }
    _POST_ORDER = {"channel": 0, "linear": 0, "ols": 1, "nested": 2, "rational": 3}

    def __init__(
        self,
        rat_max_degree:     int   = _CfgOLS.RAT_MAX_DEGREE,
        max_basis_channels: int   = _CfgOLS.RAT_MAX_BASIS_CH,
        max_features:       int   = _CfgOLS.RAT_MAX_FEATURES,
        parsimony_rel_tol:  float = _CfgOLS.PARSIMONY_REL_TOL,
        complexity_max            = _CfgOLS.COMPLEXITY_MAX,
        loss_kind:          str   = _CfgLoss.KIND,
        huber_delta_frac:   float = _CfgLoss.HUBER_DELTA_FRAC,
    ):
        self.rat_max_degree     = rat_max_degree
        self.max_basis_channels = max_basis_channels
        self.max_features       = max_features
        self.parsimony_rel_tol  = parsimony_rel_tol
        self.complexity_max     = complexity_max
        self.loss_kind          = loss_kind
        self.huber_delta_frac   = huber_delta_frac

    # ── Public entry points ───────────────────────────────────────────────────

    def evaluate(self, pred_all, true_all, loss_mode: str = "full") -> tuple:
        """Dispatch OLS evaluation.

        Returns
        -------
        (mse_norm, selected_c, ols_weights, best_channel_loss,
         lr_stub, nested, rational, valid_idx_global, analysis)
        """
        y     = true_all.squeeze().numpy()
        var_y = float(np.var(y))
        if var_y < 1e-30:
            return 1.0, 0, None, 1.0, None, None, None, None, {}
        if pred_all.ndim == 1 or pred_all.shape[-1] == 1:
            return self._eval_single(pred_all, y, var_y, loss_mode)
        return self._eval_multi(pred_all, y, var_y, loss_mode)

    def print_analysis(self, analysis: dict, formulas: list, valid_idx_global,
                       selected_c: int, f=None):
        """Print the full OLS analysis (channel / sparse / nested / rational)."""
        def _p(msg):
            print(msg)
            if f is not None:
                f.write(msg + "\n")

        var_y    = analysis.get("var_y", 1.0)
        ch_res   = analysis.get("channel_results", [])
        n_ch     = analysis.get("n_ch", len(ch_res))
        ols_norm = analysis.get("ols_mse_norm")
        ols_bias = analysis.get("ols_bias", 0.0)
        ols_wts  = analysis.get("ols_weights")
        nested   = analysis.get("nested")
        rat_cands = analysis.get("rational_candidates", [])

        _p(f"\n-- Single-channel analysis ({n_ch} channels) --")
        best_ch_norm = np.inf
        for c, norm_c in ch_res:
            f_str = formulas[c] if c < len(formulas) else "?"
            if norm_c is None:
                _p(f"  ch[{c}]: constant or NaN (skipped)  formula={f_str}")
            else:
                tag = " <<< SELECTED" if c == selected_c else ""
                _p(f"  ch[{c}]: normMSE={norm_c:.10f}  R2={1.0-norm_c:.8f}  formula={f_str}{tag}")
                if norm_c < best_ch_norm:
                    best_ch_norm = norm_c

        if ols_norm is not None and ols_wts is not None:
            _p(f"\n-- Sparse OLS fit --")
            _p(f"  normMSE={ols_norm:.10f}  R2={1.0-ols_norm:.8f}")
            for vi, w in enumerate(ols_wts):
                if abs(w) > 1e-6:
                    _p(f"    w={w:+.6f}  [{vi}] {formulas[vi] if vi < len(formulas) else '?'}")
            _p(f"    bias = {ols_bias:+.6f}")

        if nested is not None:
            norm_n = nested["mse"] / var_y
            _p(f"\n-- Nested OLS --")
            _p(f"  link={nested['link']}")
            for j in range(n_ch):
                if nested.get("valid", [])[j] if j < len(nested.get("valid", [])) else False:
                    _p(f"    w={nested['w'][j]:+.6f}  unary={nested['unaries'][j]:5s}  "
                       f"[{j}] {formulas[j] if j < len(formulas) else '?'}")
            _p(f"    bias = {nested['b']:+.6f}")
            _p(f"  normMSE={norm_n:.10f}  R2={1.0-norm_n:.8f}")
            _p(f"  Formula = {self.format_nested(nested, formulas)}")
        else:
            _p("\n-- Nested OLS: no improvement --")

        _p(f"\n-- Poly-Rational OLS (deg 1 to {self.rat_max_degree}) --")
        best_rat = None; best_rat_norm = np.inf
        for rc in rat_cands:
            deg    = rc.get("max_degree", "?")
            norm_r = rc["mse"] / var_y
            _p(f"\n  > Degre {deg}:")
            _p(f"    normMSE={norm_r:.10f}  R2={1.0-norm_r:.8f}")
            _p(f"    Formula = {self.format_rational(rc, formulas, valid_idx_global)}")
            if norm_r < best_rat_norm:
                best_rat_norm = norm_r; best_rat = rc
        if not rat_cands:
            _p("  No rational formulation improved over baseline.")

        _p("\n-- Comparison --")
        if best_ch_norm < np.inf:
            _p(f"  Best channel [{selected_c}]:  normMSE={best_ch_norm:.10f}  R2={1.0-best_ch_norm:.8f}")
        if ols_norm is not None and ols_wts is not None:
            _p(f"  Sparse OLS:              normMSE={ols_norm:.10f}  R2={1.0-ols_norm:.8f}")
        if nested is not None:
            norm_n = nested["mse"] / var_y
            _p(f"  Nested OLS:              normMSE={norm_n:.10f}  R2={1.0-norm_n:.8f}  link={nested['link']}")
        if best_rat is not None:
            _p(f"  Best Poly-Rational:      normMSE={best_rat_norm:.10f}  "
               f"R2={1.0-best_rat_norm:.8f}  deg={best_rat['max_degree']}")

    def format_nested(self, nested: dict, formulas: list, min_weight: float = 1e-4) -> str:
        """Format a nested-OLS result as a human-readable formula string."""
        link    = nested["link"]
        unaries = nested["unaries"]
        w       = nested["w"]
        b       = nested["b"]
        valid   = nested["valid"]
        valid_w = [(i, wi) for i, wi in enumerate(w)
                   if i < len(valid) and valid[i] and abs(wi) >= min_weight and i < len(formulas)]
        if not valid_w:
            return f"{b:.4f}"
        used_u = {unaries[i] for i, _ in valid_w}
        if link == "log" and used_u == {"log"}:
            terms = [f"({formulas[i]})**({wi:.4f})" for i, wi in valid_w]
            return f"{np.exp(b):.4e} * " + " * ".join(terms)
        if link == "log":
            inner = " ".join(f"{wi:+.4f}*{self._U_WRAP[unaries[i]](formulas[i])}" for i, wi in valid_w)
            return f"{np.exp(b):.4e} * exp({inner})"
        if link == "inv" and used_u == {"inv"}:
            denom = " ".join(f"{wi:+.4f}/({formulas[i]})" for i, wi in valid_w)
            if abs(b) > min_weight:
                denom += f" {b:+.4f}"
            return f"1 / ({denom})"
        inner = " ".join(f"{wi:+.4f}*{self._U_WRAP[unaries[i]](formulas[i])}" for i, wi in valid_w)
        if abs(b) > min_weight:
            inner += f" {b:+.4f}"
        return self._LINK_WRAP[link](inner)

    def format_rational(self, rational: dict, formulas: list, valid_idx_global,
                        min_weight: float = 1e-4) -> str:
        """Format a poly-rational OLS result as a human-readable formula string."""
        from collections import Counter
        a, a0  = rational["a"], rational["a0"]
        b, b0  = rational["b"], rational["b0"]
        kept_a = rational["kept_a"]
        kept_b = rational["kept_b"]
        monos  = rational["monomials"]

        def mono_str(mono):
            counts = Counter(mono)
            parts  = []
            for li, exp in counts.items():
                gi    = int(valid_idx_global[li])
                f_str = formulas[gi] if gi < len(formulas) else f"ch{gi}"
                parts.append(f"({f_str})" if exp == 1 else f"({f_str})**{exp}")
            return "*".join(parts) if parts else "1"

        num = [f"{a[k]:+.4f}*{mono_str(m)}" for k, m in enumerate(monos)
               if kept_a[k] and abs(a[k]) >= min_weight]
        if abs(a0) >= min_weight or not num:
            num.append(f"{a0:+.4f}")
        den = [f"{b[k]:+.4f}*{mono_str(m)}" for k, m in enumerate(monos)
               if kept_b[k] and abs(b[k]) >= min_weight]
        den.append(f"{b0:+.4f}")
        return f"({' '.join(num)}) / ({' '.join(den)})"

    # ── Normalised loss ───────────────────────────────────────────────────────

    def _normalized_loss(self, pred, y, var_y) -> float:
        if var_y < 1e-30:
            return 1.0
        if self.loss_kind == "huber":
            delta = self.huber_delta_frac * float(np.sqrt(var_y))
            r     = np.abs(y - pred)
            quad  = np.minimum(r, delta)
            return float(np.mean(0.5 * quad ** 2 + delta * (r - quad)) / (0.5 * var_y))
        return float(np.mean((y - pred) ** 2) / var_y)

    # ── Parsimony-aware model selection ──────────────────────────────────────

    def _eff_params(self, cand: dict, min_w: float = 1e-4) -> int:
        kind = cand["kind"]
        if kind in ("channel", "linear"):
            return 1
        if kind == "ols":
            w = cand.get("ols_weights")
            return int(np.sum(np.abs(w) >= min_w)) if w is not None else 1
        if kind == "nested":
            nd = cand["nested"]
            return int(np.sum(nd["valid"] & (np.abs(nd["w"]) >= min_w)))
        if kind == "rational":
            r = cand["rational"]
            return int(np.sum(np.abs(r["a"]) >= min_w) + np.sum(np.abs(r["b"]) >= min_w))
        return 1

    def _select_model(self, cands: list) -> dict:
        for c in cands:
            c["_k"] = self._eff_params(c)
        pool = ([c for c in cands if c["_k"] <= self.complexity_max]
                if self.complexity_max is not None else cands)
        if not pool:
            pool = [min(cands, key=lambda c: c["_k"])]
        best_mse = min(c["mse_norm"] for c in pool)
        thresh   = best_mse * (1.0 + self.parsimony_rel_tol)
        near     = [c for c in pool if c["mse_norm"] <= thresh]
        return min(near, key=lambda c: (c["_k"], self._POST_ORDER.get(c["kind"], 9), c["mse_norm"]))

    # ── Linear solvers ────────────────────────────────────────────────────────

    def _safe_lstsq(self, P, y_t):
        if P.shape[0] < P.shape[1] + 1:
            return None, None, None
        A = np.hstack([P, np.ones((P.shape[0], 1))])
        try:
            sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
        except np.linalg.LinAlgError:
            return None, None, None
        return sol[:-1], sol[-1], A @ sol

    def _sparse_lstsq(self, P, y_t, rel_thresh=1e-3, abs_thresh=1e-8, max_iter=10):
        n, k = P.shape
        if n < k + 1:
            return None, None, None, None
        kept = np.ones(k, dtype=bool)
        for _ in range(max_iter):
            Pk = P[:, kept]
            if Pk.shape[1] == 0:
                return None, None, None, None
            A = np.hstack([Pk, np.ones((n, 1))])
            try:
                sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
            except np.linalg.LinAlgError:
                return None, None, None, None
            w_k    = sol[:-1]
            thresh = max(rel_thresh * np.max(np.abs(w_k)), abs_thresh)
            mask_k = np.abs(w_k) >= thresh
            if mask_k.all():
                break
            idx_keep = np.where(kept)[0][mask_k]
            new_kept = np.zeros(k, dtype=bool); new_kept[idx_keep] = True
            if new_kept.sum() == 0:
                return None, None, None, None
            kept = new_kept
        Pk  = P[:, kept]
        A   = np.hstack([Pk, np.ones((n, 1))])
        sol, *_ = np.linalg.lstsq(A, y_t, rcond=None)
        return sol[:-1], float(sol[-1]), A @ sol, kept

    # ── Nested OLS ────────────────────────────────────────────────────────────

    def _nested_ols(self, channels, y, mse_floor):
        n_ch = channels.shape[1]
        best = None
        for link_name, (fwd, inv) in self._LINKS.items():
            with np.errstate(all='ignore'):
                y_t = fwd(y)
            if not np.all(np.isfinite(y_t)) or np.std(y_t) < 1e-12:
                continue
            y_t_c        = y_t - y_t.mean()
            unaries_pick = []
            chans_t      = np.empty_like(channels)
            valid_mask   = np.zeros(n_ch, dtype=bool)
            for j in range(n_ch):
                best_corr, best_u, best_vals = -1.0, 'id', channels[:, j]
                for u_name, u_fn in self._UNARIES.items():
                    with np.errstate(all='ignore'):
                        v = u_fn(channels[:, j])
                    if not np.all(np.isfinite(v)) or np.std(v) < 1e-12:
                        continue
                    v_c   = v - v.mean()
                    denom = np.sqrt((v_c ** 2).sum() * (y_t_c ** 2).sum()) + 1e-30
                    c     = abs((v_c * y_t_c).sum() / denom)
                    if np.isfinite(c) and c > best_corr:
                        best_corr, best_u, best_vals = c, u_name, v
                unaries_pick.append(best_u)
                if best_corr > 0:
                    chans_t[:, j] = best_vals; valid_mask[j] = True
            if valid_mask.sum() == 0:
                continue
            w_kept, b, pred_t, kept_local = self._sparse_lstsq(chans_t[:, valid_mask], y_t)
            if w_kept is None:
                continue
            valid_idx_sub = np.where(valid_mask)[0]
            new_valid     = np.zeros(n_ch, dtype=bool)
            new_valid[valid_idx_sub[kept_local]] = True
            with np.errstate(all='ignore'):
                pred = inv(pred_t)
            if not np.all(np.isfinite(pred)):
                continue
            mse = float(np.mean((y - pred) ** 2))
            if not np.isfinite(mse) or mse >= mse_floor:
                continue
            w_full = np.zeros(n_ch); w_full[new_valid] = w_kept
            best      = dict(mse=mse, link=link_name, unaries=unaries_pick,
                             w=w_full, b=b, valid=new_valid, pred=pred)
            mse_floor = mse
        return best

    # ── Poly-rational OLS ─────────────────────────────────────────────────────

    def _build_poly_features(self, channels, max_degree: int):
        n, k         = channels.shape
        feats, names = [], []
        for j in range(k):
            feats.append(channels[:, j]); names.append((j,))
        if max_degree >= 2:
            for j in range(k):
                feats.append(channels[:, j] ** 2); names.append((j, j))
            for i in range(k):
                for j in range(i + 1, k):
                    feats.append(channels[:, i] * channels[:, j]); names.append((i, j))
        if not feats:
            return np.empty((n, 0)), []
        return np.column_stack(feats), names

    def _poly_rational_ols(self, channels, y, mse_floor, max_degree,
                           rel_thresh=1e-3, abs_thresh=1e-8, max_iter=10):
        n, k = channels.shape
        if k == 0 or n < 4:
            return None
        y_c   = y - y.mean()
        sy    = (y_c ** 2).sum() ** 0.5 + 1e-30
        corrs = np.array([
            abs(((channels[:, j] - channels[:, j].mean()) * y_c).sum() /
                ((((channels[:, j] - channels[:, j].mean()) ** 2).sum() ** 0.5 + 1e-30) * sy))
            for j in range(k)])
        sel        = np.argsort(-corrs)[:min(k, self.max_basis_channels)]
        feats, mono_local = self._build_poly_features(channels[:, sel], max_degree)
        if feats.shape[1] == 0:
            return None
        ok = np.array([np.all(np.isfinite(feats[:, j])) and np.std(feats[:, j]) > 1e-15
                       for j in range(feats.shape[1])])
        if not ok.any():
            return None
        feats      = feats[:, ok]
        mono_local = [m for m, ko in zip(mono_local, ok) if ko]
        if feats.shape[1] > self.max_features:
            sc   = np.array([abs(((feats[:, j] - feats[:, j].mean()) * y_c).sum() /
                                 ((((feats[:, j] - feats[:, j].mean()) ** 2).sum() ** 0.5 + 1e-30) * sy))
                             for j in range(feats.shape[1])])
            keep       = np.argsort(-sc)[:self.max_features]
            feats      = feats[:, keep]
            mono_local = [mono_local[i] for i in keep]
        m = feats.shape[1]
        if n < 2 * m + 2:
            sc   = np.array([abs(np.corrcoef(feats[:, j], y)[0, 1])
                             if np.std(feats[:, j]) > 1e-15 else 0.0 for j in range(m)])
            keep = np.argsort(-sc)[:max(1, (n - 2) // 2)]
            feats = feats[:, keep]; mono_local = [mono_local[i] for i in keep]
            m = feats.shape[1]
            if m == 0:
                return None

        def build_A(ka, kb):
            cols = []
            if ka.any(): cols.append(feats[:, ka])
            cols.append(np.ones((n, 1)))
            if kb.any(): cols.append(-feats[:, kb] * y[:, None])
            return np.hstack(cols)

        kept_a = np.ones(m, dtype=bool); kept_b = np.ones(m, dtype=bool)
        for _ in range(max_iter):
            A = build_A(kept_a, kept_b)
            try:
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            except np.linalg.LinAlgError:
                return None
            ka   = int(kept_a.sum()); kb = int(kept_b.sum())
            a_k  = sol[:ka]; b_k = sol[ka + 1: ka + 1 + kb]
            all_abs = np.concatenate([np.abs(a_k), np.abs(b_k)]) if ka + kb else np.array([0.0])
            thresh  = max(rel_thresh * float(all_abs.max()), abs_thresh)
            na = np.abs(a_k) >= thresh; nb = np.abs(b_k) >= thresh
            if na.all() and nb.all():
                break
            new_ka = np.zeros(m, dtype=bool); new_ka[np.where(kept_a)[0][na]] = True
            new_kb = np.zeros(m, dtype=bool); new_kb[np.where(kept_b)[0][nb]] = True
            if not new_ka.any() and not new_kb.any():
                return None
            kept_a, kept_b = new_ka, new_kb

        A = build_A(kept_a, kept_b)
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        ka  = int(kept_a.sum()); kb = int(kept_b.sum())
        a   = np.zeros(m); a[kept_a] = sol[:ka]
        a0  = float(sol[ka])
        b   = np.zeros(m); b[kept_b] = sol[ka + 1: ka + 1 + kb]
        numer = feats @ a + a0
        denom = feats @ b + 1.0
        if not np.all(np.abs(denom) > 1e-10):
            return None
        pred = numer / denom
        if not np.all(np.isfinite(pred)):
            return None
        mse = float(np.mean((y - pred) ** 2))
        if not np.isfinite(mse) or mse >= mse_floor:
            return None
        mono_global = [tuple(int(sel[i]) for i in mn) for mn in mono_local]
        return dict(mse=mse, a=a, a0=a0, b=b, b0=1.0,
                    kept_a=kept_a, kept_b=kept_b,
                    monomials=mono_global, pred=pred, max_degree=max_degree)

    # ── Evaluation paths ──────────────────────────────────────────────────────

    def _eval_single(self, pred_all, y, var_y, loss_mode):
        ch = pred_all.squeeze().numpy()
        if not np.all(np.isfinite(ch)) or np.std(ch) < 1e-12:
            return 1.0, 0, None, 1.0, None, None, None, np.array([0]), {}
        if loss_mode == "channel":
            r = correl(ch, y)
            l = float(1 - r ** 2) if np.isfinite(r) else 1.0
            return l, 0, None, l, None, None, None, np.array([0]), {}

        w, b, pred = self._safe_lstsq(ch.reshape(-1, 1), y)
        if w is None:
            r = correl(ch, y)
            l = float(1 - r ** 2) if np.isfinite(r) else 1.0
            return l, 0, None, l, None, None, None, np.array([0]), {}

        mse_lin = self._normalized_loss(pred, y, var_y)
        if loss_mode == "ols":
            ana = {"var_y": var_y, "channel_results": [(0, mse_lin)], "n_ch": 1,
                   "ols_mse_norm": mse_lin, "ols_kept_global": None,
                   "ols_bias": float(b), "ols_weights": None,
                   "nested": None, "rational_candidates": []}
            return mse_lin, 0, None, mse_lin, None, None, None, np.array([0]), ana

        ch_mat    = ch.reshape(-1, 1)
        nested    = self._nested_ols(ch_mat, y, mse_floor=mse_lin * var_y)
        floor     = min(mse_lin, (nested["mse"] / var_y) if nested else mse_lin) * var_y
        rat_cands = []; rational = None; best_rat_mse = np.inf
        for deg in range(1, self.rat_max_degree + 1):
            rc = self._poly_rational_ols(ch_mat, y, floor, max_degree=deg)
            if rc is not None:
                rat_cands.append(rc)
                if rational is None or rc["mse"] < best_rat_mse:
                    rational = rc; best_rat_mse = rc["mse"]

        cands = [{"kind": "linear", "mse_norm": mse_lin}]
        if nested:   cands.append({"kind": "nested",   "mse_norm": nested["mse"] / var_y,   "nested":   nested})
        if rational: cands.append({"kind": "rational", "mse_norm": rational["mse"] / var_y, "rational": rational})
        winner = self._select_model(cands)

        ana = {"var_y": var_y, "channel_results": [(0, mse_lin)], "n_ch": 1,
               "ols_mse_norm": mse_lin, "ols_kept_global": None, "ols_bias": float(b),
               "ols_weights": None, "nested": nested, "rational_candidates": rat_cands}
        if winner["kind"] == "rational":
            return rational["mse"] / var_y, 0, None, mse_lin, None, None, rational, np.array([0]), ana
        if winner["kind"] == "nested":
            return nested["mse"] / var_y, 0, None, mse_lin, None, nested, None, np.array([0]), ana
        return mse_lin, 0, None, mse_lin, None, None, None, np.array([0]), ana

    def _eval_multi(self, pred_all, y, var_y, loss_mode):
        P    = pred_all.numpy()
        n_ch = P.shape[1]

        best_ch_loss = 1.0; selected_c = 0; ch_res = []
        for c in range(n_ch):
            if not np.all(np.isfinite(P[:, c])) or np.std(P[:, c]) < 1e-12:
                ch_res.append((c, None)); continue
            w, b, pred = self._safe_lstsq(P[:, c:c+1], y)
            if w is None:
                ch_res.append((c, None)); continue
            loss = self._normalized_loss(pred, y, var_y)
            ch_res.append((c, loss))
            if loss < best_ch_loss:
                best_ch_loss = loss; selected_c = c

        if loss_mode == "channel":
            ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
                   "ols_mse_norm": None, "ols_kept_global": None, "ols_bias": None,
                   "ols_weights": None, "nested": None, "rational_candidates": []}
            return best_ch_loss, selected_c, None, best_ch_loss, None, None, None, None, ana

        mask             = np.array([np.all(np.isfinite(P[:, c])) and np.std(P[:, c]) > 1e-15
                                      for c in range(n_ch)])
        valid_idx_global = np.where(mask)[0]
        ols_loss = 1.0; ols_weights = None; ols_bias = 0.0; ols_kept = None; lr_obj = None
        if mask.sum() > 0:
            try:
                lr   = Ridge(alpha=1e-8).fit(P[:, mask], y)
                r2   = lr.score(P[:, mask], y)
                if np.isfinite(r2) and r2 > 0:
                    ols_loss    = float(1.0 - r2)
                    ols_kept    = np.zeros(n_ch, dtype=bool); ols_kept[valid_idx_global] = True
                    ols_weights = np.zeros(n_ch); ols_weights[mask] = lr.coef_
                    ols_bias    = float(lr.intercept_)
                    class _LR: pass
                    lr_obj = _LR()
                    lr_obj.coef_ = ols_weights; lr_obj.intercept_ = lr.intercept_; lr_obj._kept = ols_kept
            except Exception:
                pass

        if loss_mode == "ols":
            ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
                   "ols_mse_norm": ols_loss, "ols_kept_global": ols_kept, "ols_bias": ols_bias,
                   "ols_weights": ols_weights, "nested": None, "rational_candidates": []}
            if ols_weights is not None and ols_loss <= best_ch_loss:
                return ols_loss, selected_c, ols_weights, best_ch_loss, lr_obj, None, None, valid_idx_global, ana
            return best_ch_loss, selected_c, None, best_ch_loss, None, None, None, valid_idx_global, ana

        # ── Full: nested + rational ───────────────────────────────────────────
        nested = None
        if mask.sum() > 0:
            P_v   = P[:, mask]
            y_c   = y - y.mean(); sy = np.sqrt((y_c ** 2).sum()) + 1e-30
            corrs = np.array([
                abs(((P_v[:, j] - P_v[:, j].mean()) * y_c).sum() /
                    ((((P_v[:, j] - P_v[:, j].mean()) ** 2).sum() ** 0.5 + 1e-30) * sy))
                if np.all(np.isfinite(P_v[:, j])) and np.std(P_v[:, j]) > 1e-15 else 0.0
                for j in range(P_v.shape[1])])
            top_local  = np.argsort(-corrs)[:self.max_basis_channels]
            top_global = valid_idx_global[top_local]
            ns = self._nested_ols(P[:, top_global], y,
                                  mse_floor=min(ols_loss, best_ch_loss) * var_y)
            if ns is not None:
                w_full = np.zeros(n_ch); w_full[top_global] = ns["w"]
                u_full = ["id"] * n_ch
                for ki, j in enumerate(top_global):
                    u_full[j] = ns["unaries"][ki]
                vf = np.zeros(n_ch, dtype=bool); vf[top_global] = ns["valid"]
                nested = dict(mse=ns["mse"], link=ns["link"], unaries=u_full,
                              w=w_full, b=ns["b"], valid=vf, pred=ns["pred"])

        rat_cands = []; rational = None; best_rat_mse = np.inf
        if mask.sum() > 0:
            floor = min(ols_loss, best_ch_loss,
                        (nested["mse"] / var_y) if nested else np.inf) * var_y
            for deg in range(1, self.rat_max_degree + 1):
                rc = self._poly_rational_ols(P[:, mask], y, floor, max_degree=deg)
                if rc is not None:
                    rat_cands.append(rc)
                    if rational is None or rc["mse"] < best_rat_mse:
                        rational = rc; best_rat_mse = rc["mse"]

        ana = {"var_y": var_y, "channel_results": ch_res, "n_ch": n_ch,
               "ols_mse_norm": ols_loss, "ols_kept_global": ols_kept, "ols_bias": ols_bias,
               "ols_weights": ols_weights, "nested": nested, "rational_candidates": rat_cands}

        cands = [{"kind": "channel", "mse_norm": best_ch_loss}]
        if ols_weights is not None: cands.append({"kind": "ols",      "mse_norm": ols_loss,              "ols_weights": ols_weights})
        if nested:                  cands.append({"kind": "nested",   "mse_norm": nested["mse"] / var_y, "nested":      nested})
        if rational:                cands.append({"kind": "rational", "mse_norm": rational["mse"] / var_y,"rational":   rational})
        winner = self._select_model(cands)

        if winner["kind"] == "rational":
            return rational["mse"] / var_y, selected_c, ols_weights, best_ch_loss, lr_obj, None, rational, valid_idx_global, ana
        if winner["kind"] == "nested":
            return nested["mse"] / var_y, selected_c, ols_weights, best_ch_loss, lr_obj, nested, None, valid_idx_global, ana
        if winner["kind"] == "ols":
            return ols_loss, selected_c, ols_weights, best_ch_loss, lr_obj, None, None, valid_idx_global, ana
        return best_ch_loss, selected_c, None, best_ch_loss, None, None, None, valid_idx_global, ana
