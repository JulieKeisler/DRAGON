import os
import time
import traceback

import torch
from Config import (Experiment as _CfgExp, Paths as _CfgPaths,
                    Dragon as _CfgDragon)
import numpy as np

from runner.Ols import OLSPostProcessor
from helpers.stats import _collect_landscape
from helpers.Helper import _build_result
from pipeline.DragonOrchestrator import DragonOrchestrator
from target_categories import get_spar_groups_for_target

def dragon_worker(method_cfg: dict, target: str, run_id: int,
                  *, _max_iters: int = None,
                  _X_preprocessed=None, _y_preprocessed=None,
                  _feature_names_preloaded=None, _feat_scores_preloaded=None) -> dict:
    """Entry point for each parallel DRAGON process.

    method_cfg : one element of DRAGON_METHODS
    run_id     : 0..N_RUNS-1, also indexes Experiment.INIT_STRATEGIES
    """
    method_id = method_cfg["id"] #todo: see if it's not redondant with the prepare_data call
    strategy = _CfgExp.INIT_STRATEGIES[run_id]
    t_start = time.time(); best_loss = np.inf; best_formula = "N/A"; loss_state = {}
    X_sel = _X_preprocessed
    y = _y_preprocessed.copy()
    feature_names = _feature_names_preloaded
    feat_scores = _feat_scores_preloaded
    seed = _CfgExp.RANDOM_SEED + run_id
    # Seed the process RNG once.  The SPAR streams are THREADS sharing one
    # global numpy/random state, so seeding inside each stream would reseed
    # that shared state concurrently and correlate the streams; the parent
    # (_run_smart_parallel) seeds instead, and a lone worker seeds here.
    if not method_cfg.get("_in_smart_stream"):
        from dragon.utils.tools import set_seed
        set_seed(seed, change_random=True)

    try:
        run_dir = os.path.join(
            _CfgPaths.OUTPUT_DIR,
            target,
            method_id,
            f"run_{run_id}",
            *( [f"stream_{method_cfg['_stream_id']}"] if method_cfg.get("_in_smart_stream") else []),
        )
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, f"{target}_{method_id}{_CfgPaths.LOG_SUFFIX}")
        save_dir = os.path.join(run_dir, "save")
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_loss, best_formula, loss_state = DragonOrchestrator.run_worker_pipeline(
            method_cfg=method_cfg,
            X_sel=X_sel,
            y=y,
            feature_names=feature_names,
            feat_scores=feat_scores,
            log_path=log_path,
            seed=seed,
            device=device,
            run_id=run_id,
            strategy=strategy,
            save_dir=save_dir,
            _max_iters=_max_iters,
            target=target,
        )

    except Exception:
        tb = traceback.format_exc()
        best_formula = f"ERROR: {tb[:200]}"
        print(f"[{method_id}/{target}/run{run_id}] ERROR:\n{tb}")

    actual_T, loss_history, landscape_svg = _collect_landscape(save_dir)
    return _build_result(target, method_cfg, run_id, strategy, best_loss, best_formula,
                         log_path, loss_state, actual_T, loss_history, landscape_svg,
                         time.time() - t_start)




def _run_smart_parallel(method_cfg, target, run_id, *,
                        _max_iters,
                        _X_preprocessed=None, _y_preprocessed=None,
                        _feature_names_preloaded=None, _feat_scores_preloaded=None) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    method_id      = method_cfg["id"]

    # Use target-aware SPAR operator groups
    _spar_groups = get_spar_groups_for_target(target)
    # Optionally cross every SPAR operator group with several loss_modes, so the
    # search is run under each objective instead of committing to one globally.
    # All modes score against the SAME target here, so cross-stream loss stays comparable.
    _loss_modes = method_cfg.get("SPAR_LOSS_MODES") or [None]
    if not isinstance(_loss_modes, (list, tuple)):
        _loss_modes = [_loss_modes]
    _n_streams     = len(_spar_groups) * len(_loss_modes)
    # No explicit budget from the caller means "use the configured one";
    # a hard-coded fallback here silently overrode N_ITERATIONS from the config.
    _iter_cap      = _max_iters if _max_iters is not None else _CfgDragon.N_ITERATIONS
    _stream_budget = max(1, _iter_cap // _n_streams)

    # One deterministic seeding per run, before any stream starts.  Before this
    # the search drew from an unseeded process-wide RNG, so no run was
    # reproducible.  Thread interleaving still orders the draws, so a run is
    # reproducible in its starting state, not bitwise; serialising the streams
    # would be required for that.
    from dragon.utils.tools import set_seed
    set_seed(_CfgExp.RANDOM_SEED + run_id, change_random=True)

    stream_results = []
    with ThreadPoolExecutor(max_workers=_n_streams) as ex:
        futs = {}
        for stream_id, ops in _spar_groups.items():
          for _lm in _loss_modes:
            _sid = stream_id if _lm is None else f"{stream_id}|{_lm}"
            sub_cfg = {**method_cfg, "operators": list(ops),
                       "_in_smart_stream": True, "_stream_id": _sid}
            if _lm is not None:
                sub_cfg["loss_mode"] = _lm
            futs[ex.submit(dragon_worker, sub_cfg, target, run_id,
                           _max_iters=_stream_budget,
                           _X_preprocessed=_X_preprocessed, _y_preprocessed=_y_preprocessed,
                           _feature_names_preloaded=_feature_names_preloaded,
                           _feat_scores_preloaded=_feat_scores_preloaded)] = stream_id
        for f in as_completed(futs):
            try:
                stream_results.append(f.result())
            except Exception:
                print(f"[spar/{target}/run{run_id}] stream {futs[f]} ERROR:")
                traceback.print_exc()

    if not stream_results:
        return {
            "target": target, "method": method_id, "run_id": run_id,
            "strategy": _CfgExp.INIT_STRATEGIES[run_id],
            "loss": float(np.inf), "r2": 0.0,
            "formula": "ERROR: all smart-parallel streams failed",
            "time_s": 0.0, "log_path": "", "description": method_cfg["description"],
            "search_space_ops": list(method_cfg.get("operators", [])),
        }

    # Copy winner payload to avoid mutating the original stream result dict.
    # Streams routinely TIE at machine precision (loss 0 or ~1e-16) when the model is
    # exact; plain min() then returns whichever future completed first, which is random
    # and often an ugly-but-equivalent rendering (or a worse model that fits equally).
    # Among near-tied streams prefer the simpler model class, then the shorter formula.
    _TIE_REL, _TIE_ABS = 0.02, 1e-10
    _ORDER = {"channel": 0, "linear": 0, "ols": 1, "nested": 2, "rational": 3}

    def _loss_of(r):
        v = r.get("loss")
        return float("inf") if v is None else v

    _lo = min((_loss_of(r) for r in stream_results), default=float("inf"))
    if _lo == float("inf"):
        _near_ties = list(stream_results)
    else:
        _cut = _lo + max(abs(_lo) * _TIE_REL, _TIE_ABS)
        _near_ties = [r for r in stream_results if _loss_of(r) <= _cut] or list(stream_results)

    def _has_var(r):
        # A bare constant ("0.0000", "-3.5e+15") is degenerate: on targets with huge
        # dynamic range it can land in the tie window with loss 0. Never prefer one.
        return any(c.isalpha() for c in str(r.get("formula") or ""))

    def _simplicity(r):
        return (0 if _has_var(r) else 1,
                _ORDER.get(r.get("winner_type"), 9),
                len(str(r.get("formula") or "")),
                _loss_of(r))

    if len(_near_ties) > 1:
        print(f"[spar/{target}/run{run_id}] {len(_near_ties)} streams within tie window "
              f"of {_lo:.6g}; picking simplest")
    best = dict(min(_near_ties, key=_simplicity))
    best.update(method=method_id, description=method_cfg["description"],
                search_space_ops=list(method_cfg.get("operators", [])))
    per_stream_T = {r.get("_stream_id", "?"): int(r["actualT"])
                    for r in stream_results if r.get("actualT") is not None}
    best["actualT_per_stream"] = per_stream_T
    best["actualT_winner"]     = best.get("actualT")
    if per_stream_T:
        best["actualT"] = int(sum(per_stream_T.values()))
    best["smart_parallel_streams"] = [
        {"stream": r.get("_stream_id", "?"), "ops": r.get("search_space_ops", []),
         "loss": r.get("loss"), "formula": r.get("formula"),
         "time_s": r.get("time_s"), "actualT": r.get("actualT")}
        for r in stream_results
    ]

    if method_cfg.get("boosted"):
        try:
            preds, sids, y_ref, ref_order = [], [], None, None
            for r in stream_results:
                p  = r.get("_best_pred_np"); yy = r.get("_best_y_np")
                if p is None or yy is None: continue
                p  = np.asarray(p,  dtype=np.float64).ravel()
                yy = np.asarray(yy, dtype=np.float64).ravel()
                if p.shape != yy.shape or p.size < 4: continue
                if not np.all(np.isfinite(p)): continue
                if y_ref is None:
                    y_ref = yy.copy(); ref_order = np.argsort(y_ref, kind="stable")
                else:
                    if yy.size != y_ref.size: continue
                    if not np.allclose(np.sort(yy), np.sort(y_ref), rtol=1e-8, atol=1e-12): continue
                p_aligned = np.empty_like(p)
                p_aligned[ref_order] = p[np.argsort(yy, kind="stable")]
                preds.append(p_aligned); sids.append(r.get("_stream_id", "?"))
            if len(preds) >= 2:
                P = np.column_stack(preds).astype(np.float64)
                (mse_b, sel_c_b, w_b, ch_loss_b,
                 _lr_b, nested_b, rational_b, _vidx_b, ana_b) = OLSPostProcessor().evaluate(
                    torch.tensor(P), torch.tensor(y_ref).reshape(-1, 1), "full")
                if np.isfinite(mse_b) and mse_b < best.get("loss", float("inf")):
                    if rational_b is not None:
                        wtype = "boosted-polyrat"
                        f_str = (f"poly-rational(deg≤{rational_b.get('max_degree','?')}) "
                                 f"over {len(sids)} streams")
                    elif nested_b is not None:
                        wtype = "boosted-nested"; f_str = f"nested-OLS over {len(sids)} streams"
                    elif w_b is not None and np.any(np.asarray(w_b) != 0):
                        wtype = "boosted-ols"
                        wts  = np.asarray(w_b).ravel()
                        bias = float(ana_b.get("ols_bias", 0.0)) if ana_b else 0.0
                        terms = [f"{float(w):+.4g}·ŷ_{sid}"
                                 for sid, w in zip(sids, wts) if abs(float(w)) > 1e-8]
                        if abs(bias) > 1e-8: terms.append(f"{bias:+.4g}")
                        f_str = " ".join(terms) if terms else "(boosted-OLS)"
                    else:
                        wtype = "boosted-channel"; f_str = f"best ŷ_{sids[sel_c_b]} (boosted)"
                    best.update(loss=float(mse_b),
                                r2=float(1.0 - mse_b) if mse_b <= 1.0 else 0.0,
                                formula="boosted: " + f_str, winner_type=wtype,
                                alignment_loss=float(ch_loss_b) if ch_loss_b is not None else float(mse_b),
                                boosted_streams=list(sids))
        except Exception as _e:
            print(f"[{method_id}/{target}/run{run_id}] boost FAILED: {_e}")

    for r in stream_results:
        r.pop("_best_pred_np", None); r.pop("_best_y_np", None)
    best.pop("_best_pred_np", None); best.pop("_best_y_np", None)
    return best