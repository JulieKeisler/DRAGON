from __future__ import annotations

import os
import json
import multiprocessing as mp

import numpy as np

from Config import (
    Experiment as _CfgExp,
    Paths as _CfgPaths,
    PySR as _CfgPySR,
    DRAGON_METHODS as DRAGON_METHOD_CONFIGS,
)
from dataprocessing.Data import DatasetLoader
from runner.Dragon import run_dragon_method
from helpers.html_builder import build_html
from helpers.Helper import apply_text_config
from pathlib import Path


def _get_run_pysr():
    from runner.pysr_runner import run_pysr
    return run_pysr


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_all(targets=None, n_runs=None, resume=False):
    """Run Dragon-only, then PySR-only, avoiding duplicate code paths."""
    if targets is None:
        targets = _CfgExp.TARGETS
    if n_runs is None:
        n_runs = _CfgExp.N_RUNS
    os.makedirs(_CfgPaths.OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(_CfgPaths.OUTPUT_DIR, "results.json")
    if not resume and os.path.exists(results_path):
        try:
            os.remove(results_path)
            print(f"[run_all] Removed existing results.json for a fresh run: {results_path}")
        except Exception as _e:
            print(f"[run_all] Could not remove existing results.json ({_e}); continuing anyway")
    elif resume:
        print("[run_all] Resuming from existing results.json")

    run_dragon_only(targets, n_runs)
    return run_pysr_only(targets, n_runs)


def run_pysr_only(targets=None, n_runs=None):
    """Complete missing PySR entries in an existing results.json, then rebuild HTML.

    Entries with formula starting with 'ERROR:' or 'SKIP' are treated as
    failed and will be re-run (removed from results before re-running).
    """
    if targets is None:
        targets = _CfgExp.TARGETS
    if n_runs is None:
        n_runs = _CfgExp.N_RUNS
    os.makedirs(_CfgPaths.OUTPUT_DIR, exist_ok=True)
    add_noise_setting = getattr(_CfgPySR, "ADD_NOISE", None)
    if add_noise_setting is True:
        add_noise_values = (True,)
    elif add_noise_setting is False:
        add_noise_values = (False,)
    else:
        add_noise_values = (False, True)

    results_path = os.path.join(_CfgPaths.OUTPUT_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_path}")
    else:
        results = []
        print("No existing results.json found — starting from scratch.")

    def _is_failed(r):
        fo = str(r.get("formula", "")).strip()
        if fo in ("", "N/A", "TIMEOUT", "PySR: no result"):
            return True
        return fo.startswith(("ERROR:", "SKIP", "PySR: no result"))

    # Remove failed PySR entries so they can be re-run
    results = [r for r in results
               if r.get("method") not in ("pysr", "pysr_noise") or not _is_failed(r)]

    done = {(r["target"], int(r["run_id"]), r["method"]) for r in results}
    loader = DatasetLoader()

    for target in targets:
        for run_id in range(n_runs):
            _seed_data = _CfgExp.RANDOM_SEED + run_id
            _X_base, _y_clean = loader.load(target, run_id=run_id)
            _noise_rng = np.random.default_rng(_seed_data + 9999)
            _y_noisy   = _y_clean + _noise_rng.normal(
                0, _CfgExp.NOISE_STD * float(_y_clean.std()), len(_y_clean)
            )
            for add_noise in add_noise_values:
                method_id = "pysr_noise" if add_noise else "pysr"
                if (target, run_id, method_id) in done:
                    print(f"  [SKIP] {target}/run_{run_id}/{method_id} already present")
                    continue
                print(f"  [RUN]  {target}/run_{run_id}/{method_id}")
                r = _get_run_pysr()(target, run_id, add_noise=add_noise,
                                     _X_preloaded=_X_base, _y_clean_preloaded=_y_clean,
                                     _y_noisy_preloaded=_y_noisy)
                results.append(r)
                done.add((target, run_id, method_id))
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                build_html(results)
    return results


def run_dragon_only(targets=None, n_runs=None, build_html_full=True):
    """Complete missing Dragon entries in an existing results.json, then rebuild HTML.

    Entries with formula 'N/A', '' or starting with 'ERROR:' are treated as
    failed and will be re-run.
    """
    if targets is None:
        targets = _CfgExp.TARGETS
    if n_runs is None:
        n_runs = _CfgExp.N_RUNS

    results_path = Path(_CfgPaths.OUTPUT_DIR, "results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    print(
        f"Loaded {len(results)} existing results from {results_path}"
        if results_path.exists()
        else "No existing results.json found — starting from scratch."
    )

    dragon_method_ids = {cfg["id"] for cfg in DRAGON_METHOD_CONFIGS}

    def _is_failed(r):
        fo = str(r.get("formula", "")).strip()
        return fo in ("", "N/A") or fo.startswith("ERROR:")

    # Remove failed Dragon entries so they can be re-run
    results = [r for r in results
               if r.get("method") not in dragon_method_ids or not _is_failed(r)]

    done = {(r["target"], int(r["run_id"]), r["method"]) for r in results}
    new_results = []

    for target in targets:
        for run_id in range(n_runs):
            for cfg in DRAGON_METHOD_CONFIGS:
                method_id = cfg["id"]
                if (target, run_id, method_id) in done:
                    print(f"  [SKIP] {target}/run_{run_id}/{method_id} already present")
                    continue
                print(f"  [RUN]  {target}/run_{run_id}/{method_id}")
                r = run_dragon_method(cfg, target, run_id)
                results.append(r)
                new_results.append(r)
                done.add((target, run_id, method_id))
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                if build_html_full:
                    build_html(results)
                else:
                    build_html(new_results, full_leaderboard=False)
    if not build_html_full and not new_results and results:
        print("[run_dragon_only] no new Dragon results were generated; rendering full leaderboard instead.")
        build_html(results, full_leaderboard=True)
    return results if build_html_full else (new_results if new_results else results)


if __name__ == "__main__":
    import argparse
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dragonsr", action="store_true",
                        help="Run DragonSR only (portable CLI mode)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to a CSV dataset used directly for all non-synthetic targets")
    parser.add_argument("--config_file_path", type=str, default=None,
                        help="Path to a text config file overriding values in Config.py")
    parser.add_argument("--pysr-only", action="store_true",
                        help="Skip DragonSR; only run missing PySR entries and rebuild HTML")
    parser.add_argument("--dragon-only", action="store_true",
                        help="Skip PySR; only run missing Dragon entries and rebuild HTML")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Build the full leaderboard HTML from existing results; can be combined with --dragon-only, --pysr-only, or --continue")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Resume run_all() from an existing results.json instead of starting fresh")
    args = parser.parse_args()

    if args.config_file_path:
        apply_text_config(args.config_file_path)
        if isinstance(_CfgExp.INIT_STRATEGIES, list) and len(_CfgExp.INIT_STRATEGIES) < _CfgExp.N_RUNS:
            _CfgExp.INIT_STRATEGIES += [_CfgExp.INIT_STRATEGIES[-1]] * (_CfgExp.N_RUNS - len(_CfgExp.INIT_STRATEGIES))

    if args.data_path:
        _CfgPaths.EXTERNAL_DATA_CSV = args.data_path

    if args.leaderboard:
        if args.dragon_only:
            run_dragon_only()
        elif args.pysr_only:
            run_pysr_only()
        else:
            results_path = Path(_CfgPaths.OUTPUT_DIR, "results.json")
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                print(f"Generating full leaderboard HTML from {len(results)} existing results")
            else:
                results = []
                print("Generating full leaderboard HTML from an empty results set")
            build_html(results)
    elif args.run_dragonsr:
        run_dragon_only(build_html_full=False)
    elif args.dragon_only:
        run_dragon_only()
    elif args.pysr_only:
        run_pysr_only()
    else:
        run_all(resume=args.resume)
