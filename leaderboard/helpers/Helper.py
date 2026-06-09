from __future__ import annotations
import os
import sys
import random
import numpy as np
import torch.nn as nn
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")))
from dragon.search_space.dag_encoding import AdjMatrix, SymbolicNode
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import (
    SelectFeatures, Inverse, Negate, Power, SumFeatures, ConstantBrick,
)


def _build_random_seed_dags(feature_names, operator_keys, seed=0, n_seeds=200):
    """Build a population of seed DAGs for the 'random' init strategy."""
    rng = random.Random(seed)
    if not feature_names:
        return None

    op_pool = [
        ("Identity", Identity, {}, nn.Identity()),
        ("Inverse", Inverse, {}, nn.Identity()),
        ("Negate", Negate, {}, nn.Identity()),
        ("SumFeatures", SumFeatures, {}, nn.Identity()),
        *[(f"Power_{exp}", Power, {"exponent": exp}, nn.Identity())
          for exp in (-3, -2, -1, 1, 2, 3)],
        *[(f"Sel_{i}", SelectFeatures, {"indices": [i]}, nn.Identity())
          for i in range(len(feature_names))],
        *[(f"Sel_{i}_{j}", SelectFeatures, {"indices": [i, j]}, nn.Identity())
          for i, j in combinations(range(len(feature_names)), 2)],
    ]
    if "const" in operator_keys:
        op_pool.append(("Const", ConstantBrick, {}, nn.Identity()))

    combiner_patterns = (
        lambda k: "add",
        lambda k: "mul",
        lambda k: "add" if k % 2 == 0 else "mul",
        lambda k: "mul" if k % 2 == 0 else "add",
    )
    topologies = ("chain", "fan", "skip", "rand")

    seed_dags = []
    for s in range(n_seeds):
        size = rng.randint(3, 7)
        topo = topologies[s % len(topologies)]
        comb_fn = combiner_patterns[s % len(combiner_patterns)]
        nodes = [
            SymbolicNode(
                combiner=comb_fn(k),
                operation=op_cls,
                hp=dict(hp),
                activation=act,
            )
            for k in range(size)
            for _, op_cls, hp, act in [op_pool[rng.randrange(len(op_pool))]]
        ]

        M = np.zeros((size, size), dtype=int)
        if topo in ("chain", "skip", "rand"):
            M[np.arange(size - 1), np.arange(1, size)] = 1
        if topo == "fan":
            M[0, 1:] = 1
        if topo == "skip":
            M[np.arange(size - 2), np.arange(2, size)] = 1
        if topo == "rand":
            for i in range(size):
                for j in range(i + 1, size):
                    if rng.random() < 0.3:
                        M[i, j] = 1
        try:
            seed_dags.append(AdjMatrix(operations=nodes, matrix=M))
        except Exception:
            continue

    return [[m] for m in seed_dags] if seed_dags else None


def _extract_best_formula_from_log(log_path: str) -> str:
    """Return the formula with the smallest logged loss."""
    if not os.path.exists(log_path):
        return "N/A"
    best_loss = np.inf; best_formula = "N/A"
    current_formula = None; current_loss = None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- Idx="):
                    try:
                        current_loss = float(line.split("Loss=")[1].split()[0].rstrip("---").strip())
                    except Exception:
                        current_loss = None
                    current_formula = None
                elif "<<< SELECTED" in line and current_loss is not None:
                    try:
                        formula_part = line.split("Formula[")[1].split("]: ", 1)[1]
                        formula_part = formula_part.replace(" <<< SELECTED", "").strip()
                        current_formula = formula_part
                    except Exception:
                        pass
                    if current_loss is not None and current_loss < best_loss:
                        best_loss = current_loss; best_formula = current_formula or "N/A"
    except Exception:
        pass
    return best_formula


def _build_result(target, method_cfg, run_id, strategy, best_loss, best_formula,
                  log_path, loss_state, actual_T, loss_history, landscape_svg, elapsed) -> dict:
    result = {
        "target": target, "method": method_cfg["id"], "run_id": run_id,
        "strategy": strategy, "_stream_id": method_cfg.get("_stream_id"),
        "loss": float(best_loss), "r2": float(1.0 - best_loss) if best_loss <= 1.0 else 0.0,
        "formula": str(best_formula), "time_s": elapsed, "log_path": log_path,
        "description": method_cfg["description"],
        "search_space_ops": list(method_cfg.get("operators", [])),
        "loss_mode":      method_cfg.get("loss_mode", "full"),
        "winner_type":    loss_state.get("winner_type", "channel"),
        "corr_value":     float(loss_state.get("corr_value", 0.0)),
        "alignment_loss": float(loss_state.get("alignment_loss", 1.0)),
        "rat_degree":     loss_state.get("rat_degree"),
        "loss_history":   loss_history,
        "actualT":        int(actual_T) if actual_T is not None else None,
        "all_channel_formulas": loss_state.get("all_channel_formulas", []),
        "ops_used":         loss_state.get("ops_used"),
        "has_const":        loss_state.get("has_const"),
        "dag_size":         loss_state.get("dag_size"),
        "dag_text":         loss_state.get("dag_text"),
        "dag_svg":          loss_state.get("dag_svg"),
        "dag_data":         loss_state.get("dag_data", []),
        "formula_channel":  loss_state.get("formula_channel"),
        "formula_ols":      loss_state.get("formula_ols"),
        "formula_nested":   loss_state.get("formula_nested"),
        "formula_polyrat":  loss_state.get("formula_polyrat", {}),
        "loss_channel":     loss_state.get("loss_channel"),
        "loss_ols":         loss_state.get("loss_ols"),
        "loss_nested":      loss_state.get("loss_nested"),
        "loss_polyrat":     loss_state.get("loss_polyrat", {}),
        "mse_channel":      loss_state.get("mse_channel"),
        "mse_ols":          loss_state.get("mse_ols"),
        "mse_nested":       loss_state.get("mse_nested"),
        "mse_polyrat":      loss_state.get("mse_polyrat", {}),
        "landscape_svg":    landscape_svg,
    }
    if method_cfg.get("_in_smart_stream"):
        result["_best_pred_np"] = loss_state.get("best_pred_np")
        result["_best_y_np"]    = loss_state.get("best_y_np")
    return result
