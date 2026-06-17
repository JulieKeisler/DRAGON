from __future__ import annotations
import os
import ast
import re
import numpy as np


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
        "max_complexity_reached": loss_state.get("max_complexity_reached"),
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


def _parse_text_value(raw: str):
    txt = raw.strip()
    low = txt.lower()
    if low == "none":
        return None
    if low in ("true", "false"):
        return low == "true"
    if txt.startswith("[") or txt.startswith("{") or txt.startswith("("):
        return ast.literal_eval(txt)
    if "," in txt:
        return [p.strip() for p in txt.split(",") if p.strip()]
    try:
        return int(txt)
    except ValueError:
        pass
    try:
        return float(txt)
    except ValueError:
        pass
    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        return txt[1:-1]
    return txt


def apply_text_config(config_file_path: str) -> None:
    """Apply runtime overrides from a plain text config file.

    Supported keys:
    - Flat key (preferred): TARGETS = n4,n5,n6
    - Class attribute:      Experiment.TARGETS = n4,n5,n6
    - Method registry:      DRAGON_METHODS[0].loss_mode = full
    - Method field (default method): loss_mode = full
    """

    # Local import keeps Helper generic and avoids import-order coupling.
    from Config import Experiment, Dragon, Paths, Loss, OLS, PySR, MCDropout, Sampling, DRAGON_METHODS

    flat_key_priority = [Experiment, Dragon, PySR, Paths, Loss, OLS, MCDropout, Sampling]

    def _set_class_attr(cls, attr_name: str, value, line_no: int, key_label: str):
        if not hasattr(cls, attr_name):
            raise ValueError(f"Unknown config attribute at line {line_no}: {key_label}")
        if attr_name in ("TARGETS", "INIT_STRATEGIES") and isinstance(value, str):
            value = [value]
        setattr(cls, attr_name, value)

    def _set_default_method_field(field: str, value, line_no: int):
        if not DRAGON_METHODS:
            raise ValueError(f"No DRAGON_METHODS defined (line {line_no})")
        DRAGON_METHODS[0][field] = value

    with open(config_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            raise ValueError(f"Invalid config line {line_no}: missing '='")

        key, raw_val = s.split("=", 1)
        key = key.strip()
        value = _parse_text_value(raw_val)

        m = re.match(r"^DRAGON_METHODS\[(\d+)\]\.([A-Za-z_]\w*)$", key)
        if m:
            idx = int(m.group(1))
            field = m.group(2)
            if idx < 0 or idx >= len(DRAGON_METHODS):
                raise IndexError(f"Invalid DRAGON_METHODS index at line {line_no}: {idx}")
            DRAGON_METHODS[idx][field] = value
            continue

        m = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)$", key)
        if m:
            class_name, attr_name = m.group(1), m.group(2)
            cls = globals().get(class_name)
            if cls is None:
                cls = {
                    "Experiment": Experiment,
                    "Dragon": Dragon,
                    "Paths": Paths,
                    "Loss": Loss,
                    "OLS": OLS,
                    "PySR": PySR,
                    "MCDropout": MCDropout,
                    "Sampling": Sampling,
                }.get(class_name)
            if cls is None:
                raise ValueError(f"Unknown config class at line {line_no}: {class_name}")
            _set_class_attr(cls, attr_name, value, line_no, key)
            continue

        candidates = [cls for cls in flat_key_priority if hasattr(cls, key)]
        if len(candidates) == 1:
            _set_class_attr(candidates[0], key, value, line_no, key)
            continue
        if len(candidates) > 1:
            _set_class_attr(candidates[0], key, value, line_no, key)
            continue

        _set_default_method_field(key, value, line_no)
