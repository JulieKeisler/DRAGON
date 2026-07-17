#!/usr/bin/env python3
"""Batch launcher for sequential DragonSR / PySR runs.

This script runs one of:
    python3 main.py --run_dragonsr --config_file_path ... [--data_path ...]
    python3 main.py --pysr-only   --config_file_path ... [--data_path ...]

for one or many config files, sequentially. After each run it:
1) archives generated HTML files into a dedicated folder,
2) rotates (renames) the `leaderboard_runs` folder to avoid name collisions.

You can provide runs either directly with --configs (DragonSR by default) or
via --plan-file.
Plan file format (one run per line, comments with '#'):
    /path/to/config_a.txt
    label_for_run|/path/to/config_b.txt
    label2|/path/to/config_c.txt|/path/to/other_data.csv
    pysr|target|/path/to/config_pysr.txt
    dragonsr|target|/path/to/config_dragon.txt
    dragonsr|target|/path/to/config_dragon.txt|label
    dragonsr|target|/path/to/config_dragon.txt|label|/path/to/other_data.csv

When a target is provided in the plan, this script auto-generates a resolved
config for that run with ``TARGETS = <target>`` appended. This makes plan-file
the source of truth for which formula/target is executed, independent of
TARGETS declared in the base config file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_label(raw: str) -> str:
    keep = []
    for ch in raw.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "run"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _parse_plan_file(plan_file: Path) -> list[dict]:
    engines = {"dragonsr", "pysr"}
    runs = []
    with plan_file.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split("|")]
            if len(parts) == 1:
                engine = "dragonsr"
                cfg = parts[0]
                label = Path(cfg).stem
                target = None
                data_path = None
            elif len(parts) == 2:
                if parts[0].lower() in engines:
                    engine = parts[0].lower()
                    cfg = parts[1]
                    label = Path(cfg).stem
                    target = None
                    data_path = None
                else:
                    engine = "dragonsr"
                    label, cfg = parts
                    target = None
                    data_path = None
            elif len(parts) == 3:
                if parts[0].lower() in engines:
                    engine = parts[0].lower()
                    target, cfg = parts[1], parts[2]
                    label = target
                    data_path = None
                else:
                    engine = "dragonsr"
                    label, cfg, data_path = parts
                    target = None
            elif len(parts) == 4:
                engine = parts[0].lower()
                if engine not in engines:
                    raise ValueError(
                        f"Invalid engine at line {line_no}: '{parts[0]}'. "
                        "Expected 'dragonsr' or 'pysr'."
                    )
                target, cfg, fourth = parts[1], parts[2], parts[3]
                if ("/" in fourth) or ("\\" in fourth) or fourth.lower().endswith(".csv"):
                    label = target
                    data_path = fourth
                else:
                    label = fourth
                    data_path = None
            elif len(parts) == 5:
                engine = parts[0].lower()
                if engine not in engines:
                    raise ValueError(
                        f"Invalid engine at line {line_no}: '{parts[0]}'. "
                        "Expected 'dragonsr' or 'pysr'."
                    )
                target, cfg, label, data_path = parts[1], parts[2], parts[3], parts[4]
            else:
                raise ValueError(
                    f"Invalid plan format at line {line_no}: '{s}'. "
                    "Expected: config OR label|config OR label|config|data_path "
                    "OR engine|target|config[|label][|data_path]"
                )
            runs.append(
                {
                    "engine": engine,
                    "label": _sanitize_label(label),
                    "target": (target.strip() if isinstance(target, str) and target.strip() else None),
                    "config": str(Path(cfg).expanduser().resolve()),
                    "data_path": (str(Path(data_path).expanduser().resolve()) if data_path else None),
                }
            )
    return runs


def _materialize_run_config(
    *,
    base_config: Path,
    target_override: str | None,
    work_dir: Path,
    idx: int,
    label: str,
    engine: str,
) -> Path:
    """Create an effective config for one run (with optional TARGETS override)."""
    if not target_override:
        return base_config

    resolved_dir = work_dir / "batch_resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    out = resolved_dir / f"run_{idx:03d}_{engine}_{_sanitize_label(label)}_{base_config.name}"
    out = _unique_path(out)

    txt = base_config.read_text(encoding="utf-8")
    if txt and not txt.endswith("\n"):
        txt += "\n"
    txt += "\n# Auto-generated by run_dragonsr_batch.py\n"
    txt += f"TARGETS = {target_override}\n"
    out.write_text(txt, encoding="utf-8")
    return out


def _collect_runs(args: argparse.Namespace) -> list[dict]:
    runs = []

    if args.plan_file:
        runs.extend(_parse_plan_file(Path(args.plan_file).expanduser().resolve()))

    for cfg in args.configs or []:
        cfg_path = Path(cfg).expanduser().resolve()
        runs.append(
            {
                "engine": "dragonsr",
                "label": _sanitize_label(cfg_path.stem),
                "config": str(cfg_path),
                "data_path": None,
            }
        )

    if not runs:
        raise ValueError("No runs provided. Use --configs and/or --plan-file.")

    for run in runs:
        cfg = Path(run["config"])
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")

    return runs


def _rotate_leaderboard_runs(
    work_dir: Path,
    runs_archive_dir: Path,
    label: str,
    idx: int,
    mode: str,
) -> str | None:
    src = work_dir / "leaderboard_runs"
    if not src.exists():
        return None

    if mode == "delete":
        shutil.rmtree(src)
        return None

    target = runs_archive_dir / f"run_{idx:03d}_{label}_{_timestamp()}_leaderboard_runs"
    target = _unique_path(target)
    shutil.move(str(src), str(target))
    return str(target)


def _archive_recent_htmls(
    work_dir: Path,
    html_archive_dir: Path,
    label: str,
    idx: int,
    start_time: dt.datetime,
) -> list[str]:
    archived = []
    # Only archive HTML regenerated by THIS run. Using the run start time as the
    # cutoff prevents copying a stale HTML from a previous run (which would
    # otherwise be mislabeled with this run's target/engine).
    cutoff = start_time.timestamp()

    # HTML is produced in two places in this codebase:
    # 1) leaderboard root (user/custom outputs),
    # 2) helpers/ (leaderboard_standalone.html + template file).
    root_html = [p for p in work_dir.glob("*.html") if p.is_file()]
    helpers_dir = work_dir / "helpers"
    helpers_html = [p for p in helpers_dir.glob("*.html") if p.is_file()] if helpers_dir.exists() else []

    all_html = root_html + helpers_html
    selected = [p for p in all_html if p.stat().st_mtime >= cutoff]
    selected = sorted(selected, key=lambda x: x.stat().st_mtime)

    if not selected:
        print(
            f"  [WARN] No HTML was regenerated during run '{label}' (idx {idx}); "
            "nothing archived. The run likely failed before writing its HTML."
        )
        return archived

    for html in selected:
        dst = html_archive_dir / f"run_{idx:03d}_{label}_{_timestamp()}_{html.name}"
        dst = _unique_path(dst)
        shutil.copy2(str(html), str(dst))
        archived.append(str(dst))
        print(f"  [HTML] archived: {html} -> {dst}")

    return archived


def _run_one(
    *,
    idx: int,
    run: dict,
    main_script: Path,
    default_data_path: Path | None,
    work_dir: Path,
    html_archive_dir: Path,
    runs_archive_dir: Path,
    rotate_mode: str,
) -> dict:
    label = run["label"]
    engine = str(run.get("engine", "dragonsr")).lower()
    config_path = Path(run["config"])
    target_override = run.get("target")
    data_path = Path(run["data_path"]).resolve() if run.get("data_path") else default_data_path
    if data_path is not None and not data_path.exists():
        raise FileNotFoundError(f"Data path not found for run '{label}': {data_path}")

    if engine not in {"dragonsr", "pysr"}:
        raise ValueError(f"Unsupported engine '{engine}' for run '{label}'")

    effective_config = _materialize_run_config(
        base_config=config_path,
        target_override=target_override,
        work_dir=work_dir,
        idx=idx,
        label=label,
        engine=engine,
    )

    started_at = dt.datetime.now()

    # Rotate pre-existing leaderboard_runs before launching next run.
    pre_rotated = _rotate_leaderboard_runs(
        work_dir=work_dir,
        runs_archive_dir=runs_archive_dir,
        label=f"pre_{label}",
        idx=idx,
        mode=rotate_mode,
    )

    cmd = [
        sys.executable,
        str(main_script),
        "--config_file_path",
        str(effective_config),
    ]
    cmd.insert(2, "--run_dragonsr" if engine == "dragonsr" else "--pysr-only")
    if data_path is not None:
        cmd.extend(["--data_path", str(data_path)])

    print("=" * 80)
    print(f"[RUN {idx}] engine={engine} label={label}")
    if target_override:
        print(f"Target override: {target_override}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    proc = subprocess.run(cmd, cwd=str(work_dir))
    return_code = int(proc.returncode)

    html_files = _archive_recent_htmls(
        work_dir=work_dir,
        html_archive_dir=html_archive_dir,
        label=label,
        idx=idx,
        start_time=started_at,
    )

    post_rotated = _rotate_leaderboard_runs(
        work_dir=work_dir,
        runs_archive_dir=runs_archive_dir,
        label=label,
        idx=idx,
        mode=rotate_mode,
    )

    return {
        "index": idx,
        "engine": engine,
        "label": label,
        "target": target_override,
        "config": str(config_path),
        "effective_config": str(effective_config),
        "data_path": (str(data_path) if data_path is not None else None),
        "started_at": started_at.isoformat(timespec="seconds"),
        "return_code": return_code,
        "status": "ok" if return_code == 0 else "failed",
        "html_archives": html_files,
        "pre_rotated_runs": pre_rotated,
        "post_rotated_runs": post_rotated,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially launch multiple DragonSR/PySR runs from one or more config files, "
            "archive HTML outputs, and rotate leaderboard_runs between runs."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="List of config.txt files to run sequentially with DragonSR (duplicates allowed).",
    )
    parser.add_argument(
        "--plan-file",
        default=None,
        help=(
            "Optional plan file with one run per line: "
            "config OR label|config OR label|config|data_path OR "
            "engine|target|config[|label][|data_path] where engine is dragonsr or pysr"
        ),
    )
    parser.add_argument(
        "--data-path",
        required=False,
        default=None,
        help=(
            "Default CSV data path used when a run does not specify one in plan-file. "
            "Optional if every plan entry provides its own data_path."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Directory containing main.py (default: current directory).",
    )
    parser.add_argument(
        "--main-script",
        default="main.py",
        help="Path to main.py relative to --work-dir or absolute.",
    )
    parser.add_argument(
        "--html-archive-dir",
        default="batch_html_archives",
        help="Folder where generated HTML files are copied.",
    )
    parser.add_argument(
        "--runs-archive-dir",
        default="batch_runs_archives",
        help="Folder where leaderboard_runs snapshots are moved.",
    )
    parser.add_argument(
        "--rotate-runs-mode",
        choices=["rename", "delete"],
        default="rename",
        help="What to do with leaderboard_runs before/after each run.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"work-dir does not exist: {work_dir}")

    main_script = Path(args.main_script)
    if not main_script.is_absolute():
        main_script = (work_dir / main_script).resolve()
    if not main_script.exists():
        raise FileNotFoundError(f"main script not found: {main_script}")

    data_path = None
    if args.data_path:
        data_path = Path(args.data_path).expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"data path not found: {data_path}")

    html_archive_dir = Path(args.html_archive_dir)
    if not html_archive_dir.is_absolute():
        html_archive_dir = (work_dir / html_archive_dir).resolve()
    html_archive_dir.mkdir(parents=True, exist_ok=True)

    runs_archive_dir = Path(args.runs_archive_dir)
    if not runs_archive_dir.is_absolute():
        runs_archive_dir = (work_dir / runs_archive_dir).resolve()
    runs_archive_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(args)

    print(f"Work dir           : {work_dir}")
    print(f"Main script        : {main_script}")
    print(
        "Default data path  : "
        f"{data_path if data_path is not None else 'None (internal target generation or per-run data_path)'}"
    )
    print(f"HTML archive dir   : {html_archive_dir}")
    print(f"Runs archive dir   : {runs_archive_dir}")
    print(f"Rotate mode        : {args.rotate_runs_mode}")
    print(f"Total runs         : {len(runs)}")

    results = []
    for idx, run in enumerate(runs, start=1):
        result = _run_one(
            idx=idx,
            run=run,
            main_script=main_script,
            default_data_path=data_path,
            work_dir=work_dir,
            html_archive_dir=html_archive_dir,
            runs_archive_dir=runs_archive_dir,
            rotate_mode=args.rotate_runs_mode,
        )
        results.append(result)

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "work_dir": str(work_dir),
        "main_script": str(main_script),
        "default_data_path": (str(data_path) if data_path is not None else None),
        "html_archive_dir": str(html_archive_dir),
        "runs_archive_dir": str(runs_archive_dir),
        "rotate_runs_mode": args.rotate_runs_mode,
        "runs": results,
    }

    summary_path = work_dir / f"batch_summary_{_timestamp()}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok
    print("\nBatch finished.")
    print(f"Succeeded: {ok}")
    print(f"Failed   : {failed}")
    print(f"Summary  : {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
