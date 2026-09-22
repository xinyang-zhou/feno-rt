"""Run the complete formal CUDA Graph A/B matrix in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "benchmarks/benchmark_graph_ab.py"
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_graph_ab import (  # noqa: E402
    display_path,
    ensure_external_new_output,
    file_sha256,
    git_identity,
    load_checked_manifest,
    resolve_model_paths,
    verified_dependency_versions,
)
from benchmarks.graph_workload import DEFAULT_CONFIG, load_config  # noqa: E402
from benchmarks.summarize_graph_ab import (  # noqa: E402
    summarize_directory,
    write_json_atomic,
)
from benchmarks.validate_result import validate_document  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_plan(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    configured_modes = set(config["cuda_graph"]["modes"])
    if configured_modes != {"off", "on"}:
        raise ValueError("Graph A/B matrix requires exactly off and on modes")
    batches = [int(value) for value in config["workload"]["batch_sizes"]]
    repeats = int(config["measurement"]["independent_runs"])
    plan: List[Dict[str, Any]] = []
    sequence = 1
    for repeat_index in range(1, repeats + 1):
        batch_order = batches if repeat_index % 2 else list(reversed(batches))
        mode_order = ("off", "on") if repeat_index % 2 else ("on", "off")
        for batch_size in batch_order:
            pair_id = f"batch{batch_size}_run{repeat_index}"
            for pair_position, mode in enumerate(mode_order, start=1):
                plan.append(
                    {
                        "sequence": sequence,
                        "pair_id": pair_id,
                        "pair_position": pair_position,
                        "mode": mode,
                        "batch_size": batch_size,
                        "repeat_index": repeat_index,
                        "result_file": (
                            f"graph_{mode}_batch{batch_size}_run{repeat_index}.json"
                        ),
                        "log_file": (
                            f"logs/graph_{mode}_batch{batch_size}_run{repeat_index}.log"
                        ),
                        "status": "pending",
                        "return_code": None,
                        "started_at_utc": None,
                        "completed_at_utc": None,
                        "attempts": [],
                    }
                )
                sequence += 1
    return plan


def child_command(
    config_path: Path, output_dir: Path, entry: Mapping[str, Any]
) -> List[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config_path.resolve()),
        "--mode",
        str(entry["mode"]),
        "--batch-size",
        str(entry["batch_size"]),
        "--repeat-index",
        str(entry["repeat_index"]),
        "--output",
        str((output_dir / entry["result_file"]).resolve()),
    ]


def new_session(
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    git_commit: str,
    model_artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    plan = build_plan(config)
    return {
        "schema_version": "1.0.0",
        "benchmark": config["benchmark"],
        "status": "running",
        "created_at_utc": utc_now(),
        "completed_at_utc": None,
        "source": {
            "git_commit": git_commit,
            "git_dirty": False,
            "command": [sys.executable, *sys.argv],
            "resume_commands": [],
            "config_path": display_path(config_path),
            "config_sha256": file_sha256(config_path),
        },
        "model_artifacts": dict(model_artifacts),
        "matrix": {
            "modes": list(config["cuda_graph"]["modes"]),
            "batch_sizes": list(config["workload"]["batch_sizes"]),
            "independent_runs": int(config["measurement"]["independent_runs"]),
            "requests_per_run": int(config["workload"]["request_count"]),
            "total_processes": len(plan),
            "process_isolation": "one_configuration_per_fresh_process",
            "pairing": "adjacent_counterbalanced_graph_off_on",
        },
        "output_directory": str(output_dir.resolve()),
        "execution_order": plan,
        "summary": {"json": None, "markdown": None},
        "errors": [],
    }


def load_resumable_session(
    session_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    git_commit: str,
    model_artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    with session_path.open("r", encoding="utf-8") as handle:
        session = json.load(handle)
    if session.get("schema_version") != "1.0.0":
        raise ValueError("cannot resume an unsupported session schema")
    source = session.get("source", {})
    if source.get("git_commit") != git_commit:
        raise ValueError("cannot resume with a different Git commit")
    if source.get("config_sha256") != file_sha256(config_path):
        raise ValueError("cannot resume with a different benchmark configuration")
    if session.get("model_artifacts") != model_artifacts:
        raise ValueError("cannot resume with different model artifacts")
    expected_plan = build_plan(config)
    actual_plan = session.get("execution_order")
    if not isinstance(actual_plan, list) or len(actual_plan) != len(expected_plan):
        raise ValueError("cannot resume a session with a different matrix")
    identity_fields = (
        "sequence",
        "pair_id",
        "pair_position",
        "mode",
        "batch_size",
        "repeat_index",
        "result_file",
        "log_file",
    )
    for expected, actual in zip(expected_plan, actual_plan):
        if any(expected[name] != actual.get(name) for name in identity_fields):
            raise ValueError("cannot resume a session with a changed execution plan")
    session["status"] = "running"
    session["completed_at_utc"] = None
    session["source"].setdefault("resume_commands", []).append(
        [sys.executable, *sys.argv]
    )
    return session


def write_log(
    path: Path, command: List[str], completed: subprocess.CompletedProcess[str]
) -> None:
    content = "\n".join(
        (
            f"command: {shlex.join(command)}",
            f"return_code: {completed.returncode}",
            "",
            "[stdout]",
            completed.stdout.rstrip(),
            "",
            "[stderr]",
            completed.stderr.rstrip(),
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def attempt_log_file(entry: Mapping[str, Any], attempt_index: int) -> str:
    base = Path(str(entry["log_file"]))
    if attempt_index == 1:
        return str(base)
    return str(base.with_name(f"{base.stem}.attempt{attempt_index}{base.suffix}"))


def validate_preconditions(
    config: Mapping[str, Any], config_path: Path
) -> Tuple[str, Dict[str, Any]]:
    errors, kind, _schema_status = validate_document(config, kind="config")
    if errors:
        raise ValueError(f"invalid {kind}: {'; '.join(errors)}")
    git_commit, git_dirty = git_identity()
    if git_dirty:
        raise RuntimeError("formal benchmark requires a clean Git working tree")
    for batch_size in config["workload"]["batch_sizes"]:
        load_checked_manifest(config, int(batch_size))
    checkpoint, normalization = resolve_model_paths(config)
    verified_dependency_versions()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    model_artifacts = {
        "checkpoint": {
            "identifier": checkpoint.name,
            "sha256": file_sha256(checkpoint),
        },
        "normalization": {
            "identifier": normalization.name,
            "sha256": file_sha256(normalization),
        },
    }
    return git_commit, model_artifacts


def existing_result_status(
    output_dir: Path, entry: Mapping[str, Any]
) -> Optional[str]:
    result_path = output_dir / entry["result_file"]
    if not result_path.is_file():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    errors, _kind, _schema_status = validate_document(result, kind="result")
    if errors:
        raise ValueError(f"existing result is invalid: {result_path}: {'; '.join(errors)}")
    expected_mode = entry["mode"] == "on"
    matches = (
        result["execution"]["graph_enabled"] == expected_mode
        and result["workload"]["batch_size"] == entry["batch_size"]
        and result["repeat_index"] == entry["repeat_index"]
    )
    if not matches:
        raise ValueError(f"existing result does not match its matrix entry: {result_path}")
    return str(result["status"])


def run_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = args.config.resolve()
    config = load_config(config_path)
    git_commit, model_artifacts = validate_preconditions(config, config_path)
    output_dir = args.output_dir.resolve()
    session_path = output_dir / "session.json"

    if args.resume:
        try:
            output_dir.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("formal run output must be outside the repository")
        if not session_path.is_file():
            raise FileNotFoundError(f"resume session not found: {session_path}")
        session = load_resumable_session(
            session_path,
            config,
            config_path,
            git_commit,
            model_artifacts,
        )
    else:
        ensure_external_new_output(output_dir)
        output_dir.mkdir(parents=True)
        session = new_session(
            config,
            config_path,
            output_dir,
            git_commit,
            model_artifacts,
        )
    write_json_atomic(session_path, session)

    infrastructure_failed = False
    try:
        for entry in session["execution_order"]:
            existing_status = existing_result_status(output_dir, entry)
            if existing_status is not None:
                entry["status"] = existing_status
                write_json_atomic(session_path, session)
                print(
                    f"[{entry['sequence']:02d}/{len(session['execution_order'])}] "
                    f"skip existing {entry['result_file']}",
                    flush=True,
                )
                continue
            if entry["status"] in {"passed", "failed"}:
                raise ValueError(
                    f"session marks a missing result as complete: {entry['result_file']}"
                )
            result_path = output_dir / entry["result_file"]
            command = child_command(config_path, output_dir, entry)
            attempts = entry.setdefault("attempts", [])
            attempt_index = len(attempts) + 1
            current_log_file = attempt_log_file(entry, attempt_index)
            attempt = {
                "attempt_index": attempt_index,
                "log_file": current_log_file,
                "started_at_utc": utc_now(),
                "completed_at_utc": None,
                "return_code": None,
            }
            attempts.append(attempt)
            entry["status"] = "running"
            entry["started_at_utc"] = attempt["started_at_utc"]
            write_json_atomic(session_path, session)
            print(
                f"[{entry['sequence']:02d}/{len(session['execution_order'])}] "
                f"Graph {entry['mode']}, batch {entry['batch_size']}, "
                f"repeat {entry['repeat_index']}",
                flush=True,
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as error:
                completed = subprocess.CompletedProcess(
                    command,
                    returncode=127,
                    stdout="",
                    stderr=str(error),
                )
            write_log(output_dir / current_log_file, command, completed)
            entry["return_code"] = completed.returncode
            entry["completed_at_utc"] = utc_now()
            attempt["return_code"] = completed.returncode
            attempt["completed_at_utc"] = entry["completed_at_utc"]
            if result_path.is_file():
                try:
                    result_status = existing_result_status(output_dir, entry)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    entry["status"] = "error"
                    session["errors"].append(f"{entry['result_file']}: {error}")
                    infrastructure_failed = True
                else:
                    expected_return_code = 0 if result_status == "passed" else 1
                    if completed.returncode != expected_return_code:
                        entry["status"] = "error"
                        session["errors"].append(
                            f"{entry['result_file']}: return code {completed.returncode} "
                            f"does not match result status {result_status}"
                        )
                        infrastructure_failed = True
                    else:
                        entry["status"] = result_status
            else:
                entry["status"] = "error"
                session["errors"].append(
                    f"{entry['result_file']} was not produced; see {current_log_file}"
                )
                infrastructure_failed = True
            write_json_atomic(session_path, session)
            if infrastructure_failed:
                break
    except KeyboardInterrupt:
        session["status"] = "interrupted"
        session["completed_at_utc"] = utc_now()
        write_json_atomic(session_path, session)
        raise
    except Exception as error:
        session["status"] = "failed"
        session["completed_at_utc"] = utc_now()
        session["errors"].append(str(error))
        write_json_atomic(session_path, session)
        raise

    result_paths = sorted(output_dir.glob("graph_*_batch*_run*.json"))
    if result_paths:
        try:
            summary = summarize_directory(
                config_path,
                output_dir,
                output_dir / "summary.json",
                output_dir / "summary.md",
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            session["errors"].append(f"summary: {error}")
            session["status"] = "failed"
            infrastructure_failed = True
        else:
            session["summary"] = {
                "json": "summary.json",
                "markdown": "summary.md",
            }
            session["status"] = summary["status"]
    else:
        session["status"] = "failed"
    if infrastructure_failed:
        session["status"] = "failed"
    session["completed_at_utc"] = utc_now()
    write_json_atomic(session_path, session)
    return session


def print_plan(config: Mapping[str, Any], config_path: Path, output_dir: Path) -> None:
    plan = build_plan(config)
    for entry in plan:
        command = child_command(config_path, output_dir, entry)
        print(
            f"[{entry['sequence']:02d}/{len(plan)}] {shlex.join(command)}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.dry_run:
        raise ValueError("--resume and --dry-run cannot be used together")
    if args.dry_run:
        config = load_config(args.config.resolve())
        errors, kind, _schema_status = validate_document(config, kind="config")
        if errors:
            raise ValueError(f"invalid {kind}: {'; '.join(errors)}")
        print_plan(config, args.config.resolve(), args.output_dir.resolve())
        return 0
    session = run_matrix(args)
    print(f"session: {(args.output_dir.resolve() / 'session.json')}")
    if session["summary"]["json"]:
        print(f"summary: {(args.output_dir.resolve() / session['summary']['json'])}")
    print(f"status: {session['status']}")
    for error in session["errors"]:
        print(f"error: {error}", file=sys.stderr)
    return 0 if session["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
