#!/usr/bin/env python3
"""RSI Step0 command-line interface (M7).

Subcommands (see the design doc):

* ``transcribe`` - turn heuresis run logs into trajectory records
* ``validate``   - aggregate-validate trajectory records and JSON schemas
* ``train``      - train the pi router (draft/improve/crossover) from trajectories
* ``replay``     - deterministic replay of a run
* ``evaluate``   - run / compare an ablation config (heuresis-aide vs heuresis-aira)
* ``check``      - self-check (engine availability, package imports, Python version)

Every command writes a structured manifest under ``output/rsi_step0/manifests`` and
prints a JSON summary. Commands exit 0 on success and 1 on failure.

Only ``rsi_step0.contracts`` and ``rsi_step0.validation`` (both always present)
are imported eagerly. The other ``rsi_step0`` modules (reward/router/trajectory/
scheduler/training/heuresis_adapter) are generated in parallel by other agents and
may not exist yet, so they are imported lazily *inside* each command handler; a
missing module produces a clear message and exit code 1 instead of a crash.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from rsi_step0 import contracts as C
from rsi_step0 import validation as V


MANIFESTS_DIR = "output/rsi_step0/manifests"

DEFAULT_MINIMAL_CFG = "configs/rsi_step0/minimal.yaml"
DEFAULT_REPLAY_CFG = "configs/rsi_step0/replay.yaml"
DEFAULT_TRAIN_CFG = "configs/rsi_step0/train_pi.yaml"
DEFAULT_ABLATION_CFG = "configs/rsi_step0/ablation.yaml"
DEFAULT_TRAJ_DIR = "output/rsi_step0/trajectories"
DEFAULT_SCHEMA = "rsi_step0/schemas/trajectory.schema.json"


# --------------------------------------------------------------------------- utils
def _load_module(name: str) -> Any:
    """Import a module lazily; returns ``None`` (with reason printed) on failure."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "command": "cli",
                    "ok": False,
                    "error": f"module {name!r} is not available yet: {exc}",
                    "hint": (
                        f"rsi_step0.{name.rsplit('.', 1)[-1]} is generated in "
                        "parallel; rerun once it exists"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return None


def _call_module(
    module_name: str,
    entrypoints: Tuple[str, ...],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Optional[Any], Optional[str], List[str]]:
    """Call the first matching entrypoint of a lazily-imported module.

    Returns ``(result, entrypoint_used, failures)``. A ``TypeError`` is treated as
    a signature mismatch and the next candidate is tried; any other exception is a
    genuine module failure and stops the search. Returns ``(None, None, failures)``
    when the module is missing or no entrypoint matched.
    """
    module = _load_module(module_name)
    if module is None:
        return None, None, [f"module {module_name!r} could not be imported"]

    failures: List[str] = []
    for entrypoint in entrypoints:
        func: Optional[Callable[..., Any]] = getattr(module, entrypoint, None)
        if not callable(func):
            continue
        try:
            result = func(*args, **kwargs)
            return result, f"{module_name}.{entrypoint}", failures
        except TypeError as exc:
            failures.append(f"{module_name}.{entrypoint}: signature mismatch: {exc}")
            continue
        except Exception as exc:  # genuine module failure - do not mask it
            failures.append(
                f"{module_name}.{entrypoint} raised {type(exc).__name__}: {exc}"
            )
            return None, f"{module_name}.{entrypoint}", failures

    expected = ", ".join(entrypoints)
    failures.append(
        f"{module_name} exposes none of the expected entrypoints ({expected}); "
        "check the module API in the design doc"
    )
    return None, None, failures


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_manifest(command: str, payload: Dict[str, Any]) -> str:
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    path = os.path.join(MANIFESTS_DIR, f"{command}_{_now_stamp()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def _manifest(command: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "command": command,
        "timestamp": _now_stamp(),
        "python_version": platform.python_version(),
        "schema_version": C.SCHEMA_VERSION,
    }
    payload.update(extra)
    return payload


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


def _summarize_result(result: Any) -> Any:
    """Keep manifest payloads bounded no matter what the module returned."""
    if isinstance(result, dict):
        summary: Dict[str, Any] = {"type": "dict"}
        for key in ("ok", "status", "count", "counts", "stable", "hash", "metrics",
                    "n_records", "n_files", "epochs", "run_id", "task_id"):
            if key in result:
                summary[key] = result[key]
        if "metrics" in summary:
            summary.pop("metrics")
        return summary
    if isinstance(result, list):
        return {"type": "list", "len": len(result)}
    if isinstance(result, (int, float, str, bool)) or result is None:
        return result
    return {"type": type(result).__name__, "repr": str(result)[:200]}


# --------------------------------------------------------------------------- commands
def cmd_check(args: argparse.Namespace) -> int:
    module_names = (
        "rsi_step0.contracts",
        "rsi_step0.validation",
        "rsi_step0.reward",
        "rsi_step0.router",
        "rsi_step0.trajectory",
        "rsi_step0.scheduler",
        "rsi_step0.training",
        "rsi_step0.heuresis_adapter",
    )
    modules: Dict[str, bool] = {}
    for name in module_names:
        modules[name] = _import_ok(name)

    engine_available = False
    engine_hint = "heuresis_adapter not available"
    adapter = _import_quiet("rsi_step0.heuresis_adapter")
    if adapter is not None:
        for probe in ("is_engine_available", "engine_available", "check_engine",
                      "is_available", "available"):
            func = getattr(adapter, probe, None)
            if callable(func):
                try:
                    engine_available = bool(func())
                    engine_hint = f"probed via heuresis_adapter.{probe}()"
                    break
                except Exception as exc:
                    engine_hint = f"engine probe {probe}() failed: {exc}"

    base_ok = modules["rsi_step0.contracts"] and modules["rsi_step0.validation"]
    report = {
        "command": "check",
        "ok": base_ok,
        "python_version": platform.python_version(),
        "engine_available": engine_available,
        "engine_hint": engine_hint,
        "modules": modules,
    }
    # `report` already carries the "command" key; strip it before spreading into
    # `_manifest` so the positional `command` argument is not duplicated.
    report_extra = dict(report)
    report_extra.pop("command", None)
    manifest_path = _write_manifest("check", _manifest("check", **report_extra))
    report["manifest"] = manifest_path
    _print_summary(report)
    return 0 if base_ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    schema_path = args.schema or DEFAULT_SCHEMA
    data_dir = args.data or DEFAULT_TRAJ_DIR
    report = V.run_validation(data_dir, schema_path)
    manifest_path = _write_manifest(
        "validate",
        _manifest(
            "validate",
            schema=schema_path,
            data_dir=data_dir,
            **report,
        ),
    )
    summary = {
        "command": "validate",
        "ok": report["ok"],
        "schema": schema_path,
        "data_dir": data_dir,
        "counts": report["counts"],
        "n_errors": len(report["errors"]),
        "first_errors": report["errors"][:10],
        "manifest": manifest_path,
    }
    _print_summary(summary)
    return 0 if report["ok"] else 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    out_dir = args.out or DEFAULT_TRAJ_DIR
    if not os.path.isdir(run_dir):
        _print_summary(
            {
                "command": "transcribe",
                "ok": False,
                "error": f"run dir not found: {run_dir}",
            }
        )
        return 1

    result = None
    used = None
    failures: List[str] = []
    for module_name in ("rsi_step0.heuresis_adapter", "rsi_step0.trajectory"):
        result, used, failures = _call_module(
            module_name,
            ("transcribe_run", "transcribe", "run"),
            run_dir,
            out_dir,
        )
        if result is not None or used is not None:
            break
        # second try: config-dict calling convention
        result, used, failures = _call_module(
            module_name,
            ("transcribe_run", "transcribe", "run"),
            {"run_dir": run_dir, "output_dir": out_dir, "log_dir": run_dir},
        )
        if result is not None or used is not None:
            break

    if result is None and used is None:
        _print_summary(
            {
                "command": "transcribe",
                "ok": False,
                "run_dir": run_dir,
                "out_dir": out_dir,
                "error": "no transcribe entrypoint available",
                "details": failures,
            }
        )
        return 1

    manifest_path = _write_manifest(
        "transcribe",
        _manifest(
            "transcribe",
            run_dir=run_dir,
            out_dir=out_dir,
            entrypoint=used,
            result=_summarize_result(result),
        ),
    )
    _print_summary(
        {
            "command": "transcribe",
            "ok": True,
            "run_dir": run_dir,
            "out_dir": out_dir,
            "entrypoint": used,
            "result": _summarize_result(result),
            "manifest": manifest_path,
        }
    )
    return 0


def _run_config_command(
    command: str,
    config_path: str,
    module_name: str,
    entrypoints: Tuple[str, ...],
) -> int:
    if not os.path.isfile(config_path):
        _print_summary(
            {
                "command": command,
                "ok": False,
                "error": f"config file not found: {config_path}",
            }
        )
        return 1
    try:
        config = V.load_yaml(config_path)
    except Exception as exc:
        _print_summary(
            {
                "command": command,
                "ok": False,
                "config": config_path,
                "error": f"failed to load config: {type(exc).__name__}: {exc}",
            }
        )
        return 1

    result, used, failures = _call_module(module_name, entrypoints, config)
    if result is None and used is None:
        _print_summary(
            {
                "command": command,
                "ok": False,
                "config": config_path,
                "error": "no usable entrypoint for this command",
                "details": failures,
            }
        )
        return 1

    manifest_path = _write_manifest(
        command,
        _manifest(
            command,
            config=config_path,
            entrypoint=used,
            config_summary=config,
            result=_summarize_result(result),
        ),
    )
    _print_summary(
        {
            "command": command,
            "ok": True,
            "config": config_path,
            "entrypoint": used,
            "result": _summarize_result(result),
            "manifest": manifest_path,
        }
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    return _run_config_command(
        "train",
        args.config,
        "rsi_step0.training",
        ("train_pi", "train_pipeline", "run", "train", "train_pi_pipeline"),
    )


def cmd_replay(args: argparse.Namespace) -> int:
    return _run_config_command(
        "replay",
        args.config,
        "rsi_step0.scheduler",
        ("replay", "run_replay", "replay_run", "run"),
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    return _run_config_command(
        "evaluate",
        args.config,
        "rsi_step0.reward",
        ("evaluate", "evaluate_ablation", "evaluate_config", "run"),
    )


# --------------------------------------------------------------------------- parser
def _import_ok(name: str) -> bool:
    return _import_quiet(name) is not None


def _import_quiet(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi-step0",
        description="RSI Step0 CLI: transcribe / validate / train / replay / evaluate / check.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_transcribe = sub.add_parser("transcribe", help="transcribe heuresis logs into trajectory records")
    p_transcribe.add_argument("--run-dir", required=True, help="heuresis run log directory")
    p_transcribe.add_argument("--out", default=DEFAULT_TRAJ_DIR, help="trajectory output directory")
    p_transcribe.add_argument("--config", default=DEFAULT_MINIMAL_CFG, help="optional config file")
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_validate = sub.add_parser("validate", help="validate trajectory records and schemas")
    p_validate.add_argument("--schema", default=DEFAULT_SCHEMA, help="JSON Schema file or schema dir")
    p_validate.add_argument("--data", default=DEFAULT_TRAJ_DIR, help="trajectory records dir")
    p_validate.add_argument("--config", default=DEFAULT_MINIMAL_CFG, help="optional config file")
    p_validate.set_defaults(func=cmd_validate)

    p_train = sub.add_parser("train", help="train the pi router from trajectories")
    p_train.add_argument("--config", default=DEFAULT_TRAIN_CFG, help="training config YAML")
    p_train.set_defaults(func=cmd_train)

    p_replay = sub.add_parser("replay", help="deterministic replay of a run")
    p_replay.add_argument("--config", default=DEFAULT_REPLAY_CFG, help="replay config YAML")
    p_replay.set_defaults(func=cmd_replay)

    p_evaluate = sub.add_parser("evaluate", help="evaluate / compare an ablation config")
    p_evaluate.add_argument("--config", default=DEFAULT_ABLATION_CFG, help="ablation config YAML")
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_check = sub.add_parser("check", help="self-check: engine availability, imports, Python version")
    p_check.set_defaults(func=cmd_check)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Optional[Callable[[argparse.Namespace], int]] = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
