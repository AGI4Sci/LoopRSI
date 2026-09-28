"""RSI Step0 validation module (M7).

Pure-stdlib validation helpers for actions, trajectory records, JSON Schema
compliance (``jsonschema`` when available, builtin fallback otherwise), pi/f_theta
data isolation, replay stability and aggregate run validation.

PyYAML is optional: :func:`load_yaml` prefers ``yaml.safe_load`` and falls back to
a tiny builtin parser so that config loading never crashes on an ``ImportError``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from rsi_step0 import contracts as C


# --------------------------------------------------------------------------- constants
ACTION_REQUIRED_FIELDS: tuple = (
    "action_id",
    "run_id",
    "round",
    "step",
    "operator",
    "mode",
    "target",
    "decision",
    "budget",
    "provenance",
)
BUDGET_FIELDS: tuple = ("max_seconds", "max_gpu_seconds", "max_cost")
PROVENANCE_FIELDS: tuple = ("parent_action_id", "model", "seed")


# --------------------------------------------------------------------------- helpers
def _import_optional(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError:
        return None


def _read_json_file(path: str) -> tuple:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith(".jsonl"):
                records = []
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                return records, None
            return json.load(fh), None
    except OSError as exc:
        return None, f"cannot read file: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"


def _collect_json_files(path: str) -> List[str]:
    """Enumerate actual ``.json`` files under ``path`` (file or directory)."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    files: List[str] = []
    for root, _dirs, names in os.walk(path):
        for name in sorted(names):
            if name.endswith(".json") or name.endswith(".jsonl"):
                files.append(os.path.join(root, name))
    return sorted(files)


# --------------------------------------------------------------------------- action
def validate_action(action: dict) -> list[str]:
    """Validate a single action dict and return a list of error strings."""
    errors: List[str] = []
    if not isinstance(action, dict):
        return [f"action must be a dict, got {type(action).__name__}"]

    for field in ACTION_REQUIRED_FIELDS:
        if field not in action:
            errors.append(f"missing required field: {field}")

    operator = action.get("operator")
    if operator is not None and not C.is_valid_operator(operator):
        errors.append(f"invalid operator {operator!r}; allowed: {', '.join(C.OPERATORS)}")

    mode = action.get("mode")
    if mode is not None and not C.is_valid_mode(mode):
        errors.append(f"invalid mode {mode!r}; allowed: {', '.join(C.MODES)}")

    for field in ("round", "step"):
        value = action.get(field)
        if value is not None and not isinstance(value, int):
            errors.append(f"field {field} must be an int, got {type(value).__name__}")

    for field in ("action_id", "run_id", "target"):
        value = action.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"field {field} must be a str, got {type(value).__name__}")

    decision = action.get("decision")
    if decision is not None and not isinstance(decision, dict):
        errors.append(f"field decision must be a dict, got {type(decision).__name__}")

    budget = action.get("budget")
    if isinstance(budget, dict):
        for field in BUDGET_FIELDS:
            if field not in budget:
                errors.append(f"budget missing required field: {field}")
        for field in BUDGET_FIELDS:
            value = budget.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                if field in budget:
                    errors.append(
                        f"budget.{field} must be a number, got {type(value).__name__}"
                    )
    elif budget is not None:
        errors.append(f"budget must be a dict, got {type(budget).__name__}")

    provenance = action.get("provenance")
    if isinstance(provenance, dict):
        for field in PROVENANCE_FIELDS:
            if field not in provenance:
                errors.append(f"provenance missing required field: {field}")
        for field in ("parent_action_id", "model"):
            value = provenance.get(field)
            if value is not None and not isinstance(value, str):
                errors.append(
                    f"provenance.{field} must be a str, got {type(value).__name__}"
                )
        seed = provenance.get("seed")
        if isinstance(seed, bool) or seed is not None and not isinstance(seed, int):
            errors.append(
                f"provenance.seed must be an int, got {type(seed).__name__}"
            )
    elif provenance is not None:
        errors.append(f"provenance must be a dict, got {type(provenance).__name__}")

    return errors


# --------------------------------------------------------------------------- schema
def _format_jsonschema_errors(iterable: Iterable[Any]) -> List[str]:
    messages: List[str] = []
    for err in iterable:
        loc = "/".join(str(part) for part in err.path) or "$"
        messages.append(f"{loc}: {err.message}")
    return messages


def _builtin_schema_check(data: Any, schema: dict, schema_path: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return [
            f"{schema_path}: data is not an object (type {type(data).__name__}); "
            f"expected {schema.get('type', 'any')}"
        ]

    for required in schema.get("required", []) or []:
        if required not in data:
            errors.append(f"{schema_path}: missing required property {required!r}")

    properties = schema.get("properties", {}) or {}
    if not isinstance(properties, dict):
        properties = {}

    for prop, subschema in properties.items():
        if prop not in data or not isinstance(subschema, dict):
            continue
        value = data[prop]
        enum = subschema.get("enum")
        if isinstance(enum, list) and value not in enum:
            errors.append(
                f"{schema_path}: property {prop!r} value {value!r} not in enum {enum}"
            )
        ptype = subschema.get("type")
        if isinstance(ptype, str) and ptype != "any":
            if ptype == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    errors.append(
                        f"{schema_path}: property {prop!r} must be integer, "
                        f"got {type(value).__name__}"
                    )
            elif ptype == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(
                        f"{schema_path}: property {prop!r} must be number, "
                        f"got {type(value).__name__}"
                    )
            elif ptype == "string" and not isinstance(value, str):
                errors.append(
                    f"{schema_path}: property {prop!r} must be string, "
                    f"got {type(value).__name__}"
                )
            elif ptype == "boolean" and not isinstance(value, bool):
                errors.append(
                    f"{schema_path}: property {prop!r} must be boolean, "
                    f"got {type(value).__name__}"
                )
            elif ptype == "object" and not isinstance(value, dict):
                errors.append(
                    f"{schema_path}: property {prop!r} must be object, "
                    f"got {type(value).__name__}"
                )
            elif ptype == "array" and not isinstance(value, list):
                errors.append(
                    f"{schema_path}: property {prop!r} must be array, "
                    f"got {type(value).__name__}"
                )
    return errors


def validate_schema(data, schema_path: str) -> list[str]:
    """Validate ``data`` against the JSON Schema at ``schema_path``.

    Uses ``jsonschema`` when importable, otherwise a builtin structural check for
    dict instances (``required`` and ``enum``/``type`` on ``properties``).
    """
    schema, read_error = _read_json_file(schema_path)
    if read_error:
        return [f"schema {schema_path}: {read_error}"]
    if not isinstance(schema, dict):
        return [
            f"schema {schema_path} must be a JSON object, "
            f"got {type(schema).__name__}"
        ]

    if _import_optional("jsonschema") is not None:
        try:
            from jsonschema.validators import validator_for

            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
            return _format_jsonschema_errors(validator.iter_errors(data))
        except ImportError:
            pass  # jsonschema present but incomplete -> fall back to builtin
        except Exception as exc:  # invalid schema or other validator problem
            return [f"schema {schema_path} failed jsonschema validation: {exc}"]

    return _builtin_schema_check(data, schema, schema_path)


def _check_schema_file(path: str) -> List[str]:
    schema, read_error = _read_json_file(path)
    if read_error:
        return [f"schema {path}: {read_error}"]
    if not isinstance(schema, dict):
        return [f"schema {path} must be a JSON object, got {type(schema).__name__}"]
    if _import_optional("jsonschema") is not None:
        try:
            from jsonschema.validators import validator_for

            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            return []
        except ImportError:
            pass
        except Exception as exc:
            return [f"schema {path} is not a valid JSON Schema: {exc}"]
    # Builtin sanity: must expose at least one schema keyword.
    if not any(key in schema for key in ("type", "properties", "required", "enum")):
        return [f"schema {path} has none of the recognized keywords: type/properties/required/enum"]
    return []


# --------------------------------------------------------------------------- isolation
def _identity_keys(records: list) -> set:
    keys: set = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            keys.add(f"record:{index}:{C.hash_json(record)}")
            continue
        key = record.get("dedupe_key") or record.get("record_id")
        if key is None:
            key = C.hash_json(record)
        keys.add(str(key))
    return keys


def validate_isolation(pi_records: list, ftheta_records: list) -> list[str]:
    """Verify pi and f_theta data share no overlapping identity keys."""
    if not isinstance(pi_records, list) or not isinstance(ftheta_records, list):
        return ["pi_records and ftheta_records must both be lists"]
    pi_keys = _identity_keys(pi_records)
    ftheta_keys = _identity_keys(ftheta_records)
    shared = sorted(pi_keys & ftheta_keys)
    if not shared:
        return []
    return [f"key overlap between pi and f_theta data: {key}" for key in shared]


# --------------------------------------------------------------------------- replay
def validate_replay(records: list) -> dict:
    """Validate replay stability across event sequences via ``C.hash_json``.

    * ``records`` as a list of lists -> each inner list is one replay attempt; the
      hashes of every attempt must match for the replay to be ``stable``.
    * ``records`` as a flat event list -> checks that no step key is repeated with
      conflicting content (an internal determinism conflict).

    Always returns ``{"stable": bool, "hash": str, "deterministic": True}``.
    """
    if not isinstance(records, list):
        return {"stable": False, "hash": "", "deterministic": True}

    if records and all(isinstance(item, list) for item in records):
        hashes = [C.hash_json(item) for item in records]
        base = hashes[0]
        stable = all(item == base for item in hashes)
        return {"stable": stable, "hash": base, "deterministic": True}

    errors: List[str] = []
    seen: Dict[tuple, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index}: expected object, got {type(record).__name__}")
            continue
        key = tuple(
            str(record.get(field, ""))
            for field in ("round", "step", "operator", "dedupe_key", "record_id")
        )
        content_hash = C.hash_json(record)
        if key in seen:
            if content_hash != seen[key]:
                errors.append(f"record {index}: conflicting content for step key {key!r}")
        else:
            seen[key] = content_hash

    return {
        "stable": not errors,
        "hash": C.hash_json(records),
        "deterministic": True,
    }


# --------------------------------------------------------------------------- aggregate
def _validate_record(
    record: Any,
    schema_path: Optional[str],
    errors: List[str],
    counts: Dict[str, int],
    seen_dedupe: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(record, dict):
        errors.append(
            f"record #{counts['records']}: expected object, got {type(record).__name__}"
        )
        counts["records"] += 1
        return
    counts["records"] += 1
    rid = record.get("record_id") or f"record#{counts['records']}"

    if schema_path:
        errors.extend(validate_schema(record, schema_path))

    version = record.get("schema_version")
    if version is not None and version != C.SCHEMA_VERSION:
        errors.append(f"{rid}: schema_version {version!r} != {C.SCHEMA_VERSION!r}")

    if "action" not in record:
        errors.append(f"{rid}: missing action")
    else:
        errors.extend(validate_action(record.get("action")))

    label = record.get("label")
    if label is not None and not C.is_valid_label(label):
        errors.append(f"{rid}: invalid label {label!r}")

    outcome = record.get("outcome")
    if (
        isinstance(outcome, dict)
        and outcome.get("status") is not None
        and not C.is_valid_status(outcome["status"])
    ):
        errors.append(f"{rid}: invalid outcome.status {outcome['status']!r}")

    if seen_dedupe is not None:
        dedupe_key = record.get("dedupe_key")
        if dedupe_key is not None:
            if dedupe_key in seen_dedupe:
                errors.append(f"{rid}: duplicate dedupe_key {dedupe_key!r}")
            else:
                seen_dedupe[dedupe_key] = True


def run_validation(traj_dir: str, schema_dir: str) -> dict:
    """Aggregate-validate every record file under ``traj_dir`` and the schemas.

    ``schema_dir`` may be a directory (all ``*.json`` schemas) or a single schema
    file. Returns ``{"ok": bool, "errors": list[str], "counts": dict}``.
    """
    errors: List[str] = []
    counts = {"files": 0, "records": 0, "schemas": 0, "errors": 0}

    if not os.path.isdir(traj_dir):
        errors.append(f"trajectory dir not found: {traj_dir}")

    schema_files: List[str] = []
    if isinstance(schema_dir, str) and schema_dir:
        if not os.path.exists(schema_dir):
            errors.append(f"schema dir/file not found: {schema_dir}")
        else:
            schema_files = _collect_json_files(schema_dir)
            if not schema_files:
                errors.append(f"no JSON schema files found under {schema_dir}")

    schema_path = schema_files[0] if schema_files else None
    for schema_file in schema_files:
        counts["schemas"] += 1
        errors.extend(_check_schema_file(schema_file))

    if os.path.isdir(traj_dir):
        record_files = _collect_json_files(traj_dir)
        counts["files"] = len(record_files)
        seen_dedupe: Dict[str, Any] = {}
        for record_file in record_files:
            obj, read_error = _read_json_file(record_file)
            if read_error:
                errors.append(f"{record_file}: {read_error}")
                continue
            records = obj if isinstance(obj, list) else [obj]
            for record in records:
                _validate_record(record, schema_path, errors, counts, seen_dedupe)

    counts["errors"] = len(errors)
    return {"ok": not errors, "errors": errors, "counts": counts}


# --------------------------------------------------------------------------- yaml
def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment outside of quotes (naive but sufficient)."""
    quote = None
    for index, char in enumerate(line):
        if char in "\"'":
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _split_inline_list(inner: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    quote = None
    for char in inner:
        if char in "\"'":
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            current.append(char)
        elif char == "," and quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part != ""]


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        inner = token[1:-1]
        return inner.replace("\\\"", "\"").replace("\\'", "'").replace("\\\\", "\\")
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_inline_list(inner)]
    lowered = token.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", token):
        return float(token)
    return token


def _simple_yaml(path: str) -> dict:
    """Minimal fallback YAML parser: 2-level key/value, numbers, bools, lists."""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = _strip_comment(line).rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if stripped in ("---", "..."):
                continue

            if indent >= 1 and current_section is not None:
                if stripped.startswith("- "):
                    container = result.get(current_section)
                    if not isinstance(container, list):
                        container = []
                        result[current_section] = container
                    container.append(_parse_scalar(stripped[2:].strip()))
                    continue
                subkey, sep, value = stripped.partition(":")
                if sep:
                    if value.strip() == "":
                        continue  # unsupported 3rd-level header, skip
                    container = result.get(current_section)
                    if not isinstance(container, dict):
                        container = {}
                        result[current_section] = container
                    container[subkey.strip()] = _parse_scalar(value.strip())
                    continue
                continue  # stray content at depth 1, ignore

            if indent >= 1 and current_section is None:
                continue  # content without a section header, ignore

            key, sep, value = stripped.partition(":")
            if not sep:
                continue  # malformed top-level line, ignore
            key = key.strip()
            if value.strip() == "":
                current_section = key
                result[key] = None
            else:
                current_section = None
                result[key] = _parse_scalar(value.strip())

    # Empty section headers become empty dicts.
    for key, value in list(result.items()):
        if value is None:
            result[key] = {}
    return result


def load_yaml(path: str) -> dict:
    """Load a YAML config dict. Prefers ``yaml.safe_load``; never crashes on ImportError."""
    yaml = _import_optional("yaml")
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(
                f"config {path} must be a mapping, got {type(data).__name__}"
            )
        return data
    return _simple_yaml(path)


__all__ = [
    "validate_action",
    "validate_schema",
    "validate_isolation",
    "validate_replay",
    "run_validation",
    "load_yaml",
]
