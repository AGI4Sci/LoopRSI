"""Preparation and dry preflight for isolated VCC25 coding candidates."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class CandidateBoundaryError(ValueError):
    """Raised when a candidate request crosses an experiment boundary."""


_REQUEST_FIELDS = {
    "allowed_paths",
    "hypothesis",
    "activation_diagnostics",
    "train_command",
    "validation_command",
    "expected_artifacts",
}
_EDITABLE_ROOTS = frozenset({"crpm", "scripts", "tests"})
_IMPLEMENTATION_PREFIX = Path("tasks/vcc25/implementation")
_FINAL_EXPRESSION_MARKERS = (
    "adata_test.h5ad",
    "test_expression",
    "test_h5ad",
    "competition_test",
    "real_de.csv",
    "cell_eval_raw_vcc_test_full",
)
_VALIDATION_ENTRYPOINT = "official_h1_autonomous_research_loop.py"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _string_list(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CandidateBoundaryError(f"{name} must be a non-empty string list")
    return tuple(item.strip() for item in value)


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _safe_diagnostic(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in value)
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized or not normalized[0].isalpha():
        normalized = "diagnostic_" + normalized
    return normalized


@dataclass(frozen=True)
class PreparedCandidate:
    workspace: Path
    request_path: Path
    implementation_request_path: Path
    allowed_paths: tuple[Path, ...]
    train_command: tuple[str, ...]
    validation_command: tuple[str, ...]
    expected_artifacts: tuple[str, ...]


class CandidateExecutor:
    """Validate and materialize requests without invoking Codex or a GPU."""

    def prepare(self, request: Mapping[str, Any], workspace: str | Path) -> PreparedCandidate:
        if not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS:
            raise CandidateBoundaryError("coding request fields do not match the bounded contract")
        resolved_workspace = Path(workspace).resolve()
        if not resolved_workspace.is_dir():
            raise CandidateBoundaryError("experiment workspace must exist")
        hypothesis = request.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise CandidateBoundaryError("hypothesis must be a non-empty string")
        allowed_raw = _string_list("allowed_paths", request.get("allowed_paths"))
        diagnostics = _string_list("activation_diagnostics", request.get("activation_diagnostics"))
        train_command = _string_list("train_command", request.get("train_command"))
        validation_command = _string_list("validation_command", request.get("validation_command"))
        expected_artifacts = _string_list("expected_artifacts", request.get("expected_artifacts"))
        self._guard_search_command(train_command, require_validation_entrypoint=False)
        self._guard_search_command(validation_command, require_validation_entrypoint=True)

        allowed_paths: list[Path] = []
        for raw in allowed_raw:
            relative = Path(raw)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise CandidateBoundaryError(f"candidate path escapes workspace: {raw}")
            if relative.parts[0] not in _EDITABLE_ROOTS:
                raise CandidateBoundaryError(f"candidate path is not declared editable: {raw}")
            target = (resolved_workspace / _IMPLEMENTATION_PREFIX / relative).resolve()
            if not _under(target, resolved_workspace):
                raise CandidateBoundaryError(f"candidate path resolves outside workspace: {raw}")
            policy_root = (resolved_workspace / _IMPLEMENTATION_PREFIX / relative.parts[0]).resolve()
            if not _under(target, policy_root):
                raise CandidateBoundaryError(f"candidate path escapes its editable root: {raw}")
            allowed_paths.append(target)

        request_value = json.loads(json.dumps(dict(request), ensure_ascii=False))
        request_path = resolved_workspace / "artifacts/candidate_request.json"
        _write_json(request_path, request_value)
        request_id = "knowledge-" + hashlib.sha256(
            json.dumps(request_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        implementation_request = {
            "request_id": request_id,
            "hypothesis": hypothesis.strip(),
            "change_scope": ["implementation"],
            "allowed_paths": list(allowed_raw),
            "required_capabilities": ["static_checks", "validation_only"],
            "activation_diagnostics": [_safe_diagnostic(value) for value in diagnostics],
            "trial_proposal": {
                "hypothesis": hypothesis.strip(),
                "change_scope": ["implementation"],
                "parameters": {
                    "candidate_variant": "autonomous_research_candidate",
                },
                "expected_effect": {"validation_delta": "positive"},
                "acceptance_criteria": {},
                "resource_request": {"gpu_count": 0, "cpu": 1, "memory_mb": 1024, "max_runtime_minutes": 30},
            },
        }
        implementation_request_path = resolved_workspace / "artifacts/implementation_request.json"
        _write_json(implementation_request_path, implementation_request)
        return PreparedCandidate(
            workspace=resolved_workspace,
            request_path=request_path,
            implementation_request_path=implementation_request_path,
            allowed_paths=tuple(allowed_paths),
            train_command=train_command,
            validation_command=validation_command,
            expected_artifacts=expected_artifacts,
        )

    def _guard_search_command(
        self,
        command: Sequence[str],
        *,
        require_validation_entrypoint: bool,
    ) -> None:
        lowered = " ".join(command).lower()
        if any(marker in lowered for marker in _FINAL_EXPRESSION_MARKERS):
            raise CandidateBoundaryError("search command references final-test expression data")
        if any(token.lower() == "run_trial" for token in command):
            raise CandidateBoundaryError("canonical run_trial is forbidden during search")
        if require_validation_entrypoint and not any(
            token.endswith(_VALIDATION_ENTRYPOINT) for token in command
        ):
            raise CandidateBoundaryError(
                f"validation command must target {_VALIDATION_ENTRYPOINT}"
            )

    def run_preflight(
        self,
        prepared: PreparedCandidate,
        *,
        required_tools: Sequence[str] = (),
        required_assets: Sequence[str | Path] = (),
    ) -> dict[str, Any]:
        tools = {
            tool: shutil.which(tool) if not os.path.isabs(tool) else (tool if os.access(tool, os.X_OK) else None)
            for tool in required_tools
        }
        assets = {
            str(Path(asset)): Path(asset).is_file() and os.access(Path(asset), os.R_OK)
            for asset in required_assets
        }
        missing_tools = sorted(tool for tool, path in tools.items() if path is None)
        missing_assets = sorted(path for path, readable in assets.items() if not readable)
        result = {
            "status": "ok" if not missing_tools and not missing_assets else "failed",
            "request_path": str(prepared.request_path),
            "implementation_request_path": str(prepared.implementation_request_path),
            "tools": tools,
            "assets": assets,
            "missing_tools": missing_tools,
            "missing_assets": missing_assets,
            "codex_invoked": False,
            "gpu_invoked": False,
        }
        _write_json(prepared.workspace / "artifacts/candidate_preflight.json", result)
        return result
