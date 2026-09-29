"""Auditable phase gates for the knowledge-driven VCC25 experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ai4ai.plugin_manifest import load_task_plugin_manifest

from .evaluation_authority import EvaluationRecord, promotion_eligible
from .knowledge_bridge import KnowledgeBridge
from .knowledge_worker import KnowledgeDrivenVcc25Worker
from .layer_agent import LayerDecisionBackend, RecordingDecisionBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("L1", "L2", "L3", "L4", "L5")


class ExperimentGateError(ValueError):
    """Raised when an experiment phase lacks immutable prerequisites."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentGateError(f"required artifact is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentGateError(f"artifact must contain an object: {path.name}")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class KnowledgeExperiment:
    def __init__(
        self,
        config: Mapping[str, Any],
        config_path: Path,
        workspace: Path,
        backend: LayerDecisionBackend,
    ) -> None:
        self.config = dict(config)
        self.config_path = config_path.resolve()
        self.workspace = workspace.resolve()
        self.backend = backend

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        workspace: str | Path,
        backend: LayerDecisionBackend | None = None,
    ) -> "KnowledgeExperiment":
        path = Path(config_path).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ExperimentGateError("experiment config must be an object")
        configured_backend = config.get("decision_backend", "recording")
        if backend is None:
            if configured_backend != "recording":
                raise ExperimentGateError(
                    "local preflight supports decision_backend=recording unless a backend is supplied"
                )
            backend = RecordingDecisionBackend()
        return cls(config, path, Path(workspace), backend)

    def _validate_workspace_boundary(self) -> None:
        for raw in self.config.get("source_checkouts") or []:
            source = Path(str(raw)).resolve()
            if _is_within(self.workspace, source):
                raise ExperimentGateError(
                    f"experiment workspace must not equal or be inside source checkout: {source}"
                )

    def _git_revision(self) -> str | None:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    def _source_revision(self) -> str | None:
        return self._git_revision() or (
            str(self.config["source_revision"])
            if self.config.get("source_revision")
            else None
        )

    def run_preflight(self) -> dict[str, Any]:
        self._validate_workspace_boundary()
        if self.workspace.exists() and any(self.workspace.iterdir()):
            raise ExperimentGateError("preflight workspace must be empty for append-only artifacts")
        self.workspace.mkdir(parents=True, exist_ok=True)
        task_plugin = REPOSITORY_ROOT / str(self.config["task_plugin"])
        bridge = KnowledgeBridge(load_task_plugin_manifest(task_plugin))
        worker = KnowledgeDrivenVcc25Worker(bridge, self.backend)
        results = [
            worker.step(layer, 0, index, {"task_id": "vcc25"})
            for index, layer in enumerate(LAYERS, start=1)
        ]

        config_bytes = self.config_path.read_bytes()
        manifest = {
            "schema_version": "vcc25-knowledge-experiment/v1",
            "experiment_id": str(self.config["experiment_id"]),
            "phase": "preflight",
            "evaluation_authority": None,
            "decision_backend": type(self.backend).__name__,
            "config_path": str(self.config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_revision": self._source_revision(),
            "source_checkouts": list(self.config.get("source_checkouts") or []),
            "knowledge_root": str(self.config.get("knowledge_root")),
            "gpu_commands": [],
            "final_budget_consumed": False,
        }
        _write_json_once(self.workspace / "run_manifest.json", manifest)

        rows = []
        for result in results:
            knowledge = result.action["knowledge"]
            rows.append(
                {
                    "layer": result.layer,
                    "round": result.round,
                    "step": result.step,
                    "skill_id": knowledge["skill_id"],
                    "card_ids": knowledge["card_ids"],
                    "content_hash": knowledge["content_hash"],
                    "token_budget": knowledge["token_budget"],
                    "source": knowledge["source"],
                    "evaluation_authority": result.action["evaluation_authority"],
                    "decision": result.action["decision"],
                    "prior_decision_count": result.action["prior_decision_count"],
                }
            )
        with (self.workspace / "layer_decisions.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        candidate_request = rows[-1]["decision"]["coding_request"]
        _write_json_once(self.workspace / "candidate_request.json", candidate_request)
        _write_text_once(self.workspace / "candidate.patch", "")
        _write_json_once(
            self.workspace / "candidate_code_sha256.json",
            {"status": "not_generated", "candidate_hash": None, "frozen": False},
        )
        _write_json_once(
            self.workspace / "split_audit.json",
            {
                "schema_version": "vcc25-split-audit/v1",
                "status": "not_run",
                "contents": "target_identifiers_only",
                "splits": {},
            },
        )
        result = {
            "status": "ok",
            "phase": "preflight",
            "layers_completed": list(LAYERS),
            "backend": type(self.backend).__name__,
            "executed_commands": [],
            "gpu_invoked": False,
            "candidate_executed": False,
        }
        _write_json_once(self.workspace / "preflight_result.json", result)
        _write_json_once(
            self.workspace / "summary.json",
            {
                "status": "ok",
                "phase": "preflight",
                "promotion_eligible": False,
                "final_budget_consumed": False,
                "candidate_generated": False,
            },
        )
        return result

    def run_smoke(self, result: Mapping[str, Any]) -> dict[str, Any]:
        preflight = _read_json(self.workspace / "preflight_result.json")
        if preflight.get("status") != "ok":
            raise ExperimentGateError("successful preflight is required before smoke")
        candidate = _read_json(self.workspace / "candidate_code_sha256.json")
        if not candidate.get("candidate_hash"):
            raise ExperimentGateError("generated candidate hash is required before smoke")
        record = dict(result)
        record.update({"phase": "smoke", "metric_source": "smoke_only", "reward": None})
        _write_json_once(self.workspace / "smoke_result.json", record)
        return record

    def run_validation(self, result: Mapping[str, Any]) -> dict[str, Any]:
        _read_json(self.workspace / "smoke_result.json")
        record = EvaluationRecord.from_mapping(result)
        if record.authority != "search_validation" or not record.full_validation:
            raise ExperimentGateError("full search_validation record is required")
        value = dict(result)
        value["promotion_contract_eligible"] = promotion_eligible(record)
        _write_json_once(self.workspace / "validation_result.json", value)
        return value

    def run_final(self) -> dict[str, Any]:
        candidate_path = self.workspace / "candidate_code_sha256.json"
        if not candidate_path.is_file():
            raise ExperimentGateError("frozen candidate is required before final")
        candidate = _read_json(candidate_path)
        if not candidate.get("candidate_hash") or candidate.get("frozen") is not True:
            raise ExperimentGateError("frozen candidate is required before final")
        promotion_path = self.workspace / "promotion_record.json"
        if not promotion_path.is_file():
            raise ExperimentGateError("promotion record is required before final")
        promotion = _read_json(promotion_path)
        if promotion.get("promotion_eligible") is not True:
            raise ExperimentGateError("promotion record is not eligible")
        if promotion.get("candidate_hash") != candidate.get("candidate_hash"):
            raise ExperimentGateError("promotion record candidate hash mismatch")
        budget = self.workspace / "final_budget.json"
        if budget.is_file() and _read_json(budget).get("consumed") is True:
            raise ExperimentGateError("single-use final budget has already been consumed")
        return {
            "status": "ready",
            "phase": "final",
            "candidate_hash": candidate["candidate_hash"],
            "final_budget_remaining": 1,
            "selection_feedback_allowed": False,
        }
