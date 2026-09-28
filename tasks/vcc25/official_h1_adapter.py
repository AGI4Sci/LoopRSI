#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tasks.adapter_contract import AdapterError, TaskAdapter, adapter_main, load_json_or_jsonl, run_command
from ai4ai.candidate_execution import execute_candidate
from ai4ai.plugin_protocols import (
    CandidateExecutionRequest, CandidateExecutionResult, EvaluationResult, ValidationResult,
)

ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1")
ASSETS = ROOT / "assets/official_2025"
REQUIRED_METRICS = (
    "overlap_at_N", "de_spearman_sig", "de_spearman_lfc_sig", "pr_auc",
    "pearson_delta", "mae", "discrimination_score_l1",
)

CANDIDATE_G_VARIANT = "candidate_g_promoted_delta_model"
AUTONOMOUS_VARIANT = "autonomous_research_candidate"
ALLOWED_CANDIDATE_VARIANTS = (CANDIDATE_G_VARIANT, AUTONOMOUS_VARIANT)
AUTONOMOUS_ROUTE_PARAMETERS = frozenset({
    "autonomous_runner",
    "candidate_id",
    "candidate_origin",
    "controlled_change",
    "external_mode",
    "historical_candidate_reuse",
    "max_eval_samples",
    "max_train_samples",
    "research_seed",
    "shuffled_target_control",
    "smoke_mode",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VCC25OfficialH1TaskAdapter(TaskAdapter):
    task_name = "vcc25"
    trial_parameters = frozenset({"method", "candidate_variant", "seed"}) | AUTONOMOUS_ROUTE_PARAMETERS
    method_capabilities = frozenset({"candidate"})
    method_capability_details = {
        "candidate": {
            "available": True,
            "protocol": "official_h1_cell_eval",
            "candidate_variants": list(ALLOWED_CANDIDATE_VARIANTS),
            "variant_roles": {
                CANDIDATE_G_VARIANT: "historical_promoted_delta_model",
                AUTONOMOUS_VARIANT: (
                    "generic admission route for a validated ResearchStudio/Heuresis "
                    "candidate; not a predefined architecture"
                ),
            },
            "autonomous_candidate_contract": {
                "candidate_origin": "autonomous_research",
                "research_seed": "lingshu_skill",
                "historical_candidate_reuse": False,
                "requires_generated_runner": "scripts/run_official_h1_autonomous_candidate.sh",
                "must_not_fallback_to": [CANDIDATE_G_VARIANT],
            },
        }
    }
    parameter_capabilities = {
        "method": {"type": "string", "enum": ["candidate"]},
        "candidate_variant": {"type": "string", "enum": list(ALLOWED_CANDIDATE_VARIANTS)},
        "seed": {"type": "integer", "enum": [20260907, 20260908, 20260909]},
        "candidate_origin": {"type": "string", "enum": ["autonomous_research"]},
        "autonomous_runner": {"type": "string"},
        "candidate_id": {"type": "string"},
        "controlled_change": {"type": "string"},
        "external_mode": {"type": "string", "enum": ["hesc_latent_impute"]},
        "historical_candidate_reuse": {"type": "boolean"},
        "max_eval_samples": {"type": "integer", "minimum": 0},
        "max_train_samples": {"type": "integer", "minimum": 0},
        "research_seed": {"type": "string", "enum": ["lingshu_skill"]},
        "shuffled_target_control": {"type": "boolean"},
        "smoke_mode": {"type": "boolean"},
    }

    def __init__(self) -> None:
        self.project = Path(__file__).resolve().parent
        self.implementation = self.project / "implementation"

    def initialization_capabilities(self, spec: dict[str, Any]) -> dict[str, Any]:
        capabilities = super().initialization_capabilities(spec)
        capabilities.update({
            "protocols": ["official_h1_cell_eval"],
            "default_protocol": "official_h1_cell_eval",
            "baseline": {
                "name": "candidate_g_three_seed_ensemble",
                "metric": "pearson_delta",
                "value": 0.0926758799981326,
                "role": "historical_reference_not_single_trial_baseline",
            },
            "scientific_boundaries": {
                "train_targets": 150,
                "validation_targets": 50,
                "test_targets": 100,
                "genes": 18080,
                "test_expression_for_selection": False,
                "evaluator": "Cell-Eval 0.8.2",
            },
            "autonomous_research_admission": {
                "candidate_variant": AUTONOMOUS_VARIANT,
                "purpose": (
                    "Allow a validated novel ResearchStudio/Heuresis proposal "
                    "to enter implementation and trial routing without being "
                    "collapsed onto a historical candidate."
                ),
                "candidate_origin": "autonomous_research",
                "research_seed": "lingshu_skill",
                "historical_candidate_reuse": False,
                "route_is_architecture": False,
                "requires_generated_runner": "scripts/run_official_h1_autonomous_candidate.sh",
                "forbidden_fallbacks": [CANDIDATE_G_VARIANT],
                "strict_admission_env": "REQUIRE_AUTONOMOUS_CANDIDATE=1",
                "strict_admission_active": os.environ.get("REQUIRE_AUTONOMOUS_CANDIDATE") == "1",
            },
        })
        return capabilities

    def proposal_to_trial(self, proposal: dict[str, Any], defaults: dict[str, Any], name: str) -> dict[str, Any]:
        trial = super().proposal_to_trial(proposal, defaults, name)
        values = trial["cli_overrides"]
        if values.get("method") != "candidate" or not values.get("candidate_variant"):
            raise AdapterError("official H1 execution requires method=candidate and a candidate_variant")
        variant = values["candidate_variant"]
        if variant not in ALLOWED_CANDIDATE_VARIANTS:
            raise AdapterError(f"unsupported official H1 candidate_variant: {variant!r}")
        if os.environ.get("REQUIRE_AUTONOMOUS_CANDIDATE") == "1" and variant != AUTONOMOUS_VARIANT:
            raise AdapterError(
                "this run requires candidate_variant='autonomous_research_candidate'; "
                f"historical fallback {variant!r} is not admissible"
            )
        if variant == AUTONOMOUS_VARIANT:
            values.setdefault("candidate_origin", "autonomous_research")
            values.setdefault("research_seed", "lingshu_skill")
            values.setdefault("historical_candidate_reuse", False)
            if values.get("candidate_origin") != "autonomous_research":
                raise AdapterError("autonomous route requires candidate_origin='autonomous_research'")
            if values.get("research_seed") != "lingshu_skill":
                raise AdapterError("autonomous route requires research_seed='lingshu_skill'")
            if values.get("historical_candidate_reuse") is not False:
                raise AdapterError("autonomous route requires historical_candidate_reuse=false")
            trial.setdefault("candidate_provenance", {
                "candidate_origin": "autonomous_research",
                "research_seed": "lingshu_skill",
                "historical_candidate_reuse": False,
                "adapter_route_only": True,
            })
        return trial

    def _required_paths(self) -> dict[str, Path]:
        return {
            "train_h5ad": ASSETS / "train/adata_Training.h5ad",
            "validation_h5ad": ASSETS / "validation/adata_Validation.h5ad",
            "test_h5ad": ASSETS / "test/adata_Test.h5ad",
            "train_targets": ASSETS / "train/pert_counts_Training.csv",
            "validation_targets": ASSETS / "validation/pert_counts_Validation.csv",
            "test_targets": ASSETS / "test/pert_counts_Test.csv",
            "gene_names": ASSETS / "gene_names.csv",
            "gene_embeddings": ROOT / "assets/lingshu_hf_b77f980/gene_embeddings.npz",
            "real_de": ROOT / "runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv",
        }

    def prepare_data(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.validate_data(request)

    def validate_data(self, request: dict[str, Any]) -> dict[str, Any]:
        paths = self._required_paths()
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise AdapterError(f"official H1 artifacts missing: {missing}")
        small_hashes = {name: _sha256(path) for name, path in paths.items() if path.stat().st_size < 10 * 1024 * 1024}
        return {
            "status": "ok", "dataset_ref": "vcc25-official-h1@2025",
            "paths": {name: str(path) for name, path in paths.items()},
            "small_artifact_sha256": small_hashes,
            "targets": {"train": 150, "validation": 50, "test": 100}, "genes": 18080,
        }

    def run_baseline(self, request: dict[str, Any]) -> dict[str, Any]:
        raise AdapterError("official H1 historical references are immutable; run_trial is required")

    def run_trial(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("method") != "candidate":
            raise AdapterError("official H1 run requires method=candidate")
        variant = str(request.get("candidate_variant") or "")
        if variant not in ALLOWED_CANDIDATE_VARIANTS:
            raise AdapterError(f"unsupported official H1 candidate_variant: {variant!r}")
        seed = int(request.get("seed", 20260907))
        if seed not in {20260907, 20260908, 20260909}:
            raise AdapterError("official H1 seed is outside the frozen three-seed set")
        output = Path(str(request["output"])).resolve()
        run_dir = output.parent / f"{output.stem}_official_h1"
        if run_dir.exists():
            raise AdapterError(f"official H1 output already exists: {run_dir}")
        if variant == AUTONOMOUS_VARIANT:
            script = self.implementation / "scripts/run_official_h1_autonomous_candidate.sh"
            if not script.is_file():
                raise AdapterError(
                    "autonomous route is admitted but no generated runner is present; "
                    f"expected {script.relative_to(self.implementation)}. "
                    "Do not fallback to candidate_g_promoted_delta_model."
                )
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir()
            request_json = run_dir / "autonomous_trial_request.json"
            request_json.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            command = ["bash", str(script), str(seed), str(run_dir), str(ROOT), str(request_json)]
            if request.get("external_mode"):
                command = ["env", f"EXTERNAL_MODE={request['external_mode']}", *command]
        else:
            script = self.implementation / "scripts/run_official_h1_candidate_g.sh"
            command = ["bash", str(script), str(seed), str(run_dir), str(ROOT)]
        eval_profile = request.get("eval_profile")
        eval_skip_metrics = request.get("eval_skip_metrics")
        env_prefix = []
        if eval_profile == "native":
            # In-loop native evaluation: skip the cell-eval subprocess entirely
            # (VCC_NATIVE_EVAL=1); only the memory-safe native evaluator runs.
            env_prefix.append("VCC_NATIVE_EVAL=1")
        elif eval_profile:
            env_prefix.append(f"CELL_EVAL_PROFILE={eval_profile}")
        if eval_skip_metrics:
            env_prefix.append(f"CELL_EVAL_SKIP_METRICS={eval_skip_metrics}")
        if env_prefix:
            command = ["env", *env_prefix, *command]
        run_command(command, self.implementation, output.with_suffix(".backend"))
        result = load_json_or_jsonl(run_dir / "official_h1_result.json")
        if request.get("smoke_mode") is True:
            return {
                **result,
                "status": "ok" if result.get("status") == "pass" else result.get("status", "blocked"),
                "method": "candidate",
                "candidate_variant": variant,
                "candidate_origin": "autonomous_research",
                "research_seed": "lingshu_skill",
                "historical_candidate_reuse": False,
                "dataset_ref": "vcc25-official-h1@2025",
                "scientific_evidence": False,
            }
        metrics = result.get("metrics") or {}
        skipped = {
            m.strip() for m in str(request.get("eval_skip_metrics") or "").split(",") if m.strip()
        }
        required = tuple(m for m in REQUIRED_METRICS if m not in skipped)
        if result.get("status") != "pass" or any(name not in metrics for name in required):
            raise AdapterError("official H1 result failed canonical metric completeness")
        return {
            **result,
            "status": "ok",
            "method": "candidate",
            "candidate_variant": variant,
            "candidate_origin": (
                "autonomous_research" if variant == AUTONOMOUS_VARIANT else "historical_promoted_candidate"
            ),
            "research_seed": "lingshu_skill" if variant == AUTONOMOUS_VARIANT else None,
            "historical_candidate_reuse": variant != AUTONOMOUS_VARIANT,
            "dataset_ref": "vcc25-official-h1@2025",
            "protocol": {
                "name": "official_h1_cell_eval", "train_targets": 150,
                "validation_targets": 50, "test_targets": 100, "genes": 18080,
                "test_expression_used_for_selection": False,
            },
            "scientific_evidence": True,
        }

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        raw = load_json_or_jsonl(Path(str(request.get("raw_result", ""))).resolve())
        metrics = raw.get("metrics") or {}
        skipped = {
            m.strip() for m in str(request.get("eval_skip_metrics") or "").split(",") if m.strip()
        }
        required = tuple(m for m in REQUIRED_METRICS if m not in skipped)
        if any(name not in metrics for name in required):
            raise AdapterError("canonical official H1 metrics are incomplete")
        return {"status": "ok", "metrics": metrics, "scientific_evidence": raw.get("scientific_evidence") is True}

    def summarize_results(self, request: dict[str, Any]) -> dict[str, Any]:
        paths = [Path(value) for value in str(request.get("results", "")).split(",") if value]
        rows = [load_json_or_jsonl(path) for path in paths]
        rows.sort(key=lambda row: float((row.get("metrics") or {}).get("pearson_delta", float("-inf"))), reverse=True)
        return {"status": "ok", "primary_metric": "pearson_delta", "rows": rows, "best": rows[0] if rows else None}


class VCC25CandidateAdapter:
    """Opt-in native candidate boundary; production still uses the legacy adapter."""

    def __init__(self) -> None:
        self.project = Path(__file__).resolve().parent
        self.implementation = self.project / "implementation"

    def validate_package(self, package: Any) -> ValidationResult:
        return ValidationResult(passed=isinstance(package, (dict, str, Path)))

    def prepare_execution(self, package: Any, context: dict[str, Any]) -> CandidateExecutionRequest:
        request = {**dict(package), **dict(context)}
        variant = str(request.get("candidate_variant") or AUTONOMOUS_VARIANT)
        seed = int(request.get("seed", 20260907))
        if variant not in ALLOWED_CANDIDATE_VARIANTS:
            raise AdapterError(f"unsupported official H1 candidate_variant: {variant!r}")
        if seed not in {20260907, 20260908, 20260909}:
            raise AdapterError("official H1 seed is outside the frozen three-seed set")
        output = Path(str(request["output"])).resolve()
        run_dir = output.parent / f"{output.stem}_official_h1"
        if variant == AUTONOMOUS_VARIANT:
            runner = self.implementation / "scripts/run_official_h1_autonomous_candidate.sh"
            request_json = run_dir / "autonomous_trial_request.json"
            command = ("bash", str(runner), str(seed), str(run_dir), str(ROOT), str(request_json))
            required = (str(runner), str(request.get("autonomous_runner", "")))
        else:
            runner = self.implementation / "scripts/run_official_h1_candidate_g.sh"
            request_json = None
            command = ("bash", str(runner), str(seed), str(run_dir), str(ROOT))
            required = (str(runner),)
        if request_json is not None:
            run_dir.mkdir(parents=True, exist_ok=False)
            request_json.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateExecutionRequest(
            candidate_id=variant,
            command=command,
            working_directory=str(self.implementation),
            output_location=str(run_dir),
            environment={"SKIP_EVAL": "1"},
            required_inputs=tuple(path for path in required if path),
            expected_artifacts=(
                str(run_dir / "official_h1_prediction.h5ad"),
                str(run_dir / "official_h1_contract.json"),
            ),
            runtime={"seed": seed, "evaluation": False, "authority": "native_candidate_adapter"},
        )

    def execute(self, request: CandidateExecutionRequest) -> CandidateExecutionResult:
        return execute_candidate(request)

    def validate_artifacts(self, result: CandidateExecutionResult) -> ValidationResult:
        expected = tuple(result.artifact_manifest.get("expected", ()))
        missing = tuple(path for path in expected if path not in result.produced_artifacts)
        return ValidationResult(passed=result.return_code == 0 and not missing, errors=missing)

    def prepare_existing_artifact(
        self, prediction_path: str | Path, contract_path: str | Path,
    ) -> CandidateExecutionRequest:
        """Bind an existing prediction; this never starts a candidate runner."""
        prediction = Path(prediction_path).resolve()
        contract = Path(contract_path).resolve()
        return CandidateExecutionRequest(
            candidate_id="vcc25_existing_prediction_replay",
            command=("true",),
            working_directory=str(self.implementation),
            output_location=str(prediction.parent),
            required_inputs=(str(prediction), str(contract)),
            expected_artifacts=(str(prediction), str(contract)),
            runtime={"mode": "existing_artifact_replay", "evaluation": False},
        )

    def materialize_existing_artifact(
        self, request: CandidateExecutionRequest,
    ) -> CandidateExecutionResult:
        """Validate and bind an existing H1 artifact without legacy execution."""
        import h5py

        prediction, contract = (Path(path) for path in request.required_inputs)
        errors: list[str] = []
        metadata: dict[str, Any] = {"mode": "existing_artifact_replay"}
        if not prediction.is_file() or not contract.is_file():
            errors.append("prediction/contract input is missing")
        else:
            try:
                contract_obj = json.loads(contract.read_text(encoding="utf-8"))
                metadata["contract_status"] = contract_obj.get("status")
                metadata["contract_protocol"] = contract_obj.get("protocol_id")
                gates = contract_obj.get("gates") or {}
                required_gates = {
                    "condition_targets_100", "gene_count_18080", "gene_order_matches_official",
                    "obs_count_matches_reference_test", "obs_order_matches_reference_test",
                    "prediction_targets_match_condition_csv", "reference_targets_match_condition_csv",
                    "target_column_present",
                }
                errors.extend(f"contract gate failed: {gate}" for gate in sorted(required_gates) if gates.get(gate) is not True)
                with h5py.File(prediction, "r") as handle:
                    if "X" not in handle:
                        errors.append("prediction H5AD has no X dataset")
                    else:
                        shape = tuple(int(value) for value in handle["X"].shape)
                        metadata["shape"] = shape
                        metadata["dtype"] = str(handle["X"].dtype)
                        if shape != (170846, 18080):
                            errors.append(f"unexpected prediction shape: {shape}")
                        else:
                            import numpy as np
                            dataset = handle["X"]
                            finite = True
                            nonnegative = True
                            block = max(1, min(1024, int(dataset.chunks[0]) if dataset.chunks else 1024))
                            for start in range(0, shape[0], block):
                                values = dataset[start:min(start + block, shape[0]), ...]
                                finite = finite and bool(np.isfinite(values).all())
                                nonnegative = nonnegative and bool((values >= 0).all())
                                if not finite or not nonnegative:
                                    break
                            metadata["finite"] = finite
                            metadata["nonnegative"] = nonnegative
                            if not finite:
                                errors.append("prediction contains non-finite values")
                            if not nonnegative:
                                errors.append("prediction contains negative values")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"artifact validation error: {exc}")
        produced = tuple(path for path in request.expected_artifacts if Path(path).is_file())
        manifest = {
            "mode": "existing_artifact_replay",
            "prediction": str(prediction),
            "contract": str(contract),
            "metadata": metadata,
            "errors": errors,
        }
        return CandidateExecutionResult(
            candidate_id=request.candidate_id,
            return_code=0 if not errors else 1,
            stdout="native materialization replay\n",
            stderr="\n".join(errors),
            execution_metadata=metadata,
            produced_artifacts=produced,
            artifact_manifest=manifest,
        )


class OfficialH1Evaluator:
    """Evaluator protocol facade; canonical Cell-Eval remains task-local."""

    def validate_input(self, prediction: Any) -> ValidationResult:
        if not isinstance(prediction, dict):
            return ValidationResult(passed=False, errors=("prediction must be a mapping",))
        return ValidationResult(passed=True)

    def evaluate(self, prediction: Any, context: dict[str, Any]) -> EvaluationResult:
        metrics = prediction.get("metrics", {}) if isinstance(prediction, dict) else {}
        return EvaluationResult(task_id="vcc25", status="ok", metrics=metrics, evidence={"official_h1": True})

    def normalize(self, result: EvaluationResult) -> EvaluationResult:
        return result

    def archive_record(self, result: EvaluationResult) -> dict[str, Any]:
        return {"task": result.task_id, "status": result.status, "metrics": dict(result.metrics), "evidence": dict(result.evidence)}


if __name__ == "__main__":
    raise SystemExit(adapter_main(VCC25OfficialH1TaskAdapter()))
