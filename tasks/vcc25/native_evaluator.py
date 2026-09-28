"""Native VCC evaluator replay over existing official artifacts.

This module deliberately does not call the task adapter, candidate runner or
Cell-Eval process during replay. Existing Cell-Eval output is treated as the
deterministic evaluator artifact and validated as such.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ai4ai.plugin_protocols import EvaluatorRequest, EvaluatorResult, ValidationResult


REQUIRED_METRICS = (
    "overlap_at_N", "de_spearman_sig", "de_spearman_lfc_sig", "pr_auc",
    "pearson_delta", "mae", "discrimination_score_l1",
)


class OfficialH1Evaluator:
    def validate_input(self, request: EvaluatorRequest) -> ValidationResult:
        errors = []
        for path in (request.prediction_artifact, request.contract_artifact):
            if not Path(path).is_file():
                errors.append(f"missing input: {path}")
        if request.existing_evaluator_artifact and not Path(request.existing_evaluator_artifact).is_file():
            errors.append(f"missing evaluator artifact: {request.existing_evaluator_artifact}")
        if request.existing_cell_eval_artifact and not Path(request.existing_cell_eval_artifact).is_file():
            errors.append(f"missing Cell-Eval artifact: {request.existing_cell_eval_artifact}")
        return ValidationResult(passed=not errors, errors=tuple(errors))

    def evaluate(self, request: EvaluatorRequest) -> EvaluatorResult:
        validation = self.validate_input(request)
        if not validation.passed:
            return EvaluatorResult(request.task_id, "invalid", validation=validation.__dict__)
        contract = json.loads(Path(request.contract_artifact).read_text(encoding="utf-8"))
        gates = contract.get("gates") or {}
        contract_ok = contract.get("status") == "ready" and all(gates.values())
        if not contract_ok:
            return EvaluatorResult(request.task_id, "invalid", validation={"contract": contract})

        metrics: dict[str, float] = {}
        fresh_metadata: dict[str, Any] = {}
        fresh_agg: Path | None = None
        if request.config.get("fresh_cell_eval") is True:
            fresh_agg, fresh_metadata = self._run_fresh_cell_eval(request)
            with fresh_agg.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            mean = next((row for row in rows if row.get("statistic") == "mean"), None)
            if mean:
                metrics.update({name: float(mean[name]) for name in REQUIRED_METRICS if name in mean})
        elif request.existing_evaluator_artifact:
            result = json.loads(Path(request.existing_evaluator_artifact).read_text(encoding="utf-8"))
            metrics.update({name: float((result.get("metrics") or {})[name]) for name in REQUIRED_METRICS if name in (result.get("metrics") or {})})
        elif request.existing_cell_eval_artifact:
            with Path(request.existing_cell_eval_artifact).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            mean = next((row for row in rows if row.get("statistic") == "mean"), None)
            if mean:
                metrics.update({name: float(mean[name]) for name in REQUIRED_METRICS if name in mean})
        complete = all(name in metrics for name in REQUIRED_METRICS)
        fresh_result = None
        if fresh_agg is not None and complete:
            fresh_result = Path(request.output_location) / "official_h1_result.json"
            fresh_result.parent.mkdir(parents=True, exist_ok=True)
            fresh_result.write_text(json.dumps({
                "protocol_id": "vcc-h1-native-fresh-cell-eval-replay-v1",
                "status": "pass", "targets": 100, "genes": 18080,
                "metrics": metrics, "artifacts": {"aggregate_csv": str(fresh_agg)},
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return EvaluatorResult(
            task_id=request.task_id,
            status="ok" if complete else "invalid",
            metrics=metrics,
            artifacts={"prediction": request.prediction_artifact, "contract": request.contract_artifact, "cell_eval": str(fresh_agg or request.existing_cell_eval_artifact or ""), "official_result": str(fresh_result or request.existing_evaluator_artifact or "")},
            metric_completeness=complete,
            validation={"contract": "ready", "cell_eval_replay": bool(request.existing_cell_eval_artifact), "official_result_replay": bool(request.existing_evaluator_artifact)},
            metadata={**({"cell_eval_execution": "fresh_process", **fresh_metadata} if fresh_agg else {"cell_eval_execution": "reused_existing_artifact"}), "scientific_execution": False},
        )

    def _run_fresh_cell_eval(self, request: EvaluatorRequest) -> tuple[Path, dict[str, Any]]:
        config = request.config
        python = Path(str(config["cell_eval_python"]))
        source = Path(str(config["cell_eval_source"]))
        reference = Path(str(config["reference_test"]))
        real_de = Path(str(config["real_de"]))
        output = Path(request.output_location).resolve() / "cell_eval_target_metrics"
        output.mkdir(parents=True, exist_ok=False)
        for path in (python, source, reference, real_de):
            if not path.exists():
                raise FileNotFoundError(path)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{source}:{env.get('PYTHONPATH', '')}"
        cache_dir = Path(str(config.get("numba_cache_dir", "/tmp/vcc25_native_cell_eval_numba"))).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["NUMBA_CACHE_DIR"] = str(cache_dir)
        mpl_dir = Path(str(config.get("mpl_config_dir", "/tmp/vcc25_native_cell_eval_mpl"))).resolve()
        mpl_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_dir)
        command = [
            str(python), "-m", "cell_eval", "run",
            "-ap", str(Path(request.prediction_artifact).resolve()),
            "-ar", str(reference), "-dr", str(real_de),
            "--pert-col", "target_gene", "--control-pert", "non-targeting",
            "--profile", "full", "--num-threads", str(config.get("num_threads", 16)),
            "-o", str(output),
        ]
        started = time.monotonic()
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        trace = Path(request.output_location).resolve() / "cell_eval_execution.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(json.dumps({
            "execution_mode": "native_protocol", "evaluator_mode": "native",
            "command": command, "pid": "child-process", "return_code": completed.returncode,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:],
            "runtime_seconds": time.monotonic() - started,
        }, indent=2) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"fresh Cell-Eval failed with code {completed.returncode}: {completed.stderr[-1000:]}")
        aggregate = output / "agg_results.csv"
        if not aggregate.is_file():
            raise FileNotFoundError(aggregate)
        return aggregate, {"cell_eval_run_id": output.name, "cell_eval_pid": "child-process", "trace": str(trace)}

    def normalize(self, result: EvaluatorResult) -> EvaluatorResult:
        return result

    def archive_record(self, result: EvaluatorResult) -> dict[str, Any]:
        return {"task": result.task_id, "status": result.status, "metrics": dict(result.metrics), "artifacts": dict(result.artifacts), "metric_completeness": result.metric_completeness, "metadata": dict(result.metadata)}
