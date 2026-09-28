from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_ar.autoresearch import (
    StrategyBridge, _activation_report, _candidate_records, _next_live_proposal,
    _proposal_execution_partition, _validate_execution_policy,
    _run_coding_agent_with_retry,
    _coding_control_suite, _execution_context_for_parents, _request_fingerprint,
    _run_stability_validation, _small_smoke_proposal, _evaluate_coding_smoke,
    _executed_parameter_report,
    resume_autoresearch, run_autoresearch,
)
from omni_ar.finalizer import choose_winner
from omni_ar.initialization import RoughIdeaEngine
from omni_ar.project_records import ProjectRecords
from omni_ar.preflight import PreflightError, run_preflight


REPOSITORY = Path(__file__).resolve().parents[2]


class UnifiedAutoResearchTests(unittest.TestCase):
    def test_selected_parent_reuses_nearest_generated_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            worktree = Path(temporary) / "candidate"
            task = worktree / "tasks/sst2-small/task_spec.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task: {}\n", encoding="utf-8")
            archive = [
                {"candidate_id": "coded", "candidate_worktree": str(worktree), "parent_ids": []},
                {"candidate_id": "tuned", "parent_ids": ["coded"]},
            ]
            repo, inherited_task, code_parent = _execution_context_for_parents(
                REPOSITORY, REPOSITORY / "tasks/sst2-small/task_spec.yaml",
                archive, ["tuned"],
            )
        self.assertEqual(repo, worktree.resolve())
        self.assertEqual(inherited_task, task.resolve())
        self.assertEqual(code_parent, "coded")

    def test_coding_control_suite_has_disabled_baseline_and_enabled(self) -> None:
        request = {
            "request_id": "control-test",
            "trial_proposal": {
                "hypothesis": "Use SP-MLP.",
                "change_scope": ["model"],
                "parameters": {"method": "sp_mlp", "seed": 42},
                "expected_effect": {}, "acceptance_criteria": {},
                "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
            },
        }
        proposal, roles = _coding_control_suite(
            REPOSITORY, REPOSITORY / "tasks/sst2-small/task_spec.yaml", request,
            {"max_train_samples": 1000},
        )
        self.assertEqual(roles[0], ["baseline", "disabled"])
        self.assertEqual(roles[1], ["enabled"])
        self.assertEqual(proposal["experiment_proposals"][0]["parameters"]["method"], "multinomial_naive_bayes")
        self.assertIn(["combination"], roles)

    def test_coding_smoke_compares_disabled_and_enabled_effect(self) -> None:
        wrapper = {
            "proposal_id": "smoke", "task_name": "sst2_small", "round": 0,
            "verdict": "smoke", "evidence_gaps": [], "risks": [],
            "experiment_proposals": [
                {"hypothesis": "disabled", "change_scope": ["model"],
                 "parameters": {"method": "multinomial_naive_bayes"},
                 "expected_effect": {}, "acceptance_criteria": {}, "resource_request": {}},
                {"hypothesis": "enabled", "change_scope": ["model"],
                 "parameters": {"method": "linear_svm", "c": 0.1,
                                "max_train_samples": 0, "max_eval_samples": 0},
                 "expected_effect": {}, "acceptance_criteria": {}, "resource_request": {}},
            ],
        }
        smoke = _small_smoke_proposal(
            REPOSITORY, REPOSITORY / "tasks/sst2-small/task_spec.yaml", wrapper,
        )
        self.assertIsNotNone(smoke)
        self.assertEqual(len(smoke["experiment_proposals"]), 2)
        self.assertEqual(smoke["experiment_proposals"][0]["parameters"]["max_train_samples"], 1024)
        self.assertEqual(smoke["experiment_proposals"][1]["parameters"]["max_train_samples"], 1024)
        self.assertEqual(smoke["experiment_proposals"][1]["parameters"]["max_eval_samples"], 256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for index, score in enumerate((0.75, 0.60)):
                raw = root / f"raw{index}.json"
                raw.write_text(json.dumps({
                    "activation_diagnostics": {"hybrid_active": index == 1},
                    "executed_parameters": smoke["experiment_proposals"][index]["parameters"],
                }), encoding="utf-8")
                standardized = root / f"standardized{index}.json"
                standardized.write_text(json.dumps({
                    "metrics": {"accuracy": {"role": "primary", "value": score,
                                               "direction": "maximize"}},
                    "protocol": {"executed_parameters": smoke["experiment_proposals"][index]["parameters"]},
                    "artifacts": {"raw_result": str(raw)},
                }), encoding="utf-8")
                jobs.append({"archive_gate": {"failure_class": None},
                             "standardized_result": str(standardized)})
            trace = root / "trace.json"
            trace.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            report = _evaluate_coding_smoke(trace, smoke, ["hybrid_active"])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["effect_passed"])

    def test_executed_parameter_report_rejects_missing_combination_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps({
                "protocol": {"executed_parameters": {"method": "hybrid"}},
            }), encoding="utf-8")
            report = _executed_parameter_report(
                path, {"method": "hybrid", "parent_c": 0.1},
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing"], ["parent_c"])

    def test_failed_code_request_fingerprint_is_stable(self) -> None:
        request = {
            "hypothesis": "new loss", "change_scope": ["loss"],
            "required_capabilities": ["loss"],
            "trial_proposal": {"parameters": {"method": "x", "alpha": 1}},
        }
        reordered = json.loads(json.dumps(request))
        reordered["trial_proposal"]["parameters"] = {"alpha": 1, "method": "x"}
        self.assertEqual(_request_fingerprint(request), _request_fingerprint(reordered))

    def test_winner_requires_budget_controls_and_requested_seed_count(self) -> None:
        common = {
            "bucket": "accepted", "scientific_evidence": True,
            "direction": "maximize", "activation_validation": {"passed": True},
            "resource_usage": {"within_budget": True}, "protocol": {},
        }
        entries = [
            {**common, "candidate_id": "single", "score": 0.95, "seed_count": 1},
            {**common, "candidate_id": "stable", "score": 0.90, "seed_count": 3,
             "stability": {"mean": 0.90, "std": 0.01}},
            {**common, "candidate_id": "over-budget", "score": 0.99, "seed_count": 3,
             "resource_usage": {"within_budget": False}},
            {**common, "candidate_id": "uncontrolled-code", "score": 0.98,
             "seed_count": 3, "route": "coding_agent", "control_validation": {"passed": False}},
        ]
        self.assertEqual(
            choose_winner(entries, required_seeds=3)["candidate_id"], "stable"
        )

    def test_final_stability_validation_writes_aggregate_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standardized = []
            for index, score in enumerate((0.80, 0.82, 0.84)):
                path = root / f"seed_{index}.json"
                path.write_text(json.dumps({
                    "status": "ok", "task": "sst2_small", "trial_id": f"s{index}",
                    "metrics": {"accuracy": {
                        "value": score, "direction": "maximize", "role": "primary",
                    }},
                    "resource_usage": {"runtime_seconds": 1.0, "within_budget": True},
                    "protocol": {"stability": {"status": "single_seed", "num_seeds": 1}},
                    "artifacts": {},
                }), encoding="utf-8")
                standardized.append(path)
            trace = root / "trace.json"
            trace.write_text(json.dumps({
                "execution_mode": "local", "scientific_evidence": True,
                "jobs": [{
                    "job_name": f"s{index}",
                    "archive_gate": {"eligible": True},
                    "standardized_result": str(path),
                } for index, path in enumerate(standardized)],
            }), encoding="utf-8")
            records = ProjectRecords(root / "project")
            records.initialize({"task": "sst2_small"})
            winner = {
                "candidate_id": "best", "bucket": "accepted", "score": 0.82,
                "direction": "maximize", "scientific_evidence": True,
                "parameters": {"method": "multinomial_naive_bayes", "seed": 42},
                "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
                "activation_validation": {"passed": True, "required": []},
                "dataset_lineage": {}, "control_validation": None,
            }
            strategy = StrategyBridge(REPOSITORY, "islands", "maximize")
            with patch("omni_ar.autoresearch._execute", return_value=trace):
                candidate = _run_stability_validation(
                    REPOSITORY, REPOSITORY / "tasks/sst2-small/task_spec.yaml",
                    records, strategy, winner, 3, "local", 5,
                )
            aggregate = json.loads(Path(candidate["standardized_result"]).read_text())
        self.assertAlmostEqual(candidate["score"], 0.82)
        self.assertEqual(candidate["seed_count"], 3)
        self.assertEqual(aggregate["protocol"]["stability"]["status"], "multi_seed")
        self.assertEqual(aggregate["resource_usage"]["stability_runs"], 3)

    def fresh_vcc_rough_idea(self, root: Path) -> Path:
        source = REPOSITORY / "research_initializations/vcc25-flow-combination-search-20260819"
        answers = json.loads((source / "raw_answers.json").read_text())
        rough = RoughIdeaEngine(REPOSITORY).build(
            REPOSITORY / "tasks/vcc25/task_spec.yaml", answers,
            initialization_id="autoresearch-unit", confirmed=True,
        )
        path = root / "rough_idea.yaml"
        path.write_text(__import__("yaml").safe_dump(rough, sort_keys=False), encoding="utf-8")
        return path

    def test_rjob_dry_run_is_not_training_failure_or_scientific_evidence(self) -> None:
        proposal = {
            "proposal_id": "dry-run-proposal",
            "verdict": "technical validation only",
            "experiment_proposals": [{"hypothesis": "candidate"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.json"
            trace.write_text(json.dumps({
                "execution_mode": "rjob_dry_run",
                "scientific_evidence": False,
                "jobs": [{
                    "job_name": "dry-job",
                    "archive_gate": {"eligible": False, "decision": "reject"},
                    "standardized_result": None,
                }],
            }), encoding="utf-8")
            strategy = StrategyBridge(REPOSITORY, "islands", "maximize")
            candidates = _candidate_records(
                proposal, Path(temporary) / "proposal.json", trace, 0, [], strategy,
            )
        self.assertEqual(candidates[0]["bucket"], "dry_run")
        self.assertFalse(candidates[0]["scientific_evidence"])

    def test_activation_verification_result_cannot_enter_scientific_archive(self) -> None:
        proposal = {
            "proposal_id": "verification-proposal", "verdict": "verify activation",
            "experiment_proposals": [{"hypothesis": "feature is active"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result.json"
            result.write_text(json.dumps({
                "status": "ok",
                "metrics": {"accuracy": {
                    "value": 0.99, "direction": "maximize", "role": "primary",
                }},
                "protocol": {"name": "activation_verification"},
            }), encoding="utf-8")
            trace = root / "trace.json"
            trace.write_text(json.dumps({
                "execution_mode": "rjob", "scientific_evidence": True,
                "jobs": [{
                    "job_name": "verification-job",
                    "archive_gate": {"eligible": True, "archive_bucket": "accepted"},
                    "standardized_result": str(result),
                }],
            }), encoding="utf-8")
            strategy = StrategyBridge(REPOSITORY, "islands", "maximize")
            candidate = _candidate_records(
                proposal, root / "proposal.json", trace, 0, [], strategy,
            )[0]
        self.assertEqual(candidate["bucket"], "verification")
        self.assertIsNone(candidate["score"])
        self.assertEqual(candidate["observed_score"], 0.99)
        self.assertFalse(candidate["scientific_evidence"])
        self.assertEqual(candidate["strategy_bucket"], "activation_verified")
        self.assertTrue(candidate["strategy_metadata"]["excluded_from_search_archive"])
        self.assertEqual(
            candidate["archive_gate"]["failure_class"], "non_scientific_evidence",
        )

    def test_generated_candidate_requires_active_runtime_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.json"
            standardized = root / "standardized.json"
            raw.write_text(json.dumps({
                "activation_diagnostics": {"new_model_active": True, "forward_calls": 3},
            }), encoding="utf-8")
            standardized.write_text(json.dumps({"artifacts": {"raw_result": str(raw)}}), encoding="utf-8")
            passed = _activation_report(standardized, ["new_model_active", "forward_calls"])
            failed = _activation_report(standardized, ["new_model_active", "gradient_updates"])
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["missing"], ["gradient_updates"])

    def test_inactive_generated_candidate_cannot_enter_archive(self) -> None:
        proposal = {
            "proposal_id": "generated", "verdict": "candidate",
            "experiment_proposals": [{"hypothesis": "new model"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.json"
            standardized = root / "standardized.json"
            trace = root / "trace.json"
            raw.write_text(json.dumps({"activation_diagnostics": {"new_model_active": False}}), encoding="utf-8")
            standardized.write_text(json.dumps({
                "metrics": {"accuracy": {"role": "primary", "value": 0.99, "direction": "maximize"}},
                "artifacts": {"raw_result": str(raw)},
            }), encoding="utf-8")
            trace.write_text(json.dumps({
                "execution_mode": "local", "scientific_evidence": True,
                "jobs": [{
                    "job_name": "generated-job",
                    "archive_gate": {"eligible": True, "decision": "accept"},
                    "standardized_result": str(standardized),
                }],
            }), encoding="utf-8")
            strategy = StrategyBridge(REPOSITORY, "islands", "maximize")
            candidates = _candidate_records(
                proposal, root / "proposal.json", trace, 0, [], strategy,
                route="coding_agent", required_activation_diagnostics=["new_model_active"],
            )
        self.assertEqual(candidates[0]["bucket"], "invalid")
        self.assertIsNone(candidates[0]["score"])
        self.assertEqual(candidates[0]["archive_gate"]["failure_class"], "method_inactive")

    def test_inherited_code_candidate_requires_executed_parameter_receipt(self) -> None:
        proposal = {
            "proposal_id": "inherited", "verdict": "candidate",
            "experiment_proposals": [{
                "hypothesis": "inherited hybrid", "parameters": {"char_weight": 0.75},
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.json"
            standardized = root / "standardized.json"
            trace = root / "trace.json"
            raw.write_text(json.dumps({
                "activation_diagnostics": {"hybrid_active": 1},
            }), encoding="utf-8")
            standardized.write_text(json.dumps({
                "metrics": {"accuracy": {"role": "primary", "value": 0.90,
                                            "direction": "maximize"}},
                "protocol": {"executed_parameters": {"char_weight": 0.5}},
                "artifacts": {"raw_result": str(raw)},
            }), encoding="utf-8")
            trace.write_text(json.dumps({
                "execution_mode": "local", "scientific_evidence": True,
                "jobs": [{"archive_gate": {"eligible": True},
                          "standardized_result": str(standardized)}],
            }), encoding="utf-8")
            candidate = _candidate_records(
                proposal, root / "proposal.json", trace, 1, ["coded"],
                StrategyBridge(REPOSITORY, "islands", "maximize"),
                route="adapter", candidate_worktree=str(root),
                required_activation_diagnostics=["hybrid_active"],
            )[0]
        self.assertEqual(candidate["bucket"], "invalid")
        self.assertFalse(candidate["activation_validation"]["parameter_validation"]["passed"])
        self.assertEqual(
            candidate["activation_validation"]["parameter_validation"]["mismatched"],
            ["char_weight"],
        )

    def test_coding_apply_retries_independent_attempts_until_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "request.json"
            request.write_text("{}", encoding="utf-8")
            calls = []

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                calls.append(output)
                ready = len(calls) == 2
                (output / "coding_route.json").write_text(json.dumps({
                    "schema_version": "omni-ar-coding-route/v1",
                    "status": "ready" if ready else "invalid",
                    "candidate_worktree": str(root / "candidate") if ready else None,
                }), encoding="utf-8")
                return SimpleNamespace(returncode=0 if ready else 2, stdout="", stderr="")

            with patch.dict(os.environ, {"OMNI_AR_CODING_MAX_ATTEMPTS": "3"}), patch(
                "omni_ar.autoresearch.subprocess.run", side_effect=fake_run,
            ):
                route_path = _run_coding_agent_with_retry(
                    REPOSITORY, REPOSITORY / "tasks/example/task_spec.yaml",
                    request, root / "coding", "apply",
                )
            route = json.loads(route_path.read_text())
        self.assertEqual(route["status"], "ready")
        self.assertEqual(route["attempts_used"], 2)
        self.assertEqual(len(route["attempt_history"]), 2)

    def test_next_round_inherits_unified_boyue_model_and_timeout_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context.json"
            context.write_text(json.dumps({"round": 0}), encoding="utf-8")
            records = ProjectRecords(root / "project")
            records.initialize({})
            captured: dict = {}

            def fake_stage(command, **kwargs):
                captured.update(kwargs["env"])
                output = Path(command[command.index("--out") + 1])
                output.write_text("{}\n", encoding="utf-8")

            with patch.dict(os.environ, {
                "OMNI_AR_BOYUE_MODEL": "deepseek-v4-flash",
                "HEURESIS_SUGGESTION_TIMEOUT_SEC": "17",
            }, clear=False), patch(
                "omni_ar.autoresearch._run_external_stage", side_effect=fake_stage,
            ):
                _next_live_proposal(
                    REPOSITORY, context, root / "round", 1,
                    records, [], "",
                    active_task_spec=REPOSITORY / "tasks/sst2-small/task_spec.yaml",
                )

            self.assertEqual(captured["BOYUE_MODEL_NAME"], "deepseek-v4-flash")
            self.assertEqual(captured["HEURESIS_SUGGESTION_TIMEOUT_SEC"], "17")
            next_context = json.loads((root / "round/heuresis_context.json").read_text())
            self.assertIn("sp_mlp", next_context["dataset"]["capabilities"]["method_capabilities"])
            self.assertEqual(next_context["active_code"]["task_spec"], str(REPOSITORY / "tasks/sst2-small/task_spec.yaml"))

    def test_resume_restores_recorded_planning_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            records = ProjectRecords(project)
            records.initialize({
                "repository": str(REPOSITORY),
                "rough_idea": str(Path(temporary) / "rough_idea.yaml"),
                "strategy": "omni_epic", "planning_mode": "live",
                "execution_mode": "rjob", "coding_mode": "apply",
                "round_budget": 3, "planning_model": "deepseek-v4-flash",
            })
            observed: dict[str, str | None] = {}

            def fake_run(*args, **kwargs):
                observed["model"] = os.environ.get("OMNI_AR_BOYUE_MODEL")
                return project / "autoresearch_result.json"

            with patch.dict(os.environ, {"OMNI_AR_BOYUE_MODEL": "caller-model"}), patch(
                "omni_ar.autoresearch.run_autoresearch", side_effect=fake_run,
            ):
                resume_autoresearch(project)
                self.assertEqual(os.environ["OMNI_AR_BOYUE_MODEL"], "caller-model")

            self.assertEqual(observed["model"], "deepseek-v4-flash")

    def test_qa_trial_limit_counts_deferred_method_once(self) -> None:
        trial = {
            "parameters": {"method": "new_model"},
            "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
        }
        proposal = {
            "experiment_proposals": [trial, {
                "parameters": {"method": "new_model", "control": True},
                "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
            }],
            "implementation_requests": [{"trial_proposal": trial}],
        }
        _validate_execution_policy(proposal, {
            "max_trials_per_round": 1, "max_gpus": 0,
            "max_runtime_minutes": 10,
        })

    def test_qa_trial_limit_can_defer_existing_trial_for_new_mechanism(self) -> None:
        proposal = {
            "experiment_proposals": [{
                "parameters": {"method": "existing_model"},
                "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
            }],
            "implementation_requests": [{"trial_proposal": {
                "parameters": {"method": "new_model"},
                "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
            }}],
        }
        _validate_execution_policy(proposal, {
            "max_trials_per_round": 1, "max_gpus": 0,
            "max_runtime_minutes": 10,
        })

    def test_implemented_request_is_promoted_to_direct_trial(self) -> None:
        trial = {
            "hypothesis": "Run the registered SP-MLP.",
            "change_scope": ["model"],
            "parameters": {"method": "sp_mlp", "hidden_units": 64},
            "expected_effect": {}, "acceptance_criteria": {},
            "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
        }
        proposal = {
            "experiment_proposals": [],
            "implementation_requests": [{"trial_proposal": trial}],
        }
        task_path = REPOSITORY / "tasks/sst2-small/task_spec.yaml"
        scripts = REPOSITORY / "controller_bash/scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from task_contract import load_task_spec

        direct, unresolved = _proposal_execution_partition(
            REPOSITORY, load_task_spec(task_path), task_path, proposal,
        )
        self.assertEqual(len(direct), 1)
        self.assertEqual(direct[0]["parameters"]["method"], "sp_mlp")
        self.assertEqual(unresolved, [])

    def test_simulation_is_never_a_scientific_winner(self) -> None:
        self.assertIsNone(choose_winner([{
            "bucket": "accepted", "scientific_evidence": False,
            "score": 99.0, "direction": "maximize",
        }]))

    def test_native_omni_epic_parent_and_generation(self) -> None:
        strategy = StrategyBridge(REPOSITORY, "omni_epic", "maximize")
        parents, _ = strategy.select(0)
        first = strategy.on_result(
            "c0", 0.5, idea="first idea", parent_ids=parents, round_number=0,
        )
        parents, context = strategy.select(1)
        second = strategy.on_result(
            "c1", 0.6, idea="second idea", parent_ids=parents, round_number=1,
        )
        self.assertEqual(first["generation"], 0)
        self.assertEqual(parents, ["c0"])
        self.assertEqual(second["generation"], 1)
        self.assertIsInstance(context, str)

    def test_coding_plan_normalizes_implementation_and_interface_paths(self) -> None:
        request = {
            "request_id": "unit-coding-plan",
            "hypothesis": "A task-local mechanism can be registered safely.",
            "change_scope": ["model"],
            "allowed_paths": ["train.py", "tasks/vcc25/adapter.py"],
            "required_capabilities": ["new_model"],
            "activation_diagnostics": ["new_model_active"],
            "trial_proposal": {
                "hypothesis": "Compile the post-code trial.",
                "change_scope": ["model"], "parameters": {},
                "expected_effect": {}, "acceptance_criteria": {},
                "resource_request": {},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            proc = subprocess.run([
                sys.executable,
                str(REPOSITORY / "controller_bash/scripts/run_coding_agent.py"),
                "--task", str(REPOSITORY / "tasks/vcc25/task_spec.yaml"),
                "--request", str(request_path), "--output-dir", str(root / "out"),
                "--mode", "plan",
            ], cwd=REPOSITORY, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            route = json.loads((root / "out/coding_route.json").read_text())
            self.assertEqual(route["status"], "planned")
            self.assertEqual(route["workspace_allowed_paths"], [
                "tasks/vcc25/implementation/train.py",
                "tasks/vcc25/adapter.py",
            ])

    def test_three_round_vcc_simulation_writes_lineage_but_no_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rough = self.fresh_vcc_rough_idea(Path(temporary))
            result_path = run_autoresearch(
                REPOSITORY, rough, Path(temporary) / "project", rounds=3,
                strategy_name="omni_epic", planning_mode="dry_run",
                execution_mode="simulated", coding_mode="off",
            )
            result = json.loads(result_path.read_text())
            lineage = json.loads(
                (Path(temporary) / "project/records/lineage.json").read_text()
            )
            manifest = json.loads(
                (Path(temporary) / "project/final_package/manifest.json").read_text()
            )
            self.assertEqual(result["termination"]["rounds_completed"], 3)
            self.assertEqual(len(lineage["nodes"]), 3)
            self.assertGreaterEqual(len(lineage["edges"]), 2)
            self.assertIsNone(result["winner"])
            self.assertEqual(manifest["status"], "no_accepted_candidate")

    def test_round_failure_is_finalized_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "omni_ar.autoresearch._execute", side_effect=RuntimeError("synthetic failure")
        ):
            rough = self.fresh_vcc_rough_idea(Path(temporary))
            result_path = run_autoresearch(
                REPOSITORY, rough, Path(temporary) / "project", rounds=2,
                strategy_name="islands", planning_mode="dry_run",
                execution_mode="simulated", coding_mode="off",
            )
            result = json.loads(result_path.read_text())
            project = json.loads(
                (Path(temporary) / "project/records/project.json").read_text()
            )
            failed_round = json.loads(
                (Path(temporary) / "project/records/rounds/round_000.json").read_text()
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["termination"]["reason"], "fatal_round_failure")
            self.assertEqual(project["status"], "failed")
            self.assertEqual(failed_round["status"], "failed")
            self.assertTrue((Path(temporary) / "project/final_package/manifest.json").is_file())

    def test_failed_project_resumes_from_first_incomplete_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rough = self.fresh_vcc_rough_idea(root)
            project = root / "project"
            with patch("omni_ar.autoresearch._execute", side_effect=RuntimeError("once")):
                failed = json.loads(run_autoresearch(
                    REPOSITORY, rough, project, rounds=2, strategy_name="islands",
                    planning_mode="dry_run", execution_mode="simulated", coding_mode="off",
                ).read_text())
            self.assertEqual(failed["status"], "failed")
            resumed = json.loads(run_autoresearch(
                REPOSITORY, rough, project, rounds=2, strategy_name="islands",
                planning_mode="dry_run", execution_mode="simulated", coding_mode="off",
                resume=True,
            ).read_text())
            self.assertEqual(resumed["status"], "ok")
            self.assertEqual(resumed["termination"]["rounds_completed"], 2)
            stages = json.loads((project / "records/stages.json").read_text())["stages"]
            self.assertEqual(stages["planning"]["status"], "completed")
            self.assertTrue(stages["planning"].get("reused"))

    def test_rjob_preflight_rejects_output_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rough_path = self.fresh_vcc_rough_idea(root)
            rough = __import__("yaml").safe_load(rough_path.read_text())
            with self.assertRaises(PreflightError):
                run_preflight(
                    REPOSITORY, rough, REPOSITORY / "tasks/vcc25/task_spec.yaml",
                    root / "preflight.json", planning_mode="dry_run",
                    execution_mode="rjob_dry_run", coding_mode="plan",
                    project_dir=root / "outside-project",
                )
            result = json.loads((root / "preflight.json").read_text())
            self.assertIn("rjob_output_location", result["blockers"])


if __name__ == "__main__":
    unittest.main()
