from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "controller_bash/scripts"
sys.path.insert(0, str(SCRIPTS))

from task_contract import ContractError, load_task_spec, normalize_result, validate_proposal  # noqa: E402


class TaskAgnosticPromptContract(unittest.TestCase):
    def test_runtime_prompts_do_not_name_project_tasks(self) -> None:
        prompt_sources = [
            ROOT / "ResearchStudio-main/scripts/ideaspark_boyue_author.py",
            ROOT / "controller_bash/scripts/heuresis_suggest.py",
            ROOT / "controller_bash/prompts/codex_build_prototype.md",
            ROOT / "controller_bash/prompts/codex_apply_suggestion.md",
            ROOT / "ResearchStudio-main/ResearchStudio-Idea/skills/idea_spark/references/system-prompts/ideate_generate.txt",
            ROOT / "ResearchStudio-main/ResearchStudio-Reel/skills/paper2poster/references/content_patterns.md",
        ]
        forbidden = ("CIFAR", "VCC25", "case01_cv", "guide_id", "target_gene")
        for source in prompt_sources:
            text = source.read_text(encoding="utf-8")
            for term in forbidden:
                self.assertNotIn(term, text, f"{source} contains task-specific term {term}")

    def test_researchstudio_context_is_configuration_driven(self) -> None:
        source = (ROOT / "ResearchStudio-main/scripts/ideaspark_boyue_author.py").read_text(
            encoding="utf-8"
        )
        for variable in (
            "IDEASPARK_TASK_CONTEXT",
            "IDEASPARK_RESOURCE_CONTEXT",
            "IDEASPARK_CONSTRAINTS",
        ):
            self.assertIn(variable, source)

    def test_relative_allowed_paths_resolve_under_configured_root(self) -> None:
        module_path = ROOT / "controller_bash/scripts/validate_suggestion.py"
        spec = importlib.util.spec_from_file_location("validate_suggestion", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        allowed = Path("/tmp/ar-contract/implementation")
        resolved = module.resolve_allowed_path("model/train.py", Path("/tmp/ar-contract"), allowed)
        self.assertEqual(resolved, allowed / "model/train.py")

    def test_simulation_never_synthesizes_scientific_metrics(self) -> None:
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metric = module.simulated_metric_for(
            {"name": "arbitrary_domain_trial", "command": "run --output {output}"},
            0,
            Path("/tmp/ar-contract/result.json"),
        )
        self.assertIsNone(metric["primary_metric"])
        self.assertEqual(metric["simulation_basis"], "controller_plumbing_only")
        self.assertNotIn("score", metric)

    def test_structured_metric_envelope_and_legacy_scalar_are_supported(self) -> None:
        module_path = ROOT / "controller_bash/scripts/summarize_round.py"
        spec = importlib.util.spec_from_file_location("summarize_round", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        structured = {
            "primary_metric": {"name": "quality", "value": 1.25, "direction": "maximize"}
        }
        self.assertEqual(module.extract_primary_metric(structured), ("quality", 1.25, "maximize"))
        self.assertEqual(module.extract_primary_metric({"score": 0.5}), ("score", 0.5, None))

    def test_cifar_and_vcc_use_the_same_task_spec_contract(self) -> None:
        cifar_path = ROOT / "tasks/cifar/task_spec.yaml"
        vcc_path = ROOT / "tasks/vcc25/task_spec.yaml"
        cifar = load_task_spec(cifar_path)
        vcc = load_task_spec(vcc_path)
        self.assertEqual(set(cifar), set(vcc))
        self.assertEqual(cifar["schema_version"], "omni-ar-task/v2")
        self.assertEqual(vcc["schema_version"], "omni-ar-task/v2")
        self.assertNotEqual(cifar["task"]["modality"], vcc["task"]["modality"])

    def test_proposal_is_limited_by_task_search_policy(self) -> None:
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        task = load_task_spec(task_path)
        proposal = {
            "schema_version": "omni-ar-proposal/v1",
            "proposal_id": "vcc25-round-0-contract-test",
            "task_name": "vcc25",
            "round": 0,
            "verdict": "contract test",
            "evidence_gaps": [],
            "idea_variants": [],
            "ablation_plan": [],
            "sweep_trials": [
                {
                    "name": "evaluate_contract",
                    "entrypoint": "evaluate",
                    "cli_overrides": {},
                    "purpose": "check result plumbing",
                    "expected_signal": "none",
                }
            ],
            "codex_tasks": [
                {
                    "name": "edit_model",
                    "task": "small model edit",
                    "allowed_paths": ["crpm"],
                    "acceptance_checks": ["static check"],
                }
            ],
            "priority_order": ["evaluate_contract"],
            "risks": [],
        }
        validate_proposal(proposal, task, task_path)
        proposal["codex_tasks"][0]["allowed_paths"] = ["data"]
        with self.assertRaises(ContractError):
            validate_proposal(proposal, task, task_path)

    def test_new_trial_capability_requires_explicit_task_adapter_edit(self) -> None:
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        task = load_task_spec(task_path)
        implementation = {
            "request_id": "new-capability",
            "hypothesis": "Test a genuinely new task-local mechanism.",
            "change_scope": ["model"],
            "allowed_paths": ["train.py"],
            "required_capabilities": ["new_mechanism"],
            "activation_diagnostics": ["new_mechanism_active"],
            "trial_proposal": {
                "hypothesis": "The new mechanism improves the primary metric.",
                "change_scope": ["model"],
                "parameters": {"new_mechanism": True},
                "expected_effect": {}, "acceptance_criteria": {},
                "resource_request": {"gpu_count": 1, "max_runtime_minutes": 30},
            },
        }
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": "vcc25-new-capability",
            "task_name": "vcc25", "round": 0, "verdict": "contract test",
            "evidence_gaps": [],
            "experiment_proposals": [{
                "hypothesis": "Keep a registered control trial.",
                "change_scope": ["model"], "parameters": {},
                "expected_effect": {}, "acceptance_criteria": {},
                "resource_request": {},
            }],
            "implementation_requests": [implementation], "risks": [],
        }
        with self.assertRaises(ContractError):
            validate_proposal(proposal, task, task_path)
        implementation["allowed_paths"].append("tasks/vcc25/adapter.py")
        validate_proposal(proposal, task, task_path)

    def test_unimplemented_method_and_list_parameters_route_through_adapter_edit(self) -> None:
        task_path = ROOT / "tasks/sst2-small/task_spec.yaml"
        task = load_task_spec(task_path)
        trial = {
            "hypothesis": "A new text model uses configurable cue lists.",
            "change_scope": ["model"],
            "parameters": {
                "method": "new_text_model", "cue_words": ["not", "but"],
            },
            "expected_effect": {}, "acceptance_criteria": {},
            "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
        }
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": "sst2-new-method", "task_name": "sst2_small",
            "round": 0, "verdict": "implementation route", "evidence_gaps": [],
            "experiment_proposals": [trial],
            "implementation_requests": [{
                "request_id": "implement-new-text-model",
                "hypothesis": "Implement the proposed text model.",
                "change_scope": ["model"],
                "allowed_paths": ["model.py", "tasks/sst2-small/adapter.py"],
                "required_capabilities": ["new_text_model"],
                "activation_diagnostics": ["new_text_model_active"],
                "trial_proposal": trial,
            }],
            "risks": [],
        }
        validate_proposal(proposal, task, task_path)

        proposal["implementation_requests"][0]["allowed_paths"] = ["model.py"]
        with self.assertRaises(ContractError):
            validate_proposal(proposal, task, task_path)

    def test_registered_implementation_request_compiles_as_runnable_trial(self) -> None:
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials_registered_impl", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task_path = ROOT / "tasks/mnist-small/task_spec.yaml"
        trial = {
            "hypothesis": "Run the registered image candidate.",
            "change_scope": ["model"],
            "parameters": {"method": "is3_cnn", "lambda_invariance": 0.1},
            "expected_effect": {}, "acceptance_criteria": {},
            "resource_request": {"gpu_count": 0, "max_runtime_minutes": 10},
        }
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "experiment_proposals": [],
            "implementation_requests": [{"trial_proposal": trial}],
        }
        with patch.dict(os.environ, {"TASK_SPEC": str(task_path)}):
            compiled = module.trials_from_suggestion(proposal)
        self.assertEqual(len(compiled), 1)
        self.assertEqual(compiled[0]["cli_overrides"]["method"], "is3_cnn")

    def test_vcc_raw_metrics_normalize_to_shared_result_schema(self) -> None:
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        task = load_task_spec(task_path)
        raw = {
            "status": "ok",
            "seed": 7,
            "runtime_seconds": 2.5,
            "baseline_metrics": {
                "mean_delta_pearson": 0.55,
                "mse": 0.5,
                "delta_mse": 0.35,
                "top_k_delta_overlap": 0.3,
            },
            "metrics": {
                "mean_delta_pearson": 0.61,
                "mse": 0.42,
                "delta_mse": 0.31,
                "top_k_delta_overlap": 0.36,
            },
        }
        result = normalize_result(raw, task, Path("raw.json"), "proposal-1", "trial-1")
        self.assertEqual(result["schema_version"], "omni-ar-result/v2")
        self.assertEqual(result["task"], "vcc25")
        primary = result["metrics"]["mean_delta_pearson"]
        self.assertEqual(primary["role"], "primary")
        self.assertEqual(primary["direction"], "maximize")
        self.assertTrue(primary["beats_baseline"])
        self.assertTrue(result["resource_usage"]["within_budget"])
        self.assertEqual(result["protocol"]["stability"]["status"], "single_seed")
        self.assertEqual(result["protocol"]["stability"]["num_seeds"], 1)
        self.assertIsNone(result["protocol"]["stability"]["std"])
        self.assertEqual(result["features"]["status"], "not_requested")

    def test_vcc_go_feature_provenance_is_complete_and_active(self) -> None:
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        task = load_task_spec(task_path)
        raw = {
            "status": "ok", "method": "pseudobulk_go_hierarchy_prototype_flow_verification",
            "seed": 7, "runtime_seconds": 1.0,
            "metrics": {
                "mean_delta_pearson": 0.61, "mse": 0.42,
                "delta_mse": 0.31, "top_k_delta_overlap": 0.36,
            },
            "baseline_metrics": {
                "mean_delta_pearson": 0.55, "mse": 0.5,
                "delta_mse": 0.35, "top_k_delta_overlap": 0.3,
            },
            "protocol": {
                "name": "activation_verification", "split": "heldout_guide",
                "requested_split": "heldout_guide", "n_train": 10, "n_eval": 4,
            },
            "executed_parameters": {
                "method": "pseudobulk_go_hierarchy_prototype_flow_verification",
            },
            "backend_metadata": {"phenotype_activation": {"candidate_active": True}},
        }
        with patch.dict(os.environ, {"TASK_SPEC": str(task_path)}, clear=False):
            result = normalize_result(
                raw, task, ROOT / "tmp-vcc-raw.json", "proposal", "go-trial",
            )
        feature = result["features"]
        self.assertEqual(feature["status"], "active")
        self.assertEqual(feature["records"][0]["builder"]["id"], "gene_ontology")
        self.assertTrue(feature["records"][0]["enabled"])
        self.assertIn("prepared", feature["records"][0]["input_hashes"])
        self.assertIn("edge_table.csv", feature["records"][0]["feature_file_hashes"])
        self.assertEqual(feature["records"][0]["data_usage_scope"]["n_eval"], 4)

    def test_explicit_raw_feature_selection_overrides_method_binding(self) -> None:
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        task = load_task_spec(task_path)
        raw = {
            "status": "ok", "seed": 7, "runtime_seconds": 1.0,
            "metrics": {
                "mean_delta_pearson": 0.61, "mse": 0.42,
                "delta_mse": 0.31, "top_k_delta_overlap": 0.36,
            },
            "baseline_metrics": {
                "mean_delta_pearson": 0.55, "mse": 0.5,
                "delta_mse": 0.35, "top_k_delta_overlap": 0.3,
            },
            "executed_parameters": {
                "method": "pseudobulk_go_hierarchy_prototype_flow_verification",
                "feature_id": "raw",
            },
            "backend_metadata": {"phenotype_activation": {"candidate_active": True}},
        }
        with patch.dict(os.environ, {"TASK_SPEC": str(task_path)}, clear=False):
            result = normalize_result(
                raw, task, ROOT / "tmp-vcc-raw.json", "proposal", "raw-trial",
            )
        self.assertEqual(result["features"]["status"], "not_requested")
        self.assertEqual(result["features"]["records"], [])

    def test_cifar_constraint_and_budget_metadata_are_domain_agnostic(self) -> None:
        task = load_task_spec(ROOT / "tasks/cifar/task_spec.yaml")
        raw = {
            "status": "ok", "seed": 3, "runtime_seconds": 61,
            "metrics": {"primary_accuracy": 0.8, "mean_cost": 0.7},
            "baseline_metrics": {"primary_accuracy": 0.75, "mean_cost": 0.8},
            "resource_usage": {"gpu_count": 1},
        }
        result = normalize_result(
            raw, task, Path("cifar.json"), "p", "t",
            resource_request={"gpu_count": 1, "max_runtime_minutes": 1},
        )
        self.assertTrue(result["metrics"]["primary_accuracy"]["beats_baseline"])
        self.assertEqual(result["metrics"]["mean_cost"]["role"], "constraint")
        self.assertFalse(result["metrics"]["mean_cost"]["constraint_satisfied"])
        self.assertFalse(result["resource_usage"]["within_budget"])

    def test_generic_selector_reads_roles_directions_budget_and_stability_only(self) -> None:
        module_path = ROOT / "controller_bash/scripts/select_results.py"
        spec = importlib.util.spec_from_file_location("select_results", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task = load_task_spec(ROOT / "tasks/vcc25/task_spec.yaml")

        def candidate(trial: str, value: float, within_budget: bool = True):
            raw = {
                "status": "ok", "seed": 7, "runtime_seconds": 2,
                "metrics": {
                    "mean_delta_pearson": value, "mse": 0.4,
                    "delta_mse": 0.3, "top_k_delta_overlap": 0.35,
                },
                "baseline_metrics": {
                    "mean_delta_pearson": 0.5, "mse": 0.5,
                    "delta_mse": 0.4, "top_k_delta_overlap": 0.3,
                },
            }
            result = normalize_result(raw, task, Path(trial), "p", trial)
            result["resource_usage"]["within_budget"] = within_budget
            return result

        selected = module.select_results([
            candidate("lower", 0.6), candidate("higher", 0.7), candidate("over", 0.9, False)
        ])
        self.assertEqual(selected["winners"]["vcc25"]["trial_id"], "p:higher")
        strict = module.select_results([candidate("single", 0.7)], require_multiseed=True)
        self.assertEqual(strict["winners"], {})
        self.assertIn("multi_seed_stability_required", strict["decisions"][0]["reasons"])

    def test_multiseed_stability_metadata_is_preserved(self) -> None:
        task = load_task_spec(ROOT / "tasks/vcc25/task_spec.yaml")
        raw = {
            "status": "ok",
            "metrics": {
                "mean_delta_pearson": 0.62, "mse": 0.4,
                "delta_mse": 0.3, "top_k_delta_overlap": 0.35,
            },
            "protocol": {
                "seed": None,
                "stability": {
                    "status": "multi_seed", "num_seeds": 3,
                    "mean": 0.62, "std": 0.01, "min": 0.61, "max": 0.63,
                },
            },
        }
        result = normalize_result(raw, task, Path("multi.json"), "p", "multi")
        self.assertEqual(result["protocol"]["stability"]["status"], "multi_seed")
        self.assertEqual(result["protocol"]["stability"]["num_seeds"], 3)

    def test_task_entrypoint_expands_result_path_and_overrides(self) -> None:
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials_entrypoint", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        with patch.dict(os.environ, {"TASK_SPEC": str(task_path), "TRIAL_EXECUTION_MODE": "local"}):
            command = module.command_string_for(
                {"name": "entrypoint_test", "entrypoint": "evaluate", "cli_overrides": {"seed": 11}},
                Path("/tmp/result.json"),
            )
        self.assertIn("/tmp/result.json", command)
        self.assertIn("--seed 11", command)

        with patch.dict(os.environ, {"TASK_SPEC": str(task_path), "TRIAL_EXECUTION_MODE": "local"}):
            trial_command = module.command_string_for(
                {
                    "name": "adapter_trial",
                    "entrypoint": "run_trial",
                    "cli_overrides": {"method": "pseudobulk", "epochs": 250},
                },
                Path("/tmp/trial.json"),
            )
        self.assertIn("tasks.vcc25.adapter run_trial", trial_command)
        self.assertIn("--method pseudobulk", trial_command)
        self.assertIn("--epochs 250", trial_command)

    def test_local_trial_prefers_controller_python_environment(self) -> None:
        controller_python = Path(sys.executable)
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials_python", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {"TRIAL_EXECUTION_MODE": "local"}):
            command = module.command_string_for(
                {"command": "{python_executable} adapter.py --output {result_path}"},
                Path("/tmp/result.json"),
            )
        self.assertTrue(command.startswith(str(controller_python)))

    def test_rjob_log_recovers_adapter_json_result(self) -> None:
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials_recovery", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "trial.rjob.log"
            log.write_text(
                'prefix >> {"@level":"info","@msg":"platform"}\n'
                'worker >> {"status":"ok","metrics":{"quality":0.75},"seed":7}\n',
                encoding="utf-8",
            )
            recovered = module.recover_json_result_from_rjob_log(log)
        self.assertEqual(recovered["metrics"]["quality"], 0.75)

    def test_structured_v2_proposal_compiles_via_task_adapter(self) -> None:
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials_structured", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task_path = ROOT / "tasks/vcc25/task_spec.yaml"
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": "vcc25-round-3-structured",
            "task_name": "vcc25",
            "round": 3,
            "verdict": "test adapter compilation",
            "evidence_gaps": [],
            "experiment_proposals": [{
                "hypothesis": "Guide dropout improves robustness.",
                "change_scope": ["model"],
                "parameters": {"guide_dropout": 0.1},
                "expected_effect": {
                    "mean_delta_pearson": {"direction": "increase", "minimum_change": 0.01}
                },
                "acceptance_criteria": {
                    "mean_delta_pearson": {"operator": ">=", "value": 0.68}
                },
                "resource_request": {
                    "gpu_count": 1, "cpu": 8, "memory_mb": 20000,
                    "max_runtime_minutes": 60
                },
            }],
            "risks": [],
        }
        task = load_task_spec(task_path)
        validate_proposal(proposal, task, task_path)
        with patch.dict(os.environ, {"TASK_SPEC": str(task_path)}):
            trials = module.trials_from_suggestion(proposal)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["entrypoint"], "run_trial")
        self.assertEqual(trials[0]["cli_overrides"]["batch_size"], 256)
        self.assertEqual(trials[0]["cli_overrides"]["epochs"], 250)
        self.assertEqual(trials[0]["cli_overrides"]["max_steps"], 0)
        self.assertEqual(trials[0]["cli_overrides"]["guide_dropout"], 0.1)
        self.assertNotIn("command", proposal["experiment_proposals"][0])

        proposal["experiment_proposals"][0]["command"] = "python train.py"
        with self.assertRaises(ContractError):
            validate_proposal(proposal, task, task_path)
        del proposal["experiment_proposals"][0]["command"]
        proposal["experiment_proposals"][0]["parameters"] = {"unknown_task_flag": 1}
        with self.assertRaises(ContractError):
            validate_proposal(proposal, task, task_path)
        proposal["experiment_proposals"][0]["parameters"] = {"guide_dropout": 0.1}
        proposal["experiment_proposals"][0]["resource_request"]["gpu_count"] = 2
        with self.assertRaises(ContractError):
            validate_proposal(proposal, task, task_path)

    def test_structured_acceptance_criteria_are_evaluated(self) -> None:
        module_path = ROOT / "controller_bash/scripts/run_trials.py"
        spec = importlib.util.spec_from_file_location("run_trials_acceptance", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.json"
            raw.write_text('{"status":"ok","metrics":{"quality":0.75}}', encoding="utf-8")
            result = module.evaluate_acceptance(
                raw, {"quality": {"operator": ">=", "value": 0.7}}
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["actual"], 0.75)

    def test_vcc_official_h1_autonomous_candidate_route_is_admitted_not_fallback(self) -> None:
        task_path = ROOT / "tasks/vcc25/task_spec_official_h1.yaml"
        task = load_task_spec(task_path)
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": "vcc25-autonomous-admission",
            "task_name": "vcc25",
            "round": 0,
            "verdict": "contract test for autonomous candidate admission",
            "evidence_gaps": [],
            "experiment_proposals": [{
                "hypothesis": (
                    "A ResearchStudio/Heuresis-generated novel candidate can "
                    "enter the official H1 route without historical fallback."
                ),
                "change_scope": ["model"],
                "parameters": {
                    "candidate_variant": "autonomous_research_candidate",
                    "seed": 20260907,
                },
                "expected_effect": {
                    "pearson_delta": {"direction": "increase", "minimum_change": 0.0}
                },
                "acceptance_criteria": {
                    "pearson_delta": {"operator": ">=", "value": 0.0}
                },
                "resource_request": {
                    "gpu_count": 1,
                    "cpu": 8,
                    "memory_mb": 20000,
                    "max_runtime_minutes": 180,
                },
            }],
            "implementation_requests": [],
            "risks": [],
        }
        validate_proposal(proposal, task, task_path)

        adapter_path = ROOT / "tasks/vcc25/official_h1_adapter.py"
        spec = importlib.util.spec_from_file_location("official_h1_adapter_contract", adapter_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        adapter = module.VCC25OfficialH1TaskAdapter()
        trial = adapter.proposal_to_trial(
            proposal["experiment_proposals"][0],
            task["adapter"]["trial_defaults"],
            "contract_check",
        )
        self.assertEqual(trial["cli_overrides"]["candidate_variant"], "autonomous_research_candidate")
        self.assertEqual(trial["candidate_provenance"]["candidate_origin"], "autonomous_research")
        self.assertFalse(trial["candidate_provenance"]["historical_candidate_reuse"])

        proposal["experiment_proposals"][0]["parameters"]["candidate_variant"] = "candidate_g_promoted_delta_model"
        with patch.dict(os.environ, {"REQUIRE_AUTONOMOUS_CANDIDATE": "1"}, clear=False):
            with self.assertRaises(ContractError):
                validate_proposal(proposal, task, task_path)


if __name__ == "__main__":
    unittest.main()
