#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_HINT = Path(__file__).resolve().parents[2]
if str(REPO_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_HINT))

from tasks.adapter_contract import (  # noqa: E402
    AdapterError,
    TaskAdapter,
    adapter_main,
    load_json_or_jsonl,
    repository_root,
    run_command,
)
from datasets.registry import DatasetRegistry  # noqa: E402


class VCC25TaskAdapter(TaskAdapter):
    task_name = "vcc25"
    trial_parameters = frozenset({
        "dataset_ref", "prepared_artifact", "method", "split_strategy", "device", "seed", "epochs",
        "max_steps", "batch_size", "rank", "learning_rate", "weight_decay",
        "shrinkage", "guide_shrinkage", "guide_shrinkage_type", "guide_dropout",
        "heldout_guide_prior_scale", "delta_loss_weight", "loss_type", "huber_delta",
        "target_balanced_sampling", "aggregate_training", "aggregate_count_power",
        "target_sampling_power", "target_sampling_max_weight", "no_batch_calibration",
        "target_batch_interaction", "interaction_shrinkage", "min_controls_per_batch",
        "top_k", "group_metrics", "prototype_rank", "prototype_blend",
        "prototype_iterations", "prototype_target_pseudocount",
        "prototype_batch_pseudocount", "heldout_target_fraction", "train_limit",
        "eval_limit", "use_guide_prior", "heldout_batch_fraction",
        "hierarchy_embedding_dim", "hierarchy_aggregation",
        "hierarchy_alpha_learnable", "hierarchy_shuffle",
        "hierarchy_annotation", "hierarchy_negative_control",
        "hierarchy_dropout", "hierarchy_depth_weight_power", "hierarchy_layer_mode",
        "flow_condition_dim", "flow_hidden_dim", "flow_depth",
        "flow_inference_steps",
    })
    required_keys = {
        "x_train", "x_val", "target_train", "target_val", "batch_train",
        "batch_val", "guide_train", "guide_val", "gene_names", "target_names",
        "guide_names", "batch_names",
    }

    def __init__(self) -> None:
        self.repo = repository_root(__file__)
        self.project = self.repo / "tasks/vcc25"
        self.implementation = self.project / "implementation"
        self.dataset_registry = DatasetRegistry(self.repo / "datasets")

    def _hierarchy_verification(self) -> tuple[bool, str]:
        marker = self.project / "hierarchy_activation_verified.json"
        if not marker.exists():
            return False, "verification marker is missing"
        try:
            verification = json.loads(marker.read_text(encoding="utf-8"))
            expected_hashes = verification.get("implementation_sha256") or {}
            required_implementation = (
                "tasks/vcc25/implementation/train.py",
                "tasks/vcc25/implementation/crpm/model.py",
            )
            verified = (
                verification.get("status") == "verified"
                and verification.get("contract_version")
                == "vcc25.guide_hierarchy_activation.v1"
                and all(
                    expected_hashes.get(relative)
                    == hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()
                    for relative in required_implementation
                )
            )
        except (OSError, json.JSONDecodeError):
            return False, "verification marker is unreadable"
        if verification.get("status") != "verified":
            return False, str(verification.get("rejection_reason") or verification.get("status") or "not verified")
        return (
            (True, "verified implementation hashes and activation contract")
            if verified
            else (False, "verification marker is stale or does not match the implementation")
        )

    def _go_hierarchy_verification(self) -> tuple[bool, str]:
        marker = self.project / "go_hierarchy_activation_verified.json"
        if not marker.exists():
            return False, "GO DAG hierarchy verification marker is missing"
        try:
            verification = json.loads(marker.read_text(encoding="utf-8"))
            expected_hashes = verification.get("implementation_sha256") or {}
            required = (
                "tasks/vcc25/implementation/train.py",
                "tasks/vcc25/implementation/crpm/model.py",
                "datasets/vcc25/annotations/go-bp-20260818/edge_table.csv",
            )
            verified = (
                verification.get("status") == "verified"
                and verification.get("contract_version")
                == "vcc25.go_dag_hierarchy_activation.v1"
                and all(
                    expected_hashes.get(relative)
                    == hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()
                    for relative in required
                )
            )
        except (OSError, json.JSONDecodeError):
            return False, "GO DAG hierarchy verification marker is unreadable"
        if verification.get("status") != "verified":
            return False, str(verification.get("rejection_reason") or verification.get("status") or "not verified")
        return (
            (True, "verified GO DAG implementation, annotation, and activation contract")
            if verified else (False, "GO DAG verification marker is stale")
        )

    def _flow_verification(self, hierarchical: bool) -> tuple[bool, str]:
        marker_name = (
            "go_hierarchy_flow_activation_verified.json"
            if hierarchical else "flow_activation_verified.json"
        )
        contract = (
            "vcc25.go_hierarchy_flow_activation.v1"
            if hierarchical else "vcc25.flow_matching_activation.v1"
        )
        marker = self.project / marker_name
        if not marker.exists():
            return False, f"{contract} verification marker is missing"
        try:
            verification = json.loads(marker.read_text(encoding="utf-8"))
            expected_hashes = verification.get("implementation_sha256") or {}
            required = [
                "tasks/vcc25/implementation/train.py",
                "tasks/vcc25/implementation/crpm/flow.py",
            ]
            if hierarchical:
                required.append("datasets/vcc25/annotations/go-bp-20260818/edge_table.csv")
            verified = (
                verification.get("status") == "verified"
                and verification.get("contract_version") == contract
                and all(
                    expected_hashes.get(relative)
                    == hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()
                    for relative in required
                )
            )
        except (OSError, json.JSONDecodeError):
            return False, f"{contract} verification marker is unreadable"
        if verification.get("status") != "verified":
            return False, str(verification.get("rejection_reason") or verification.get("status") or "not verified")
        return (
            (True, f"verified {contract} implementation and activation contract")
            if verified else (False, f"{contract} verification marker is stale")
        )

    def _prototype_flow_verification(self, hierarchical: bool) -> tuple[bool, str]:
        marker_name = (
            "go_hierarchy_prototype_flow_activation_verified.json"
            if hierarchical else "prototype_flow_activation_verified.json"
        )
        contract = (
            "vcc25.go_hierarchy_prototype_flow_activation.v1"
            if hierarchical else "vcc25.prototype_flow_activation.v1"
        )
        marker = self.project / marker_name
        if not marker.exists():
            return False, f"{contract} verification marker is missing"
        try:
            verification = json.loads(marker.read_text(encoding="utf-8"))
            expected_hashes = verification.get("implementation_sha256") or {}
            required = [
                "tasks/vcc25/implementation/train.py",
                "tasks/vcc25/implementation/crpm/flow.py",
                "tasks/vcc25/implementation/crpm/prototype.py",
            ]
            if hierarchical:
                required.append("datasets/vcc25/annotations/go-bp-20260818/edge_table.csv")
            verified = (
                verification.get("status") == "verified"
                and verification.get("contract_version") == contract
                and all(
                    expected_hashes.get(relative)
                    == hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()
                    for relative in required
                )
            )
        except (OSError, json.JSONDecodeError):
            return False, f"{contract} verification marker is unreadable"
        if verification.get("status") != "verified":
            return False, str(verification.get("rejection_reason") or verification.get("status") or "not verified")
        return (
            (True, f"verified {contract} implementation and activation contract")
            if verified else (False, f"{contract} verification marker is stale")
        )

    def _go_prototype_flow_activation(self) -> tuple[bool, str]:
        """Validate activation evidence separately from scientific eligibility."""
        marker = self.project / "go_hierarchy_prototype_flow_activation_verified.json"
        contract = "vcc25.go_hierarchy_prototype_flow_activation.v1"
        if not marker.exists():
            return False, f"{contract} verification marker is missing"
        try:
            verification = json.loads(marker.read_text(encoding="utf-8"))
            expected_hashes = verification.get("implementation_sha256") or {}
            required = [
                "tasks/vcc25/implementation/train.py",
                "tasks/vcc25/implementation/crpm/flow.py",
                "tasks/vcc25/implementation/crpm/prototype.py",
                "datasets/vcc25/annotations/go-bp-20260818/edge_table.csv",
            ]
            valid = (
                verification.get("activation_verified") is True
                and verification.get("contract_version") == contract
                and all(
                    expected_hashes.get(relative)
                    == hashlib.sha256((self.repo / relative).read_bytes()).hexdigest()
                    for relative in required
                )
            )
        except (OSError, json.JSONDecodeError):
            return False, f"{contract} activation marker is unreadable"
        return (
            (True, "current implementation hashes match activation evidence; performance remains rejected")
            if valid else (False, f"{contract} activation evidence is stale")
        )

    def initialization_capabilities(self, spec: dict[str, Any]) -> dict[str, Any]:
        dataset_ref = str((spec.get("data") or {}).get("dataset_ref", "vcc25@current"))
        modes = self.dataset_registry.load_adapter(dataset_ref).usage_modes()
        baseline_path = self.repo / "Heuresis_PJLAB-boyue/src/heuresis/tasks/vcc25/baseline_scores.yaml"
        baseline_scores = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
        reference_path = self.repo / "regression_anchors/vcc25/optimization_summary.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        protocols = [
            name for name in ("heldout_guide", "heldout_target", "heldout_batch")
            if name in modes and modes[name].get("status") != "declared_not_materialized"
        ]
        hierarchy_verified, hierarchy_verification_reason = self._hierarchy_verification()
        go_hierarchy_verified, go_hierarchy_verification_reason = self._go_hierarchy_verification()
        flow_verified, flow_verification_reason = self._flow_verification(False)
        go_flow_verified, go_flow_verification_reason = self._flow_verification(True)
        prototype_flow_verified, prototype_flow_verification_reason = self._prototype_flow_verification(False)
        go_prototype_flow_verified, go_prototype_flow_verification_reason = self._prototype_flow_verification(True)
        go_prototype_flow_active, go_prototype_flow_active_reason = self._go_prototype_flow_activation()
        shared_hierarchy_parameters = {
            "hierarchy_embedding_dim", "hierarchy_aggregation",
            "hierarchy_alpha_learnable",
        }
        root_hierarchy_parameters = {"hierarchy_shuffle"}
        go_hierarchy_parameters = {
            "hierarchy_annotation", "hierarchy_negative_control",
            "hierarchy_dropout", "hierarchy_depth_weight_power", "hierarchy_layer_mode",
        }
        flow_parameters = {
            "flow_condition_dim", "flow_hidden_dim", "flow_depth",
            "flow_inference_steps",
        }
        exposed_parameters = set(self.trial_parameters)
        if not hierarchy_verified:
            exposed_parameters -= root_hierarchy_parameters
        if not (hierarchy_verified or go_hierarchy_verified or go_flow_verified or go_prototype_flow_verified or go_prototype_flow_active):
            exposed_parameters -= shared_hierarchy_parameters
        if not (go_hierarchy_verified or go_flow_verified or go_prototype_flow_verified or go_prototype_flow_active):
            exposed_parameters -= go_hierarchy_parameters
        if not (flow_verified or go_flow_verified or prototype_flow_verified or go_prototype_flow_verified or go_prototype_flow_active):
            exposed_parameters -= flow_parameters
        return {
            "metrics": spec.get("metrics") or {},
            "protocols": protocols,
            "default_protocol": "heldout_guide",
            "baseline": {
                "name": "global_mean",
                "metric": str(baseline_scores["metric"]),
                "value": float(baseline_scores["baseline"]),
                "protocol": str(baseline_scores["protocol"]),
                "source": str(baseline_path.relative_to(self.repo)),
                "source_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            },
            "reference_results": [{
                "name": "pseudobulk_frequency_shrinkage_multiseed",
                "metric": "mean_delta_pearson",
                "value": float(reference["summary"]["mean_delta_pearson"]["mean"]),
                "protocol": "heldout_guide",
                "role": "historical_best_not_baseline",
                "source": str(reference_path.relative_to(self.repo)),
                "source_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            }],
            "protected_items": [
                "dataset_version", "dataset_payload", "data_split",
                "evaluation_protocol", "metric_definition",
                "phenotype_activation_contract",
                *((spec.get("search") or {}).get("protected_paths") or []),
            ],
            "operation_capabilities": [
                "representation", "model", "objective", "regularization",
                "sampling", "optimization", "inference", "evaluation",
            ],
            "trial_parameters": sorted(exposed_parameters),
            "trial_defaults": dict((spec.get("adapter") or {}).get("trial_defaults") or {}),
            "method_capabilities": [
                "crpm", "guide_crpm", "target_prototype", "pseudobulk",
                *(["pseudobulk_hierarchy"] if hierarchy_verified else []),
                *(["pseudobulk_go_hierarchy"] if go_hierarchy_verified else []),
                *(["pseudobulk_flow_matching"] if flow_verified else []),
                *(["pseudobulk_go_hierarchy_flow"] if go_flow_verified else []),
                *(["pseudobulk_prototype_flow_matching"] if prototype_flow_verified else []),
                *(["pseudobulk_go_hierarchy_prototype_flow"] if go_prototype_flow_verified else []),
                *(["pseudobulk_go_hierarchy_prototype_flow_verification"] if go_prototype_flow_active else []),
            ],
            "method_capability_details": {
                "pseudobulk_hierarchy": {
                    "available": hierarchy_verified,
                    "backend_method": "hierarchy_crpm",
                    "hierarchy": "root_target_guide",
                    "ancestor_aggregation": "weighted_sum",
                    "supports_shuffled_parent_control": True,
                    "activation_contract": "vcc25.guide_hierarchy_activation.v1",
                    "limitations": [
                        "No pathway or gene-program annotations are present in vcc25@2.",
                        "This is a CRPM hierarchy contribution, not the Idea Card's flow-matching transport model.",
                    ],
                },
                "pseudobulk_go_hierarchy": {
                    "available": go_hierarchy_verified,
                    "backend_method": "go_hierarchy_crpm",
                    "hierarchy": "root_go_biological_process_target_guide_dag",
                    "supports_multiple_ancestors": True,
                    "supports_negative_controls": [
                        "shuffle_target_annotations", "degree_preserving_target_swap",
                    ],
                    "activation_contract": "vcc25.go_dag_hierarchy_activation.v1",
                    "limitations": [
                        "GO Biological Process membership is not a curated single-parent pathway tree.",
                        "This method remains CRPM and is not flow matching.",
                    ],
                },
                "pseudobulk_flow_matching": {
                    "available": flow_verified,
                    "backend_method": "flow_matching",
                    "activation_contract": "vcc25.flow_matching_activation.v1",
                    "conditioning": ["target", "guide", "batch"],
                },
                "pseudobulk_go_hierarchy_flow": {
                    "available": go_flow_verified,
                    "backend_method": "go_hierarchy_flow_matching",
                    "activation_contract": "vcc25.go_hierarchy_flow_activation.v1",
                    "conditioning": ["target", "guide", "batch", "go_dag_hierarchy"],
                    "limitations": ["GO Biological Process is a DAG, not a curated pathway tree."],
                },
                "pseudobulk_prototype_flow_matching": {
                    "available": prototype_flow_verified,
                    "backend_method": "prototype_flow_matching",
                    "activation_contract": "vcc25.prototype_flow_activation.v1",
                    "combination": ["target_batch_low_rank_prototype", "conditional_flow_matching"],
                    "conditioning": ["target", "guide", "batch"],
                },
                "pseudobulk_go_hierarchy_prototype_flow": {
                    "available": go_prototype_flow_verified,
                    "backend_method": "go_hierarchy_prototype_flow_matching",
                    "activation_contract": "vcc25.go_hierarchy_prototype_flow_activation.v1",
                    "combination": ["target_batch_low_rank_prototype", "conditional_flow_matching", "go_dag_hierarchy"],
                    "conditioning": ["target", "guide", "batch", "go_dag_hierarchy"],
                },
                "pseudobulk_go_hierarchy_prototype_flow_verification": {
                    "available": go_prototype_flow_active,
                    "role": "activation_verification_only",
                    "scientific_evidence": False,
                    "backend_method": "go_hierarchy_prototype_flow_matching",
                    "activation_contract": "vcc25.go_hierarchy_prototype_flow_activation.v1",
                    "limitations": [
                        "The previous full comparison rejected the GO hierarchy performance effect.",
                        "This route reports activation and metrics but cannot enter the accepted archive.",
                    ],
                },
            },
            "parameter_capabilities": {
                "hierarchy_embedding_dim": {"type": "integer", "minimum": 1},
                "hierarchy_aggregation": {"type": "string", "enum": ["weighted_sum"]},
                "hierarchy_alpha_learnable": {"type": "boolean"},
                "hierarchy_shuffle": {
                    "type": "boolean",
                    "role": "negative_control_only",
                },
                **({
                    "hierarchy_annotation": {"type": "string", "enum": ["guide_hierarchy"]},
                    "hierarchy_negative_control": {
                        "type": "string",
                        "enum": ["none", "shuffle_target_annotations", "degree_preserving_target_swap"],
                    },
                    "hierarchy_dropout": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                    "hierarchy_depth_weight_power": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    "hierarchy_layer_mode": {
                        "type": "string",
                        "enum": ["target_only", "biological_process_only", "full"],
                    },
                } if go_hierarchy_verified or go_flow_verified or go_prototype_flow_verified or go_prototype_flow_active else {}),
                **({
                    "flow_condition_dim": {"type": "integer", "enum": [32, 64, 128]},
                    "flow_hidden_dim": {"type": "integer", "enum": [128, 256, 384]},
                    "flow_depth": {"type": "integer", "enum": [1, 2, 3, 4]},
                    "flow_inference_steps": {"type": "integer", "enum": [4, 8, 12, 16]},
                    "prototype_rank": {"type": "integer", "enum": [8, 16, 32]},
                    "prototype_blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                } if flow_verified or go_flow_verified or prototype_flow_verified or go_prototype_flow_verified or go_prototype_flow_active else {}),
            } if hierarchy_verified or go_hierarchy_verified or flow_verified or go_flow_verified or prototype_flow_verified or go_prototype_flow_verified or go_prototype_flow_active else {},
            "hierarchy_activation_verified": hierarchy_verified,
            "hierarchy_activation_verification_reason": hierarchy_verification_reason,
            "go_hierarchy_activation_verified": go_hierarchy_verified,
            "go_hierarchy_activation_verification_reason": go_hierarchy_verification_reason,
            "flow_activation_verified": flow_verified,
            "flow_activation_verification_reason": flow_verification_reason,
            "go_hierarchy_flow_activation_verified": go_flow_verified,
            "go_hierarchy_flow_activation_verification_reason": go_flow_verification_reason,
            "prototype_flow_activation_verified": prototype_flow_verified,
            "prototype_flow_activation_verification_reason": prototype_flow_verification_reason,
            "go_hierarchy_prototype_flow_activation_verified": go_prototype_flow_verified,
            "go_hierarchy_prototype_flow_activation_verification_reason": go_prototype_flow_verification_reason,
            "go_hierarchy_prototype_flow_verification_only_available": go_prototype_flow_active,
            "go_hierarchy_prototype_flow_verification_only_reason": go_prototype_flow_active_reason,
            "resources": spec.get("resources") or {},
            "dataset_modes": modes,
            "activation_contract": {"required": True, "policy": "fail_closed"},
            "literature_search": {
                "query": "single cell perturbation response prediction guide hierarchy perturbation loss held out guide",
                "queries": [
                    "single cell perturbation prediction guide target hierarchical modeling",
                    "perturbation specific loss gene expression response prediction",
                    "held out guide evaluation single cell perturbation generalization",
                    "low rank residual perturbation modeling batch calibration",
                    "guide aware sampling shrinkage single cell CRISPR response",
                ],
                "task_context": "Single-cell genetic perturbation response prediction using target, guide, batch, and cell-context metadata.",
                "resource_context": "One GPU per trial; require bounded experiments and preserve the configured task budget.",
                "constraints": "Use the registered VCC25 dataset only; preserve held-out-guide evaluation, perturbation-sensitive metrics, baseline comparisons, and phenotype activation diagnostics.",
            },
        }

    def proposal_to_trial(
        self,
        proposal: dict[str, Any],
        defaults: dict[str, Any],
        name: str,
    ) -> dict[str, Any]:
        trial = super().proposal_to_trial(proposal, defaults, name)
        parameters = trial["cli_overrides"]
        hierarchy_keys = {
            "hierarchy_embedding_dim", "hierarchy_aggregation",
            "hierarchy_alpha_learnable", "hierarchy_shuffle",
        }
        go_hierarchy_keys = {
            "hierarchy_annotation", "hierarchy_negative_control",
            "hierarchy_dropout", "hierarchy_depth_weight_power", "hierarchy_layer_mode",
        }
        flow_keys = {
            "flow_condition_dim", "flow_hidden_dim", "flow_depth",
            "flow_inference_steps",
        }
        prototype_keys = {
            "prototype_rank", "prototype_blend", "prototype_iterations",
            "prototype_target_pseudocount", "prototype_batch_pseudocount",
        }
        method = parameters.get("method")
        flow_requested = method in {
            "pseudobulk_flow_matching", "pseudobulk_flow_matching_verification",
            "pseudobulk_prototype_flow_matching",
            "pseudobulk_prototype_flow_matching_verification",
        }
        go_flow_requested = method in {
            "pseudobulk_go_hierarchy_flow",
            "pseudobulk_go_hierarchy_flow_verification",
            "pseudobulk_go_hierarchy_prototype_flow",
            "pseudobulk_go_hierarchy_prototype_flow_verification",
        }
        prototype_flow_requested = method in {
            "pseudobulk_prototype_flow_matching",
            "pseudobulk_prototype_flow_matching_verification",
            "pseudobulk_go_hierarchy_prototype_flow",
            "pseudobulk_go_hierarchy_prototype_flow_verification",
        }
        hierarchy_requested = parameters.get("method") == "pseudobulk_hierarchy"
        go_hierarchy_requested = parameters.get("method") in {
            "pseudobulk_go_hierarchy", "pseudobulk_go_hierarchy_verification",
        } or go_flow_requested
        if hierarchy_requested and not self._hierarchy_verification()[0]:
            raise AdapterError("pseudobulk_hierarchy is unavailable until its current implementation passes activation verification")
        if not hierarchy_requested and hierarchy_keys.intersection(proposal.get("parameters") or {}):
            if not go_hierarchy_requested:
                raise AdapterError("hierarchy parameters require a hierarchy method")
        if parameters.get("method") == "pseudobulk_go_hierarchy" and not self._go_hierarchy_verification()[0]:
            raise AdapterError("pseudobulk_go_hierarchy is unavailable until its current implementation passes activation verification")
        if method == "pseudobulk_flow_matching" and not self._flow_verification(False)[0]:
            raise AdapterError("pseudobulk_flow_matching is unavailable until activation verification passes")
        if method == "pseudobulk_go_hierarchy_flow" and not self._flow_verification(True)[0]:
            raise AdapterError("pseudobulk_go_hierarchy_flow is unavailable until activation verification passes")
        if method == "pseudobulk_prototype_flow_matching" and not self._prototype_flow_verification(False)[0]:
            raise AdapterError("pseudobulk_prototype_flow_matching is unavailable until activation verification passes")
        if method == "pseudobulk_go_hierarchy_prototype_flow" and not self._prototype_flow_verification(True)[0]:
            raise AdapterError("pseudobulk_go_hierarchy_prototype_flow is unavailable until activation verification passes")
        if not go_hierarchy_requested and go_hierarchy_keys.intersection(proposal.get("parameters") or {}):
            raise AdapterError("GO hierarchy parameters require method=pseudobulk_go_hierarchy")
        if hierarchy_requested:
            dimension = parameters.get("hierarchy_embedding_dim", 32)
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise AdapterError("hierarchy_embedding_dim must be a positive integer")
            if parameters.get("hierarchy_aggregation", "weighted_sum") != "weighted_sum":
                raise AdapterError("hierarchy_aggregation currently supports only weighted_sum")
            for key in ("hierarchy_alpha_learnable", "hierarchy_shuffle"):
                if key in parameters and not isinstance(parameters[key], bool):
                    raise AdapterError(f"{key} must be boolean")
        if go_hierarchy_requested:
            if parameters.get("hierarchy_annotation", "guide_hierarchy") != "guide_hierarchy":
                raise AdapterError("hierarchy_annotation currently supports only guide_hierarchy")
            if parameters.get("hierarchy_negative_control", "none") not in {
                "none", "shuffle_target_annotations", "degree_preserving_target_swap",
            }:
                raise AdapterError("unsupported hierarchy_negative_control")
            dropout = parameters.get("hierarchy_dropout", 0.0)
            if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
                raise AdapterError("hierarchy_dropout must be in [0, 1)")
            depth_power = parameters.get("hierarchy_depth_weight_power", 0.5)
            if isinstance(depth_power, bool) or not isinstance(depth_power, (int, float)) or depth_power < 0:
                raise AdapterError("hierarchy_depth_weight_power must be non-negative")
            if parameters.get("hierarchy_layer_mode", "full") not in {
                "target_only", "biological_process_only", "full",
            }:
                raise AdapterError("unsupported hierarchy_layer_mode")
        if not (flow_requested or go_flow_requested) and flow_keys.intersection(proposal.get("parameters") or {}):
            raise AdapterError("flow parameters require a flow-matching method")
        if flow_requested or go_flow_requested:
            if "learning_rate" not in (proposal.get("parameters") or {}):
                parameters["learning_rate"] = 0.001
            for key, default in (
                ("flow_condition_dim", 64), ("flow_hidden_dim", 256),
                ("flow_depth", 2), ("flow_inference_steps", 8),
            ):
                value = parameters.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise AdapterError(f"{key} must be a positive integer")
        if prototype_keys.intersection(proposal.get("parameters") or {}) and not (
            prototype_flow_requested or method == "target_prototype"
        ):
            raise AdapterError("prototype parameters require a prototype-based method")
        if prototype_flow_requested:
            rank = parameters.get("prototype_rank", 16)
            blend = parameters.get("prototype_blend", 1.0)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise AdapterError("prototype_rank must be a non-negative integer")
            if isinstance(blend, bool) or not isinstance(blend, (int, float)) or not 0 <= blend <= 1:
                raise AdapterError("prototype_blend must be in [0, 1]")
        if parameters.get("method") not in {
            "pseudobulk", "pseudobulk_hierarchy", "pseudobulk_go_hierarchy",
            "pseudobulk_go_hierarchy_verification",
            "pseudobulk_flow_matching", "pseudobulk_flow_matching_verification",
            "pseudobulk_go_hierarchy_flow", "pseudobulk_go_hierarchy_flow_verification",
            "pseudobulk_prototype_flow_matching", "pseudobulk_prototype_flow_matching_verification",
            "pseudobulk_go_hierarchy_prototype_flow", "pseudobulk_go_hierarchy_prototype_flow_verification",
            "pseudobulk_prototype_flow_matching", "pseudobulk_prototype_flow_matching_verification",
            "pseudobulk_go_hierarchy_prototype_flow", "pseudobulk_go_hierarchy_prototype_flow_verification",
        } and not parameters.get("aggregate_training"):
            parameters["aggregate_count_power"] = 0.0
        return trial

    def dataset_paths(self, request: dict[str, Any]) -> tuple[Path, Path]:
        dataset_ref = str(request.get("dataset_ref", "vcc25@current"))
        try:
            manifest = self.dataset_registry.manifest(dataset_ref)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise AdapterError(f"unavailable VCC25 dataset ref: {dataset_ref}") from exc
        if str((manifest.get("dataset") or {}).get("id")) != "vcc25":
            raise AdapterError("VCC25 task requires a vcc25 Dataset Pack")
        prepared_artifact = str(request.get("prepared_artifact", "full_512g"))
        try:
            source_binding = self.dataset_registry.binding(dataset_ref, artifact="raw_h5")
            prepared_binding = self.dataset_registry.binding(dataset_ref, artifact=prepared_artifact)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise AdapterError(
                f"unavailable VCC25 artifact {prepared_artifact!r} in {dataset_ref}"
            ) from exc
        source = Path(source_binding["resolved"]["artifacts"][0]["path"])
        prepared = Path(prepared_binding["resolved"]["artifacts"][0]["path"])
        if prepared.suffix != ".npz":
            raise AdapterError("prepared artifact must resolve to NPZ")
        return source, prepared

    def _lineage(self, source: Path, prepared: Path) -> dict[str, Any]:
        if not prepared.exists():
            raise AdapterError(f"prepared dataset missing: {prepared}")
        if source == prepared:
            if not source.exists():
                raise AdapterError(f"source dataset missing: {source}")
            return {"source": str(source), "prepared": str(prepared)}
        metadata_path = prepared.with_suffix(".json")
        if not metadata_path.exists():
            raise AdapterError(f"prepared dataset lineage metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if Path(str(metadata.get("source", ""))).name != source.name:
            raise AdapterError("prepared dataset metadata does not reference competition_train.h5")
        expected_size = metadata.get("source_size_bytes")
        source_available = source.exists()
        if source_available and expected_size is not None and int(expected_size) != source.stat().st_size:
            raise AdapterError("competition_train.h5 size does not match prepared-data lineage")
        expected_prepared_hash = str(metadata.get("prepared_sha256", ""))
        if not expected_prepared_hash:
            raise AdapterError("prepared dataset lineage has no SHA-256")
        digest = hashlib.sha256()
        with prepared.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_prepared_hash = digest.hexdigest()
        if actual_prepared_hash != expected_prepared_hash:
            raise AdapterError("prepared dataset SHA-256 does not match H5 lineage metadata")
        return {
            "source": str(source),
            "source_available_in_worker": source_available,
            "source_sha256": metadata.get("source_sha256"),
            "source_size_bytes": source.stat().st_size if source_available else expected_size,
            "prepared": str(prepared),
            "prepared_sha256": actual_prepared_hash,
            "prepared_sha256_verified": True,
            "source_cells": metadata.get("source_cells"),
            "source_genes": metadata.get("source_genes"),
            "selected_genes": metadata.get("genes"),
        }

    def prepare_data(self, request: dict[str, Any]) -> dict[str, Any]:
        source, prepared = self.dataset_paths(request)
        lineage = self._lineage(source, prepared)
        return {
            "status": "ok",
            "dataset": str(source),
            "prepared_dataset": str(prepared),
            "prepared": True,
            "lineage": lineage,
            "note": "Adapter uses the declared H5 source through a verified task-owned training artifact.",
        }

    def validate_data(self, request: dict[str, Any]) -> dict[str, Any]:
        source, prepared = self.dataset_paths(request)
        lineage = self._lineage(source, prepared)
        with np.load(prepared, allow_pickle=False) as data:
            keys = sorted(data.files)
            missing = sorted(self.required_keys.difference(keys))
            if missing:
                raise AdapterError(f"VCC25 NPZ is missing required keys: {missing}")
            expression_key = "x_train"
            shape = list(data[expression_key].shape)
            if len(shape) != 2 or min(shape) < 1:
                raise AdapterError(f"invalid expression shape: {shape}")
            if data["x_val"].ndim != 2 or data["x_val"].shape[1] != shape[1]:
                raise AdapterError("train/validation expression widths differ")
            for prefix in ("target", "batch", "guide"):
                if len(data[f"{prefix}_train"]) != shape[0]:
                    raise AdapterError(f"{prefix}_train length does not match x_train")
        return {
            "status": "ok",
            "dataset": str(source),
            "prepared_dataset": str(prepared),
            "format": "npz",
            "keys": keys,
            "expression_key": expression_key,
            "n_rows": shape[0],
            "n_features": shape[1],
            "lineage": lineage,
        }

    def _run(self, request: dict[str, Any], baseline: bool) -> dict[str, Any]:
        output = Path(request["output"])
        backend_output = output.with_suffix(".backend.json")
        source_dataset, prepared_dataset = self.dataset_paths(request)
        lineage = self._lineage(source_dataset, prepared_dataset)
        method = str(request.get("method", "global_mean" if baseline else "crpm"))
        pseudobulk = method in {
            "pseudobulk", "pseudobulk_hierarchy", "pseudobulk_go_hierarchy",
            "pseudobulk_go_hierarchy_verification",
            "pseudobulk_flow_matching", "pseudobulk_flow_matching_verification",
            "pseudobulk_go_hierarchy_flow", "pseudobulk_go_hierarchy_flow_verification",
            "pseudobulk_prototype_flow_matching", "pseudobulk_prototype_flow_matching_verification",
            "pseudobulk_go_hierarchy_prototype_flow", "pseudobulk_go_hierarchy_prototype_flow_verification",
        }
        hierarchy_method = method == "pseudobulk_hierarchy"
        go_hierarchy_method = method == "pseudobulk_go_hierarchy"
        go_hierarchy_verification_method = method == "pseudobulk_go_hierarchy_verification"
        flow_method = method == "pseudobulk_flow_matching"
        flow_verification_method = method == "pseudobulk_flow_matching_verification"
        go_flow_method = method == "pseudobulk_go_hierarchy_flow"
        go_flow_verification_method = method == "pseudobulk_go_hierarchy_flow_verification"
        prototype_flow_method = method == "pseudobulk_prototype_flow_matching"
        prototype_flow_verification_method = method == "pseudobulk_prototype_flow_matching_verification"
        go_prototype_flow_method = method == "pseudobulk_go_hierarchy_prototype_flow"
        go_prototype_flow_verification_method = method == "pseudobulk_go_hierarchy_prototype_flow_verification"
        prototype_flow_execution = prototype_flow_method or prototype_flow_verification_method
        go_prototype_flow_execution = go_prototype_flow_method or go_prototype_flow_verification_method
        flow_execution = flow_method or flow_verification_method or prototype_flow_execution
        go_flow_execution = go_flow_method or go_flow_verification_method or go_prototype_flow_execution
        go_hierarchy_execution = (
            go_hierarchy_method or go_hierarchy_verification_method or go_flow_execution
        )
        if hierarchy_method and not self._hierarchy_verification()[0]:
            raise AdapterError("pseudobulk_hierarchy execution is blocked because activation verification is missing or stale")
        if go_hierarchy_method and not self._go_hierarchy_verification()[0]:
            raise AdapterError("pseudobulk_go_hierarchy execution is blocked because activation verification is missing or stale")
        if flow_method and not self._flow_verification(False)[0]:
            raise AdapterError("pseudobulk_flow_matching execution is blocked because activation verification is missing or stale")
        if go_flow_method and not self._flow_verification(True)[0]:
            raise AdapterError("pseudobulk_go_hierarchy_flow execution is blocked because activation verification is missing or stale")
        if prototype_flow_method and not self._prototype_flow_verification(False)[0]:
            raise AdapterError("pseudobulk_prototype_flow_matching execution is blocked because activation verification is missing or stale")
        if go_prototype_flow_method and not self._prototype_flow_verification(True)[0]:
            raise AdapterError("pseudobulk_go_hierarchy_prototype_flow execution is blocked because activation verification is missing or stale")
        backend_method = (
            "go_hierarchy_prototype_flow_matching" if go_prototype_flow_execution
            else "prototype_flow_matching" if prototype_flow_execution
            else "go_hierarchy_flow_matching" if go_flow_execution
            else "flow_matching" if flow_execution
            else "go_hierarchy_crpm" if go_hierarchy_execution
            else "hierarchy_crpm" if hierarchy_method
            else "guide_crpm" if pseudobulk else method
        )
        allowed = {"global_mean", "batch_control_mean"} if baseline else {
            "crpm", "guide_crpm", "target_prototype", "pseudobulk",
            "pseudobulk_hierarchy", "pseudobulk_go_hierarchy",
            "pseudobulk_go_hierarchy_verification",
            "pseudobulk_flow_matching", "pseudobulk_flow_matching_verification",
            "pseudobulk_go_hierarchy_flow", "pseudobulk_go_hierarchy_flow_verification",
            "pseudobulk_prototype_flow_matching", "pseudobulk_prototype_flow_matching_verification",
            "pseudobulk_go_hierarchy_prototype_flow", "pseudobulk_go_hierarchy_prototype_flow_verification",
        }
        if method not in allowed:
            raise AdapterError(f"unsupported {'baseline' if baseline else 'trial'} method: {method}")
        command = [
            sys.executable,
            "train.py",
            "--data-root", str(prepared_dataset.parent),
            "--dataset", prepared_dataset.name,
            "--method", backend_method,
            "--split-strategy", str(request.get("split_strategy", "heldout_guide")),
            "--device", str(request.get("device", "cuda")),
            "--seed", str(int(request.get("seed", 20250805))),
            "--epochs", str(int(request.get("epochs", 0 if baseline else 1))),
            # Full trials must run their declared epoch budget. A step cap is a
            # smoke-test control and is applied only when explicitly requested.
            "--max-steps", str(int(request.get("max_steps", 0))),
            "--batch-size", str(int(request.get("batch_size", 128))),
            "--rank", str(int(request.get("rank", 16))),
            "--metrics-out", str(backend_output),
        ]
        hierarchy_graph_path = None
        if go_hierarchy_execution:
            hierarchy_name = str(request.get("hierarchy_annotation", "guide_hierarchy"))
            dataset_adapter = self.dataset_registry.load_adapter(
                str(request.get("dataset_ref", "vcc25@current"))
            )
            validation = dataset_adapter.validate_hierarchy(
                hierarchy=hierarchy_name, deep=True
            )
            if validation.get("valid") is not True:
                raise AdapterError("registered GO hierarchy failed Dataset Adapter validation")
            graph = dict(dataset_adapter.load_hierarchy(
                hierarchy=hierarchy_name, include_graph=True
            ))
            graph_payload = {
                "schema_version": "omni-ar-materialized-hierarchy/v1",
                "hierarchy": hierarchy_name,
                "root": graph["descriptor"]["root"],
                "nodes": graph["nodes"],
                "edges": [list(edge) for edge in graph["edges"]],
                "metadata": {
                    "hierarchy_id": graph["descriptor"]["hierarchy_id"],
                    "negative_control": "none",
                    "unchanged_annotation_fraction": 1.0,
                },
            }
            negative_control = str(request.get("hierarchy_negative_control", "none"))
            if negative_control != "none":
                negative = dict(dataset_adapter.build_negative_control(
                    hierarchy=hierarchy_name,
                    strategy=negative_control,
                    seed=int(request.get("seed", 20250805)),
                    include_edges=True,
                ))
                graph_payload["edges"] = [list(edge) for edge in negative.pop("edges")]
                graph_payload["metadata"].update({
                    "negative_control": negative_control,
                    **negative,
                })
            hierarchy_graph_path = output.with_suffix(".hierarchy_input.json")
            hierarchy_graph_path.parent.mkdir(parents=True, exist_ok=True)
            hierarchy_graph_path.write_text(
                json.dumps(graph_payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            command.extend(["--hierarchy-graph", str(hierarchy_graph_path)])
        scalar_flags = {
            "heldout_target_fraction": "--heldout-target-fraction",
            "heldout_batch_fraction": "--heldout-batch-fraction",
            "train_limit": "--train-limit",
            "eval_limit": "--eval-limit",
            "learning_rate": "--learning-rate",
            "weight_decay": "--weight-decay",
            "shrinkage": "--shrinkage",
            "guide_shrinkage": "--guide-shrinkage",
            "guide_shrinkage_type": "--guide-shrinkage-type",
            "guide_dropout": "--guide-dropout",
            "heldout_guide_prior_scale": "--heldout-guide-prior-scale",
            "delta_loss_weight": "--delta-loss-weight",
            "loss_type": "--loss-type",
            "huber_delta": "--huber-delta",
            "target_sampling_power": "--target-sampling-power",
            "target_sampling_max_weight": "--target-sampling-max-weight",
            "aggregate_count_power": "--aggregate-count-power",
            "interaction_shrinkage": "--interaction-shrinkage",
            "min_controls_per_batch": "--min-controls-per-batch",
            "top_k": "--top-k",
            "prototype_rank": "--prototype-rank",
            "prototype_blend": "--prototype-blend",
            "prototype_iterations": "--prototype-iterations",
            "prototype_target_pseudocount": "--prototype-target-pseudocount",
            "prototype_batch_pseudocount": "--prototype-batch-pseudocount",
            "hierarchy_embedding_dim": "--hierarchy-embedding-dim",
            "hierarchy_aggregation": "--hierarchy-aggregation",
            "hierarchy_dropout": "--hierarchy-dropout",
            "hierarchy_depth_weight_power": "--hierarchy-depth-weight-power",
            "hierarchy_layer_mode": "--hierarchy-layer-mode",
            "flow_condition_dim": "--flow-condition-dim",
            "flow_hidden_dim": "--flow-hidden-dim",
            "flow_depth": "--flow-depth",
            "flow_inference_steps": "--flow-inference-steps",
        }
        for key, flag in scalar_flags.items():
            if key in request:
                command.extend([flag, str(request[key])])
        boolean_flags = {
            "group_metrics": "--group-metrics",
            "target_balanced_sampling": "--target-balanced-sampling",
            "target_batch_interaction": "--target-batch-interaction",
            "aggregate_training": "--aggregate-training",
            "no_batch_calibration": "--no-batch-calibration",
            "use_guide_prior": "--use-guide-prior",
            "hierarchy_alpha_learnable": "--hierarchy-alpha-learnable",
            "hierarchy_shuffle": "--hierarchy-shuffle",
        }
        if pseudobulk:
            request = {**request, "aggregate_training": True}
            if "aggregate_count_power" not in request:
                command.extend(["--aggregate-count-power", "1.0"])
        for key, flag in boolean_flags.items():
            if request.get(key):
                command.append(flag)
        run_command(command, self.implementation, output.with_suffix(".backend"))
        raw = load_json_or_jsonl(backend_output)
        metrics = raw.get("metrics")
        if raw.get("status") != "ok" or not isinstance(metrics, dict):
            raise AdapterError("VCC25 backend did not emit status=ok with a metrics object")
        activation = raw.get("phenotype_activation")
        if hierarchy_method or go_hierarchy_execution or flow_execution:
            if not isinstance(activation, dict):
                raise AdapterError("hierarchy phenotype diagnostics missing")
            expected_contract = (
                "vcc25.go_hierarchy_prototype_flow_activation.v1" if go_prototype_flow_execution
                else "vcc25.prototype_flow_activation.v1" if prototype_flow_execution
                else "vcc25.go_hierarchy_flow_activation.v1" if go_flow_execution
                else "vcc25.flow_matching_activation.v1" if flow_execution
                else "vcc25.go_dag_hierarchy_activation.v1" if go_hierarchy_execution
                else "vcc25.guide_hierarchy_activation.v1"
            )
            if activation.get("contract_version") != expected_contract:
                raise AdapterError("hierarchy phenotype contract version mismatch")
            if activation.get("candidate_active") is not True:
                raise AdapterError("hierarchy phenotype is not active")
            if not (flow_execution or go_flow_execution) and activation.get("hierarchy_aggregation") != request.get("hierarchy_aggregation", "weighted_sum"):
                raise AdapterError("hierarchy aggregation diagnostic mismatch")
            if not go_hierarchy_execution and bool(activation.get("hierarchy_shuffle")) != bool(request.get("hierarchy_shuffle", False)):
                raise AdapterError("hierarchy shuffle diagnostic mismatch")
            mechanisms = activation.get("mechanisms")
            if not isinstance(mechanisms, list) or len(mechanisms) != 1:
                raise AdapterError("hierarchy mechanism diagnostics missing")
            diagnostics = mechanisms[0].get("diagnostics") if isinstance(mechanisms[0], dict) else None
            required_positive = (
                (
                    "activation_events", "affected_rows", "transformed_feature_count",
                    "gradient_norm_max", "parameter_update_l2", "velocity_abs_sum",
                    "trajectory_displacement_abs_sum",
                    *((
                        "hierarchy_parameter_update_l2",
                        "hierarchy_contribution_abs_sum",
                        "hierarchy_embedding_norm",
                    ) if go_flow_execution else ()),
                )
                if flow_execution or go_flow_execution else (
                    "activation_events", "affected_rows", "transformed_feature_count",
                    "gradient_norm_max", "parameter_update_l2", "contribution_abs_sum",
                    "embedding_norm",
                )
            )
            if not isinstance(diagnostics, dict) or any(
                not isinstance(diagnostics.get(key), (int, float))
                or isinstance(diagnostics.get(key), bool)
                or not math.isfinite(float(diagnostics[key]))
                or float(diagnostics[key]) <= 0
                for key in required_positive
            ):
                raise AdapterError("hierarchy phenotype diagnostics are missing or non-positive")
            if prototype_flow_execution or go_prototype_flow_execution:
                if activation.get("prototype_residual_active") is not True:
                    raise AdapterError("prototype residual base is not active")
                if not isinstance(activation.get("prototype_base_contribution_abs_sum"), (int, float)) or float(
                    activation["prototype_base_contribution_abs_sum"]
                ) <= 0:
                    raise AdapterError("prototype residual contribution diagnostic is missing or non-positive")
                if not isinstance(activation.get("prototype_rank_effective"), int) or activation[
                    "prototype_rank_effective"
                ] <= 0:
                    raise AdapterError("prototype residual rank diagnostic is missing or non-positive")
            if go_hierarchy_execution:
                expected_control = str(request.get("hierarchy_negative_control", "none"))
                if activation.get("negative_control") != expected_control:
                    raise AdapterError("GO hierarchy negative-control diagnostic mismatch")
                unchanged = activation.get("unchanged_annotation_fraction")
                if not isinstance(unchanged, (int, float)):
                    raise AdapterError("GO hierarchy annotation-match diagnostic missing")
                if expected_control == "none" and float(unchanged) != 1.0:
                    raise AdapterError("real GO hierarchy annotations changed unexpectedly")
                if expected_control != "none" and float(unchanged) >= 1.0:
                    raise AdapterError("GO hierarchy negative control did not change annotations")
                layer_nonzero = activation.get("layer_membership_nonzero")
                layer_mode = str(request.get("hierarchy_layer_mode", "full"))
                if activation.get("hierarchy_layer_mode") != layer_mode:
                    raise AdapterError("GO hierarchy layer-mode diagnostic mismatch")
                expected_layers = {
                    "target_only": {"root", "target"},
                    "biological_process_only": {"root", "biological_process"},
                    "full": {"root", "biological_process", "target"},
                }[layer_mode]
                if not isinstance(layer_nonzero, dict) or any(
                    not isinstance(layer_nonzero.get(layer), int)
                    or (layer_nonzero[layer] <= 0 if layer in expected_layers else layer_nonzero[layer] != 0)
                    for layer in ("root", "biological_process", "target")
                ):
                    raise AdapterError("GO hierarchy layer activation diagnostics missing")
                if hierarchy_graph_path is None or activation.get("graph_sha256") != hashlib.sha256(
                    hierarchy_graph_path.read_bytes()
                ).hexdigest():
                    raise AdapterError("GO hierarchy graph hash diagnostic mismatch")
            elif hierarchy_method:
                match_rate = activation.get("guide_parent_match_rate")
                if not isinstance(match_rate, (int, float)):
                    raise AdapterError("hierarchy parent match diagnostic missing")
                if request.get("hierarchy_shuffle", False) and float(match_rate) >= 1.0:
                    raise AdapterError("shuffled hierarchy did not change any parent assignment")
                if not request.get("hierarchy_shuffle", False) and float(match_rate) != 1.0:
                    raise AdapterError("real hierarchy parent mapping changed unexpectedly")
        actual_split = str(raw.get("split_strategy") or request.get("split_strategy"))
        evidence_scope = str(raw.get("evidence_scope") or "task_adapter_evaluation")
        verification_execution = any((
            go_hierarchy_verification_method, flow_verification_method,
            go_flow_verification_method, prototype_flow_verification_method,
            go_prototype_flow_verification_method,
        ))
        if verification_execution:
            evidence_scope = "activation_verification"
        elif actual_split in {"heldout_target", "heldout_guide", "heldout_batch"} or "heldout_guide" in actual_split:
            evidence_scope = "scientific_candidate"
        return {
            "status": "ok",
            "backend": "vcc25_crpm",
            "method": method,
            "dataset_ref": request.get("dataset_ref", "vcc25@current"),
            "dataset": source_dataset.name,
            "prepared_dataset": prepared_dataset.name,
            "data_lineage": lineage,
            "seed": raw.get("seed"),
            "runtime_seconds": raw.get("runtime_seconds"),
            "metrics": metrics,
            "baseline_metrics": raw.get("global_mean_baseline_metrics") or {},
            "resource_usage": {
                "runtime_seconds": raw.get("runtime_seconds"),
                "gpu_count": 1 if str(request.get("device", "cuda")).startswith("cuda") else 0,
                "epochs_requested": raw.get("epochs_requested"),
                "epochs_completed": raw.get("epochs_completed"),
                "optimizer_steps": raw.get("optimizer_steps"),
                "batch_size": int(request.get("batch_size", 128)),
            },
            "protocol": {
                "name": evidence_scope,
                "split": actual_split,
                "requested_split": request.get("split_strategy"),
                "seed": raw.get("seed"),
                "n_train": raw.get("n_train"),
                "n_eval": raw.get("n_eval"),
                "dataset_ref": request.get("dataset_ref", "vcc25@current"),
                "prepared_artifact": request.get("prepared_artifact", "full_512g"),
            },
            "artifacts": {
                "source_dataset": str(source_dataset),
                "prepared_dataset": str(prepared_dataset),
                "backend_result": str(backend_output),
                **({"hierarchy_input": str(hierarchy_graph_path)} if hierarchy_graph_path else {}),
            },
            "backend_metadata": {
                "split_strategy": raw.get("split_strategy"),
                "evidence_scope": raw.get("evidence_scope"),
                "n_train": raw.get("n_train"),
                "n_eval": raw.get("n_eval"),
                "phenotype_activation": activation,
            },
        }

    def run_baseline(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._run(request, baseline=True)

    def run_trial(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._run(request, baseline=False)

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_path = Path(str(request.get("raw_result", ""))).resolve()
        if not str(request.get("raw_result", "")):
            raise AdapterError("evaluate requires --raw-result")
        raw = load_json_or_jsonl(raw_path)
        metrics = raw.get("metrics")
        if not isinstance(metrics, dict):
            raise AdapterError("raw result has no metrics object")
        return {
            "status": "ok",
            "raw_result": str(raw_path),
            "metrics": metrics,
            "primary_metric": {
                "name": "mean_delta_pearson",
                "value": metrics.get("mean_delta_pearson"),
                "direction": "maximize",
            },
        }

    def summarize_results(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_paths = str(request.get("results", "")).split(",") if request.get("results") else []
        if request.get("results_dir"):
            raw_paths.extend(str(path) for path in Path(str(request["results_dir"])).glob("*.json"))
        rows = []
        for raw_path in raw_paths:
            if not raw_path:
                continue
            row = load_json_or_jsonl(Path(raw_path))
            metrics = row.get("metrics") or {}
            value = metrics.get("mean_delta_pearson")
            if isinstance(value, (int, float)):
                rows.append({
                    "path": raw_path,
                    "method": row.get("method"),
                    "mean_delta_pearson": value,
                    "mse": metrics.get("mse"),
                })
        rows.sort(key=lambda row: row["mean_delta_pearson"], reverse=True)
        return {"status": "ok", "primary_metric": "mean_delta_pearson", "rows": rows, "best": rows[0] if rows else None}


if __name__ == "__main__":
    raise SystemExit(adapter_main(VCC25TaskAdapter()))
