from __future__ import annotations

from pathlib import Path

from omni_ar.loop import ResearchLoop


ROOT = Path(__file__).resolve().parents[2]


def proposal() -> dict:
    return {
        "hypothesis": "A bounded guide-aware smoke remains executable.",
        "change_scope": ["model"],
        "parameters": {"method": "guide_crpm", "epochs": 1, "max_steps": 1},
        "expected_effect": {"mean_delta_pearson": {"direction": "increase", "minimum_change": 0.0}},
        "acceptance_criteria": {"mean_delta_pearson": {"operator": ">=", "value": -1.0}},
        "resource_request": {"gpu_count": 1, "cpu": 8, "memory_mb": 20000, "max_runtime_minutes": 20},
    }


def test_vcc_loading_uses_predeclared_mode_without_paths() -> None:
    result = ResearchLoop(ROOT).dataset_loading("vcc25@1", mode="heldout_guide", inspect_limit=2)
    assert result["status"] == "ok"
    assert result["payload"]["mode_spec"]["split_template"] == "single_cell.paired_guide_holdout"
    assert "path" not in str(result["payload"])
    assert result["payload"]["ready_for_training"] is True


def test_design_code_compiles_but_does_not_execute() -> None:
    identified = {**proposal(), "proposal_id": "identified-smoke"}
    result = ResearchLoop(ROOT).design_code("tasks/vcc25/task_spec.yaml", identified)
    assert result["payload"]["trial"]["cli_overrides"]["max_steps"] == 1
    assert result["payload"]["trial"]["name"] == "identified-smoke"
    assert result["payload"]["execution_authorized"] is False


def test_dataset_tool_uses_same_service_boundary() -> None:
    result = ResearchLoop(ROOT).dataset_tool("vcc25@1", "list_modes")
    assert "heldout_guide" in result["payload"]["result"]["payload"]["usage_modes"]


def test_v1_declared_unmaterialized_mode_is_not_reported_ready() -> None:
    result = ResearchLoop(ROOT).dataset_loading("vcc25@1", mode="heldout_target", inspect_limit=1)
    assert result["payload"]["ready_for_training"] is False


def test_v2_semantic_mode_is_materialized_and_ready() -> None:
    result = ResearchLoop(ROOT).dataset_loading("vcc25@2", mode="heldout_target", inspect_limit=1)
    assert result["payload"]["ready_for_training"] is True
    assert result["payload"]["binding"]["resolved"]["artifacts"][0]["name"] == "heldout_target_512g"


def test_dataset_mode_is_compiled_into_trial() -> None:
    service = ResearchLoop(ROOT)
    target = service.design_code("tasks/vcc25/task_spec.yaml", proposal(), dataset_mode="heldout_target")
    batch = service.design_code("tasks/vcc25/task_spec.yaml", proposal(), dataset_mode="heldout_batch")
    assert target["payload"]["trial"]["cli_overrides"]["split_strategy"] == "artifact"
    assert batch["payload"]["trial"]["cli_overrides"]["split_strategy"] == "artifact"
    assert target["payload"]["trial"]["cli_overrides"]["prepared_artifact"] == "heldout_target_512g"
    assert batch["payload"]["trial"]["cli_overrides"]["prepared_artifact"] == "heldout_batch_512g"
