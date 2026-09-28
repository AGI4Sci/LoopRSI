from __future__ import annotations

import json
from pathlib import Path

import yaml

from omni_ar.initialization import RoughIdeaEngine, dry_run_pipeline
from omni_ar.initialization.llm_questions import generic_capabilities, option_catalog, validate_question_plan


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/vcc25/task_spec.yaml"


def answers() -> dict:
    return {
        "research_goal": "explore",
        "priorities": ["metric_quality", "stability"],
        "exploration_level": "balanced",
        "requested_scopes": ["representation", "objective", "sampling", "inference"],
        "evaluation_protocol": "heldout_guide",
        "success_criteria": ["outperform_baseline", "minimum_improvement", "budget_compliance"],
        "minimum_improvement": 0.005,
        "resource_profile": "standard",
        "rough_idea": "Combine perturbation loss and guide hierarchy.",
        "forbidden_directions": "No external data.",
        "requested_mutations": [],
    }


def test_vcc25_capabilities_are_normalized_into_confirmed_rough_idea() -> None:
    rough = RoughIdeaEngine(ROOT).build(TASK, answers(), initialization_id="vcc25-test", confirmed=True)
    assert rough["confirmation"]["status"] == "confirmed"
    assert rough["dataset"]["ref"] == "vcc25@2"
    assert rough["evaluation"]["primary_metric"]["name"] == "mean_delta_pearson"
    assert rough["evaluation"]["protocol"] == "heldout_guide"
    assert rough["evaluation"]["baseline"]["name"] == "global_mean"
    assert rough["constraints"]["require_phenotype_activation"] is True
    assert "data_split" in rough["constraints"]["protected_items"]
    assert rough["schema_version"] == "omni-ar-rough-idea/v2"
    assert rough["execution_policy"]["submit_rjob"] is False
    assert rough["execution_policy"]["auto_retry"] is True


def test_protected_mutation_prevents_confirmation() -> None:
    value = answers()
    value["requested_mutations"] = ["data_split"]
    rough = RoughIdeaEngine(ROOT).build(TASK, value, initialization_id="vcc25-conflict", confirmed=True)
    assert rough["confirmation"]["status"] == "pending"
    assert rough["constraints"]["conflicts"] == ["requested mutation is protected: data_split"]


def test_vcc25_dry_run_is_traceable_and_never_executes(tmp_path: Path) -> None:
    engine = RoughIdeaEngine(ROOT)
    value = answers()
    rough = engine.build(TASK, value, initialization_id="vcc25-dry-run-test", confirmed=True)
    bundle = engine.write_bundle(tmp_path / "initialization", value, rough)
    result = dry_run_pipeline(ROOT, bundle["rough_idea"], tmp_path / "pipeline")
    trace = json.loads(result["trace"].read_text(encoding="utf-8"))
    proposal = json.loads(result["artifacts"]["heuresis_proposal"].read_text(encoding="utf-8"))
    compiled = json.loads(result["artifacts"]["compiled_trial"].read_text(encoding="utf-8"))
    assert trace["status"] == "ok"
    assert trace["checks"]["training_started"] is False
    assert trace["checks"]["rjob_submitted"] is False
    assert proposal["schema_version"] == "omni-ar-proposal/v2"
    assert compiled["payload"]["execution_authorized"] is False
    assert compiled["payload"]["trial"]["cli_overrides"]["dataset_ref"] == "vcc25@2"
    assert yaml.safe_load(bundle["rough_idea"].read_text())["confirmation"]["status"] == "confirmed"


def generated_plan() -> dict:
    return {
        "schema_version": "omni-ar-generated-questions/v1",
        "summary": "明确研究目的、改动范围和评估方式。",
        "questions": [
            {
                "id": "research_goal", "type": "single_choice",
                "title": "这次更偏向提升还是探索？", "why": "明确研究目的。",
                "options": [
                    {"id": "improve", "label": "提升"},
                    {"id": "explore", "label": "探索"},
                ],
                "default": "improve",
            },
            {
                "id": "requested_scopes", "type": "multiple_choice",
                "title": "允许系统改动哪些层面？", "why": "划定搜索边界。",
                "options": [
                    {"id": "model", "label": "模型"},
                    {"id": "objective", "label": "目标函数"},
                ],
                "default": ["model", "objective"],
            },
            {
                "id": "evaluation_protocol", "type": "single_choice",
                "title": "使用哪种验证协议？", "why": "保证结果可比较。",
                "options": [
                    {"id": "heldout_guide", "label": "协议一"},
                    {"id": "heldout_target", "label": "协议二"},
                ],
                "default": "heldout_guide",
            },
            {
                "id": "priorities", "type": "multiple_choice",
                "title": "最看重什么？", "why": "确定优化侧重点。",
                "options": [{"id": "metric_quality", "label": "效果"}],
                "default": ["metric_quality"],
            },
            {
                "id": "research_details", "type": "free_text",
                "title": "还有什么具体预期？", "why": "补足想法细节。",
                "options": [], "default": "",
            },
        ],
    }


def test_llm_questions_start_from_user_idea_and_remain_contract_bounded() -> None:
    engine = RoughIdeaEngine(ROOT)
    _path, _spec, capabilities = engine.task_context(TASK)
    plan = validate_question_plan(
        generated_plan(),
        schema_path=ROOT / "omni_ar/schemas/generated_questions.schema.json",
        catalog=option_catalog(capabilities),
        question_count=5,
    )
    prompts = iter([
        "Try a more robust representation without changing the split.",
        "2", "1,2", "1", "1", "Keep the mechanism broadly reusable.",
    ])
    observed: list[str] = []

    def fake_generator(*_args, **_kwargs):
        return plan

    def fake_resolver(*_args, **_kwargs):
        return {
            "schema_version": "omni-ar-task-resolution/v1",
            "status": "selected", "task_id": "vcc25", "confidence": 1.0,
            "reason": "explicit test selection", "question": "",
            "task_profile": None,
            "generator": {"provider": "test", "model": "fake"},
        }

    value = engine.ask(
        TASK, input_fn=lambda prompt: (observed.append(prompt), next(prompts))[1],
        output_fn=lambda _line: None, question_generator=fake_generator,
        task_resolver=fake_resolver,
    )
    assert observed[0].startswith("粗略想法")
    assert value["rough_idea"].startswith("Try a more robust")
    assert value["research_goal"] == "explore"
    assert value["requested_scopes"] == ["model", "objective"]
    assert value["clarifications"]["research_details"] == "Keep the mechanism broadly reusable."
    rough = engine.build(TASK, value, initialization_id="llm-question-test", confirmed=True)
    assert rough["confirmation"]["status"] == "confirmed"
    assert rough["user_notes"]["clarifications"]["research_details"]


def test_adapter_overrides_llm_protocol_options_and_default() -> None:
    engine = RoughIdeaEngine(ROOT)
    _path, _spec, capabilities = engine.task_context(TASK)
    plan = generated_plan()
    protocol_question = plan["questions"][2]
    protocol_question["options"] = [{"id": "heldout_target", "label": "LLM preferred target"}]
    protocol_question["default"] = "heldout_target"
    validated = validate_question_plan(
        plan,
        schema_path=ROOT / "omni_ar/schemas/generated_questions.schema.json",
        catalog=option_catalog(capabilities),
        question_count=5,
    )
    protocol_question = validated["questions"][2]
    assert protocol_question["default"] == "heldout_guide"
    assert [item["id"] for item in protocol_question["options"]] == [
        "heldout_guide", "heldout_target", "heldout_batch",
    ]


def test_adapter_adds_protocol_binding_question_when_llm_omits_it() -> None:
    engine = RoughIdeaEngine(ROOT)
    _path, _spec, capabilities = engine.task_context(TASK)
    plan = generated_plan()
    plan["questions"][2] = {
        "id": "exploration_level", "type": "single_choice",
        "title": "允许多大改动？", "why": "控制研究边界。",
        "options": [{"id": "balanced", "label": "平衡"}], "default": "balanced",
    }
    plan = validate_question_plan(
        plan,
        schema_path=ROOT / "omni_ar/schemas/generated_questions.schema.json",
        catalog=option_catalog(capabilities),
        question_count=5,
    )
    prompts = iter([
        "把 vcc 跑好", "2", "1,2", "1", "1", "优先保证泛化。", "",
    ])

    def fake_resolver(*_args, **_kwargs):
        return {
            "schema_version": "omni-ar-task-resolution/v1",
            "status": "selected", "task_id": "vcc25", "confidence": 1.0,
            "reason": "任务明确。", "question": "", "task_profile": None,
            "generator": {"provider": "test", "model": "fake"},
        }

    value = engine.ask(
        None, input_fn=lambda _prompt: next(prompts), output_fn=lambda _line: None,
        question_generator=lambda *_args, **_kwargs: plan,
        task_resolver=fake_resolver,
    )
    assert value["evaluation_protocol"] == "heldout_guide"
    binding = value["_question_plan"]["binding_questions"][0]
    assert binding["source"] == "task_adapter"
    assert binding["default"] == "heldout_guide"


def test_task_conflict_is_resolved_before_adapter_questions() -> None:
    engine = RoughIdeaEngine(ROOT)
    prompts = iter([
        "提升 CIFAR 的分类结果。",
        "",  # accept the LLM-inferred cifar task instead of the vcc25 hint
        "1", "1,2", "1", "1", "Prefer reusable mechanisms.",
    ])
    plan = generated_plan()
    plan["questions"][2] = {
        "id": "exploration_level", "type": "single_choice",
        "title": "允许多大改动？", "why": "控制研究边界。",
        "options": [{"id": "balanced", "label": "平衡"}], "default": "balanced",
    }

    def fake_resolver(*_args, **_kwargs):
        return {
            "schema_version": "omni-ar-task-resolution/v1",
            "status": "conflict", "task_id": "cifar", "confidence": 0.99,
            "reason": "想法明确指向 cifar，而提示为 vcc25。",
            "question": "检测到任务不一致，实际要研究哪个任务？",
            "task_profile": None,
            "generator": {"provider": "test", "model": "fake"},
        }

    value = engine.ask(
        TASK, input_fn=lambda _prompt: next(prompts), output_fn=lambda _line: None,
        question_generator=lambda *_args, **_kwargs: plan,
        task_resolver=fake_resolver,
    )
    assert value["_task_spec"] == "tasks/cifar/task_spec.yaml"
    assert value["evaluation_protocol"] == "default"
    assert value["_question_plan"]["task_resolution"]["selected_task_id"] == "cifar"


def test_unregistered_task_builds_researchstudio_rough_idea_without_fake_adapter(tmp_path: Path) -> None:
    engine = RoughIdeaEngine(ROOT)
    plan = generated_plan()
    plan["questions"][2] = {
        "id": "exploration_level", "type": "single_choice",
        "title": "允许多大改动？", "why": "控制研究边界。",
        "options": [{"id": "balanced", "label": "平衡"}], "default": "balanced",
    }
    plan = validate_question_plan(
        plan,
        schema_path=ROOT / "omni_ar/schemas/generated_questions.schema.json",
        catalog=option_catalog(generic_capabilities()),
        question_count=5,
    )
    prompts = iter([
        "研究卫星遥感序列中的异常事件预测。",
        "2", "1,2", "1", "1", "希望适用于不同地区。",
    ])

    def fake_resolver(*_args, **_kwargs):
        return {
            "schema_version": "omni-ar-task-resolution/v1",
            "status": "unregistered", "task_id": None, "confidence": 0.96,
            "reason": "该任务不属于现有 CIFAR 或 VCC25。", "question": "",
            "task_profile": {
                "name": "remote-sensing-event-prediction", "type": "classification",
                "modality": "time_series", "description": "预测遥感时间序列中的异常事件。",
                "literature_queries": [
                    "remote sensing time series anomaly event prediction",
                    "satellite temporal anomaly detection machine learning",
                ],
            },
            "generator": {"provider": "test", "model": "fake"},
        }

    value = engine.ask(
        None, input_fn=lambda _prompt: next(prompts), output_fn=lambda _line: None,
        question_generator=lambda *_args, **_kwargs: plan,
        task_resolver=fake_resolver,
    )
    assert value["_task_spec"] is None
    rough = engine.build_open(value, initialization_id="unregistered-test", confirmed=True)
    assert rough["schema_version"] == "omni-ar-rough-idea/v2"
    assert rough["task"]["binding_status"] == "unbound"
    assert rough["dataset"]["ref"] is None
    assert rough["execution_readiness"]["researchstudio_ready"] is True
    assert rough["execution_readiness"]["training_ready"] is False
    bundle = engine.write_bundle(tmp_path / "initialization", value, rough)
    result = dry_run_pipeline(ROOT, bundle["rough_idea"], tmp_path / "pipeline")
    trace = json.loads(result["trace"].read_text(encoding="utf-8"))
    assert trace["outcome"] == "researchstudio_ready_adapter_required"
    assert trace["checks"]["heuresis_started"] is False
