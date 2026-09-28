from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jsonschema
import yaml


class InitializationError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InitializationError(f"expected a mapping in {path}")
    return value


def _controller_api(repository: Path):
    script_dir = repository / "controller_bash/scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from task_contract import load_task_adapter, load_task_spec, validate_proposal

    return load_task_adapter, load_task_spec, validate_proposal


class RoughIdeaEngine:
    """Normalize novice-friendly answers against task and dataset contracts."""

    def __init__(self, repository: str | Path):
        self.repository = Path(repository).resolve()
        self.questions_path = Path(__file__).with_name("questions.yaml")
        self.schema_path = Path(__file__).parents[1] / "schemas/rough_idea.schema.json"
        self.questions = _mapping(self.questions_path)["questions"]

    def task_context(self, task_spec: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        path = Path(task_spec)
        if not path.is_absolute():
            path = self.repository / path
        path = path.resolve()
        if path != self.repository and self.repository not in path.parents:
            raise InitializationError(f"task spec escapes repository: {task_spec}")
        load_adapter, load_spec, _ = _controller_api(self.repository)
        spec = load_spec(path)
        adapter = load_adapter(spec, path)
        capabilities = adapter.initialization_capabilities(spec)
        if not capabilities.get("operation_capabilities"):
            raise InitializationError("task adapter exposes no research operation capabilities")
        return path, spec, capabilities

    def registered_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for path in sorted((self.repository / "tasks").glob("*/task_spec.yaml")):
            spec = _mapping(path)
            task = spec.get("task") or {}
            task_id = str(task.get("name") or path.parent.name)
            tasks.append({
                "id": task_id,
                "type": str(task.get("type") or ""),
                "modality": str(task.get("modality") or ""),
                "description": str(task.get("description") or ""),
                "dataset_ref": str((spec.get("data") or {}).get("dataset_ref") or ""),
                "task_spec": str(path.relative_to(self.repository)),
            })
        if len({item["id"] for item in tasks}) != len(tasks):
            raise InitializationError("registered task ids must be unique")
        return tasks

    @staticmethod
    def append_qa_event(
        answers: dict[str, Any], *, event: str, source: str,
        question_id: str | None = None, question: str | None = None,
        options: list[Any] | None = None, default: Any = None,
        raw_answer: Any = None, answer: Any = None, details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transcript = answers.setdefault("_qa_transcript", [])
        record = {
            "schema_version": "omni-ar-qa-event/v1",
            "sequence": len(transcript) + 1,
            "timestamp": _now(), "event": event, "source": source,
            "question_id": question_id, "question": question,
            "options": list(options or []), "default": default,
            "raw_answer": raw_answer, "answer": answer,
            "details": dict(details or {}),
        }
        transcript.append(record)
        return record

    @staticmethod
    def constraint_sources(answers: dict[str, Any]) -> dict[str, Any]:
        """Map each effective research constraint to its QA or Adapter origin."""
        events = {
            str(item.get("question_id")): item
            for item in answers.get("_qa_transcript") or []
            if item.get("event") in {"answer", "authorization", "confirmation"}
            and item.get("question_id")
        }

        def from_question(question_id: str, fallback_source: str, detail: str) -> dict[str, Any]:
            item = events.get(question_id)
            return {
                "source": str(item.get("source")) if item else fallback_source,
                "question_id": question_id if item else None,
                "qa_sequence": item.get("sequence") if item else None,
                "detail": detail,
            }

        mapping = {
            "research_goal.mode": from_question("research_goal", "fixed_safety_check", "research goal"),
            "priorities": from_question("priorities", "fixed_safety_check", "selection priorities"),
            "search_policy.exploration_level": from_question("exploration_level", "fixed_safety_check", "change level"),
            "search_policy.requested_scopes": from_question("requested_scopes", "fixed_safety_check", "editable research scopes"),
            "evaluation.protocol": from_question("evaluation_protocol", "task_adapter", "Adapter-supported evaluation protocol"),
            "evaluation.primary_metric": {"source": "task_adapter", "question_id": None, "qa_sequence": None, "detail": "metric contract"},
            "evaluation.baseline": {"source": "task_adapter", "question_id": None, "qa_sequence": None, "detail": "registered baseline"},
            "evaluation.acceptance_reference": {"source": "task_adapter", "question_id": None, "qa_sequence": None, "detail": "best comparable registered reference"},
            "evaluation.success_criteria": from_question("success_criteria", "fixed_safety_check", "acceptance rule"),
            "constraints.protected_items": {"source": "task_adapter", "question_id": None, "qa_sequence": None, "detail": "protected task contract"},
            "constraints.require_phenotype_activation": {"source": "task_adapter", "question_id": None, "qa_sequence": None, "detail": "activation contract"},
            "resources.profile": from_question("resource_profile", "fixed_safety_check", "resource profile"),
            "resources.limits": {"source": "task_adapter", "question_id": None, "qa_sequence": None, "detail": "Adapter limits bound the selected profile"},
            "execution_policy.external_services": from_question("authorize_external_services", "fixed_safety_check", "external service permission"),
            "execution_policy.submit_rjob": from_question("authorize_rjob", "fixed_safety_check", "rjob permission"),
            "execution_policy.code_changes": from_question("authorize_code_changes", "fixed_safety_check", "Coding Agent permission"),
        }
        answers["_constraint_sources"] = mapping
        return mapping

    def ask(
        self,
        task_spec: str | Path | None = None,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        question_generator: Callable[..., dict[str, Any]] | None = None,
        task_resolver: Callable[..., dict[str, Any]] | None = None,
        model: str | None = None,
        question_count: int = 5,
        initial_rough_idea: str | None = None,
    ) -> dict[str, Any]:
        transcript_holder: dict[str, Any] = {"_qa_transcript": []}
        if initial_rough_idea is None:
            output_fn("\n先用一两句话描述你的粗略研究想法。")
            rough_raw = input_fn("粗略想法（必填）：")
            rough_idea = rough_raw.strip()
        else:
            rough_idea = initial_rough_idea.strip()
            rough_raw = initial_rough_idea
            output_fn(f"\n已使用前置数据接入阶段的粗略想法：{rough_idea}")
        if not rough_idea:
            raise InitializationError("rough_idea is required before LLM follow-up questions")
        self.append_qa_event(
            transcript_holder, event="answer", source="fixed_safety_check",
            question_id="rough_idea", question="先用一两句话描述你的粗略研究想法。",
            raw_answer=rough_raw, answer=rough_idea,
        )

        candidates = self.registered_tasks()
        hint_id: str | None = None
        if task_spec is not None:
            hinted_path, hinted_spec, _hinted_capabilities = self.task_context(task_spec)
            hint_id = str(hinted_spec["task"]["name"])
        if task_resolver is None:
            from .llm_questions import resolve_task

            task_resolver = resolve_task
        resolution = task_resolver(
            self.repository, rough_idea, candidates, task_hint=hint_id, model=model
        )
        resolution_source = (
            "llm" if (resolution.get("generator") or {}).get("provider") == "boyue"
            else "fixed_safety_check"
        )
        self.append_qa_event(
            transcript_holder, event="task_resolution", source=resolution_source,
            question_id="task_resolution", answer=resolution.get("task_id"),
            details={key: value for key, value in resolution.items() if key != "generator"},
        )
        task_clarifications: list[str] = []
        resolved_idea = rough_idea
        for _attempt in range(2):
            if resolution.get("status") != "needs_clarification":
                break
            output_fn(f"\n{resolution['question']}")
            clarification_raw = input_fn("补充说明（必填）：")
            clarification = clarification_raw.strip()
            if not clarification:
                raise InitializationError("task clarification cannot be empty")
            task_clarifications.append(clarification)
            self.append_qa_event(
                transcript_holder, event="answer", source="llm",
                question_id=f"task_clarification_{len(task_clarifications)}",
                question=str(resolution["question"]), raw_answer=clarification_raw,
                answer=clarification,
            )
            resolved_idea = rough_idea + "\n用户补充：" + "；".join(task_clarifications)
            resolution = task_resolver(
                self.repository, resolved_idea, candidates,
                task_hint=hint_id, model=model,
            )
            self.append_qa_event(
                transcript_holder, event="task_resolution",
                source="llm" if (resolution.get("generator") or {}).get("provider") == "boyue" else "fixed_safety_check",
                question_id="task_resolution", answer=resolution.get("task_id"),
                details={key: value for key, value in resolution.items() if key != "generator"},
            )
        if resolution.get("status") == "needs_clarification":
            raise InitializationError("task remains ambiguous after two LLM clarification rounds")
        candidate_by_id = {item["id"]: item for item in candidates}
        selected_id = resolution.get("task_id") if resolution.get("status") == "selected" else None
        unregistered_profile = resolution.get("task_profile") if resolution.get("status") == "unregistered" else None
        if selected_id is None and unregistered_profile is None:
            output_fn(f"\n{resolution['question']}")
            if resolution.get("status") == "conflict":
                output_fn(f"  检测到任务提示与想法可能冲突：{resolution['reason']}")
            for index, candidate in enumerate(candidates, 1):
                output_fn(
                    f"  [{index}] {candidate['id']} — {candidate['description']} "
                    f"({candidate['type']}/{candidate['modality']})"
                )
            default_id = str(resolution.get("task_id") or hint_id or "")
            shown_default = default_id or "无"
            raw_task = input_fn(f"回答（回车使用默认值 {shown_default}）：").strip()
            if not raw_task:
                if not default_id:
                    raise InitializationError("task selection is required when the rough idea is ambiguous")
                selected_id = default_id
            elif raw_task.isdigit() and 1 <= int(raw_task) <= len(candidates):
                selected_id = candidates[int(raw_task) - 1]["id"]
            elif raw_task in candidate_by_id:
                selected_id = raw_task
            else:
                raise InitializationError(f"unsupported task selection {raw_task!r}")
            self.append_qa_event(
                transcript_holder, event="answer", source="fixed_safety_check",
                question_id="task_selection", question=str(resolution.get("question") or "选择研究任务"),
                options=[{"id": item["id"], "label": item["description"]} for item in candidates],
                default=default_id or None, raw_answer=raw_task, answer=selected_id,
            )
        if unregistered_profile is not None:
            from .llm_questions import generic_capabilities

            spec = {"task": unregistered_profile}
            capabilities = generic_capabilities()
            output_fn(
                f"\n识别为尚未注册的任务：{unregistered_profile['name']}。"
                "本阶段生成 ResearchStudio Rough Idea，不会绑定实验 Adapter。"
            )
        else:
            if selected_id not in candidate_by_id:
                raise InitializationError("LLM task resolution escaped the registered task catalog")
            selected_task_spec = self.repository / candidate_by_id[selected_id]["task_spec"]
            _, spec, capabilities = self.task_context(selected_task_spec)
            output_fn(f"\n已绑定任务：{selected_id}（{candidate_by_id[selected_id]['dataset_ref']}）")

        use_dynamic_followups = question_generator is None
        if question_generator is None:
            from .llm_questions import generate_question_plan

            question_generator = generate_question_plan
        plan = question_generator(
            self.repository,
            resolved_idea,
            spec["task"],
            capabilities,
            model=model,
            question_count=question_count,
        )
        plan["task_resolution"] = {
            **resolution,
            "selected_task_id": selected_id,
            "selected_task_spec": candidate_by_id[selected_id]["task_spec"] if selected_id else None,
            "selected_by_user": resolution.get("status") not in {"selected", "unregistered"},
        }
        output_fn(f"\n系统希望进一步明确：{plan['summary']}")
        available = list(dict.fromkeys(capabilities.get("operation_capabilities") or []))
        protocols = list(capabilities.get("protocols") or ["default"])
        answers: dict[str, Any] = {
            "rough_idea": rough_idea,
            "research_goal": "improve",
            "task_objective": capabilities.get("selected_task_candidate"),
            "priorities": ["metric_quality", "stability"],
            "exploration_level": "balanced",
            "requested_scopes": available[: min(4, len(available))],
            "evaluation_protocol": capabilities.get("default_protocol") or protocols[0],
            "success_criteria": ["outperform_baseline", "budget_compliance"],
            "resource_profile": "standard",
            "forbidden_directions": "",
            "requested_mutations": [],
            "clarifications": {"task_clarifications": task_clarifications} if task_clarifications else {},
            "_question_plan": plan,
            "_task_spec": candidate_by_id[selected_id]["task_spec"] if selected_id else None,
            "_task_profile": unregistered_profile,
            "_binding_status": "bound" if selected_id else "unbound",
            "_qa_transcript": transcript_holder["_qa_transcript"],
        }
        for question in plan["questions"]:
            output_fn(f"\n{question['title']}")
            output_fn(f"  目的：{question['why']}")
            options = list(question.get("options") or [])
            for index, option in enumerate(options, 1):
                output_fn(f"  [{index}] {option['label']} ({option['id']})")
            default = question.get("default", "")
            shown_default = ",".join(default) if isinstance(default, list) else str(default)
            raw = input_fn(f"回答（回车使用默认值 {shown_default}）：").strip()
            if question["type"] == "free_text":
                value: Any = raw or default
            else:
                tokens = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
                if not tokens:
                    value = default
                else:
                    allowed_ids = {item["id"] for item in options}
                    values = []
                    for token in tokens:
                        if token.isdigit() and 1 <= int(token) <= len(options):
                            values.append(options[int(token) - 1]["id"])
                        elif token in allowed_ids:
                            values.append(token)
                        else:
                            raise InitializationError(f"{question['id']}: unsupported choice {token!r}")
                    value = values if question["type"] == "multiple_choice" else values[0]
            if question["id"] == "research_details":
                answers["clarifications"]["research_details"] = value
            else:
                answers[question["id"]] = value
            self.append_qa_event(
                answers, event="answer", source=str(question.get("source") or "llm"),
                question_id=str(question["id"]), question=str(question["title"]),
                options=options, default=default, raw_answer=raw, answer=value,
                details={"why": question.get("why")},
            )
        asked_ids = {question["id"] for question in plan["questions"]}
        if use_dynamic_followups and "success_criteria" not in asked_ids and answers["research_goal"] in {"improve", "explore"}:
            from .llm_questions import option_catalog

            success_options = option_catalog(capabilities)["success_criteria"]
            default_success = ["outperform_baseline", "budget_compliance"]
            if "stability" in answers.get("priorities", []):
                default_success.insert(1, "multi_seed_stability")
            output_fn("\n根据你前面的研究目标，还需要明确怎样判断实验成功：")
            for index, option in enumerate(success_options, 1):
                output_fn(f"  [{index}] {option['label']} ({option['id']})")
            shown = ",".join(default_success)
            raw = input_fn(f"回答（回车使用默认值 {shown}）：").strip()
            tokens = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
            if not tokens:
                selected_success = default_success
            else:
                allowed = {item["id"] for item in success_options}
                selected_success = []
                for token in tokens:
                    if token.isdigit() and 1 <= int(token) <= len(success_options):
                        selected_success.append(success_options[int(token) - 1]["id"])
                    elif token in allowed:
                        selected_success.append(token)
                    else:
                        raise InitializationError(f"success_criteria: unsupported choice {token!r}")
            answers["success_criteria"] = list(dict.fromkeys(selected_success))
            dynamic = {
                "id": "success_criteria", "source": "fixed_safety_check",
                "trigger": "research_goal requires an explicit acceptance rule",
                "options": success_options, "default": default_success,
                "answer": answers["success_criteria"],
            }
            plan.setdefault("dynamic_questions", []).append(dynamic)
            self.append_qa_event(
                answers, event="answer", source="fixed_safety_check",
                question_id="success_criteria",
                question="根据研究目标，怎样判断实验成功？",
                options=success_options, default=default_success,
                raw_answer=raw, answer=answers["success_criteria"],
                details={"trigger": dynamic["trigger"]},
            )
        if use_dynamic_followups and answers.get("exploration_level") == "aggressive" and "forbidden_directions" not in asked_ids:
            output_fn("\n你选择了较大改动，请补充明确禁止改变的内容（可留空）：")
            raw = input_fn("禁止项：")
            answers["forbidden_directions"] = raw.strip()
            dynamic = {
                "id": "forbidden_directions", "source": "fixed_safety_check",
                "trigger": "aggressive exploration selected", "options": [],
                "default": "", "answer": answers["forbidden_directions"],
            }
            plan.setdefault("dynamic_questions", []).append(dynamic)
            self.append_qa_event(
                answers, event="answer", source="fixed_safety_check",
                question_id="forbidden_directions",
                question="较大改动中有哪些内容禁止改变？", raw_answer=raw,
                answer=answers["forbidden_directions"], details={"trigger": dynamic["trigger"]},
            )
        if len(protocols) > 1 and "evaluation_protocol" not in asked_ids:
            default_protocol = str(capabilities.get("default_protocol") or protocols[0])
            output_fn("\n实验执行前需要选择评估协议：")
            for index, protocol_name in enumerate(protocols, 1):
                output_fn(f"  [{index}] {protocol_name.replace('_', ' ')} ({protocol_name})")
            raw = input_fn(f"回答（回车使用默认值 {default_protocol}）：").strip()
            if not raw:
                selected_protocol = default_protocol
            elif raw.isdigit() and 1 <= int(raw) <= len(protocols):
                selected_protocol = protocols[int(raw) - 1]
            elif raw in protocols:
                selected_protocol = raw
            else:
                raise InitializationError(f"evaluation_protocol: unsupported choice {raw!r}")
            answers["evaluation_protocol"] = selected_protocol
            plan["binding_questions"] = [{
                "id": "evaluation_protocol",
                "source": "task_adapter",
                "options": protocols,
                "default": default_protocol,
                "answer": selected_protocol,
            }]
            self.append_qa_event(
                answers, event="answer", source="task_adapter",
                question_id="evaluation_protocol", question="实验执行前需要选择评估协议",
                options=[{"id": item, "label": item.replace("_", " ")} for item in protocols],
                default=default_protocol, raw_answer=raw, answer=selected_protocol,
            )
        self.constraint_sources(answers)
        return answers

    def build_open(
        self,
        answers: dict[str, Any],
        *,
        initialization_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Build a ResearchStudio-ready brief without pretending execution is configured."""
        profile = answers.get("_task_profile")
        if not isinstance(profile, dict):
            raise InitializationError("open initialization requires an inferred task profile")
        requested = list(dict.fromkeys(answers.get("requested_scopes") or []))
        from .llm_questions import generic_capabilities

        capabilities = generic_capabilities()
        available = capabilities["operation_capabilities"]
        rejected = sorted(set(requested).difference(available))
        allowed = [item for item in requested if item in available]
        conflicts = [f"unsupported research scope: {item}" for item in rejected]
        if not allowed:
            conflicts.append("no domain-general research scope was selected")
        resources = capabilities["resources"]
        resource_profile = str(answers.get("resource_profile") or "standard")
        created_at = _now()
        resolution = dict((answers.get("_question_plan") or {}).get("task_resolution") or {})
        rough = {
            "schema_version": "omni-ar-rough-idea/v2",
            "initialization_id": initialization_id,
            "task": {
                "name": str(profile["name"]),
                "type": str(profile["type"]),
                "modality": str(profile["modality"]),
                "task_spec": None,
                "binding_status": "unbound",
            },
            "dataset": {
                "ref": None, "manifest_sha256": None,
                "usage_mode": None, "binding_status": "unbound",
            },
            "research_goal": {
                "mode": str(answers.get("research_goal") or "explore"),
                "custom_answer": "",
            },
            "priorities": list(dict.fromkeys(answers.get("priorities") or ["metric_quality", "stability"])),
            "search_policy": {
                "exploration_level": str(answers.get("exploration_level") or "balanced"),
                "requested_scopes": requested, "allowed_scopes": allowed,
                "rejected_scopes": rejected,
            },
            "evaluation": {
                "protocol": None, "supported_protocols": [],
                "primary_metric": {}, "secondary_metrics": [],
                "baseline": None, "reference_results": [],
                "acceptance_reference": None, "minimum_improvement": None,
                "success_criteria": list(dict.fromkeys(answers.get("success_criteria") or ["runnable"])),
            },
            "constraints": {
                "protected_items": capabilities["protected_items"],
                "requested_mutations": [], "conflicts": conflicts,
                "require_budget_compliance": True,
                "require_phenotype_activation": False,
            },
            "resources": {
                "profile": resource_profile,
                "gpu_count": 0,
                "max_runtime_minutes": int(resources["max_runtime_minutes"]),
                "max_trials": 1,
            },
            "execution_policy": {
                "external_services": bool((answers.get("execution_policy") or {}).get("external_services", False)),
                "submit_rjob": False, "code_changes": False,
                "auto_retry": bool((answers.get("execution_policy") or {}).get("auto_retry", True)),
                "max_attempts": int((answers.get("execution_policy") or {}).get("max_attempts", 2)),
                "max_rounds": 1, "max_trials_per_round": 1, "max_gpus": 0,
                "max_runtime_minutes": int(resources["max_runtime_minutes"]),
            },
            "user_notes": {
                "rough_idea": str(answers.get("rough_idea") or ""),
                "forbidden_directions": str(answers.get("forbidden_directions") or ""),
                "clarifications": dict(answers.get("clarifications") or {}),
            },
            "research_context": {
                "task_description": str(profile["description"]),
                "literature_queries": list(profile["literature_queries"]),
            },
            "execution_readiness": {
                "researchstudio_ready": not conflicts,
                "heuresis_ready": False,
                "training_ready": False,
                "blockers": ["task_adapter_not_registered", "dataset_not_bound", "evaluation_contract_not_defined"],
            },
            "confirmation": {
                "status": "confirmed" if confirmed and not conflicts else "pending",
                "confirmed_at": created_at if confirmed and not conflicts else None,
            },
            "provenance": {
                "created_at": created_at, "task_spec_sha256": None,
                "questions_sha256": _sha_file(self.questions_path),
                "raw_answers_sha256": _canonical_sha(answers),
                "question_plan_sha256": _canonical_sha(answers.get("_question_plan") or {}),
                "question_generator": dict((answers.get("_question_plan") or {}).get("generator") or {}),
                "task_resolution": resolution,
                "constraint_sources": self.constraint_sources(answers),
                "qa_transcript_sha256": _canonical_sha(answers.get("_qa_transcript") or []),
            },
        }
        self.validate(rough)
        return rough

    def build(
        self,
        task_spec: str | Path,
        answers: dict[str, Any],
        *,
        initialization_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        spec_path, spec, capabilities = self.task_context(task_spec)
        task = spec["task"]
        dataset_ref = str((spec.get("data") or {}).get("dataset_ref", ""))
        if not dataset_ref:
            raise InitializationError("task spec does not bind a dataset_ref")
        from datasets.registry import DatasetRegistry

        registry = DatasetRegistry(self.repository / "datasets")
        manifest_path = registry.manifest_path(dataset_ref)
        supported = list(capabilities.get("protocols") or [])
        protocol = str(answers.get("evaluation_protocol") or capabilities.get("default_protocol") or "")
        requested = list(dict.fromkeys(answers.get("requested_scopes") or []))
        available = list(dict.fromkeys(capabilities.get("operation_capabilities") or []))
        rejected = sorted(set(requested).difference(available))
        allowed = [scope for scope in requested if scope in available]
        protected = list(dict.fromkeys(capabilities.get("protected_items") or []))
        mutations = list(dict.fromkeys(answers.get("requested_mutations") or []))
        protected_conflicts = sorted(set(mutations).intersection(protected))
        conflicts = [f"unsupported research scope: {item}" for item in rejected]
        conflicts.extend(f"requested mutation is protected: {item}" for item in protected_conflicts)
        if protocol not in supported:
            conflicts.append(f"unsupported evaluation protocol: {protocol}")
        if not allowed:
            conflicts.append("no requested research scope is allowed by the task adapter")
        task_candidates = {
            str(item.get("id")): item
            for item in (capabilities.get("task_candidates") or [])
            if item.get("id")
        }
        selected_task_candidate = str(capabilities.get("selected_task_candidate") or "")
        requested_task_candidate = str(
            answers.get("task_objective") or selected_task_candidate
        )
        requested_candidate = task_candidates.get(requested_task_candidate)
        if task_candidates and requested_candidate is None:
            conflicts.append(f"unsupported task objective: {requested_task_candidate}")
        elif requested_candidate and not requested_candidate.get("reviewed_template", False):
            conflicts.append(
                f"task objective requires a validated generated Adapter: {requested_task_candidate}"
            )
        elif selected_task_candidate and requested_task_candidate != selected_task_candidate:
            conflicts.append(
                "selected task objective does not match the generated Task Adapter: "
                f"{requested_task_candidate}"
            )

        metrics = capabilities.get("metrics") or {}
        primary = dict(metrics.get("primary") or {})
        secondary = list(metrics.get("secondary") or [])
        baseline = capabilities.get("baseline")
        references = [
            dict(item) for item in (capabilities.get("reference_results") or [])
            if item.get("metric") == primary.get("name")
            and item.get("protocol", protocol) == protocol
        ]
        comparable = [
            item for item in [baseline, *references]
            if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
        ]
        if comparable:
            acceptance_reference = dict(
                max(comparable, key=lambda item: float(item["value"]))
                if primary.get("direction") == "maximize"
                else min(comparable, key=lambda item: float(item["value"]))
            )
        else:
            acceptance_reference = None
        success = list(dict.fromkeys(answers.get("success_criteria") or ["outperform_baseline"]))
        minimum = answers.get("minimum_improvement") if "minimum_improvement" in success else None
        resources = capabilities.get("resources") or {}
        profile = str(answers.get("resource_profile") or "standard")
        max_trials_limit = max(1, int(resources.get("max_trials_per_round", 1)))
        max_runtime_limit = max(1, int(resources.get("max_runtime_minutes", 60)))
        profile_trials = {"quick": 1, "standard": min(5, max_trials_limit), "deep": max_trials_limit}
        profile_runtime = {"quick": min(20, max_runtime_limit), "standard": min(60, max_runtime_limit), "deep": max_runtime_limit}
        created_at = _now()
        raw_sha = _canonical_sha(answers)
        rough = {
            "schema_version": "omni-ar-rough-idea/v2",
            "initialization_id": initialization_id,
            "task": {
                "name": str(task["name"]), "type": str(task["type"]),
                "modality": str(task["modality"]),
                "task_spec": str(spec_path.relative_to(self.repository)),
            },
            "dataset": {
                "ref": dataset_ref, "manifest_sha256": _sha_file(manifest_path),
                "usage_mode": protocol,
            },
            "research_goal": {
                "mode": str(answers.get("research_goal") or "improve"),
                "custom_answer": str(answers.get("research_goal_custom") or ""),
            },
            "priorities": list(dict.fromkeys(answers.get("priorities") or ["metric_quality"])),
            "search_policy": {
                "exploration_level": str(answers.get("exploration_level") or "balanced"),
                "requested_scopes": requested, "allowed_scopes": allowed,
                "rejected_scopes": rejected,
            },
            "evaluation": {
                "protocol": protocol, "supported_protocols": supported,
                "primary_metric": primary, "secondary_metrics": secondary,
                "baseline": baseline, "reference_results": references,
                "acceptance_reference": acceptance_reference,
                "minimum_improvement": float(minimum) if minimum is not None else None,
                "success_criteria": success,
            },
            "constraints": {
                "protected_items": protected, "requested_mutations": mutations,
                "conflicts": conflicts,
                "require_budget_compliance": True,
                "require_phenotype_activation": bool((capabilities.get("activation_contract") or {}).get("required", False)),
            },
            "resources": {
                "profile": profile, "gpu_count": int(resources.get("gpu_count", 0)),
                "max_runtime_minutes": profile_runtime[profile],
                "max_trials": profile_trials[profile],
            },
            "execution_policy": {
                "external_services": bool((answers.get("execution_policy") or {}).get("external_services", False)),
                "submit_rjob": bool((answers.get("execution_policy") or {}).get("submit_rjob", False)),
                "code_changes": bool((answers.get("execution_policy") or {}).get("code_changes", False)),
                "auto_retry": bool((answers.get("execution_policy") or {}).get("auto_retry", True)),
                "max_attempts": int((answers.get("execution_policy") or {}).get("max_attempts", 2)),
                "max_rounds": int((answers.get("execution_policy") or {}).get("max_rounds", 3)),
                "max_trials_per_round": min(
                    int((answers.get("execution_policy") or {}).get("max_trials_per_round", profile_trials[profile])),
                    profile_trials[profile],
                ),
                "max_gpus": min(
                    int((answers.get("execution_policy") or {}).get("max_gpus", resources.get("gpu_count", 0))),
                    int(resources.get("gpu_count", 0)),
                ),
                "max_runtime_minutes": min(
                    int((answers.get("execution_policy") or {}).get("max_runtime_minutes", profile_runtime[profile])),
                    profile_runtime[profile],
                ),
            },
            "user_notes": {
                "rough_idea": str(answers.get("rough_idea") or ""),
                "forbidden_directions": str(answers.get("forbidden_directions") or ""),
                "clarifications": {
                    **dict(answers.get("clarifications") or {}),
                    "task_objective": requested_task_candidate,
                },
            },
            "confirmation": {
                "status": "confirmed" if confirmed and not conflicts else "pending",
                "confirmed_at": created_at if confirmed and not conflicts else None,
            },
            "provenance": {
                "created_at": created_at, "task_spec_sha256": _sha_file(spec_path),
                "questions_sha256": _sha_file(self.questions_path), "raw_answers_sha256": raw_sha,
                "question_plan_sha256": _canonical_sha(answers.get("_question_plan") or {}),
                "question_generator": dict((answers.get("_question_plan") or {}).get("generator") or {"provider": "provided_answers"}),
                "constraint_sources": self.constraint_sources(answers),
                "qa_transcript_sha256": _canonical_sha(answers.get("_qa_transcript") or []),
            },
        }
        self.validate(rough)
        return rough

    def validate(self, rough: dict[str, Any]) -> None:
        schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(rough)
        conflicts = rough["constraints"]["conflicts"]
        if rough["confirmation"]["status"] == "confirmed" and conflicts:
            raise InitializationError("a confirmed rough idea cannot contain conflicts")

    def write_bundle(self, output_dir: Path, answers: dict[str, Any], rough: dict[str, Any]) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "raw_answers.json"
        rough_path = output_dir / "rough_idea.yaml"
        brief_path = output_dir / "research_brief.md"
        validation_path = output_dir / "validation.json"
        question_plan_path = output_dir / "question_plan.json"
        transcript_path = output_dir / "qa_transcript.jsonl"
        generation_path = output_dir / "question_generation.json"
        raw_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rough_path.write_text(yaml.safe_dump(rough, sort_keys=False, allow_unicode=True), encoding="utf-8")
        brief_path.write_text(render_brief(rough), encoding="utf-8")
        if answers.get("_question_plan"):
            question_plan_path.write_text(json.dumps(answers["_question_plan"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            generation_path.write_text(json.dumps(
                (answers["_question_plan"].get("generator") or {}),
                ensure_ascii=False, indent=2,
            ) + "\n", encoding="utf-8")
        with transcript_path.open("w", encoding="utf-8") as handle:
            for event in answers.get("_qa_transcript") or []:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        artifact_paths = {
            "raw_answers": raw_path, "rough_idea": rough_path, "research_brief": brief_path,
            "qa_transcript": transcript_path,
        }
        if question_plan_path.exists():
            artifact_paths["question_plan"] = question_plan_path
            artifact_paths["question_generation"] = generation_path
        validation = {
            "schema_version": "omni-ar-initialization-validation/v1",
            "status": "ok" if not rough["constraints"]["conflicts"] else "conflict",
            "confirmed": rough["confirmation"]["status"] == "confirmed",
            "conflicts": rough["constraints"]["conflicts"],
            "artifacts": {name: {"path": str(path), "sha256": _sha_file(path)} for name, path in artifact_paths.items()},
        }
        validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = {"raw_answers": raw_path, "rough_idea": rough_path, "research_brief": brief_path, "validation": validation_path}
        result["qa_transcript"] = transcript_path
        if question_plan_path.exists():
            result["question_plan"] = question_plan_path
            result["question_generation"] = generation_path
        return result


def render_brief(rough: dict[str, Any]) -> str:
    primary = rough["evaluation"]["primary_metric"]
    baseline = rough["evaluation"].get("baseline") or {}
    reference = rough["evaluation"].get("acceptance_reference") or baseline
    scopes = ", ".join(rough["search_policy"]["allowed_scopes"])
    protected = ", ".join(rough["constraints"]["protected_items"])
    policy = rough.get("execution_policy") or {}
    conflicts = rough["constraints"]["conflicts"]
    is_bound = bool(rough["task"].get("task_spec"))
    dataset_label = rough["dataset"].get("ref") or "unbound (planning only)"
    protocol_label = rough["evaluation"].get("protocol") or "to be defined after Adapter binding"
    metric_name = primary.get("name") or "to be defined"
    metric_direction = primary.get("direction") or "unknown"
    lines = [
        f"# Research Brief: {rough['initialization_id']}", "",
        f"- Task: `{rough['task']['name']}`", f"- Dataset: `{dataset_label}`",
        f"- Protocol: `{protocol_label}`",
        f"- Primary metric: `{metric_name}` ({metric_direction})",
        f"- Baseline: `{baseline.get('name', 'unavailable')}={baseline.get('value', 'unavailable')}`",
        f"- Acceptance reference: `{reference.get('name', 'unavailable')}={reference.get('value', 'unavailable')}`",
        f"- Exploration: `{rough['search_policy']['exploration_level']}`", f"- Allowed scopes: {scopes}",
        f"- Protected: {protected}",
        (
            f"- Budget: {rough['resources']['max_trials']} trial(s), {rough['resources']['gpu_count']} GPU, {rough['resources']['max_runtime_minutes']} min/trial"
            if is_bound else "- Execution: `planning only; Adapter and dataset binding required before Heuresis`"
        ),
        f"- Confirmation: `{rough['confirmation']['status']}`", "", "## User idea", "",
        rough["user_notes"]["rough_idea"] or "No concrete mechanism supplied; ResearchStudio should propose evidence-grounded directions.",
        "", "## Clarifications", "",
        json.dumps(rough["user_notes"].get("clarifications") or {}, ensure_ascii=False),
        "", "## Exclusions", "", rough["user_notes"]["forbidden_directions"] or "None supplied.",
    ]
    if policy:
        lines[11:11] = [
            f"- External services authorized: `{policy.get('external_services', False)}`",
            f"- Real rjob authorized: `{policy.get('submit_rjob', False)}`",
            f"- Coding Agent changes authorized: `{policy.get('code_changes', False)}`",
            f"- Automatic retry: `{policy.get('auto_retry', False)}` (max {policy.get('max_attempts', 1)} attempt(s))",
        ]
    if conflicts:
        lines.extend(["", "## Blocking conflicts", "", *[f"- {item}" for item in conflicts]])
    sources = (rough.get("provenance") or {}).get("constraint_sources") or {}
    if is_bound:
        lines.extend([
            "", "## System-supplied settings shown for confirmation", "",
            f"- Evaluation protocol: `{protocol_label}` (source: `{(sources.get('evaluation.protocol') or {}).get('source', 'task_adapter')}`)",
            f"- Baseline: `{baseline.get('name', 'unavailable')}={baseline.get('value', 'unavailable')}` (source: `task_adapter`)",
            f"- Budget: `{rough['resources']['max_trials']} trial(s), {rough['resources']['gpu_count']} GPU, {rough['resources']['max_runtime_minutes']} min/trial` (bounded by Task Adapter)",
            f"- External services: `{policy.get('external_services', False)}`; rjob: `{policy.get('submit_rjob', False)}`; code changes: `{policy.get('code_changes', False)}` (source: fixed safety checks or explicit CLI flags)",
            "", "## QA constraint trace", "",
        ])
        for path, origin in sorted(sources.items()):
            question = origin.get("question_id") or "automatic binding"
            lines.append(f"- `{path}` ← `{origin.get('source')}` / `{question}`")
    return "\n".join(lines) + "\n"


def render_researchstudio_dry_run(rough: dict[str, Any], output_dir: Path, source: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "researchstudio_request.json"
    card_path = output_dir / "idea.detail.en.md"
    method_path = output_dir / "method_view.json"
    request = {
        "schema_version": "omni-ar-researchstudio-request/v1", "mode": "dry_run",
        "rough_idea": str(source), "rough_idea_sha256": _sha_file(source),
        "task": rough["task"], "dataset": rough["dataset"], "evaluation": rough["evaluation"],
        "search_policy": rough["search_policy"], "constraints": rough["constraints"],
        "execution_authorized": False,
    }
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    idea = rough["user_notes"]["rough_idea"] or "Explore evidence-grounded combinations within the adapter-declared operation capabilities."
    card_path.write_text(
        "# Initialization-grounded Idea Card (Dry Run)\n\n"
        f"**Task:** {rough['task']['name']}  \n**Dataset:** {rough['dataset']['ref']}  \n"
        f"**Protocol:** {rough['evaluation']['protocol']}  \n\n"
        f"## Rough research direction\n\n{idea}\n\n"
        "## Search boundary\n\n"
        f"Allowed scopes: {', '.join(rough['search_policy']['allowed_scopes'])}.\n\n"
        f"Protected items: {', '.join(rough['constraints']['protected_items'])}.\n\n"
        "## Dry-run limitation\n\nThis card validates the ResearchStudio entry contract only; it contains no generated literature claims.\n",
        encoding="utf-8",
    )
    method = {
        "schema_version": "omni-ar-method-view/v1", "mode": "dry_run",
        "research_direction": idea, "change_scope": rough["search_policy"]["allowed_scopes"],
        "evaluation_protocol": rough["evaluation"]["protocol"], "execution_authorized": False,
    }
    method_path.write_text(json.dumps(method, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"request": request_path, "idea_card": card_path, "method_view": method_path}


def _proposal_from_rough(rough: dict[str, Any], idea_card_text: str) -> dict[str, Any]:
    metric = rough["evaluation"]["primary_metric"]
    metric_name = str(metric["name"])
    direction = str(metric["direction"])
    baseline = rough["evaluation"].get("acceptance_reference") or rough["evaluation"].get("baseline") or {}
    delta = rough["evaluation"].get("minimum_improvement") or 0.0
    value = float(baseline.get("value", 0.0))
    operator = ">=" if direction == "maximize" else "<="
    target = value + float(delta) if direction == "maximize" else value - float(delta)
    idea = rough["user_notes"]["rough_idea"] or "Combine adapter-declared mechanisms while preserving the fixed evaluation contract."
    if idea not in idea_card_text:
        raise InitializationError("ResearchStudio Idea Card does not preserve the confirmed research direction")
    experiment = {
        "hypothesis": idea,
        "change_scope": rough["search_policy"]["allowed_scopes"],
        "parameters": {},
        "expected_effect": {metric_name: {"direction": "increase" if direction == "maximize" else "decrease", "minimum_change": float(delta)}},
        "acceptance_criteria": {metric_name: {"operator": operator, "value": target}},
        "resource_request": {
            "gpu_count": rough["resources"]["gpu_count"], "cpu": 8,
            "memory_mb": 20000, "max_runtime_minutes": rough["resources"]["max_runtime_minutes"],
        },
    }
    return {
        "schema_version": "omni-ar-proposal/v2",
        "proposal_id": f"{rough['initialization_id']}-dry-run",
        "task_name": rough["task"]["name"], "round": 0,
        "verdict": "Dry-run proposal generated from a confirmed rough research brief.",
        "evidence_gaps": [{"claim": idea, "gap": "No trial has been executed from this initialization.", "why_it_matters": "Execution remains unauthorized until proposal review."}],
        "experiment_proposals": [experiment],
        "risks": ["Dry-run Idea Card has not performed literature retrieval or novelty verification."],
    }


def dry_run_pipeline(repository: Path, rough_path: Path, output_dir: Path) -> dict[str, Any]:
    repository, rough_path, output_dir = repository.resolve(), rough_path.resolve(), output_dir.resolve()
    rough = _mapping(rough_path)
    engine = RoughIdeaEngine(repository)
    engine.validate(rough)
    if rough["confirmation"]["status"] != "confirmed":
        raise InitializationError("dry-run requires a confirmed rough idea")
    if rough["constraints"]["conflicts"]:
        raise InitializationError("dry-run refuses a rough idea with unresolved conflicts")
    output_dir.mkdir(parents=True, exist_ok=True)

    researchstudio_dir = output_dir / "idea_card"
    entrypoint = repository / "ResearchStudio-main/scripts/rough_idea_entry.py"
    proc = subprocess.run(
        [sys.executable, str(entrypoint), "--rough-idea", str(rough_path), "--output-dir", str(researchstudio_dir), "--dry-run"],
        cwd=repository, text=True, capture_output=True, check=False,
    )
    (output_dir / "researchstudio.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "researchstudio.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise InitializationError(f"ResearchStudio dry-run entrypoint failed: {proc.stderr.strip()}")

    idea_card_path = researchstudio_dir / "idea.detail.en.md"
    if not rough["task"].get("task_spec"):
        trace = {
            "schema_version": "omni-ar-initialization-trace/v1",
            "status": "ok",
            "outcome": "researchstudio_ready_adapter_required",
            "initialization_id": rough["initialization_id"],
            "execution_authorized": False,
            "stages": [
                {"name": "initialization", "inputs": [], "outputs": ["rough_idea"]},
                {"name": "researchstudio", "inputs": ["rough_idea"], "outputs": ["idea_card", "method_view"]},
            ],
            "checks": {
                "rough_idea_schema": "ok", "confirmed": True,
                "task_adapter_bound": False, "heuresis_started": False,
                "training_started": False, "rjob_submitted": False,
            },
            "blockers": list((rough.get("execution_readiness") or {}).get("blockers") or []),
        }
        trace_path = output_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "trace": trace_path,
            "artifacts": {
                "idea_card": idea_card_path,
                "method_view": researchstudio_dir / "method_view.json",
            },
        }
    proposal = _proposal_from_rough(rough, idea_card_path.read_text(encoding="utf-8"))
    proposal_path = output_dir / "heuresis_proposal.json"
    proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _, load_spec, validate_proposal = _controller_api(repository)
    task_spec = (repository / rough["task"]["task_spec"]).resolve()
    spec = load_spec(task_spec)
    validate_proposal(proposal, spec, task_spec)
    from omni_ar.loop import ResearchLoop

    designed = ResearchLoop(repository).design_code(
        task_spec, {"proposal_id": proposal["proposal_id"], **proposal["experiment_proposals"][0]},
        dataset_mode=rough["dataset"]["usage_mode"],
    )
    compiled_path = output_dir / "compiled_trial.json"
    compiled_path.write_text(json.dumps(designed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = {
        "rough_idea": rough_path, "researchstudio_request": researchstudio_dir / "researchstudio_request.json",
        "idea_card": idea_card_path, "method_view": researchstudio_dir / "method_view.json",
        "heuresis_proposal": proposal_path, "compiled_trial": compiled_path,
    }
    for name, filename in (("raw_answers", "raw_answers.json"), ("research_brief", "research_brief.md"), ("initialization_validation", "validation.json")):
        candidate = rough_path.parent / filename
        if candidate.exists():
            artifacts[name] = candidate
    artifact_records = {name: {"path": str(path), "sha256": _sha_file(path)} for name, path in artifacts.items()}
    trace = {
        "schema_version": "omni-ar-initialization-trace/v1", "status": "ok", "mode": "dry_run",
        "initialization_id": rough["initialization_id"], "execution_authorized": False,
        "stages": [
            {"name": "initialization", "inputs": [name for name in ("raw_answers",) if name in artifacts], "outputs": ["rough_idea", "research_brief", "initialization_validation"]},
            {"name": "researchstudio", "inputs": ["rough_idea"], "outputs": ["researchstudio_request", "idea_card", "method_view"]},
            {"name": "heuresis", "inputs": ["idea_card", "rough_idea"], "outputs": ["heuresis_proposal"]},
            {"name": "task_adapter_compile", "inputs": ["heuresis_proposal"], "outputs": ["compiled_trial"]},
        ],
        "artifacts": artifact_records,
        "contract_pins": {
            "task_spec_sha256": rough["provenance"]["task_spec_sha256"],
            "dataset_manifest_sha256": rough["dataset"]["manifest_sha256"],
            "questions_sha256": rough["provenance"]["questions_sha256"],
        },
        "checks": {
            "rough_idea_schema": "ok", "confirmed": True, "conflicts": [],
            "researchstudio_entrypoint": str(entrypoint), "proposal_schema": "ok",
            "task_adapter_compile": "ok", "training_started": False, "rjob_submitted": False,
        },
    }
    trace_path = output_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"trace": trace_path, "artifacts": artifacts, "compiled": designed}
