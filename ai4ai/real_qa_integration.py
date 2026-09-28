from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import ResearchState
from .router import SkillRouter
from .stage_hook import SkillStageHook

ROOT = Path(__file__).resolve().parents[1]
REAL_SCRIPTS = ROOT / "controller_bash" / "scripts"
EXAMPLES = ROOT / "controller_bash" / "examples"


def _load(name: str, path: Path):
    if str(REAL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(REAL_SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"ai4ai_real_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load copied real module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def make_real_state() -> Any:
    module = _load("research_state", REAL_SCRIPTS / "research_state.py")
    return module.ResearchState(
        current_task="example_task",
        current_scientific_question="Does bounded evidence improve planning context?",
        current_best_candidate="baseline",
        current_best_evidence={"quality": 0.5},
        open_hypotheses=[{"id": "h0", "text": "baseline is runnable"}],
        known_bottlenecks=[{"id": "b0", "text": "missing matched evidence"}],
        active_blockers=[],
        next_research_action="construct proposal context",
    )


def build_contexts(router: SkillRouter) -> dict[str, Any]:
    state_module = _load("research_state", REAL_SCRIPTS / "research_state.py")
    suggest_module = _load("heuresis_suggest", REAL_SCRIPTS / "heuresis_suggest.py")
    real_state = make_real_state()
    baseline = state_module.build_planner_context(real_state)
    baseline_prompt = suggest_module.prompt_for({"task_spec": {"task": {"name": "example_task"}}, **baseline})

    local_state = ResearchState(
        task_id="example_task",
        question=real_state.current_scientific_question,
        evidence=({"id": "qa-0", "status": "observed", "quality": 0.5},),
    )
    augmented_local, injection = SkillStageHook(router).inject(
        "qa_to_idea_card", "example.evidence_summary", local_state, {"bounded": True}
    )
    skill_evidence = dict(augmented_local.evidence[-1])
    augmented = suggest_module.augment_proposal_context(
        baseline, injection.result.to_dict()
    )
    augmented_prompt = suggest_module.prompt_for({"task_spec": {"task": {"name": "example_task"}}, **augmented})
    original_keys = set(baseline)
    preserved = all(augmented[k] == baseline[k] for k in original_keys)
    provenance = {
        "stage": "QA",
        "source": "copied ResearchStudio+Heuresis source",
        "skill_id": injection.result.skill_id,
        "skill_version": injection.result.provenance.get("skill_version"),
        "backend": router.backend(injection.result.skill_id),
    }
    return {
        "real_state": real_state,
        "baseline": baseline,
        "augmented": augmented,
        "baseline_prompt": baseline_prompt,
        "augmented_prompt": augmented_prompt,
        "preserved": preserved,
        "original_context_hash": digest(baseline),
        "augmented_context_hash": digest(augmented),
        "skill_result_hash": digest(injection.result.to_dict()),
        "proposal_input_hash": digest({"task_spec": {"task": {"name": "example_task"}}, **augmented}),
        "provenance": provenance,
    }


def validate_example_proposal() -> bool:
    controller_root = ROOT / "controller_bash"
    if str(controller_root) not in sys.path:
        sys.path.insert(0, str(controller_root))
    contract = _load("task_contract", REAL_SCRIPTS / "task_contract.py")
    spec_path = EXAMPLES / "ai4ai_task_spec.yaml"
    proposal_path = EXAMPLES / "ai4ai_proposal.example.json"
    spec = contract.load_task_spec(spec_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    contract.validate_proposal(proposal, spec, spec_path)
    return True


def proposal_generation_trace(router: SkillRouter, *, qa_injection=True, proposal_injection=True) -> dict[str, Any]:
    """Run the copied Heuresis prompt/fixture path without calling a model or backend."""
    state_module = _load("research_state", REAL_SCRIPTS / "research_state.py")
    suggest_module = _load("heuresis_suggest", REAL_SCRIPTS / "heuresis_suggest.py")
    real_state = make_real_state()
    context = state_module.build_planner_context(real_state)
    trace = [{"stage": "idea_card", "context_hash": digest(context)}]
    injections = []
    local_state = ResearchState("example_task", real_state.current_scientific_question,
                                ({"id": "qa-0", "status": "observed"},))
    if qa_injection:
        local_state, qa = SkillStageHook(router).inject("qa_to_idea_card", "example.evidence_summary", local_state)
        injections.append(qa.result.to_dict())
        context = dict(context)
        context["skill_injections"] = [{"stage": "QA", "evidence": dict(qa.result.evidence),
                                         "skill_id": qa.result.skill_id, "provenance": dict(qa.result.provenance)}]
    trace.append({"stage": "after_qa", "context_hash": digest(context)})
    if proposal_injection:
        proposal_state = ResearchState("example_task", real_state.current_scientific_question,
                                       tuple(local_state.evidence) + ({"id": "proposal-context", "status": "observed"},))
        _, proposal = SkillStageHook(router).inject("idea_card_to_heuresis", "example.evidence_summary", proposal_state)
        injections.append(proposal.result.to_dict())
        context = suggest_module.augment_proposal_context(context, proposal.result.to_dict())
    final_input = {"task_spec": {"task": {"name": "example_task"}}, **context}
    prompt = suggest_module.prompt_for(final_input)
    proposal_artifact = json.loads((EXAMPLES / "ai4ai_proposal.example.json").read_text(encoding="utf-8"))
    validator_ok = validate_example_proposal()
    return {"context": context, "prompt": prompt, "proposal": proposal_artifact,
            "validator_ok": validator_ok, "trace": trace + [{"stage": "heuresis_input", "context_hash": digest(context)}],
            "injections": injections, "input_hash": digest(final_input), "proposal_hash": digest(proposal_artifact),
            "provenance": [{"skill_id": x.get("skill_id"), "skill_version": (x.get("provenance") or {}).get("skill_version")} for x in injections]}
