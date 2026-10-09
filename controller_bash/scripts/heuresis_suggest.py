#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_tools import is_tool_call, tool_schema


PROPOSAL_V2_REQUIRED = {
    "schema_version",
    "proposal_id",
    "task_name",
    "round",
    "verdict",
    "evidence_gaps",
    "experiment_proposals",
    "risks",
}


def is_complete_proposal(value: object) -> bool:
    """Reject JSON fragments so malformed model output triggers a retry."""
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "omni-ar-proposal/v2"
        and PROPOSAL_V2_REQUIRED.issubset(value)
        and isinstance(value.get("experiment_proposals"), list)
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def prompt_for(
    context: dict,
    tool_results: list[dict] | None = None,
    validation_feedback: list[str] | None = None,
) -> str:
    dataset = context.get("dataset") or {}
    tools = [tool_schema(dataset["ref"])] if dataset.get("ref") else []
    return f"""
You are Heuresis acting as a project-agnostic idea/code search controller.
The upstream system and Codex may already have produced an implementation.
Your task is to propose the next round of idea exploration, code changes,
ablations, and executable trials from the provided project context.

Return strict JSON only. Do not include prose outside JSON. Keep every proposal
feasible within the provided resource limits. Never request controller,
dataset, credential, metric, protocol, or report edits. Code requests may use
only `context.task_spec.search.editable_paths` (relative to the implementation
directory) and the exact repository-relative files, if any, in
`context.task_spec.search.interface_editable_paths`.
Never emit shell commands, entrypoints, CLI flags with leading dashes, data
paths, output paths, or rjob commands. Emit only structured experiment
proposals. The configured Task Adapter merges each proposal's `parameters` with
`context.task_spec.adapter.trial_defaults` and converts it into the concrete
`run_trial` invocation. Omit unchanged parameters instead of copying defaults.
This is mandatory for budget-sensitive defaults such as batch size and epoch
count. Parameter names use snake_case and must be supported by the task adapter.
Metric names and directions must come from context.task_spec. Simulated trials
are plumbing checks only and must not be treated as scientific evidence.
If `context.dataset.capabilities.trial_parameters` is present, it is the exact
parameter-name allowlist for ordinary `experiment_proposals`: every key there
must come from that list. Do not put unimplemented parameters into ordinary
experiment_proposals. A `trial_proposal` inside an implementation request may
name a new parameter only when that same request explicitly allows the
task-local Adapter interface needed to register and validate it. When a
materially new model, loss, or training mechanism
is justified, place it in the optional implementation_requests array. The
controller routes those requests through a Coding Agent in a separate Git
worktree, checks editable paths and activation diagnostics, and only then runs
the associated trial_proposal. Ordinary parameter changes and combinations of
registered mechanisms stay in experiment_proposals and bypass the Coding Agent.
If `context.dataset.capabilities.method_capabilities` is present, every explicit
`method` value must come from that list. A capability guarded by an activation
verification flag may be proposed only when that flag is true.
If method or parameter capability details are present, obey their enums, ranges,
roles, and limitations. Do not rename a verified mechanism or claim that a
task-local approximation implements a broader model named in the Idea Card.
When context.dataset is present, use only its registered ref, declared
artifacts, splits, and merged Dataset/Task Adapter capabilities. Data actions
and usage modes come from the Dataset Adapter; methods, trial parameters, and
activation checks come from the Task Adapter. Never invent physical data paths
or copy raw data into a research project. The Task Adapter compiles the proposal
and the controller supplies local/rjob dataset binding.
When a validated feature plan is present, select only its declared feature
combinations. Use `feature_id` and `feature_branch` exactly as exposed by the
Task Adapter. Compare raw, single-branch, and combined features when listed.
When a combination declares `task_method` and `task_parameters`, use those
exact Adapter parameters instead of adding `feature_id`; this is how a task
with an established feature-aware method consumes the registered feature.
Bindings marked `activation_verification_only` may be used only to verify that
the feature is active and to report metrics. Do not claim them as accepted
scientific candidates or as evidence of improvement.
If no Builder matches, follow the plan's fail-closed Coding Agent fallback and
never invent a feature identifier.

Available tools:
{json.dumps(tools, ensure_ascii=False)}

Before the final proposal, you may request one Dataset Adapter call at a time:
{{"schema_version":"omni-ar-tool-call/v1","tool":"dataset_adapter",
  "call_id":"unique-id","arguments":{{"action":"inspect","dataset_ref":"registered-ref","limit":10}}}}
Return either exactly one tool call or the final proposal. Tool calls are
executed by the controller, are bounded and recorded, and their results are
included below. Do not repeat an identical call. `prepare` only reports/creates
registered profiles through controller policy; it does not authorize training.

Dataset tool results from this planning turn:
{json.dumps(tool_results or [], ensure_ascii=False)}

Contract validation feedback from earlier attempts in this same turn:
{json.dumps(validation_feedback or [], ensure_ascii=False)}
When feedback is present, correct every listed issue before returning the next
proposal. Do not repeat an invalid method, parameter name, value, or route.

Required JSON shape:
{{
  "schema_version": "omni-ar-proposal/v2",
  "proposal_id": "task-round-N-short-name",
  "task_name": "exact context.task_spec.task.name",
  "round": 0,
  "verdict": "brief evidence-state assessment",
  "evidence_gaps": [
    {{"claim": "...", "gap": "...", "why_it_matters": "..."}}
  ],
  "experiment_proposals": [
    {{
      "hypothesis": "single testable claim",
      "change_scope": ["model", "loss"],
      "parameters": {{"snake_case_parameter": "value"}},
      "expected_effect": {{
        "metric_from_task_spec": {{"direction": "increase", "minimum_change": 0.01}}
      }},
      "mechanism_alignment": {{
        "core_reference_mechanisms": ["mechanism names this proposal claims to use"],
        "implemented_mechanisms": ["mechanisms actually present in the executable proposal"],
        "missing_mechanisms": ["important mechanisms intentionally absent"],
        "target_specificity": "how this proposal adds target-specific information rather than only global context",
        "approximation_gap": "why the proposal is close to or distant from the referenced mechanism",
        "evidence": "task evidence supporting this mechanism and a negative control that could falsify it"
      }},
      "acceptance_criteria": {{
        "metric_from_task_spec": {{"operator": ">=", "value": 0.5}}
      }},
      "resource_request": {{
        "gpu_count": 1,
        "cpu": 8,
        "memory_mb": 20000,
        "max_runtime_minutes": 60
      }}
    }}
  ],
  "implementation_requests": [
    {{
      "request_id": "short-unique-id",
      "hypothesis": "why new code is needed",
      "change_scope": ["model", "objective"],
      "allowed_paths": ["editable implementation-relative path or exact interface_editable_paths entry"],
      "required_capabilities": ["new_capability_name"],
      "activation_diagnostics": ["nonzero_activation_counter"],
      "trial_proposal": {{
        "hypothesis": "testable post-implementation claim",
        "change_scope": ["model"],
        "parameters": {{"newly_registered_parameter": true}},
        "expected_effect": {{
          "metric_from_task_spec": {{"direction": "increase", "minimum_change": 0.01}}
        }},
        "mechanism_alignment": {{
          "core_reference_mechanisms": ["mechanism names this trial claims to use"],
          "implemented_mechanisms": ["mechanisms actually present in the executable trial"],
          "missing_mechanisms": ["important mechanisms intentionally absent"],
          "target_specificity": "how this trial adds target-specific information rather than only global context",
          "approximation_gap": "why the trial is close to or distant from the referenced mechanism",
          "evidence": "task evidence supporting this mechanism and a negative control that could falsify it"
        }},
        "acceptance_criteria": {{
          "metric_from_task_spec": {{"operator": ">=", "value": 0.5}}
        }},
        "resource_request": {{
          "gpu_count": 1, "cpu": 8, "memory_mb": 20000,
          "max_runtime_minutes": 60
        }}
      }}
    }}
  ],
  "risks": ["..."]
}}

Context JSON:
{json.dumps(context, ensure_ascii=False)}

Machine-checkable skill constraints:
- If `context.heuresis_constraints` is present, every proposal must copy its
  `controls` into `mechanism_alignment.required_controls` and its
  `acceptance.required_gates` into `mechanism_alignment.required_gates`.
- Preserve `acceptance.test_expression_used=false` as a hard acceptance rule;
  never use official test expression for training or selection.

Mechanism-alignment guidance:
- If the context includes Lingshu-Cell or Lingshu Skill knowledge, treat the
  Lingshu reference Pearson-Delta (~0.24) as reference evidence only. Do not use
  it as an acceptance criterion or promotion gate.
- A proposal inspired by Lingshu should say whether it implements whole-gene
  representation, raw-count/discrete tokenization, explicit target/cell/control
  conditioning, masked generative modeling, classifier-free guidance, and
  target-to-gene biological prior injection.
- Do not label a generic graph-conditioned delta model as a close Lingshu
  mechanism transfer unless it explains the approximation gap and the missing
  raw-count tokenization / masked generation / CFG / prior-in-sampling pieces.
- Prefer proposals with a coherent mechanism chain over proposals that borrow a
  surface keyword such as "biological prior" or "condition-aware" without
  specifying where the condition enters and how output behavior changes.
"""


def proposal_contract_error(context: dict, proposal: dict) -> str | None:
    """Return a bounded schema/Task-Adapter error for model self-correction."""
    task_path = ((context.get("paths") or {}).get("task_spec"))
    if not task_path:
        return None
    try:
        from task_contract import load_task_spec, validate_proposal  # noqa: PLC0415

        path = Path(str(task_path)).resolve()
        validate_proposal(proposal, load_task_spec(path), path)
    except Exception as exc:  # noqa: BLE001 - turn validation into retry feedback
        return f"{type(exc).__name__}: {str(exc)[:1800]}"
    return skill_constraint_error(context, proposal) or feature_coverage_error(context, proposal)


def skill_constraint_error(context: dict, proposal: dict) -> str | None:
    """Fail closed when a proposal drops constraints emitted by Skills."""
    constraints = context.get("heuresis_constraints") or {}
    if not constraints:
        return None
    required_controls = set(constraints.get("controls") or [])
    required_gates = set(((constraints.get("acceptance") or {}).get("required_gates") or []))
    for index, item in enumerate(proposal.get("experiment_proposals") or []):
        alignment = item.get("mechanism_alignment") or {}
        controls = set(alignment.get("required_controls") or [])
        gates = set(alignment.get("required_gates") or [])
        missing_controls = sorted(required_controls - controls)
        missing_gates = sorted(required_gates - gates)
        if missing_controls or missing_gates or alignment.get("test_expression_used") is not False:
            return (
                f"SkillConstraintError: experiment_proposals[{index}] is missing "
                f"required_controls={missing_controls} required_gates={missing_gates} "
                "or test_expression_used=false"
            )
    return None


def feature_coverage_error(context: dict, proposal: dict) -> str | None:
    """Require matched controls when the validated Feature Plan supports them."""
    plan = ((context.get("dataset") or {}).get("feature_plan") or {})
    combinations = [
        item for item in (plan.get("search_combinations") or [])
        if isinstance(item, dict)
    ]
    feature_combinations = [
        item for item in combinations if item.get("feature_ids")
    ]
    if not feature_combinations:
        return None
    trials = list(proposal.get("experiment_proposals") or [])
    trials.extend(
        item.get("trial_proposal") for item in (proposal.get("implementation_requests") or [])
        if isinstance(item, dict) and isinstance(item.get("trial_proposal"), dict)
    )
    parameters = [
        dict(item.get("parameters") or {}) for item in trials if isinstance(item, dict)
    ]

    def matches_feature(values: dict, combination: dict) -> bool:
        selected = set(map(str, combination.get("feature_ids") or []))
        requested = values.get("feature_ids")
        if isinstance(requested, str):
            requested = [requested]
        if selected.intersection(map(str, requested or [])):
            return True
        if str(values.get("feature_id") or "") in selected:
            return True
        method = combination.get("task_method")
        required = dict(combination.get("task_parameters") or {})
        return bool(method) and values.get("method") == method and all(
            values.get(key) == value for key, value in required.items()
        )

    feature_flags = [
        any(matches_feature(values, combination) for combination in feature_combinations)
        for values in parameters
    ]
    raw_flags = [
        not feature and str(values.get("feature_id") or "raw") == "raw"
        for values, feature in zip(parameters, feature_flags)
    ]
    missing: list[str] = []
    if not any(raw_flags):
        missing.append("raw control")
    if not any(feature_flags):
        missing.append("registered feature candidate")

    branches = {
        str(item.get("feature_branch")) for item in feature_combinations
        if item.get("feature_branch")
    }
    if "combined" in branches:
        selected_branches = {
            str(values.get("feature_branch")) for values, feature in zip(parameters, feature_flags)
            if feature and values.get("feature_branch")
        }
        if not selected_branches.intersection(branches - {"combined"}):
            missing.append("single feature branch")
        if "combined" not in selected_branches:
            missing.append("combined feature branch")
    if missing:
        return (
            "FeaturePlanCoverageError: proposal is missing " + ", ".join(missing)
            + "; compare them under the same task defaults, split, seed, and resource budget"
        )
    return None


def augment_proposal_context(context: dict, skill_result: dict) -> dict:
    """Append bounded skill evidence without replacing planner context."""
    if not isinstance(context, dict) or not isinstance(skill_result, dict):
        raise ValueError("context and skill_result must be objects")
    forbidden = {
        "proposal", "proposal_id", "candidate", "candidate_id",
        "experiment", "rjob", "evaluator_result",
    }
    evidence = skill_result.get("evidence")
    if not isinstance(evidence, dict) or forbidden.intersection(evidence):
        raise ValueError("skill result is outside the bounded evidence contract")
    augmented = dict(context)
    existing = augmented.get("skill_injections", [])
    if not isinstance(existing, list):
        raise ValueError("skill_injections must be a list")
    augmented["skill_injections"] = [*existing, {
        "skill_id": skill_result.get("skill_id"),
        "status": skill_result.get("status"),
        "evidence": dict(evidence),
        "provenance": dict(skill_result.get("provenance") or {}),
    }]
    return augmented


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = Path(os.environ["HEURESIS_DIR"])
    sys.path.insert(0, str(root / "src"))
    load_env_file(Path(os.environ.get("HEURESIS_ENV_FILE", root / ".env")))

    from heuresis.boyue import BoyueClient, decode_first_json_value  # noqa: PLC0415

    context = json.loads(args.context.read_text(encoding="utf-8"))
    recover_raw = os.environ.get("HEURESIS_SUGGESTION_RECOVER_RAW")
    if recover_raw:
        raw_path = Path(recover_raw).resolve()
        suggestion = decode_first_json_value(raw_path.read_text(encoding="utf-8", errors="replace"))
        if not is_complete_proposal(suggestion):
            raise SystemExit(f"Recovered Heuresis raw response is not a complete proposal: {raw_path}")
        suggestion["round"] = context["round"]
        validation_error = proposal_contract_error(context, suggestion)
        if validation_error is not None:
            raise SystemExit(f"Recovered Heuresis raw response failed contract validation: {validation_error}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(suggestion, indent=2, ensure_ascii=False), encoding="utf-8")
        trace_path = args.out.with_suffix(".dataset_tool_trace.json")
        trace_path.write_text(json.dumps({
            "schema_version": "omni-ar-tool-trace/v1",
            "calls": [],
            "contract_validation_feedback": [],
            "recovered_from_raw": str(raw_path),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(args.out)
        return 0

    model = os.environ.get("BOYUE_MODEL_NAME") or os.environ.get("BOYUE_MODEL") or "gpt-4o-mini"
    client = BoyueClient(
        timeout_s=float(os.environ.get("HEURESIS_SUGGESTION_TIMEOUT_SEC", "180")),
        max_retries=int(os.environ.get("HEURESIS_SUGGESTION_MAX_RETRIES", "2")),
    )
    last_text = ""
    suggestion = None
    json_retries = max(1, int(os.environ.get("HEURESIS_SUGGESTION_JSON_RETRIES", "2")))
    tool_results: list[dict] = []
    validation_feedback: list[str] = []
    max_tool_calls = max(0, int(os.environ.get("HEURESIS_DATASET_TOOL_MAX_CALLS", "4")))
    attempt = 0
    while attempt < json_retries + max_tool_calls:
        response = client.generate_json(
            model=model,
            prompt=prompt_for(context, tool_results, validation_feedback),
            temperature=float(os.environ.get("HEURESIS_SUGGESTION_TEMPERATURE", "0.25")),
        )
        last_text = response.text
        try:
            suggestion = decode_first_json_value(response.text)
        except ValueError:
            suggestion = None
        if is_tool_call(suggestion):
            if len(tool_results) >= max_tool_calls:
                raise SystemExit("Heuresis exceeded the Dataset Adapter tool-call budget")
            dataset_ref = str((context.get("dataset") or {}).get("ref", ""))
            if not dataset_ref:
                raise SystemExit("Heuresis requested a dataset tool without a bound dataset")
            task_spec_path = Path(context["paths"]["task_spec"]).resolve()
            repository = task_spec_path.parents[2]
            if str(repository) not in sys.path:
                sys.path.insert(0, str(repository))
            from omni_ar.loop import ResearchLoop  # noqa: PLC0415

            result = ResearchLoop(repository).dataset_tool(
                dataset_ref,
                str(suggestion["arguments"].get("action", "")),
                call_id=suggestion.get("call_id", "dataset-call"),
                **{
                    key: value for key, value in suggestion["arguments"].items()
                    if key not in {"action", "dataset_ref"}
                },
            )["payload"]
            if any(item["arguments"] == result["arguments"] for item in tool_results):
                raise SystemExit("Heuresis repeated an identical Dataset Adapter tool call")
            tool_results.append(result)
            attempt += 1
            continue
        if is_complete_proposal(suggestion):
            suggestion["round"] = context["round"]
            validation_error = proposal_contract_error(context, suggestion)
            if validation_error is None:
                break
            validation_feedback.append(validation_error)
            raw_path = args.out.with_suffix(f".attempt_{attempt}.raw.txt")
            raw_path.write_text(response.text, encoding="utf-8", errors="replace")
            suggestion = None
            attempt += 1
            continue
        raw_path = args.out.with_suffix(f".attempt_{attempt}.raw.txt")
        raw_path.write_text(response.text, encoding="utf-8", errors="replace")
        attempt += 1
    if not is_complete_proposal(suggestion):
        args.out.with_suffix(".raw.txt").write_text(last_text, encoding="utf-8", errors="replace")
        raise SystemExit("Heuresis suggestion must be a complete omni-ar-proposal/v2 object")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(suggestion, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_path = args.out.with_suffix(".dataset_tool_trace.json")
    trace_path.write_text(json.dumps({
        "schema_version": "omni-ar-tool-trace/v1",
        "calls": tool_results,
        "contract_validation_feedback": validation_feedback,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
