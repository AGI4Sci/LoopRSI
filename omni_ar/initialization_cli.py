from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .initialization import InitializationError, RoughIdeaEngine, dry_run_pipeline
from .initialization.engine import render_brief


def _load_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InitializationError(f"expected mapping in {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_research")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="ask questions and create a confirmed rough_idea bundle")
    init.add_argument("--task", type=Path, help="optional task hint; the LLM checks it against the rough idea")
    init.add_argument(
        "--dataset", type=Path,
        help=(
            "unregistered dataset source to analyze with a reviewed template, "
            "a supplied Task Adapter, or the generated-Adapter route"
        ),
    )
    init.add_argument("--dataset-id", help="stable id for an automatically bootstrapped dataset")
    init.add_argument("--dataset-version", default="1")
    init.add_argument("--dataset-copy-mode", choices=("copy", "hardlink"), default="copy")
    init.add_argument("--rough-idea", help="initial research idea; avoids entering it twice during unknown-data onboarding")
    init.add_argument("--adapter-blueprint", type=Path, help="reviewed YAML/JSON blueprint for a dataset without a built-in template")
    adapter_source = init.add_mutually_exclusive_group()
    adapter_source.add_argument(
        "--task-adapter", type=Path,
        help="existing adapter.py file or directory to validate before automatic generation",
    )
    adapter_source.add_argument(
        "--generate-task-adapter", action="store_true",
        help="confirm that no Task Adapter is available and use the Coding Agent generation route",
    )
    init.add_argument("--adapter-agent", default="codex", help="Coding Agent command used for an untemplated Task Adapter")
    init.add_argument("--answers", type=Path)
    init.add_argument("--output-dir", type=Path)
    init.add_argument("--initialization-id")
    init.add_argument("--yes", action="store_true", help="confirm after validation")
    init.add_argument(
        "--question-model",
        help="Boyue model used for follow-up questions and, with --auto-run, later planning",
    )
    init.add_argument("--question-count", type=int, choices=range(3, 6), default=5)
    init.add_argument(
        "--auto-run", action="store_true",
        help="after final confirmation, continue through the unified multi-round loop",
    )
    init.add_argument("--research-output-dir", type=Path)
    init.add_argument("--rounds", type=int, default=3)
    init.add_argument("--strategy", choices=("omni_epic", "islands"), default="omni_epic")
    init.add_argument("--planning-mode", choices=("live", "dry_run"), default="live")
    init.add_argument(
        "--execution-mode", choices=("simulated", "local", "rjob_dry_run", "rjob"),
        default=None,
    )
    init.add_argument("--coding-mode", choices=("off", "plan", "apply"), default=None)
    init.add_argument("--target-accepted", type=int, default=0)
    init.add_argument("--patience", type=int, default=0)
    init.add_argument("--min-rounds", type=int, default=1)
    init.add_argument("--min-improvement", type=float, default=0.0)
    init.add_argument("--required-seeds", type=int, default=1)
    init.add_argument("--max-coding-requests-per-round", type=int, default=1)
    init.add_argument("--authorize-external-services", action="store_true")
    init.add_argument("--authorize-rjob", action="store_true")
    init.add_argument("--authorize-code-changes", action="store_true")
    init.add_argument("--max-attempts", type=int, choices=range(1, 6), default=2)
    preview = sub.add_parser("preview", help="validate and render an existing rough idea")
    preview.add_argument("--rough-idea", type=Path, required=True)
    dry = sub.add_parser("dry-run", help="compile Initialization -> ResearchStudio -> Heuresis without training")
    dry.add_argument("--rough-idea", type=Path, required=True)
    dry.add_argument("--output-dir", type=Path, required=True)
    live = sub.add_parser("live-run", help="run real literature retrieval and Boyue planning without training")
    live.add_argument("--rough-idea", type=Path, required=True)
    live.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    engine = RoughIdeaEngine(repository)
    try:
        if args.command == "preview":
            rough = _load_mapping(args.rough_idea)
            engine.validate(rough)
            print(render_brief(rough), end="")
            return 0
        if args.command == "dry-run":
            result = dry_run_pipeline(repository, args.rough_idea, args.output_dir)
            print(json.dumps({"status": "ok", "trace": str(result["trace"]), "execution_authorized": False}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "live-run":
            from .live_pipeline import run_live_pipeline

            trace = run_live_pipeline(repository, args.rough_idea, args.output_dir)
            print(json.dumps({"status": "ok", "trace": str(trace), "execution_authorized": False}, ensure_ascii=False, indent=2))
            return 0

        preloaded_answers = _load_mapping(args.answers) if args.answers else None
        qa_external_authorized = bool(getattr(args, "authorize_external_services", False))
        if args.dataset:
            if args.task:
                raise InitializationError("use either --dataset or --task, not both")
            from datasets.profiler import DatasetProfileError
            from .task_bootstrap import TaskBootstrapError, bootstrap_dataset_task
            try:
                bootstrap = bootstrap_dataset_task(
                    repository, args.dataset, dataset_id=args.dataset_id,
                    version=args.dataset_version, copy_mode=args.dataset_copy_mode,
                    run_baseline=True,
                )
            except (DatasetProfileError, TaskBootstrapError) as template_error:
                from datasets.structural_profiler import structural_profile
                from .universal_adapter import (
                    confirm_blueprint_interactive, generate_blueprint,
                    generate_universal_adapter, register_supplied_adapter,
                    validate_blueprint,
                )

                supplied_adapter = args.task_adapter
                if supplied_adapter is None and not args.generate_task_adapter:
                    if args.answers:
                        raise InitializationError(
                            "unknown-data onboarding must choose --task-adapter PATH or "
                            "--generate-task-adapter"
                        )
                    has_adapter = input(
                        "该数据没有内置模板。你能否提供现有的 Task Adapter？如果没有，系统可以自动生成。[y/N] "
                    ).strip().lower() in {"y", "yes"}
                    if has_adapter:
                        adapter_answer = input("请输入 adapter.py 文件或其目录路径：").strip()
                        if not adapter_answer:
                            raise InitializationError("Task Adapter path cannot be empty")
                        supplied_adapter = Path(adapter_answer)
                structural = structural_profile(args.dataset)
                rough_for_adapter = str(
                    args.rough_idea
                    or ((preloaded_answers or {}).get("rough_idea") or "")
                ).strip()
                if args.adapter_blueprint:
                    blueprint = validate_blueprint(repository, _load_mapping(args.adapter_blueprint))
                else:
                    if not rough_for_adapter:
                        rough_for_adapter = input(
                            "该数据没有内置模板。先用一两句话说明希望预测或生成什么："
                        ).strip()
                    if not rough_for_adapter:
                        raise InitializationError("unknown data requires a rough idea before Adapter generation")
                    if not qa_external_authorized:
                        qa_external_authorized = input(
                            "允许将粗略想法和有界数据结构摘要发送给 Boyue，用于生成 Adapter 蓝图？[y/N] "
                        ).strip().lower() in {"y", "yes"}
                    if not qa_external_authorized:
                        raise InitializationError(
                            "unknown data requires either --adapter-blueprint or authorization to generate one"
                        )
                    blueprint = generate_blueprint(
                        repository, structural, rough_for_adapter, model=args.question_model,
                    )
                if args.answers:
                    if not args.adapter_blueprint:
                        raise InitializationError(
                            "non-interactive unknown-data onboarding requires a reviewed --adapter-blueprint"
                        )
                else:
                    blueprint = confirm_blueprint_interactive(repository, blueprint)
                if supplied_adapter is not None:
                    generated = register_supplied_adapter(
                        repository, args.dataset.resolve(),
                        dataset_id=args.dataset_id or args.dataset.stem,
                        version=args.dataset_version, blueprint=blueprint,
                        adapter_path=supplied_adapter, copy_mode=args.dataset_copy_mode,
                    )
                else:
                    coding_mode = args.coding_mode
                    if coding_mode is None:
                        if args.answers:
                            coding_mode = "apply" if args.authorize_code_changes else "plan"
                        else:
                            coding_mode = (
                                "apply" if input(
                                    "没有现成 Adapter。允许 Coding Agent 在独立目录生成，并仅在全部检查通过后注册？[y/N] "
                                ).strip().lower() in {"y", "yes"} else "plan"
                            )
                    if coding_mode == "apply" and not args.authorize_code_changes and args.answers:
                        raise InitializationError(
                            "non-interactive unknown-data generation requires --authorize-code-changes"
                        )
                    generated = generate_universal_adapter(
                        repository, args.dataset.resolve(),
                        dataset_id=args.dataset_id or args.dataset.stem,
                        version=args.dataset_version, blueprint=blueprint,
                        copy_mode=args.dataset_copy_mode, mode="apply" if coding_mode == "apply" else "plan",
                        agent=args.adapter_agent, model=args.question_model,
                    )
                if generated.get("status") != "ok":
                    raise InitializationError(
                        "Adapter generation is pending and the task remains unregistered; "
                        f"template failure was: {template_error}; see {generated['pending_task_spec']}"
                    )
                bootstrap = generated
                args.rough_idea = rough_for_adapter
            args.task = Path(bootstrap["task_spec"])
            generated_spec = _load_mapping(repository / args.task)
            primary_name = generated_spec["metrics"]["primary"]["name"]
            baseline_value = ((bootstrap.get("baseline") or {}).get("metrics") or {}).get(primary_name)
            if baseline_value is None:
                acceptance = (bootstrap.get("acceptance") or {}).get("standardized_result")
                if acceptance:
                    accepted_result = json.loads(Path(acceptance).read_text(encoding="utf-8"))
                    baseline_value = ((accepted_result.get("metrics") or {}).get(primary_name) or {}).get("value")
            print(
                f"已分析并绑定新数据集：{bootstrap['dataset_ref']}；"
                f"自动 baseline {primary_name}={baseline_value:.4f}。"
            )

        if args.answers:
            if args.task is None:
                raise InitializationError("--task is required with --answers because no interactive task resolution runs")
            answers = preloaded_answers
            selected_task = args.task
        else:
            if not qa_external_authorized:
                qa_external_authorized = input(
                    "允许将接下来填写的粗略想法发送给已配置的 Boyue，用于生成和调整 QA 问题？[y/N] "
                ).strip().lower() in {"y", "yes"}
            if not qa_external_authorized:
                raise InitializationError("LLM-generated QA requires external-service authorization")
            answers = engine.ask(
                args.task, model=args.question_model, question_count=args.question_count,
                initial_rough_idea=args.rough_idea,
            )
            selected_task = repository / answers["_task_spec"] if answers.get("_task_spec") else None
        answers.setdefault("execution_policy", {})["external_services"] = qa_external_authorized
        if not any(
            item.get("question_id") == "authorize_external_services"
            for item in answers.get("_qa_transcript") or []
        ):
            engine.append_qa_event(
                answers, event="authorization", source="fixed_safety_check",
                question_id="authorize_external_services",
                question="是否允许将研究想法发送给已配置的外部规划服务？",
                raw_answer="explicit_cli_flag" if args.authorize_external_services else None,
                answer=qa_external_authorized,
            )
        if args.auto_run:
            interactive = args.answers is None
            if args.execution_mode is None and interactive:
                print("\nQA 确认后如何运行实验？\n  [1] 只检查 rjob 命令\n  [2] 自动提交真实 rjob")
                args.execution_mode = "rjob" if input("回答（回车只检查命令）：").strip() == "2" else "rjob_dry_run"
            args.execution_mode = args.execution_mode or "rjob_dry_run"
            if args.coding_mode is None and interactive:
                print("\n遇到需要新增代码的方法时如何处理？\n  [1] 自动修改允许的代码并检查\n  [2] 只生成修改计划\n  [3] 跳过代码方案")
                coding_answer = input("回答（回车只生成计划）：").strip()
                args.coding_mode = {"1": "apply", "3": "off"}.get(coding_answer, "plan")
            args.coding_mode = args.coding_mode or "plan"
            def authorize(flag: bool, prompt: str, required: bool, question_id: str) -> bool:
                if flag:
                    raw_value, value = "explicit_cli_flag", True
                elif not required or not interactive:
                    raw_value, value = None, False
                else:
                    raw_value = input(prompt).strip()
                    value = raw_value.lower() in {"y", "yes"}
                engine.append_qa_event(
                    answers, event="authorization", source="fixed_safety_check",
                    question_id=question_id, question=prompt.strip(),
                    raw_answer=raw_value, answer=value,
                    details={"required_by_selected_mode": required},
                )
                return value

            answers["execution_policy"] = {
                "external_services": authorize(
                    qa_external_authorized,
                    "允许将研究描述发送给已配置的 Boyue 和论文检索服务？[y/N] ",
                    args.planning_mode == "live", "authorize_external_services",
                ),
                "submit_rjob": authorize(
                    args.authorize_rjob,
                    "允许在确认后自动提交预算范围内的 rjob？[y/N] ",
                    args.execution_mode == "rjob", "authorize_rjob",
                ),
                "code_changes": authorize(
                    args.authorize_code_changes,
                    "允许 Coding Agent 修改任务允许的代码并运行检查？[y/N] ",
                    args.coding_mode == "apply", "authorize_code_changes",
                ),
                "auto_retry": True, "max_attempts": args.max_attempts,
                "max_rounds": args.rounds,
                "max_trials_per_round": 999,
                "max_gpus": 999,
                "max_runtime_minutes": 999999,
            }
            if args.planning_mode == "live" and not answers["execution_policy"]["external_services"]:
                raise InitializationError("live auto-run requires QA authorization for external services")
            if args.execution_mode == "rjob" and not answers["execution_policy"]["submit_rjob"]:
                raise InitializationError("real rjob auto-run requires QA authorization")
            if args.coding_mode == "apply" and not answers["execution_policy"]["code_changes"]:
                raise InitializationError("automatic code changes require QA authorization")
        task_slug = (
            selected_task.parent.name if selected_task is not None
            else re.sub(r"[^a-zA-Z0-9_.-]+", "-", str((answers.get("_task_profile") or {}).get("name") or "open-task")).strip("-") or "open-task"
        )
        init_id = args.initialization_id or f"{task_slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if selected_task is not None:
            def builder(confirmed: bool) -> dict:
                return engine.build(selected_task, answers, initialization_id=init_id, confirmed=confirmed)
        else:
            def builder(confirmed: bool) -> dict:
                return engine.build_open(answers, initialization_id=init_id, confirmed=confirmed)
        engine.constraint_sources(answers)
        draft = builder(False)
        confirmed = args.yes or bool(answers.get("confirmation", False))
        if confirmed:
            engine.append_qa_event(
                answers, event="confirmation", source="fixed_safety_check",
                question_id="final_confirmation",
                question="确认以上研究目标并允许进入 Idea Card 生成阶段？",
                raw_answer="--yes" if args.yes else "provided_answers",
                answer=True,
            )
            engine.constraint_sources(answers)
            rough = builder(True)
            print("\n" + render_brief(rough))
        else:
            print("\n" + render_brief(draft))
            if not draft["constraints"]["conflicts"]:
                confirmation_raw = input("确认以上研究目标并允许进入 Idea Card 生成阶段？[y/N] ").strip()
                confirmed = confirmation_raw.lower() in {"y", "yes"}
            else:
                confirmation_raw = None
            engine.append_qa_event(
                answers, event="confirmation", source="fixed_safety_check",
                question_id="final_confirmation",
                question="确认以上研究目标并允许进入 Idea Card 生成阶段？",
                raw_answer=confirmation_raw, answer=confirmed,
                details={"blocked_by_conflicts": bool(draft["constraints"]["conflicts"])},
            )
            engine.constraint_sources(answers)
            rough = builder(confirmed)
        output = args.output_dir or repository / "research_initializations" / init_id
        paths = engine.write_bundle(output.resolve(), answers, rough)
        response = {
            "status": "ok" if rough["confirmation"]["status"] == "confirmed" else "pending",
            "initialization_id": init_id,
            "task": rough["task"]["name"],
            "rough_idea": str(paths["rough_idea"]),
            "validation": str(paths["validation"]),
            "qa_transcript": str(paths["qa_transcript"]),
            "question_generation": str(paths["question_generation"])
            if paths.get("question_generation") else None,
        }
        if args.auto_run:
            if rough["confirmation"]["status"] != "confirmed":
                raise InitializationError("--auto-run requires final confirmation")
            from .autoresearch import run_autoresearch

            research_output = (
                args.research_output_dir.resolve()
                if args.research_output_dir
                else output.resolve() / "autoresearch"
            )
            previous_model = os.environ.get("OMNI_AR_BOYUE_MODEL")
            if args.question_model:
                os.environ["OMNI_AR_BOYUE_MODEL"] = args.question_model
            try:
                autoresearch_result = run_autoresearch(
                    repository, paths["rough_idea"], research_output,
                    rounds=args.rounds, strategy_name=args.strategy,
                    planning_mode=args.planning_mode, execution_mode=args.execution_mode,
                    coding_mode=args.coding_mode or "plan", target_accepted=args.target_accepted,
                    patience=args.patience,
                    min_rounds=args.min_rounds,
                    min_improvement=args.min_improvement,
                    required_seeds=args.required_seeds,
                    max_coding_requests_per_round=args.max_coding_requests_per_round,
                )
            finally:
                if args.question_model:
                    if previous_model is None:
                        os.environ.pop("OMNI_AR_BOYUE_MODEL", None)
                    else:
                        os.environ["OMNI_AR_BOYUE_MODEL"] = previous_model
            response["autoresearch_result"] = str(autoresearch_result)
            autoresearch_summary = json.loads(autoresearch_result.read_text(encoding="utf-8"))
            if autoresearch_summary.get("status") != "ok":
                raise InitializationError(
                    "automatic research loop terminated with failure; "
                    f"see {autoresearch_result}"
                )
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0 if rough["confirmation"]["status"] == "confirmed" else 2
    except (InitializationError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
