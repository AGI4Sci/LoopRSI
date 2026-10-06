"""CLI for the Hierarchical Looped RSI Researcher.

Subcommands:
  validate   --config <yaml|json>            topology + whitelist validation
  dry-run    --config <...> --out <dir>       deterministic local run
  run        --config <...> --worker {vcc25}  real run (cluster / rjob container)
  report     --manifest <jsonl>               summarize a previous run

Pure stdlib.  The YAML loader falls back to a minimal builtin parser when
PyYAML is unavailable (same approach as ``rsi_step0.validation``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experience import capacity_signal
from .routing import build_routing_table
from .layers import resolve_loop_range
from .layer_agent import RecordingDecisionBackend


# --------------------------------------------------------------------------- config loading
def load_yaml(path: str) -> Dict[str, Any]:
    """PyYAML if available; otherwise JSON configs are used (raise with hint)."""
    text = open(path, encoding="utf-8").read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except ImportError:
        pass
    except Exception:
        pass
    raise SystemExit(
        f"{path}: PyYAML is required to parse .yaml configs on this host; "
        "use the .json config instead (pure stdlib)"
    )


def load_config(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        return json.load(open(path, encoding="utf-8"))
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a mapping, got {type(data).__name__}")
    return data


# --------------------------------------------------------------------------- commands
def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    errors: List[str] = []
    try:
        resolve_loop_range(cfg.get("loop_range"))
    except ValueError as exc:
        errors.append(str(exc))
    try:
        table = build_routing_table(cfg.get("direct_whitelist"))
        for (s, d) in sorted(table.direct_edges):
            print(f"direct edge whitelisted: {s}->{d}")
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        print("INVALID")
        for e in errors:
            print(" -", e)
        return 1
    print("VALID")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    from . import HierarchicalLoop, DeterministicWorker

    cfg = load_config(args.config)
    if args.out:
        out = cfg.setdefault("out", {})
        out["dir"] = args.out
    loop = HierarchicalLoop(cfg, DeterministicWorker(seed=int((cfg.get("run") or {}).get("seed", 0))))
    errors = loop.validate()
    if errors:
        for e in errors:
            print("INVALID:", e)
        return 1
    summary = loop.run()
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from . import HierarchicalLoop
    from .vcc25_worker import Vcc25Worker
    from .knowledge_worker import KnowledgeDrivenVcc25Worker
    from .knowledge_bridge import KnowledgeBridge
    from ai4ai.plugin_manifest import load_task_plugin_manifest

    cfg = load_config(args.config)
    if args.out:
        out = cfg.setdefault("out", {})
        out["dir"] = args.out
    if args.worker == "knowledge_vcc25":
        from .memory import build_memory_manager
        from .layer_agent import HistoricalDecisionBackend
        plugin_path = Path(__file__).resolve().parents[1] / "tasks" / "vcc25" / "task_plugin.yaml"
        memory = build_memory_manager(cfg, (cfg.get("decisions") or {}).get("dir", "output/rsi_step0/rollout"))
        execution = Vcc25Worker(config=cfg, memory_manager=memory)
        worker = KnowledgeDrivenVcc25Worker(
            KnowledgeBridge(load_task_plugin_manifest(plugin_path)),
            HistoricalDecisionBackend(memory), execution_worker=execution,
        )
    else:
        worker = Vcc25Worker(config=cfg)
    loop = HierarchicalLoop(cfg, worker, memory_manager=memory if args.worker == "knowledge_vcc25" else None)
    errors = loop.validate()
    if errors:
        for e in errors:
            print("INVALID:", e)
        return 1
    summary = loop.run()
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows: List[Dict[str, Any]] = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    kinds: Dict[str, int] = {}
    routes: Dict[str, int] = {}
    direct: List[tuple] = []
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        routes[r["route"]] = routes.get(r["route"], 0) + 1
        if r["route"] == "direct":
            direct.append((r["from_layer"], r["to_layer"]))
    print(json.dumps({
        "messages": len(rows),
        "kinds": kinds,
        "routes": routes,
        "direct_edges": sorted(set(direct)),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    """P2: lightweight rollout (planning + reasoning only), write decisions."""
    from . import HierarchicalLoop
    from .decisions import DecisionWriter, count_decisions
    from .vcc25_worker import Vcc25Worker

    cfg = load_config(args.config)
    out = cfg.setdefault("out", {})
    out["dir"] = args.out or out.get("dir", "output/rsi_step0/hier")
    wdst = cfg.setdefault("decisions", {})
    wdst["dir"] = args.decisions or wdst.get("dir", "output/rsi_step0/rollout")
    wcfg = dict(cfg.setdefault("worker", {}))
    wcfg["mode"] = "rollout"
    cfg["worker"] = wcfg

    worker = Vcc25Worker(config=cfg)
    writer = DecisionWriter(wdst["dir"], shard=args.shard)
    loop = HierarchicalLoop(cfg, worker, run_id=args.run_id, decision_writer=writer)
    errors = loop.validate()
    if errors:
        for e in errors:
            print("INVALID:", e)
        return 1
    summary = loop.run()
    stats = count_decisions(wdst["dir"])
    print(json.dumps({
        "run_id": summary.run_id,
        "messages": len(summary.messages),
        "decisions": stats,
        "shard": args.shard,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """P3: build per-loop memory indexes from rollout decisions."""
    from .memory import build_memory_manager

    cfg = load_config(args.config)
    mem = cfg.setdefault("memory", {})
    mem.setdefault("enabled", True)
    mem.setdefault("baseline", 0.050227)
    mem.setdefault("out", {})
    mem["out"]["dir"] = args.memory or mem["out"].get("dir", "output/rsi_step0/memory")
    decisions_root = args.decisions or (cfg.get("decisions") or {}).get("dir", "output/rsi_step0/rollout")
    mgr = build_memory_manager(cfg, decisions_root)
    built = mgr.build_from_decisions()
    mgr.persist()
    print(json.dumps({"mode": "build", "per_layer": built,
                      "learn_enabled": bool((mem.get("learn") or {}).get("enabled", False))},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_knowledge_preflight(args: argparse.Namespace) -> int:
    from .knowledge_experiment import KnowledgeExperiment

    result = KnowledgeExperiment.from_config(args.config, args.workspace).run_preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_reproduce_papers(args: argparse.Namespace) -> int:
    from .paper_reproduction import run_paper_reproduction

    result = run_paper_reproduction(
        Path(args.experiment_dir), Path(args.output),
        source_manifest=Path(args.source_manifest) if args.source_manifest else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_gears_preflight(args: argparse.Namespace) -> int:
    from .gears_preflight import run_gears_preflight

    result = run_gears_preflight(
        Path(args.experiment_dir), Path(args.source_manifest), Path(args.output),
        Path(args.processed_dataset), Path(args.split_json), Path(args.gene2go),
        Path(args.gene_names),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="hier_loop", description="Hierarchical Looped RSI Researcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn, help_text in (
        ("validate", cmd_validate, "validate config topology"),
        ("dry-run", cmd_dry_run, "deterministic local run"),
        ("run", cmd_run, "real vcc25 run (cluster)"),
        ("report", cmd_report, "summarize manifest"),
        ("rollout", cmd_rollout, "P2 planning+reasoning rollout"),
        ("memory", cmd_memory, "P3 build per-loop memory"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--config", required=name in ("validate", "dry-run", "run", "rollout", "memory"))
        sp.add_argument("--out")
        if name == "run":
            sp.add_argument("--worker", choices=("vcc25", "knowledge_vcc25"), default="vcc25")
        if name == "report":
            sp.add_argument("--manifest", required=True)
        if name == "rollout":
            sp.add_argument("--decisions")
            sp.add_argument("--shard", default="shard_0000")
            sp.add_argument("--run-id")
        if name == "memory":
            sp.add_argument("--decisions")
            sp.add_argument("--memory")
        sp.set_defaults(func=fn)

    knowledge = sub.add_parser(
        "knowledge-preflight",
        help="run the five-layer knowledge injection preflight without GPU execution",
    )
    knowledge.add_argument("--config", required=True)
    knowledge.add_argument("--workspace", required=True)
    knowledge.set_defaults(func=cmd_knowledge_preflight)

    reproduction = sub.add_parser(
        "reproduce-papers",
        help="read eight VCC25 card triplets and run bounded remote source attempts",
    )
    reproduction.add_argument("--experiment-dir", required=True)
    reproduction.add_argument("--output", required=True)
    reproduction.add_argument("--source-manifest")
    reproduction.set_defaults(func=cmd_reproduce_papers)

    gears = sub.add_parser("gears-preflight", help="check pinned GEARS H1 execution inputs without reading expression")
    for name in ("experiment-dir", "source-manifest", "output", "processed-dataset",
                 "split-json", "gene2go", "gene-names"):
        gears.add_argument("--" + name, required=True)
    gears.set_defaults(func=cmd_gears_preflight)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
