#!/usr/bin/env python3
"""rjob driver for the Hierarchical Looped RSI Researcher on vcc25.

Two modes:
  inner  : run INSIDE the rjob GPU container. Loads the hier config, builds a
             HierarchicalLoop with the real Vcc25Worker, runs >=1 rounds of real
             vcc25 official H1 training/eval (pearson_delta) and writes a result
             manifest under ``out.dir``.
  submit : run ON the login node. Builds the ``rjob submit`` command per the
             documented params (namespace/group/image/mount/GPU/memory=64GB) and (optionally)
             submits and polls. With ``--dry-run`` it only prints the command.

After a successful run the result manifest records every round's best
``pearson_delta`` so the ">=2 rounds optimized" criterion is verifiable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_or_yaml(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    raise SystemExit(f"cannot load config {path!r} (use .json or install pyyaml)")




_VCC_ROOT = "/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1"
_H5PY_WHEELS = (
    "/mnt/shared-storage-gpfs2/beam-gpfs02/huangwenxuan/omni-ar-autoresearch/.runtime/wheels/"
    "h5py-3.11.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "/mnt/shared-storage-gpfs2/beam-gpfs02/huangwenxuan/omni-ar-autoresearch-backup/"
    "omni-ar-autoresearch/.runtime/lingshu-wheelhouse/"
    "h5py-3.16.0-cp312-cp312-manylinux_2_28_x86_64.whl",
)


def _ensure_runtime_py312(code_dir: str) -> str:
    """Return the py312 site-packages dir the candidate runners prepend to PYTHONPATH.

    The vcc25 implementation runners (run_official_h1_candidate_*.sh) expect
    ``tasks/vcc25/implementation/runtime_py312`` in PYTHONPATH for the image's
    Python 3.12.  The dir is not tracked in this repo; bootstrap it from a local
    cp312 h5py wheel when the dir (or the import) is missing.
    """
    runtime = os.path.join(code_dir, "tasks", "vcc25", "implementation", "runtime_py312")
    try:
        import h5py  # noqa: F401
        return runtime
    except Exception:
        pass
    if os.path.isdir(runtime) and os.path.isdir(os.path.join(runtime, "h5py")):
        return runtime
    os.makedirs(runtime, exist_ok=True)
    for wheel in _H5PY_WHEELS:
        if os.path.isfile(wheel):
            import zipfile
            with zipfile.ZipFile(wheel) as zf:
                for member in zf.namelist():
                    target = os.path.join(runtime, member)
                    if member.endswith("/"):
                        os.makedirs(target, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
            return runtime
    return runtime


def _ensure_runtime_onnx(code_dir: str) -> str:
    """Bootstrap onnxruntime (+ deps) into ``runtime_deps`` for the container.

    GPU nodes have no internet access; wheels are staged in ``runtime_wheels/``
    next to the project by the login node.  Extraction only happens when
    onnxruntime is not importable yet (mirrors the h5py bootstrap pattern) and
    each wheel is unzipped into a plain ``site-packages``-style directory that
    is prepended to PYTHONPATH.
    """
    deps = os.path.join(code_dir, "tasks", "vcc25", "implementation", "runtime_deps")
    try:
        import onnxruntime  # noqa: F401
        return deps
    except Exception:
        pass
    wheels_dir = os.path.join(code_dir, "runtime_wheels")
    if not os.path.isdir(wheels_dir):
        return deps
    os.makedirs(deps, exist_ok=True)
    import zipfile
    for name in sorted(os.listdir(wheels_dir)):
        if not name.endswith(".whl"):
            continue
        with zipfile.ZipFile(os.path.join(wheels_dir, name)) as zf:
            for member in zf.namelist():
                target = os.path.join(deps, member)
                if member.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    return deps


def _configure_runtime_env(code_dir: str) -> None:
    """Augment PYTHONPATH so the official_h1 runner can import h5py (py312) and
    the editable ``cell_eval`` package from the vcc25 data root."""
    parts = [os.environ.get("PYTHONPATH", "")]
    runtime = _ensure_runtime_py312(code_dir)
    if os.path.isdir(runtime):
        parts.append(runtime)
    runtime_onnx = _ensure_runtime_onnx(code_dir)
    if os.path.isdir(runtime_onnx):
        parts.append(runtime_onnx)
    cell_eval_src = os.path.join(_VCC_ROOT, "protocols", "lingshu_cell_h1", "tools", "cell-eval", "src")
    if os.path.isdir(cell_eval_src):
        parts.append(cell_eval_src)
    if os.path.isdir(code_dir):
        parts.append(code_dir)
    os.environ["PYTHONPATH"] = os.pathsep.join([p for p in parts if p])

# ------------------------------------------------------------------------- inner
def run_inner(args: argparse.Namespace) -> int:
    from hier_loop import HierarchicalLoop
    from hier_loop.vcc25_worker import Vcc25Worker
    from hier_loop.layers import resolve_loop_range

    _configure_runtime_env(str(Path(__file__).resolve().parent))

    cfg = _load_json_or_yaml(args.config)
    wcfg = dict(cfg.setdefault("worker", {}))
    wcfg["mode"] = "real"
    wcfg["adapter_module"] = wcfg.get("adapter_module", "tasks.vcc25.official_h1_adapter")
    wcfg["result_root"] = wcfg.get("result_root", args.result_root)
    cfg["worker"] = wcfg

    out_dir = ((cfg.get("out") or {}).get("dir")) or "output/rsi_step0/hier"
    out_dir = args.out or out_dir
    cfg.setdefault("out", {})["dir"] = out_dir

    start = time.time()
    result = {
        "command": "rjob_hier_vcc25",
        "created_at": _now(),
        "config": os.path.abspath(args.config),
        "status": "failed",
        "host": os.uname().nodename,
        "error": None,
        "rounds": [],
    }

    # --- P1/P2/P3 hooks -----------------------------------------------------
    from hier_loop.experience import CardRepository
    from hier_loop.decisions import DecisionWriter
    from hier_loop.memory import build_memory_manager

    # Per-layer experience-card repo: out/experience_cards/<layer>.jsonl
    cards_root = ((cfg.get("experience_cards") or {}).get("dir")
                  or os.path.join(out_dir, "..", "experience_cards"))
    cards_root = args.experience_cards or cards_root
    card_repo = CardRepository(cards_root)

    # Decision trajectory writer (real runs still emit decision records).
    dec_dir = args.decisions or (cfg.get("decisions") or {}).get("dir", "output/rsi_step0/rollout")
    decision_writer = DecisionWriter(dec_dir, shard=f"run_{args.run_id or 'hier'}")

    # Memory manager: only active when config memory.enabled = True.
    memory_manager = None
    mem_cfg = cfg.get("memory") or {}
    if mem_cfg.get("enabled"):
        memory_manager = build_memory_manager(cfg, dec_dir)

    loop = HierarchicalLoop(
        cfg, Vcc25Worker(config=cfg, memory_manager=memory_manager), run_id=args.run_id,
        decision_writer=decision_writer,
        card_repo=card_repo,
        memory_manager=memory_manager,
    )
    errors = loop.validate()
    if errors:
        result["error"] = "config invalid: " + "; ".join(errors)
        _write(result, loop)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    summary = loop.run()
    per_layer = summary.to_dict().get("per_layer", {})
    best_score = summary.best_score
    best_layer = summary.best_layer
    n_msgs = len(summary.messages)
    n_cards = len(summary.cards)

    result["status"] = "succeeded" if (best_score is not None and best_score > 0) else "completed_no_metric"
    result["n_messages"] = n_msgs
    result["n_experience_cards"] = n_cards
    result["best_score"] = best_score
    result["best_layer"] = best_layer
    result["memory"] = {
        "enabled": bool((cfg.get("memory") or {}).get("enabled", False)),
        "usage": dict(summary.memory_usage) if hasattr(summary, "memory_usage") else {},
        "n_decisions": (
            decision_writer.count if decision_writer is not None else 0
        ),
    }
    result["rounds"] = []
    for lid, v in per_layer.items():
        for r in (v.get("rounds") or []):
            rb = r.get("best_score")
            result["rounds"].append({
                "layer": lid, "round": r.get("round"),
                "best_score": rb,
                "exit_reason": r.get("exit_reason"),
                "status": "pass" if (rb is not None and rb > 0) else "fail",
            })
    result["n_real_l5_rounds"] = sum(1 for r in result["rounds"] if r["layer"] == "L5" and r["status"] == "pass")
    result["elapsed_seconds"] = round(time.time() - start, 3)
    result["manifest"] = os.path.join(out_dir, (cfg.get("out") or {}).get("manifest", "hier_progress.jsonl"))
    _write(result, loop)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "succeeded" else 2


def _write(result: dict, loop) -> None:
    out_dir = ((loop.out_cfg if hasattr(loop, "out_cfg") else {}) or {}).get("dir", "output/rsi_step0/hier")
    p = Path(out_dir) / "rjob_hier_result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------- submit
def rjob_command(args: argparse.Namespace) -> list[str]:
    inner = (
        f"cd {args.code_dir} && PYTHONPATH={args.code_dir} "
        f"python rjob_hier_vcc25.py inner --config {args.config} --out {args.out}"
        + (f" --run-id {args.run_id}" if args.run_id else "")
    )
    mounts = args.mount
    base = [
        "rjob", "submit",
        "--namespace", args.namespace,
        "--name", args.name,
        "--folder", args.folder,
        "--charged-group", args.charged_group,
        "--private-machine", args.private_machine,
        "--image", args.image,
        "--cpu", str(args.cpu),
        "--gpu", str(args.gpu),
        "--memory", str(args.memory),
        "--restart-policy", "never",
    ]
    for m in mounts:
        base += ["--mount", m]
    base += ["--", "bash", "-c", inner]
    return base


def submit(args: argparse.Namespace) -> int:
    cmd = rjob_command(args)
    print("CMD:", " ".join(cmd))
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "command": cmd}, ensure_ascii=False, indent=2))
        return 0
    env = dict(os.environ)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr[-3000:])
    if proc.returncode != 0:
        print("submit exit", proc.returncode)
        return 1
    print("SUBMITTED")
    return 0


def run_ab(args: argparse.Namespace) -> int:
    """P4: submit a matched A/B pair (no-memory vs memory) with identical
    seeds/loop_range/rounds/budgets, then optionally run the self-improvement
    comparison once both arms report."""
    import time as _t

    base = rjob_command
    cfg_a = _load_json_or_yaml(args.config_a)
    cfg_b = _load_json_or_yaml(args.config_b)
    # Normalize resource / budget parity (B must not get more search time).
    b_worker = cfg_b.setdefault("worker", {})
    a_worker = cfg_a.get("worker") or {}
    for k in ("trial_timeout_seconds", "trial_defaults"):
        if k in a_worker:
            b_worker[k] = a_worker[k]
    for block in ("run", "loop_range", "reward", "memory"):
        if block == "memory":
            cfg_a.setdefault("memory", {})["enabled"] = False
            cfg_b.setdefault("memory", {})["enabled"] = True
            # same capacity/retrieval knobs; keep baseline in both
            for k in ("capacity_threshold", "retrieval", "learn"):
                if k in (cfg_a.get("memory") or {}):
                    cfg_b["memory"][k] = cfg_a["memory"][k]
        else:
            if block in cfg_a and block not in cfg_b:
                cfg_b[block] = cfg_a[block]

    # Write normalized arm configs into the package folder so inner runs read them.
    import tempfile, os
    pkg_dir = args.pkg_dir
    pa = os.path.join(pkg_dir, "hier_ab_A.json")
    pb = os.path.join(pkg_dir, "hier_ab_B.json")
    Path(pa).parent.mkdir(parents=True, exist_ok=True)
    Path(pa).write_text(json.dumps(cfg_a, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(pb).write_text(json.dumps(cfg_b, ensure_ascii=False, indent=2), encoding="utf-8")

    ns_a = argparse.Namespace(**{
        "config": pa, "code_dir": args.code_dir, "name": args.name_a,
        "namespace": args.namespace, "charged_group": args.charged_group,
        "private_machine": args.private_machine, "image": args.image,
        "folder": args.folder, "cpu": args.cpu, "gpu": args.gpu,
        "memory": args.memory, "mount": args.mount, "dry_run": args.dry_run,
        "out": os.path.join(args.out, args.name_a), "run_id": args.name_a,
    })
    ns_b = argparse.Namespace(**{
        "config": pb, "code_dir": args.code_dir, "name": args.name_b,
        "namespace": args.namespace, "charged_group": args.charged_group,
        "private_machine": args.private_machine, "image": args.image,
        "folder": args.folder, "cpu": args.cpu, "gpu": args.gpu,
        "memory": args.memory, "mount": args.mount, "dry_run": args.dry_run,
        "out": os.path.join(args.out, args.name_b), "run_id": args.name_b,
    })
    cmd_a = rjob_command(ns_a)
    cmd_b = rjob_command(ns_b)
    print("CMD A:", " ".join(cmd_a))
    print("CMD B:", " ".join(cmd_b))
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "cmd_a": cmd_a, "cmd_b": cmd_b}, ensure_ascii=False, indent=2))
        return 0

    env = dict(os.environ)
    for tag, cmd in (("A", cmd_a), ("B", cmd_b)):
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        print(f"SUBMIT {tag} rc={proc.returncode}")
        if proc.stdout:
            print(proc.stdout[-2000:])
        if proc.stderr:
            print("STDERR:", proc.stderr[-2000:])
        if proc.returncode != 0:
            return 1
    print("SUBMITTED A/B")
    return 0


def show_status(args: argparse.Namespace) -> int:
    """Query rjob for hier-related jobs; prints a compact table."""
    cmd = ["rjob", "list"]
    env = dict(os.environ)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    print(proc.stdout[-6000:])
    if proc.stderr:
        print("STDERR:", proc.stderr[-2000:])
    return 0 if proc.returncode == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hierarchical Looped RSI vcc25 rjob driver")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_inner = sub.add_parser("inner")
    p_inner.add_argument("--config", required=True)
    p_inner.add_argument("--out", default="output/rsi_step0/hier")
    p_inner.add_argument("--run-id")
    p_inner.add_argument("--result-root", default="output/vcc25/hier")
    p_inner.add_argument("--decisions", default="output/rsi_step0/rollout")
    p_inner.add_argument("--experience-cards", default="")
    p_inner.set_defaults(fn=run_inner)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--config", required=True)
    p_sub.add_argument("--code-dir", required=True)
    p_sub.add_argument("--name", default="hier-vcc25")
    p_sub.add_argument("--namespace", default="ailab-deepdivegzy")
    p_sub.add_argument("--charged-group", default="deepdivegzy_gpu_pool")
    p_sub.add_argument("--private-machine", default="group")
    p_sub.add_argument("--image", default="registry.h.pjlab.org.cn/ailab-deepdivegzy-deepdivegzy_gpu/slime:v0")
    p_sub.add_argument("--folder", default="/mnt/shared-storage-user/gaozhangyang/RSI/rjob_packages/hier_vcc25")
    p_sub.add_argument("--cpu", type=int, default=8)
    p_sub.add_argument("--gpu", type=int, default=1)
    p_sub.add_argument("--memory", type=int, default=64000)
    p_sub.add_argument("--mount", action="append", default=[
        "gpfs://gpfs1/gaozhangyang:/mnt/shared-storage-user/gaozhangyang",
        "gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02",
    ], help="repeatable; each becomes a --mount flag")
    p_sub.add_argument("--dry-run", action="store_true")
    p_sub.add_argument("--out", default="output/rsi_step0/hier")
    p_sub.add_argument("--run-id")
    p_sub.set_defaults(fn=submit)

    p_ab = sub.add_parser("run-ab")
    p_ab.add_argument("--config-a", required=True)
    p_ab.add_argument("--config-b", required=True)
    p_ab.add_argument("--pkg-dir", default="rjob_packages/hier_vcc25")
    p_ab.add_argument("--name-a", default="hier-vcc25-no-mem-r18")
    p_ab.add_argument("--name-b", default="hier-vcc25-mem-r18")
    p_ab.add_argument("--code-dir", required=True)
    p_ab.add_argument("--namespace", default="ailab-deepdivegzy")
    p_ab.add_argument("--charged-group", default="deepdivegzy_gpu_pool")
    p_ab.add_argument("--private-machine", default="group")
    p_ab.add_argument("--image", default="registry.h.pjlab.org.cn/ailab-deepdivegzy-deepdivegzy_gpu/slime:v0")
    p_ab.add_argument("--folder", default="/mnt/shared-storage-user/gaozhangyang/RSI/rjob_packages/hier_vcc25")
    p_ab.add_argument("--cpu", type=int, default=8)
    p_ab.add_argument("--gpu", type=int, default=1)
    p_ab.add_argument("--memory", type=int, default=64000)
    p_ab.add_argument("--mount", action="append", default=[
        "gpfs://gpfs1/gaozhangyang:/mnt/shared-storage-user/gaozhangyang",
        "gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02",
    ])
    p_ab.add_argument("--dry-run", action="store_true")
    p_ab.add_argument("--out", default="output/rsi_step0/hier")
    p_ab.set_defaults(fn=run_ab)

    p_status = sub.add_parser("status")
    p_status.set_defaults(fn=show_status)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
