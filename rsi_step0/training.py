"""RSI Step0 training layer: SFT/RL dataset builders, PiTrainer, rollout and checkpoints.

Pure stdlib (Python 3.10+). ``torch`` is optional: importing this module never
touches torch; backend ``"torch"`` attempts the import at call time and raises
``RuntimeError("torch not available")`` when it is missing, while backend
``"offline"`` always works without torch (hard requirement for remote runs).

Sibling modules ``rsi_step0.reward`` / ``rsi_step0.router`` /
``rsi_step0.trajectory`` may be written concurrently by other agents, so they are
never imported at module level; every optional dependency is imported lazily
inside the function that needs it.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import rsi_step0.contracts as C

CHECKPOINT_KIND = "rsi_step0/checkpoint/v1"
FEATURE_DIM = 16


# --------------------------------------------------------------------------- jsonl io
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping unparseable lines."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _write_jsonl(records: List[Dict[str, Any]], path: str) -> int:
    """Write records as JSONL; returns the number of lines written."""
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def _write_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def _record_operator(rec: Dict[str, Any]) -> Optional[str]:
    """Resolve the operator of a record (trajectory or SFT shaped)."""
    op = rec.get("operator")
    if not op:
        action = rec.get("action")
        if isinstance(action, dict):
            op = action.get("operator")
    return op if isinstance(op, str) else None


def _task_id_of(rec: Dict[str, Any]) -> str:
    context = rec.get("context")
    if isinstance(context, dict) and context.get("task_id"):
        return str(context["task_id"])
    if rec.get("task_id"):
        return str(rec["task_id"])
    return "unknown"


def _serializable(obj: Any) -> Any:
    """Best-effort -> JSON serializable (drops non-serializable leaves)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            try:
                json.dumps(v)
                out[str(k)] = v
            except (TypeError, ValueError):
                out[str(k)] = _serializable(v)
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            try:
                json.dumps(v)
                out.append(v)
            except (TypeError, ValueError):
                out.append(_serializable(v))
        return out
    return str(obj)


def _features(context: Dict[str, Any], action: Dict[str, Any], dim: int = FEATURE_DIM) -> List[float]:
    """Deterministic stdlib feature vector from (context, action) hashes."""
    blob = C.hash_json({"context": _serializable(context), "action": _serializable(action)})
    raw = bytes.fromhex(blob)
    return [raw[i] / 255.0 for i in range(dim)]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- sft / rl datasets
def build_sft_dataset(traj_dir: str, out_dir: str) -> dict:
    """Build an SFT dataset from trajectory ``records.jsonl``.

    Only ``label == "positive"`` records are kept; each output record is
    ``{task_id, context, action}``. Writes ``out_dir/sft_records.jsonl`` and
    ``out_dir/manifest.json`` (stats + ``hash_json`` data fingerprint).
    """
    traj = Path(traj_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = traj / "records.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"trajectory records not found: {src}")
    raw = _read_jsonl(str(src))

    tasks: Dict[str, int] = {}
    op_counts: Dict[str, int] = {}
    n_positive = 0
    sft_records: List[Dict[str, Any]] = []
    for rec in raw:
        label = rec.get("label")
        if label != "positive":
            continue
        n_positive += 1
        task_id = _task_id_of(rec)
        sft_records.append(
            {"task_id": task_id, "context": rec.get("context") or {},
             "action": rec.get("action") or {}}
        )
        tasks[task_id] = tasks.get(task_id, 0) + 1
        op = _record_operator(rec)
        if op is not None:
            op_counts[op] = op_counts.get(op, 0) + 1

    record_path = out / "sft_records.jsonl"
    manifest_path = out / "manifest.json"
    _write_jsonl(sft_records, str(record_path))
    manifest = {
        "schema_version": C.SCHEMA_VERSION,
        "dataset": "sft",
        "source": str(src),
        "num_records": len(raw),
        "num_positive": n_positive,
        "num_sft_records": len(sft_records),
        "operator_counts": dict(sorted(op_counts.items())),
        "tasks": dict(sorted(tasks.items())),
        "fingerprint": C.hash_json(sft_records),
        "created_at": _now_iso(),
    }
    _write_json(manifest, str(manifest_path))
    return {
        "dataset": "sft",
        "record_path": str(record_path),
        "manifest_path": str(manifest_path),
        "num_records": len(raw),
        "num_positive": n_positive,
        "num_sft_records": len(sft_records),
        "tasks": dict(sorted(tasks.items())),
        "operators": dict(sorted(op_counts.items())),
        "manifest": manifest,
        "records": sft_records,
    }


def build_rl_dataset(traj_dir: str, out_dir: str) -> dict:
    """Build an RL context dataset from trajectory ``records.jsonl``.

    Keeps every labeled record; writes ``out_dir/rl_contexts.jsonl`` with parent
    provenance info and ``out_dir/manifest.json``.
    """
    traj = Path(traj_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = traj / "records.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"trajectory records not found: {src}")
    raw = _read_jsonl(str(src))

    label_counts: Dict[str, int] = {}
    op_counts: Dict[str, int] = {}
    tasks: Dict[str, int] = {}
    rl_records: List[Dict[str, Any]] = []
    for rec in raw:
        label = rec.get("label") or "unknown"
        label_counts[label] = label_counts.get(label, 0) + 1
        op = _record_operator(rec)
        if op is not None:
            op_counts[op] = op_counts.get(op, 0) + 1
        task_id = _task_id_of(rec)
        tasks[task_id] = tasks.get(task_id, 0) + 1
        provenance = rec.get("provenance") or {}
        context = rec.get("context") or {}
        parent = {
            "parent_action_id": provenance.get("parent_action_id"),
            "parent_outcome": context.get("last_outcome"),
            "round": rec.get("round"),
            "step": rec.get("step"),
        }
        rl_records.append(
            {"task_id": task_id, "context": context, "action": rec.get("action") or {},
             "outcome": rec.get("outcome") or {}, "reward": rec.get("reward") or {},
             "label": label, "parent": parent, "provenance": provenance}
        )

    record_path = out / "rl_contexts.jsonl"
    manifest_path = out / "manifest.json"
    _write_jsonl(rl_records, str(record_path))
    manifest = {
        "schema_version": C.SCHEMA_VERSION,
        "dataset": "rl",
        "source": str(src),
        "num_records": len(raw),
        "num_rl_records": len(rl_records),
        "label_counts": dict(sorted(label_counts.items())),
        "operator_counts": dict(sorted(op_counts.items())),
        "tasks": dict(sorted(tasks.items())),
        "fingerprint": C.hash_json(rl_records),
        "created_at": _now_iso(),
    }
    _write_json(manifest, str(manifest_path))
    return {
        "dataset": "rl",
        "record_path": str(record_path),
        "manifest_path": str(manifest_path),
        "num_records": len(raw),
        "num_rl_records": len(rl_records),
        "labels": dict(sorted(label_counts.items())),
        "tasks": dict(sorted(tasks.items())),
        "operators": dict(sorted(op_counts.items())),
        "manifest": manifest,
        "records": rl_records,
    }


# --------------------------------------------------------------------------- helpers
def _group_advantage(rewards: List[float]) -> List[float]:
    """Deterministic stdlib group advantage: z-score with 1e-9 std floor."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-9:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def _reward_module() -> Any:
    """Lazily import the reward module; None when it does not exist yet."""
    try:
        import rsi_step0.reward as R
        return R
    except Exception:
        return None


def _group_advantage_with_R(rewards: List[float]) -> List[float]:
    """group_advantage via R when available, else the stdlib fallback."""
    R = _reward_module()
    if R is not None and hasattr(R, "group_advantage"):
        try:
            adv = R.group_advantage(list(rewards))
            return [float(a) for a in adv]
        except TypeError:
            pass
        except Exception:
            pass
    return _group_advantage(rewards)


def _policy_from_counts(counts: Dict[str, int], epsilon: float = 0.1) -> Dict[str, float]:
    """Normalized policy weights over W2_TRAIN_OPERATORS from observed counts."""
    ops = list(C.W2_TRAIN_OPERATORS)
    total = sum(counts.get(op, 0) for op in ops)
    k = len(ops)
    weights: Dict[str, float] = {}
    for op in ops:
        c = counts.get(op, 0)
        if total > 0:
            weights[op] = (c + epsilon) / (total + k * epsilon)
        else:
            weights[op] = 1.0 / k
    return weights


def _write_checkpoint(payload: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Write a checkpoint.json and return its metadata dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in payload.items() if k != "hash"}
    digest = C.hash_json(_serializable(base))
    payload = dict(payload)
    payload["hash"] = digest
    path = out / "checkpoint.json"
    _write_json(payload, str(path))
    meta = dict(payload)
    meta["path"] = str(path)
    return meta


def make_checkpoint(weights: dict, out_dir: str) -> dict:
    """Write ``out_dir/checkpoint.json`` embedding ``weights``; return metadata."""
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    payload = {
        "kind": CHECKPOINT_KIND,
        "backend": "offline",
        "weights": _serializable(weights),
        "manifest": {
            "schema_version": C.SCHEMA_VERSION,
            "n_weight_keys": len(weights),
            "created_at": _now_iso(),
        },
        "hash": None,
    }
    meta = _write_checkpoint(payload, out_dir)
    return meta


# --------------------------------------------------------------------------- PiTrainer
class PiTrainer:
    """Behavior-clone (SFT) and policy-gradient (RL) trainer.

    ``backend="offline"``: packs records/rollouts into checkpoints without any
    real training; works everywhere, including torch-less remotes.
    ``backend="torch"``: trains real (tiny, deterministic) models; requires
    torch at call time or raises ``RuntimeError("torch not available")``.
    """

    valid_backends = ("offline", "torch")

    def __init__(self, backend: str = "offline", seed: int = 0, config: Optional[dict] = None) -> None:
        if backend not in self.valid_backends:
            raise ValueError(f"backend must be one of {self.valid_backends}, got {backend!r}")
        self.backend = backend
        self.seed = int(seed)
        self.config = dict(config) if config else {}

    # -- sft ----------------------------------------------------------------
    def train_sft(self, dataset: dict, out_dir: str) -> dict:
        """Train/fit a behavior-clone policy from an SFT dataset dict.

        ``dataset`` must contain ``records`` (list of trajectory or
        ``{task_id, context, action}`` records) or ``record_path``.
        Offline: writes ``checkpoint.json`` with W2 operator stats + policy
        weights + manifest + hash. Torch: trains a small linear BC classifier
        over deterministic (context, action) features and saves ``checkpoint.pt``.
        """
        records = self._records_from_dataset(dataset)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        w2_counts: Dict[str, int] = {op: 0 for op in C.W2_TRAIN_OPERATORS}
        other_counts: Dict[str, int] = {}
        for rec in records:
            op = _record_operator(rec)
            if op in C.W2_TRAIN_OPERATORS:
                w2_counts[op] += 1
            elif op is not None:
                other_counts[op] = other_counts.get(op, 0) + 1

        train_records = [
            rec for rec in records if _record_operator(rec) in C.W2_TRAIN_OPERATORS
        ]
        weights = _policy_from_counts(w2_counts)
        dataset_hash = C.hash_json([_serializable(rec) for rec in records])
        manifest = {
            "schema_version": C.SCHEMA_VERSION,
            "backend": self.backend,
            "seed": self.seed,
            "num_records": len(records),
            "num_train_samples": len(train_records),
            "w2_operator_counts": dict(sorted(w2_counts.items())),
            "other_operator_counts": dict(sorted(other_counts.items())),
            "dataset_hash": dataset_hash,
            "note": "offline: no real model trained" if self.backend == "offline"
                    else "torch: linear behavior-clone classifier",
        }
        payload = {
            "kind": CHECKPOINT_KIND,
            "backend": self.backend,
            "policy_kind": "behavior_clone_policy",
            "policy": {
                "operator_counts": dict(sorted(w2_counts.items())),
                "weights": dict(sorted(weights.items())),
            },
            "manifest": manifest,
            "hash": None,
        }

        if self.backend == "torch":
            return self._train_sft_torch(train_records, w2_counts, other_counts, payload, out)

        meta = _write_checkpoint(payload, str(out))
        meta["num_records"] = len(records)
        meta["num_train_samples"] = len(train_records)
        return meta

    def _train_sft_torch(self, train_records, w2_counts, other_counts, payload, out):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("torch not available") from exc
        torch.manual_seed(self.seed)
        op_index = {op: i for i, op in enumerate(C.W2_TRAIN_OPERATORS)}
        xs, ys = [], []
        for rec in train_records:
            op = _record_operator(rec)
            xs.append(_features(rec.get("context") or {}, rec.get("action") or {}))
            ys.append(op_index[op])
        if not ys:
            payload["manifest"]["note"] = "torch: no W2 samples, skipped training"
            return _write_checkpoint(payload, str(out))

        feat_dim = len(xs[0])
        n_classes = len(C.W2_TRAIN_OPERATORS)
        model = torch.nn.Linear(feat_dim, n_classes)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()
        X = torch.tensor(xs, dtype=torch.float32)
        Y = torch.tensor(ys, dtype=torch.long)
        iters = int(self.config.get("sft_iters", 100))
        for _ in range(iters):
            opt.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            opt.step()
        state = {k: v.detach() for k, v in model.state_dict().items()}
        meta = {"backend": "torch", "seed": self.seed, "n_classes": n_classes,
                "feat_dim": feat_dim, "num_train_samples": len(train_records),
                "operators": list(C.W2_TRAIN_OPERATORS)}
        pt_path = out / "checkpoint.pt"
        torch.save({"state_dict": state, "meta": meta}, str(pt_path))
        payload["artifact"] = "checkpoint.pt"
        payload["policy"]["trained"] = meta
        meta_out = _write_checkpoint(payload, str(out))
        meta_out["artifact_path"] = str(pt_path)
        return meta_out

    @staticmethod
    def _records_from_dataset(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(dataset, dict):
            raise TypeError("dataset must be a dict")
        records = dataset.get("records")
        if records is None and dataset.get("record_path"):
            records = _read_jsonl(str(dataset["record_path"]))
        if records is None:
            raise ValueError("dataset must contain 'records' list or 'record_path'")
        return [rec for rec in records if isinstance(rec, dict)]

    # -- rl ----------------------------------------------------------------
    def train_rl(self, rollout_groups: list, out_dir: str) -> dict:
        """Update policy weights from per-group advantages.

        Offline: records advantage-summed operator stats and derives updated
        policy weights. Torch: simple policy gradient over features using
        ``loss = -log p(op | context, response) * advantage``.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        op_adv: Dict[str, float] = {op: 0.0 for op in C.W2_TRAIN_OPERATORS}
        op_seen: Dict[str, int] = {op: 0 for op in C.W2_TRAIN_OPERATORS}
        for group_index, group in enumerate(rollout_groups or []):
            if not isinstance(group, dict):
                continue
            op = group.get("operator")
            if op not in C.W2_TRAIN_OPERATORS:
                op = group.get("operator") or "draft"
            if op not in C.W2_TRAIN_OPERATORS:
                continue
            context = group.get("context") or {}
            for sample in group.get("samples") or []:
                if not isinstance(sample, dict):
                    continue
                adv = float(sample.get("advantage", 0.0) or 0.0)
                reward = float(sample.get("reward", 0.0) or 0.0)
                op_adv[op] += adv
                op_seen[op] += 1
                rows.append({"group": group_index, "operator": op, "context": context,
                             "response": sample.get("response"), "reward": reward,
                             "advantage": adv})

        lr = float(self.config.get("rl_lr", 0.1))
        base_counts = {op: op_seen[op] for op in C.W2_TRAIN_OPERATORS}
        base_weights = _policy_from_counts(base_counts)
        updated: Dict[str, float] = {}
        for op in C.W2_TRAIN_OPERATORS:
            logit_raw = base_weights[op] + lr * max(0.0, op_adv[op])
            updated[op] = logit_raw
        denom = sum(updated.values()) or 1.0
        updated_weights = {op: v / denom for op, v in updated.items()}
        payload = {
            "kind": CHECKPOINT_KIND,
            "backend": self.backend,
            "policy_kind": "rl_policy",
            "policy": {
                "advantage_sums": {op: round(v, 6) for op, v in sorted(op_adv.items())},
                "sample_counts": dict(sorted(base_counts.items())),
                "weights": dict(sorted(updated_weights.items())),
            },
            "manifest": {
                "schema_version": C.SCHEMA_VERSION,
                "backend": self.backend,
                "seed": self.seed,
                "num_groups": len(rollout_groups or []),
                "num_samples": len(rows),
                "rl_lr": lr,
                "note": "offline: advantage stats recorded only"
                        if self.backend == "offline" else "torch: policy gradient update",
            },
            "hash": None,
        }
        if self.backend == "torch":
            return self._train_rl_torch(rows, payload, out)
        meta = _write_checkpoint(payload, str(out))
        meta["num_groups"] = len(rollout_groups or [])
        meta["num_samples"] = len(rows)
        return meta

    def _train_rl_torch(self, rows, payload, out):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("torch not available") from exc
        torch.manual_seed(self.seed)
        op_index = {op: i for i, op in enumerate(C.W2_TRAIN_OPERATORS)}
        n_classes = len(C.W2_TRAIN_OPERATORS)
        model = torch.nn.Linear(FEATURE_DIM, n_classes)
        opt = torch.optim.SGD(model.parameters(), lr=float(self.config.get("rl_lr", 0.1)))
        total_loss = 0.0
        n_used = 0
        for row in rows:
            op = row["operator"]
            if op not in op_index:
                continue
            x = torch.tensor([_features(row.get("context") or {}, row.get("response") or {})],
                             dtype=torch.float32)
            logits = model(x)
            logp = torch.log_softmax(logits, dim=1)[0, op_index[op]]
            adv = torch.tensor([row["advantage"]], dtype=torch.float32)
            loss = -(logp * adv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_used += 1
        payload["manifest"]["num_pg_updates"] = n_used
        payload["manifest"]["total_pg_loss"] = round(total_loss, 6)
        state = {k: v.detach() for k, v in model.state_dict().items()}
        meta = {"backend": "torch", "seed": self.seed, "n_classes": n_classes,
                "feat_dim": FEATURE_DIM, "num_updates": n_used}
        torch.save({"state_dict": state, "meta": meta}, str(out / "checkpoint.pt"))
        payload["artifact"] = "checkpoint.pt"
        meta_out = _write_checkpoint(payload, str(out))
        meta_out["artifact_path"] = str(out / "checkpoint.pt")
        meta_out["num_groups"] = payload["manifest"]["num_groups"]
        meta_out["num_samples"] = payload["manifest"]["num_samples"]
        return meta_out


# --------------------------------------------------------------------------- rollout
def rollout(router, env, config: dict, seed: int = 0) -> list:
    """Generate RL rollout groups.

    Each group shares one operator / parent program / prompt and samples
    ``config["samples"]`` responses (default 16). Every response is executed via
    ``env``, rewarded from the outcome's ``cost_aware_reward``, and assigned a
    per-group advantage via ``R.group_advantage`` (deterministic stdlib fallback).

    Returns ``[{group_index, operator, context, samples: [{response, reward,
    advantage}]}]`` where ``response`` is the router-produced Action.
    """
    config = dict(config or {})
    samples = int(config.get("samples", 16))
    groups_cfg = config.get("groups") or [config]
    tasks_cfg = config.get("tasks") or None

    rng = __import__("random").Random(seed)
    result: List[Dict[str, Any]] = []
    for group_index, gcfg in enumerate(groups_cfg):
        if not isinstance(gcfg, dict):
            continue
        operator = gcfg.get("operator", "draft")
        prompt = gcfg.get("prompt", "")
        parent = gcfg.get("parent", None)
        task_id = str(gcfg.get("task_id") or (gcfg.get("context") or {}).get("task_id") or "vcc25_h1")
        run_id = str(gcfg.get("run_id", "rollout"))

        sample_rows: List[Dict[str, Any]] = []
        rewards: List[float] = []
        for i in range(samples):
            sample_seed = seed + group_index * 1000 + i
            obs = env.reset(sample_seed, task_id)
            obs = obs if isinstance(obs, dict) else {}
            ctx: Dict[str, Any] = {
                "run_id": run_id,
                "round": int(gcfg.get("round", 0)),
                "task_id": task_id,
                "history": obs.get("history", []),
                "last_outcome": obs.get("last_outcome"),
                "available_actions": obs.get("available_actions") or [{"operator": operator}],
                "resource_state": obs.get("resource_state", {}),
                "operator": operator,
                "prompt": prompt,
                "parent": parent,
                "seed": sample_seed,
            }
            error = None
            response = None
            reward = 0.0
            try:
                action = router.decide(ctx)
                outcome = env.step(action)
                if not isinstance(outcome, dict):
                    outcome = {}
                reward = float(outcome.get("cost_aware_reward", 0.0) or 0.0)
                response = action
            except Exception as exc:  # record per-sample failures, keep rolling
                error = f"{type(exc).__name__}: {exc}"
            sample_rows.append(
                {"response": _serializable(response), "reward": reward, "advantage": 0.0,
                 **({"error": error} if error else {})}
            )
            rewards.append(reward)

        advantages = _group_advantage_with_R(rewards)
        for row, adv in zip(sample_rows, advantages):
            row["advantage"] = adv

        shared_context: Dict[str, Any] = {
            "run_id": run_id,
            "task_id": task_id,
            "operator": operator,
            "prompt": prompt,
            "parent": _serializable(parent),
        }
        result.append(
            {"group_index": group_index, "operator": operator,
             "context": shared_context, "samples": sample_rows}
        )
    return result


def train_pi(config: dict, **kwargs) -> dict:
    """CLI-compatible train entrypoint (design doc command #3).

    Builds the SFT dataset from the configured trajectory directory, then
    fits a behavior-clone policy through ``PiTrainer`` (offline by default;
    set ``train.backend: torch`` to require torch).
    """
    from pathlib import Path as _Path

    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}

    traj_dir = (
        data_cfg.get("trajectory_dir")
        or (config.get("trajectory") or {}).get("output_dir")
        or "output/rsi_step0/trajectories"
        or "sample_data/heuresis_logs/run1"
    )
    default_out = "output/rsi_step0/datasets/sft"
    sft_out = str(_Path(data_cfg.get("sft_out", default_out)))
    ckpt_out = str(_Path(train_cfg.get("checkpoint_dir", "output/rsi_step0/checkpoints")))

    dataset = build_sft_dataset(traj_dir, sft_out)
    trainer = PiTrainer(
        backend=str(train_cfg.get("backend", "offline")),
        seed=int(train_cfg.get("seed", 0)),
        config=dict(train_cfg),
    )
    return trainer.train_sft(dataset, ckpt_out)
