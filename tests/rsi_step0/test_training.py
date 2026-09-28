"""Tests for the RSI Step0 training layer and heuresis thin adapter.

Covers: ``build_sft_dataset`` keeps only positive records, offline
``PiTrainer.train_sft`` writes ``checkpoint.json`` without torch,
``make_checkpoint`` field completeness, ``DeterministicEnv`` reproducibility
for identical (seed, action) sequences, ``engine_available()`` returning False
without heuresis, and the ``backend="torch"`` RuntimeError path when torch is
missing.

The tests depend only on ``rsi_step0.contracts``, ``rsi_step0.reward``,
``rsi_step0.heuresis_adapter`` and ``rsi_step0.training``; ``trajectory`` /
``router`` are deliberately not imported (stub objects are used instead).
"""

import json
import tempfile
import unittest
from pathlib import Path

import rsi_step0.contracts as C
import rsi_step0.heuresis_adapter as H
import rsi_step0.training as T


def _make_traj_record(
    task_id: str,
    operator: str,
    label: str,
    run_id: str = "run1",
    round_no: int = 1,
    step: int = 1,
    pearson: float = 0.35,
    parent_action_id=None,
) -> dict:
    """Build a valid trajectory record shaped like ``TrajectoryRecord``."""
    action = {
        "action_id": C.new_action_id(),
        "run_id": run_id,
        "round": round_no,
        "step": step,
        "operator": operator,
        "mode": "execution",
        "target": task_id,
        "decision": {"hypothesis": f"hypothesis for {task_id}-{operator}"},
        "budget": {"max_seconds": 3600, "max_gpu_seconds": 3600, "max_cost": 1.0},
        "provenance": {"parent_action_id": parent_action_id, "model": "test", "seed": 0},
    }
    context = {
        "run_id": run_id,
        "round": round_no,
        "task_id": task_id,
        "history": [],
        "last_outcome": None,
        "available_actions": [{"operator": op} for op in C.W2_TRAIN_OPERATORS],
        "resource_state": {"gpu_count": 1, "max_seconds": 3600},
    }
    outcome = {
        "status": "ok",
        "metrics": {"pearson_delta": pearson},
        "resource_usage": {"runtime_seconds": 60.0, "gpu_count": 1},
        "cost_aware_reward": 0.05,
        "error": None,
    }
    reward = {
        "normalized_outcome": 0.06,
        "cost_compute": 60.0,
        "time_cost": 60.0,
        "total": 0.05,
        "lambda_c": 1e-6,
        "lambda_t": 1e-6,
        "metadata": {},
    }
    return {
        "schema_version": C.SCHEMA_VERSION,
        "record_id": f"rec-{task_id}-{operator}-{label}",
        "run_id": run_id,
        "round": round_no,
        "step": step,
        "operator": operator,
        "mode": "execution",
        "target": task_id,
        "decision": action["decision"],
        "context": context,
        "action": action,
        "outcome": outcome,
        "reward": reward,
        "label": label,
        "dedupe_key": C.make_dedupe_key(run_id, round_no, "p1", "v1"),
        "provenance": action["provenance"],
        "created_at": "2026-09-26T00:00:00+00:00",
    }


def _write_records(path: Path, records) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _load_jsonl(path: Path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class DatasetBuilderTests(unittest.TestCase):
    def test_build_sft_dataset_only_keeps_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj = Path(tmp) / "traj"
            traj.mkdir()
            records = [
                _make_traj_record("t1", "draft", "positive"),
                _make_traj_record("t1", "improve", "positive"),
                _make_traj_record("t1", "crossover", "negative"),
                _make_traj_record("t1", "tune", "excluded"),
            ]
            _write_records(traj / "records.jsonl", records)

            out = Path(tmp) / "sft"
            stats = T.build_sft_dataset(str(traj), str(out))

            self.assertEqual(stats["num_records"], 4)
            self.assertEqual(stats["num_positive"], 2)
            self.assertEqual(stats["num_sft_records"], 2)
            self.assertTrue((out / "sft_records.jsonl").exists())
            self.assertTrue((out / "manifest.json").exists())

            sft = _load_jsonl(out / "sft_records.jsonl")
            self.assertEqual(len(sft), 2)
            for rec in sft:
                self.assertEqual(set(rec.keys()), {"task_id", "context", "action"})
                self.assertEqual(rec["task_id"], "t1")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_positive"], 2)
            self.assertEqual(manifest["fingerprint"], C.hash_json(sft))

    def test_build_rl_dataset_keeps_all_labels_with_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj = Path(tmp) / "traj"
            traj.mkdir()
            records = [
                _make_traj_record("t1", "draft", "positive", parent_action_id="pa0"),
                _make_traj_record("t1", "improve", "negative", parent_action_id="pa1"),
                _make_traj_record("t1", "crossover", "excluded"),
            ]
            _write_records(traj / "records.jsonl", records)

            out = Path(tmp) / "rl"
            stats = T.build_rl_dataset(str(traj), str(out))

            self.assertEqual(stats["num_records"], 3)
            self.assertEqual(stats["labels"], {"positive": 1, "negative": 1, "excluded": 1})
            rl = _load_jsonl(out / "rl_contexts.jsonl")
            self.assertEqual(len(rl), 3)
            by_label = {rec["label"]: rec for rec in rl}
            self.assertEqual(set(by_label.keys()), {"positive", "negative", "excluded"})
            # the "improve" record carries the negative label here
            neg = by_label["negative"]
            self.assertEqual(neg["action"]["operator"], "improve")
            self.assertIn("parent", neg)
            self.assertEqual(neg["parent"]["parent_action_id"], "pa1")
            self.assertEqual(neg["parent"]["round"], 1)
            self.assertEqual(neg["parent"]["step"], 1)


class PiTrainerTests(unittest.TestCase):
    def _dataset(self):
        # Mirrors build_sft_dataset output: positive records only.
        return {
            "records": [
                _make_traj_record("t1", "draft", "positive"),
                _make_traj_record("t1", "improve", "positive"),
                _make_traj_record("t1", "crossover", "positive"),
            ]
        }

    def test_offline_train_sft_writes_checkpoint_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ckpt"
            trainer = T.PiTrainer(backend="offline", seed=7)
            meta = trainer.train_sft(self._dataset(), str(out))

            ckpt_path = out / "checkpoint.json"
            self.assertTrue(ckpt_path.exists())
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(ckpt["kind"], T.CHECKPOINT_KIND)
            self.assertEqual(ckpt["backend"], "offline")
            self.assertEqual(ckpt["policy"]["operator_counts"]["draft"], 1)
            self.assertEqual(ckpt["policy"]["operator_counts"]["improve"], 1)
            self.assertEqual(ckpt["policy"]["operator_counts"]["crossover"], 1)
            self.assertEqual(
                set(ckpt["policy"]["weights"].keys()), set(C.W2_TRAIN_OPERATORS)
            )
            self.assertAlmostEqual(sum(ckpt["policy"]["weights"].values()), 1.0, places=6)
            self.assertEqual(ckpt["manifest"]["num_records"], 3)
            self.assertEqual(ckpt["manifest"]["num_train_samples"], 3)
            self.assertIn("dataset_hash", ckpt["manifest"])
            self.assertEqual(ckpt["hash"], C.hash_json({k: v for k, v in ckpt.items() if k != "hash"}))
            self.assertEqual(meta["path"], str(ckpt_path))
            self.assertEqual(meta["hash"], ckpt["hash"])

    def test_make_checkpoint_fields_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ckpt"
            weights = {"draft": 0.5, "improve": 0.3, "crossover": 0.2}
            meta = T.make_checkpoint(weights, str(out))

            ckpt_path = out / "checkpoint.json"
            self.assertTrue(ckpt_path.exists())
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            for key in ("kind", "backend", "weights", "manifest", "hash"):
                self.assertIn(key, ckpt)
            self.assertEqual(ckpt["weights"], weights)
            self.assertEqual(ckpt["manifest"]["n_weight_keys"], 3)
            self.assertEqual(ckpt["backend"], "offline")
            self.assertEqual(ckpt["hash"], C.hash_json({k: v for k, v in ckpt.items() if k != "hash"}))
            for key in ("path", "kind", "weights", "manifest", "hash"):
                self.assertIn(key, meta)
            self.assertEqual(meta["path"], str(ckpt_path))

    def test_train_rl_offline_records_advantages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ckpt"
            groups = [
                {
                    "group_index": 0,
                    "operator": "draft",
                    "context": {"task_id": "t1"},
                    "samples": [
                        {"response": {"operator": "draft"}, "reward": 0.4, "advantage": 0.5},
                        {"response": {"operator": "draft"}, "reward": 0.2, "advantage": -0.3},
                    ],
                }
            ]
            trainer = T.PiTrainer(backend="offline", seed=3)
            meta = trainer.train_rl(groups, str(out))
            ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(ckpt["policy"]["sample_counts"]["draft"], 2)
            self.assertAlmostEqual(ckpt["policy"]["advantage_sums"]["draft"], 0.2, places=6)
            self.assertAlmostEqual(sum(ckpt["policy"]["weights"].values()), 1.0, places=6)
            self.assertEqual(meta["num_samples"], 2)
            self.assertEqual(meta["num_groups"], 1)

    def test_torch_backend_raises_runtime_error_without_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            torch = None
        if torch is not None:
            self.skipTest("torch is installed; RuntimeError path not applicable")
        with tempfile.TemporaryDirectory() as tmp:
            trainer = T.PiTrainer(backend="torch", seed=1)
            with self.assertRaises(RuntimeError) as ctx:
                trainer.train_sft(self._dataset(), str(Path(tmp) / "sft"))
            self.assertEqual(str(ctx.exception), "torch not available")
            with self.assertRaises(RuntimeError) as ctx:
                trainer.train_rl([], str(Path(tmp) / "rl"))
            self.assertEqual(str(ctx.exception), "torch not available")

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            T.PiTrainer(backend="nope")


class _StubRouter:
    """Minimal Router duck-type returning a fixed operator action."""

    def __init__(self, operator: str = "draft"):
        self.operator = operator

    def decide(self, context):
        return {
            "action_id": "stub-action",
            "run_id": context.get("run_id", "run"),
            "round": context.get("round", 0),
            "step": 1,
            "operator": self.operator,
            "mode": "execution",
            "target": context.get("task_id", ""),
            "decision": {"hypothesis": "stub"},
            "budget": {"max_seconds": 3600},
            "provenance": {"parent_action_id": None, "model": "stub", "seed": context.get("seed", 0)},
        }


class _StubEnv:
    """Minimal Environment duck-type with seed-dependent rewards."""

    def __init__(self):
        self.last_seed = 0
        self.task_id = None

    def reset(self, seed, task_id):
        self.last_seed = int(seed)
        self.task_id = task_id
        return {
            "task_id": task_id,
            "seed": self.last_seed,
            "history": [],
            "last_outcome": None,
            "available_actions": [{"operator": "draft"}],
            "resource_state": {"gpu_count": 1, "max_seconds": 3600},
        }

    def step(self, action):
        reward = 0.30 + 0.01 * (self.last_seed % 10)
        return {
            "status": "ok",
            "metrics": {"pearson_delta": reward},
            "resource_usage": {"runtime_seconds": 60.0, "gpu_count": 1},
            "cost_aware_reward": reward,
            "error": None,
        }


class RolloutTests(unittest.TestCase):
    def _config(self, samples=4):
        return {
            "samples": samples,
            "operator": "draft",
            "prompt": "improve correlation",
            "parent": {"parent_action_id": "pa1"},
            "task_id": "vcc25_h1",
            "run_id": "rollout-1",
        }

    def test_rollout_returns_groups_with_samples_and_advantage(self):
        groups = T.rollout(_StubRouter("draft"), _StubEnv(), self._config(samples=4), seed=7)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["group_index"], 0)
        self.assertEqual(group["operator"], "draft")
        self.assertEqual(group["context"]["prompt"], "improve correlation")
        self.assertEqual(group["context"]["parent"], {"parent_action_id": "pa1"})
        self.assertEqual(len(group["samples"]), 4)
        for sample in group["samples"]:
            self.assertIn("response", sample)
            self.assertIn("reward", sample)
            self.assertIn("advantage", sample)
            self.assertEqual(sample["response"]["operator"], "draft")
        advantages = [s["advantage"] for s in group["samples"]]
        self.assertTrue(any(a != 0.0 for a in advantages))

    def test_rollout_deterministic_for_same_seed(self):
        cfg = self._config(samples=4)
        g1 = T.rollout(_StubRouter("draft"), _StubEnv(), cfg, seed=7)
        g2 = T.rollout(_StubRouter("draft"), _StubEnv(), cfg, seed=7)
        self.assertEqual(
            [s["reward"] for s in g1[0]["samples"]],
            [s["reward"] for s in g2[0]["samples"]],
        )
        self.assertEqual(
            [s["advantage"] for s in g1[0]["samples"]],
            [s["advantage"] for s in g2[0]["samples"]],
        )

    def test_rollout_default_samples_is_16(self):
        groups = T.rollout(_StubRouter("draft"), _StubEnv(), {"operator": "draft"}, seed=1)
        self.assertEqual(len(groups[0]["samples"]), 16)


class HeuresisAdapterTests(unittest.TestCase):
    def test_engine_available_false_without_heuresis(self):
        self.assertFalse(H.engine_available())

    def test_deterministic_env_reproducible_same_seed_sequence(self):
        sequence = [
            {"action_id": "a1", "operator": "draft", "decision": {}},
            {"action_id": "a2", "operator": "improve", "decision": {}},
            {"action_id": "a3", "operator": "crossover", "decision": {}},
            {"action_id": "a4", "operator": "tune", "decision": {}},
        ]
        results = []
        for _ in range(2):
            env = H.DeterministicEnv(seed=0)
            env.reset(42, "vcc25_h1")
            outcomes = [env.step(action) for action in sequence]
            results.append(outcomes)
        first, second = results
        for o1, o2 in zip(first, second):
            self.assertEqual(o1, o2)
            self.assertEqual(o1["status"], "ok")
            self.assertEqual(o1["resource_usage"]["gpu_count"], 1)
            self.assertIsInstance(o1["cost_aware_reward"], float)

    def test_deterministic_env_metrics_increase_from_baseline(self):
        env = H.DeterministicEnv(seed=0)
        env.reset(11, "vcc25_h1")
        draft = env.step({"operator": "draft"})
        self.assertGreater(draft["metrics"]["pearson_delta"], C.DEFAULT_BASELINE)
        improved = env.step({"operator": "improve"})
        self.assertGreater(improved["metrics"]["pearson_delta"], draft["metrics"]["pearson_delta"])
        crossed = env.step({"operator": "crossover"})
        self.assertGreater(crossed["metrics"]["pearson_delta"], improved["metrics"]["pearson_delta"])

    def test_heuresis_env_falls_back_to_deterministic(self):
        env = H.HeuresisEnv(seed=0)
        self.assertFalse(env.using_real_engine)
        obs = env.reset(5, "vcc25_h1")
        self.assertEqual(obs["task_id"], "vcc25_h1")
        outcome = env.step({"operator": "draft"})
        self.assertIn("cost_aware_reward", outcome)
        self.assertEqual(outcome["resource_usage"]["gpu_count"], 1)

        other = H.HeuresisEnv(seed=0)
        other.reset(5, "vcc25_h1")
        other_outcome = other.step({"operator": "draft"})
        self.assertEqual(outcome, other_outcome)

    def test_evaluator_deterministic_report(self):
        evaluator = H.HeuresisEvaluator(prefer_real=False)
        context = {
            "run_id": "r1",
            "task_id": "t1",
            "history": [{"metrics": {"pearson_delta": 0.4}}],
        }
        report = evaluator.evaluate({"weights": {"draft": 0.5}}, context)
        for key in (
            "run_id", "task_id", "evaluator", "metrics",
            "cost_aware_reward", "resource_usage", "decision", "error",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["run_id"], "r1")
        self.assertEqual(report["task_id"], "t1")
        self.assertEqual(report["metrics"]["pearson_delta"], 0.4)
        self.assertIsInstance(report["cost_aware_reward"], float)
        self.assertTrue(report["evaluator"].startswith("heuresis"))

    def test_adapters_return_not_available_without_engine(self):
        router = H.SkillRouterAdapter()
        for resp in (router.run("task"), router.invoke("task"), router.train({"x": 1})):
            self.assertFalse(resp["ok"])
            self.assertFalse(resp["available"])
            self.assertIn("not available", resp["reason"])

        factory = H.ModelFactoryAdapter()
        for resp in (factory.run({"name": "m"}), factory.invoke("cp", {}), factory.train({"x": 1})):
            self.assertFalse(resp["ok"])
            self.assertFalse(resp["available"])
            self.assertIn("not available", resp["reason"])


if __name__ == "__main__":
    unittest.main()
