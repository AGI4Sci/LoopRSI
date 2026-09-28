import json
import os
import tempfile
import unittest

import rsi_step0.contracts as C
import rsi_step0.trajectory as T

SAMPLE = "sample_data/heuresis_logs/run1"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestTrajectory(unittest.TestCase):
    def test_transcribe_creates_records(self):
        with tempfile.TemporaryDirectory() as out:
            s = T.transcribe(SAMPLE, out)
            self.assertEqual(s["run_id"], "run1")
            self.assertGreaterEqual(s["records"], 1)
            self.assertTrue(os.path.exists(os.path.join(out, "records.jsonl")))
            recs = _load(os.path.join(out, "records.jsonl"))
            for r in recs:
                self.assertEqual(r["schema_version"], "rsi_trajectory/v1")
                self.assertIn(r["label"], ("positive", "negative", "excluded"))
                self.assertIn(r["operator"], C.OPERATORS)

    def test_transcribe_tolerates_missing_files(self):
        with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as out:
            s = T.transcribe(empty, out)
            self.assertGreater(s["skipped_files"], 0)
            self.assertGreaterEqual(s["records"], 1)

    def test_annotate_and_dedupe(self):
        recs = [
            {"dedupe_key": "k1"},
            {"dedupe_key": "k1"},
            {"dedupe_key": "k2"},
        ]
        out = T.annotate_and_dedupe(recs)
        self.assertEqual([r["dedupe_key"] for r in out], ["k1", "k2"])

    def test_build_datasets_stats(self):
        with tempfile.TemporaryDirectory() as out:
            T.transcribe(SAMPLE, out)
            stats = T.build_datasets(out, os.path.join(out, "ds"))
            self.assertEqual(stats["sft"], stats["rl"])
            self.assertTrue(os.path.exists(stats["sft_path"]))
            self.assertTrue(os.path.exists(stats["rl_path"]))
            self.assertTrue(os.path.exists(stats["scheduler_path"]))

    def test_replay_hash_stable(self):
        a = {"run_id": "r", "round": 1, "step": 1, "operator": "draft", "target": "t",
             "decision": {}, "action": {}, "outcome": {}, "reward": {}, "label": "positive"}
        b = {"run_id": "r", "round": 1, "step": 2, "operator": "improve", "target": "t",
             "decision": {}, "action": {}, "outcome": {}, "reward": {}, "label": "positive"}
        self.assertEqual(T.replay_hash([a, b]), T.replay_hash([a, b]))
        self.assertNotEqual(T.replay_hash([a, b]), T.replay_hash([b, a]))

    def test_replay_deterministic(self):
        with tempfile.TemporaryDirectory() as out:
            T.transcribe(SAMPLE, out)
            recs = _load(os.path.join(out, "records.jsonl"))
            r1 = T.replay(recs)
            r2 = T.replay(recs)
            self.assertEqual(r1["hash"], r2["hash"])
            self.assertEqual(r1["reward_total"], r2["reward_total"])
            self.assertEqual(len(r1["event_sequence"]), len(recs))


if __name__ == "__main__":
    unittest.main()


class TestTrajectoryCandidateEvents(TestTrajectory):
    def test_transcribe_expands_candidate_events_with_distinct_keys(self):
        with tempfile.TemporaryDirectory() as run, tempfile.TemporaryDirectory() as out:
            proposal = {
                "run_id": "runx", "round": 1, "step": 1, "proposal_id": "p0",
                "task_id": "taskx", "seed": 7, "idea_variant": {"name": "v0"},
            }
            with open(os.path.join(run, "proposal.json"), "w", encoding="utf-8") as f:
                json.dump(proposal, f)
            with open(os.path.join(run, "trial_results.json"), "w", encoding="utf-8") as f:
                json.dump({"run_id": "runx", "round": 1, "status": "ok"}, f)
            with open(os.path.join(run, "round_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"run_id": "runx", "round": 1, "status": "ok"}, f)
            events = [
                {"event": "candidate_recorded", "run_id": "runx", "round": 1, "step": 2,
                 "proposal_id": "p1", "idea_variant": {"name": "v1"}, "operator": "improve",
                 "status": "ok", "pearson_delta": 0.5},
                {"event": "candidate_recorded", "run_id": "runx", "round": 1, "step": 3,
                 "proposal_id": "p2", "idea_variant": {"name": "v2"}, "operator": "debug",
                 "status": "fail", "pearson_delta": 0.1, "env_failure": False},
            ]
            with open(os.path.join(run, "events.jsonl"), "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(e) for e in events) + "\n")

            s = T.transcribe(run, out)
            self.assertEqual(s["records"], 3)
            recs = _load(os.path.join(out, "records.jsonl"))
            keys = [r["dedupe_key"] for r in recs]
            self.assertEqual(len(set(keys)), 3, "candidate records must dedupe distinctly")
            by_pid = {r["context"]["proposal_id"]: r for r in recs}
            self.assertEqual(by_pid["p1"]["operator"], "improve")
            self.assertEqual((by_pid["p1"]["context"]["idea_variant"] or {}).get("name"), "v1")
            self.assertEqual(by_pid["p2"]["operator"], "debug")
