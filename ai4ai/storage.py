from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from .contracts import ResearchState

class ResearchStore:
    def __init__(self, root: Path):
        self.root=root; self.root.mkdir(parents=True, exist_ok=True)
        self.state_path=self.root/"research_state.json"; self.events_path=self.root/"experiment_graph.jsonl"
    def save_state(self, state: ResearchState) -> None:
        self.state_path.write_text(json.dumps({"protocol_version":"ai4ai-research-state/v1", **asdict(state)}, ensure_ascii=False, indent=2), encoding="utf-8")
    def load_state(self) -> ResearchState:
        value=json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.pop("protocol_version", None)!="ai4ai-research-state/v1": raise ValueError("unsupported research state protocol")
        return ResearchState(value["task_id"], value["question"], tuple(value.get("evidence",())), tuple(value.get("open_hypotheses",())), tuple(value.get("rejected_hypotheses",())), tuple(value.get("blockers",())))
    def append_event(self, event: dict[str,Any]) -> None:
        if not event.get("event_id") or not event.get("event_type"): raise ValueError("graph event requires event_id and event_type")
        with self.events_path.open("a", encoding="utf-8") as f: f.write(json.dumps({"protocol_version":"ai4ai-graph-event/v1", **event}, ensure_ascii=False, sort_keys=True)+"\n")
    def events(self):
        if not self.events_path.exists(): return []
        return [json.loads(x) for x in self.events_path.read_text(encoding="utf-8").splitlines() if x]
    def query(self, event_type=None, **fields):
        return [e for e in self.events() if (event_type is None or e.get("event_type")==event_type) and all(e.get(k)==v for k,v in fields.items())]
    def lineage(self, event_id):
        by_id={e["event_id"]:e for e in self.events()}; out=[]; cur=event_id
        while cur:
            if cur not in by_id: raise KeyError(cur)
            out.append(by_id[cur]); cur=by_id[cur].get("parent")
        return out
