from __future__ import annotations
import hashlib,json,time,uuid
from dataclasses import asdict,dataclass
from pathlib import Path
from .contracts import ResearchState,SkillContext,SkillResult
from .registry import SkillRegistry
from .router import SkillRouter
from .storage import ResearchStore

STATES={"pending","running","succeeded","failed","aborted","recovered"}
@dataclass(frozen=True)
class ExecutionRecord:
 request_id:str; invocation_id:str; skill_id:str; skill_version:str; execution_backend:str; status:str; created_at:float; started_at:float|None=None; finished_at:float|None=None; context_hash:str|None=None; result_reference:str|None=None; error:str|None=None; provenance:dict|None=None; parent_event_ids:tuple[str,...]=(); state_transition_ids:tuple[str,...]=(); graph_event_ids:tuple[str,...]=()
 def to_dict(self): return {"protocol_version":"ai4ai-execution-record/v1",**asdict(self)}
class RuntimeStore(ResearchStore):
 def __init__(self,root:Path): super().__init__(root); self.execution_path=root/"execution_records.jsonl"; self.transition_path=root/"state_transitions.jsonl"
 def _append_unique(self,path,value,key):
  records=self._read(path)
  if any(x.get(key)==value.get(key) for x in records):
   if next(x for x in records if x.get(key)==value.get(key))!=value: raise ValueError("conflicting duplicate record")
   return False
  with path.open("a",encoding="utf-8") as f: f.write(json.dumps(value,sort_keys=True,ensure_ascii=False)+"\n")
  return True
 def _read(self,path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x] if path.exists() else []
 def append_execution(self,record): return self._append_unique(self.execution_path,record,"invocation_id")
 def update_execution(self,invocation_id,**updates):
  rows=self.executions(); found=False
  for row in rows:
   if row["invocation_id"]==invocation_id: row.update(updates); found=True
  if not found: raise KeyError(invocation_id)
  with self.execution_path.open("w",encoding="utf-8") as f:
   for row in rows: f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
 def executions(self): return self._read(self.execution_path)
 def execution(self,invocation_id):
  for x in self.executions():
   if x["invocation_id"]==invocation_id:return x
  raise KeyError(invocation_id)
 def append_transition(self,value):
  value={"protocol_version":"ai4ai-state-transition/v1",**value}; return self._append_unique(self.transition_path,value,"transition_id")
 def transitions(self): return self._read(self.transition_path)
 def transition(self,transition_id):
  for x in self.transitions():
   if x["transition_id"]==transition_id:return x
  raise KeyError(transition_id)
 def audit(self,request_id=None,invocation_id=None,skill_id=None):
  return [x for x in self.executions() if (request_id is None or x["request_id"]==request_id) and (invocation_id is None or x["invocation_id"]==invocation_id) and (skill_id is None or x["skill_id"]==skill_id)]
class LifecycleManager:
 def __init__(self,router:SkillRouter,registry:SkillRegistry,store:RuntimeStore): self.router=router; self.registry=registry; self.store=store
 def _hash(self,value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=list).encode()).hexdigest()
 def invoke(self,skill_id,context,version=None,request_id=None,crash_at=None,parent_event_ids=()):
  meta=self.registry.get(skill_id,version); rid=request_id or uuid.uuid4().hex; iid=rid+":invocation"; now=time.time(); ctx_hash=self._hash(asdict(context.research_state))
  if any(x["request_id"]==rid and x["status"]=="succeeded" for x in self.store.executions()): return SkillResult(skill_id,"ok",{"deduplicated":True})
  self.store.append_execution(ExecutionRecord(rid,iid,skill_id,meta["version"],meta["execution_backend"],"running",now,now,context_hash=ctx_hash,provenance={"parent_event_ids":list(parent_event_ids)}).to_dict())
  self.store.append_event({"event_id":iid+":started","event_type":"skill_invoked","timestamp":now,"request_id":rid,"invocation_id":iid,"skill_id":skill_id,"skill_version":meta["version"],"parent_ids":list(parent_event_ids)})
  if crash_at=="started": raise RuntimeError("simulated crash at started")
  try: result=self.router.run(skill_id,context)
  except Exception as exc:
   self.store.append_event({"event_id":iid+":failed","event_type":"skill_failed","timestamp":time.time(),"request_id":rid,"invocation_id":iid,"error":str(exc),"parent_ids":[iid+":started"]}); return None
  if crash_at=="result": raise RuntimeError("simulated crash after result")
  old=self.store.load_state() if self.store.state_path.exists() else context.research_state
  new=ResearchState(old.task_id,old.question,(*old.evidence,{"request_id":rid,**result.evidence}),old.open_hypotheses,old.rejected_hypotheses,old.blockers)
  transition_id=iid+":transition"; before=self._hash(asdict(old)); after=self._hash(asdict(new))
  self.store.append_transition({"transition_id":transition_id,"event_id":iid+":completed","invocation_id":iid,"before_state_reference":before,"after_state_reference":after,"before_state":asdict(old),"after_state":asdict(new),"operation":"append_evidence","timestamp":time.time(),"provenance":{"request_id":rid}})
  self.store.save_state(new)
  if crash_at=="state": raise RuntimeError("simulated crash after state transition")
  self.store.update_execution(iid,status="succeeded",finished_at=time.time(),result_reference=iid+":evidence",state_transition_ids=[transition_id],graph_event_ids=[iid+":completed",iid+":evidence"])
  self.store.append_event({"event_id":iid+":completed","event_type":"skill_completed","timestamp":time.time(),"request_id":rid,"invocation_id":iid,"skill_id":skill_id,"status":result.status,"evidence":dict(result.evidence),"transition_id":transition_id,"parent_ids":[iid+":started"]})
  self.store.append_event({"event_id":iid+":evidence","event_type":"evidence_recorded","timestamp":time.time(),"request_id":rid,"invocation_id":iid,"evidence":dict(result.evidence),"parent_ids":[iid+":completed"]})
  if crash_at=="graph": raise RuntimeError("simulated crash after graph event")
  return result
 def recover(self):
  completed={x.get("request_id") for x in self.store.query("skill_completed")}; recovered=[]
  for rec in self.store.executions():
   if rec["request_id"] in completed: continue
   if rec["status"] in {"running","pending"}:
    iid=rec["invocation_id"]; self.store.append_event({"event_id":iid+":recovered","event_type":"skill_failed","timestamp":time.time(),"request_id":rec["request_id"],"invocation_id":iid,"status":"recovered","error":"incomplete invocation; success not provable","parent_ids":[iid+":started"]}); recovered.append(rec["request_id"])
  return recovered
 def replay(self):
  state=None
  for tr in self.store.transitions():
   if not all(k in tr for k in ("before_state","after_state","before_state_reference","after_state_reference")): raise ValueError("corrupt transition")
   if self._hash(tr["before_state"]) != tr["before_state_reference"] or self._hash(tr["after_state"]) != tr["after_state_reference"]: raise ValueError("transition hash mismatch")
   if state is not None and self._hash(asdict(state)) != tr["before_state_reference"]: raise ValueError("non-contiguous transition")
   a=tr["after_state"]; state=ResearchState(a["task_id"],a["question"],tuple(a["evidence"]),tuple(a["open_hypotheses"]),tuple(a["rejected_hypotheses"]),tuple(a["blockers"]))
  if state is None: raise ValueError("no replayable transitions")
  return state
