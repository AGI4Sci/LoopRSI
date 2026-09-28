from __future__ import annotations
import hashlib,json,time,uuid
from dataclasses import dataclass
from .contracts import ResearchState,SkillContext,SkillResult
from .registry import SkillRegistry
from .router import SkillRouter
from .storage import ResearchStore
@dataclass(frozen=True)
class SkillInvocation:
 request_id:str; skill_id:str; skill_version:str; backend:str; started_at:float; parent_event:str|None=None
 def provenance(self): return {"request_id":self.request_id,"skill_id":self.skill_id,"skill_version":self.skill_version,"execution_backend":self.backend,"started_at":self.started_at,"parent_event":self.parent_event}
class SkillLifecycleManager:
 def __init__(self,router,registry,store): self.router=router; self.registry=registry; self.store=store
 def invoke(self,skill_id,context,version=None,parent_event=None,request_id=None,crash_after_state=False):
  meta=self.registry.get(skill_id,version); rid=request_id or uuid.uuid4().hex; start=time.time(); inv=SkillInvocation(rid,skill_id,meta["version"],meta["execution_backend"],start,parent_event)
  self.store.append_event({"event_id":rid+":invoked","event_type":"skill_invoked","timestamp":start,"request_id":rid,"skill":inv.provenance(),"parent":parent_event})
  try: result=self.router.run(skill_id,context); end=time.time();
  except Exception as exc:
   self.store.append_event({"event_id":rid+":failed","event_type":"skill_failed","timestamp":time.time(),"request_id":rid,"skill":inv.provenance(),"error":str(exc),"parent":rid+":invoked"}); raise
  record={"request_id":rid,"result":result.to_dict(),"finished_at":end,"input_context_sha256":hashlib.sha256(json.dumps(context.research_state.__dict__,default=list,sort_keys=True).encode()).hexdigest()}
  state=self.store.load_state() if self.store.state_path.exists() else context.research_state
  updated=ResearchState(state.task_id,state.question,evidence=(*state.evidence,{"request_id":rid,**result.evidence}),open_hypotheses=state.open_hypotheses,rejected_hypotheses=state.rejected_hypotheses,blockers=state.blockers)
  self.store.save_state(updated)
  if crash_after_state: raise RuntimeError("simulated crash after state commit")
  self.store.append_event({"event_id":rid+":completed","event_type":"skill_completed","timestamp":end,"request_id":rid,"skill":inv.provenance(),"status":result.status,"evidence":dict(result.evidence),"record":record,"parent":rid+":invoked"})
  self.store.append_event({"event_id":rid+":evidence","event_type":"evidence_recorded","timestamp":end,"request_id":rid,"evidence":dict(result.evidence),"parent":rid+":completed"})
  return result
 def recover(self):
  done={e.get("request_id") for e in self.store.query("skill_completed")}; recovered=[]
  for e in self.store.query("skill_invoked"):
   rid=e.get("request_id")
   if rid not in done and not self.store.query("skill_failed",request_id=rid):
    self.store.append_event({"event_id":rid+":recovered","event_type":"skill_failed","timestamp":time.time(),"request_id":rid,"status":"recovered_incomplete","error":"incomplete invocation recovered fail-closed","parent":e["event_id"]}); recovered.append(rid)
  return recovered
