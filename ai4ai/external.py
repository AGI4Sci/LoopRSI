import json,subprocess
from dataclasses import dataclass
from .contracts import SkillContext,SkillManifest,SkillResult,validate_json_message
@dataclass
class ExternalSkill:
 manifest: SkillManifest
 command: tuple
 timeout_s: float=30.0
 def run(self,context):
  s=context.research_state; req={"protocol_version":"ai4ai/skill-request/v1","skill_id":self.manifest.skill_id,"skill_version":self.manifest.version,"context":{"research_state":{"task_id":s.task_id,"question":s.question,"evidence":list(s.evidence),"open_hypotheses":list(s.open_hypotheses),"rejected_hypotheses":list(s.rejected_hypotheses),"blockers":list(s.blockers)},"constraints":dict(context.constraints)}}
  try: p=subprocess.run(self.command,input=json.dumps(req),text=True,capture_output=True,timeout=self.timeout_s,check=False)
  except subprocess.TimeoutExpired as e: raise RuntimeError("skill timeout") from e
  if p.returncode: raise RuntimeError(p.stderr[-500:])
  v=validate_json_message(json.loads(p.stdout),"ai4ai-skill-result/v1"); result=SkillResult(v["skill_id"],v["status"],v["evidence"],tuple(v.get("next_actions",())),v.get("provenance",{}))
  if result.skill_id!=self.manifest.skill_id or result.provenance.get("skill_version")!=self.manifest.version: raise ValueError("external skill identity/version mismatch")
  return result
