import argparse, importlib, json, sys
from .contracts import ResearchState, SkillContext, validate_json_message
def main():
 p=argparse.ArgumentParser(); p.add_argument("--entrypoint",required=True); a=p.parse_args(); raw=validate_json_message(json.load(sys.stdin),"ai4ai/skill-request/v1"); s=raw["context"]["research_state"]; c=SkillContext(ResearchState(s["task_id"],s["question"],tuple(s.get("evidence",())),tuple(s.get("open_hypotheses",())),tuple(s.get("rejected_hypotheses",())),tuple(s.get("blockers",()))),raw["context"].get("constraints",{})); mod,name=a.entrypoint.split(":",1); result=getattr(importlib.import_module(mod),name)().run(c); print(json.dumps(result.to_dict(),ensure_ascii=False,sort_keys=True))
if __name__=="__main__": main()
