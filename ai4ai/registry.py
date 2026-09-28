from __future__ import annotations
import json
from pathlib import Path
from .contracts import SkillManifest

class SkillRegistry:
 def __init__(self,path:Path): self.path=path; self._items={}; self._load()
 def _load(self):
  if self.path.exists():
   self._items=json.loads(self.path.read_text()).get("skills",{})
 def _save(self):
  self.path.parent.mkdir(parents=True,exist_ok=True); self.path.write_text(json.dumps({"protocol_version":"ai4ai-skill-registry/v1","skills":self._items},indent=2,ensure_ascii=False))
 def register(self,m:SkillManifest,backend="in-process",runtime=None):
  m.validate(); key=f"{m.skill_id}@{m.version}"
  if key in self._items: raise ValueError(f"duplicate skill/version: {key}")
  self._items[key]={**m.to_dict(),"execution_backend":backend,"runtime":runtime or {},"status":"active"}; self._save(); return self.get(m.skill_id,m.version)
 def get(self,skill_id,version=None):
  matches=[v for k,v in self._items.items() if v["skill_id"]==skill_id and (version is None or v["version"]==version)]
  if not matches: raise KeyError(f"unknown skill/version: {skill_id}@{version or '*'}")
  if version is None: matches.sort(key=lambda x:x["version"]); return matches[-1]
  return matches[0]
 def discover(self,capability=None): return [v for v in self._items.values() if capability is None or capability in v["capabilities"]]
 def validate(self,skill_id,version): return self.get(skill_id,version)["status"]=="active"
 def version_lookup(self,skill_id): return sorted(v["version"] for v in self.discover() if v["skill_id"]==skill_id)
