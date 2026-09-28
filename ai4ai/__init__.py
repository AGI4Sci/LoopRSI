"""ai4ai: heuresis 引擎核心 —— 插件协议 / 技能路由 / 候选执行框架。

命名：来自上游项目 "AI4AI"（AI for AI），即「让 AI 做 AI 研究」的通用研究框架；
``ai4ai`` 是该框架的中间件层，提供技能 / 插件 / 候选执行的通用运行时，被
``omni_ar`` 的编排循环与 ``controller_bash`` 的入口消费。heuresis 是 AI4AI
项目下的引擎子项目。

ROLE: 引擎核心（复用，接口固定，一般不改）。其它模块从这里 import 协议与接口。

关键模块：
- ``contracts.py``   —— 协议类型（ResearchState / SkillContext / SkillManifest / SkillResult）。
- ``router.py``      —— SkillRouter：技能注册与路由（run / register / discover / backend）。
- ``registry.py``    —— SkillRegistry：技能注册表。
- ``lifecycle.py``   —— SkillLifecycleManager：技能生命周期管理。
- ``storage.py``     —— ResearchStore：研究状态 / 谱系存储。
- ``external.py``    —— ExternalSkill：外部技能封装。
"""

from .contracts import ResearchState, SkillContext, SkillManifest, SkillResult
from .router import SkillRouter
from .storage import ResearchStore
from .external import ExternalSkill
from .registry import SkillRegistry
from .lifecycle import SkillLifecycleManager

__all__ = ["ResearchState", "SkillContext", "SkillManifest", "SkillResult", "SkillRouter", "ResearchStore", "ExternalSkill", "SkillRegistry", "SkillLifecycleManager"]
