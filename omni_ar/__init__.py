"""omni_ar: heuresis 引擎核心 —— 自主研究主循环（harness / autoresearch）。

命名：来自项目 "Omni-AR"（Omni Auto-Research），即「全方位自动研究」引擎，
驱动多轮 research 回合（建议 -> 执行 -> 评估 -> 晋级）。它调用 ``ai4ai`` 的
协议 / 技能 / 候选执行中间件，并按任务插件（``tasks/``）挂载具体任务。

ROLE: 引擎核心（复用，接口固定，一般不改）。

关键模块：
- ``loop.py``            —— ResearchLoop：主循环，面向外部组合 Omni-AR 工作流的面。
- ``autoresearch.py``    —— 多轮 AutoResearch 编排（建议 -> 证据 -> 标准结果 -> 终包）。
- ``finalizer.py``       —— choose_winner：按主指标从候选里选晋级者。
- ``project_records.py`` —— 事件 / candidate 谱系记录（records/events.jsonl）。
- ``initialization/``    —— 预研问题 / 归一化 / 可追溯流程（RoughIdeaEngine）。
"""

from .loop import ResearchLoop, loop

__all__ = ["ResearchLoop", "loop"]
