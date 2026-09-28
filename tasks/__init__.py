"""tasks: 任务插件（heuresis 以任务插件挂载具体任务）。

ROLE: 插件，按任务写。核心引擎零改动。新增任务 = 仿照 vcc25 建
``tasks/<任务>/`` + ``datasets/<任务>/``，并在任务插件清单里声明。

当前唯一插件：``vcc25/``（生物任务）—— 任务规格（task_spec*.yaml）、
任务插件清单（task_plugin.yaml）、候选适配器（official_h1_adapter.py）、
评估器（native_evaluator.py）、实现脚本（implementation/）。
"""
