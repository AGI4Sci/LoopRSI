"""RSI Step0 新增层（本项目真正的产出）。

在 heuresis 引擎壳之上移植的 OpenRSI「π 训练方法论」。部署于 RSI 仓库根
（独立包，也可作为 ``omni_ar/rsi_step0`` 被 heuresis 引用）。

ROLE: RSI Step0 新增层，按 M1–M7 进度扩展。

关键模块（按职责分组，看接口先看这些模块顶部 docstring）：
- ``contracts.py``        —— 全部枚举 / 常量 / 哈希工具的唯一权威契约，其它模块
                             ``from rsi_step0 import contracts as C`` 引用。不要改。
- ``trajectory.py``       —— 决策轨迹转录与标注；把 heuresis 的 run 日志转成
                             transcribe / SFT / RL / 调度四类记录。
- ``reward.py``           —— 成本感知回报：基础分归一化、验证/测试 gap 惩罚、
                             算子 shaping、组内 advantage。
- ``router.py``           —— π 的算子路由：GuidedRouter（引导版）与
                             TrainedRouter（训练版），算子 = draft/improve/crossover。
- ``scheduler.py``        —— 回合调度（stop / converge / continue + 预算守卫）。
- ``training.py``         —— SFT / RL 训练（PiTrainer：offline 纯标准库，
                             torch 后端延迟导入）+ rollout / make_checkpoint。
- ``heuresis_adapter.py`` —— heuresis 环境 / 评估器 / 路由器 / 模型工厂的薄适配层。
- ``validation.py``       —— 动作 / 轨迹 / 回放校验与 schema 检查（接口固定）。
- ``cli.py``              —— 命令行入口（transcribe/validate/train/replay/evaluate/check），
                             每个命令在 ``output/rsi_step0/manifests`` 写结构化摘要。
- ``schemas/``            —— 动作 / 奖励 / 调度 / 轨迹的 JSON schema。

Self-contained package (pure stdlib; optional jsonschema/PyYAML/torch are used only
when available). 产物统一写入仓库根的 ``output/``（只读，勿手改）。
"""

__version__ = "0.1.0"
