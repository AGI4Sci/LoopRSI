# RSI 项目地图（人类入口）

> 远端部署目录：`/mnt/shared-storage-user/gaozhangyang/RSI`
>
> 一句话：**以 heuresis 为引擎壳，移植 OpenRSI 的「π 训练方法论」（RSI Step0），跑 vcc25 生物任务**。
>
> 本文件回答三个问题：**哪些不用改（接口已定义的核心代码）、哪些是按任务改的（插件/配置）、哪些是本项目新增的（RSI Step0）**。

## 一、三层结构总览

| 层 | 目录 | 角色 | 要不要改 |
|---|---|---|---|
| **引擎核心（复用，接口固定）** | `ai4ai/` | 插件协议 / 技能路由 / 候选执行框架 | 一般不改 |
| | `omni_ar/` | 自主研究主循环（harness / autoresearch） | 一般不改 |
| | `controller_bash/` | 集群控制器（round / loop / rjob / 脚本） | 一般不改，脚本按需读 |
| **任务插件（按任务写）** | `tasks/vcc25/` | 任务实现 / evaluator / 任务规格 | 改的是这个 |
| | `datasets/vcc25/` | 数据集 / 数据适配器 | 改的是这个 |
| **RSI Step0 新增层（本项目核心）** | `rsi_step0/` | π 训练 / 回报 / 路由 / 调度 / 校验 | 按 M1–M7 进度扩展 |
| **任务/实验配置（人类主要改）** | `configs/rsi_step0/` | 训练 / 消融 / 回放配置 | **改这里** |
| **产物（只读）** | `output/` | checkpoint / trajectory / manifest / 报告 | 只读 |
| 辅助 | `sample_data/` | heuresis 示例日志 | 只读 |
| | `tests/` | `rsi_step0` 单测 | 只读 |
| | `regression_anchors/` | 历史回归基准 | 只读 |
| | `Heuresis_PJLAB-boyue/` | vcc25 基线分数（被 adapter 引用） | 只读 |

入口脚本：`rjob_train_rsi.py`（用 rjob 拉起一次训练的最小闭环入口）。

## 二、什么是不用改、但接口已经定义好的核心代码

- `rsi_step0/contracts.py` —— 全部枚举 / 常量 / 哈希工具的**唯一权威契约**，其它模块都 `from rsi_step0 import contracts as C` 引用它。**不要改**，除非你明确要在协议上加字段。
- `rsi_step0/validation.py` —— 动作 / 轨迹 / 回放的校验与 schema 检查（接口固定）。
- `ai4ai/plugin_protocols.py`、`ai4ai/router.py` —— heuresis 的插件协议与技能路由接口（引擎壳，复用不重建）。
- `omni_ar/autoresearch.py` —— heuresis 主循环（引擎壳）。

> 各角色包的 `__init__.py` 都写了 **ROLE** 与 **关键模块** 注释，说明「这个包是干什么的、里面哪个文件是接口」。

## 三、什么是需要根据任务配置的

**不是只有 `configs` 一处要改。** 分两类：

1. **只调参不换任务**（最常见）——只改 `configs/rsi_step0/`：
   - `configs/rsi_step0/{minimal,replay,train_pi,ablation}.yaml` —— RSI Step0 的训练 / 回放 / 消融配置。
2. **新增 / 修改一个任务**（按插件方式）——还要改 `tasks/` 与 `datasets/`：
   - `tasks/vcc25/task_spec*.yaml`、`task_plugin.yaml` —— 任务规格与插件清单（主指标、数据、评估方式）。
   - `tasks/vcc25/{official_h1_adapter,native_evaluator}.py` —— 候选适配器与评估器（接口由引擎定义）。
   - `datasets/vcc25/*.yaml` —— 数据集版本与特征规格。
   - `controller_bash/configs/*.env` —— rjob / 集群 / 任务运行的参数（GPU 配额组、镜像、挂载、命令）。

**要改任务配置，优先改这些文件，而不是改核心代码。**

## 四、什么是插件（任务可插拔）

heuresis 以「任务插件」方式挂载具体任务。当前唯一的插件是 `vcc25`：

- `tasks/vcc25/task_plugin.yaml` —— 任务插件清单（插件总线依据它解析入口）。
- `tasks/vcc25/{official_h1_adapter,native_evaluator}.py` —— 候选适配器与评估器。
- `tasks/vcc25/implementation/` —— vcc25 的实际训练 / 预测实现脚本。
- `datasets/vcc25/` —— 该任务的数据与数据适配器。

新增一个任务的步骤：仿照 `vcc25` 建一个 `tasks/<新任务>/` + `datasets/<新任务>/`，并在插件清单里声明，核心引擎零改动。

## 五、两个引擎包的名字与关系（人类常问）

- **`ai4ai` = "AI4AI"（AI for AI）**，来自上游研究项目名，意为「让 AI 做 AI 研究」的通用框架。它是**中间件层**：提供插件协议、技能路由、候选执行、注册表与状态存储等通用运行时。heuresis 是 AI4AI 项目下的引擎子项目。
- **`omni_ar` = "Omni-AR"（Omni Auto-Research）**，意为「全方位自动研究」引擎，是**主循环/编排层**：驱动多轮 research 回合（建议 → 执行 → 评估 → 晋级）。
- **关系**：`omni_ar` 编排、调用 `ai4ai` 的协议 / 技能 / 候选执行中间件，再按 `tasks/` 插件挂载具体任务。两者都是引擎核心（复用不改）；而 `rsi_step0` 是在它们之上新增的「π 训练方法论」层。

## 六、RSI Step0 新增了什么（本项目真正的产出）

- `rsi_step0/` 包（纯标准库，可选依赖仅在存在时使用）：
  - `contracts.py` —— 契约（不要改）。
  - `trajectory.py` / `reward.py` —— 轨迹转录与**成本感知回报**。
  - `router.py` / `scheduler.py` —— π 的算子路由（draft/improve/crossover）与调度。
  - `training.py` / `heuresis_adapter.py` —— SFT / RL 训练与 heuresis 环境适配。
  - `validation.py` / `cli.py` —— 校验与命令行入口。
- 设计 / 学习文档见本地 `proposal/RSI/docs/`（`RSI-step0-synthesis.md`、`RSI-step0-learning-guide.md`）。

## 七、快速上手（最小闭环）

1. 读 `configs/rsi_step0/minimal.yaml` 确认任务配置。
2. 用 `python -m rsi_step0.cli check` 自查环境（引擎可用性 / Python 版本）。
3. 用 `python -m rsi_step0.cli validate` 校验轨迹。
4. 用 `python rjob_train_rsi.py` 通过 rjob 拉起一次训练，产物落在 `output/rsi_step0/`。

> 注意：`vcc25/` 顶层是 `datasets/vcc25` + `tasks/vcc25` 的重复副本，已删除；数据与任务的唯一权威位置是 `datasets/vcc25` 与 `tasks/vcc25`。
>
> 产物统一在仓库根 `output/`（`output/rsi_step0/` 是 RSI Step0 层产物，`output/vcc25/` 是 heuresis 引擎对 vcc25 的实际运行产物），全部只读。
