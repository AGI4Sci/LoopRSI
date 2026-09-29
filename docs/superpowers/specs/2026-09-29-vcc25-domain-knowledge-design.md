# VCC25 领域知识库首版设计

日期：2026-09-29
状态：待书面评审

## 1. 目标与边界

知识库为 RSI Agent 提供论文、开源代码库和可执行模型三类外部知识，使 Agent 能在 L1–L5
研究循环中选择有依据、可落地的候选方案。最终目标是在严格隔离最终测试答案的前提下，使用
统一 VCC25 评测协议寻找超过 Lingshu-Cell baseline 的方案。

VCC25 是系统固定任务，不是 Agent 可选择的知识条目。因此首版不建立 `BenchmarkCard`。
论文、仓库和模型由知识库管理；固定评测规则由 `VCC25EvaluationContract` 管理；实验成绩
继续由现有 trajectory、reward 和 `ExperienceCard` 体系记录。

首版不自动下载或执行任意第三方代码，不把大权重提交到 Git，不修改现有 reward、memory
和 trajectory 的语义。

## 2. 与现有代码的关系

现有 `ai4ai.plugin_protocols.ResearchSkill` 已定义：

```text
can_activate(context) → inject(context) → validate_proposal(proposal, context)
```

`tasks/vcc25/task_plugin.yaml` 已声明但尚未实现三个入口：

- `skills.vcc25.lingshu:LingshuAlignmentSkill`
- `skills.vcc25.lingshu:LingshuDataProcessingInsightSkill`
- `skills.vcc25.lingshu:LingshuModelDesignInsightSkill`

首版直接实现这些入口：

```text
JSON 知识卡
    ↓
KnowledgeStore 校验与查询
    ↓
Lingshu*Skill 生成 PromptFragment
    ↓
Agent 在 L1–L5 选择论文、方法、仓库或模型
    ↓
白名单 CandidateAdapter 校验执行配方
    ↓
固定 VCC25EvaluationContract 评测
    ↓
现有 ExperienceCard / trajectory / reward
```

检索结果通过 `PromptFragment.source`、`content_hash` 和 `knowledge_refs` 保留出处。
知识卡 ID 使用 `kb:*` 命名空间，不与运行期生成的 `ExperienceCard.card_id` 混用。

## 3. 后端与目录

首版使用 Git 管理的 JSON、JSON Schema 文档和 Python 标准库内存索引。当前本地环境没有
PyYAML/jsonschema，运行时代码不新增数据库、向量库或解析依赖；Schema 是稳定契约，Python
实现最小必填字段、类型、引用和安全校验。

```text
knowledge/vcc25/
  papers/*.json
  repositories/*.json
  models/*.json
  schemas/*.schema.json
  evaluation_contract.json

domain_knowledge/
  cards.py
  store.py
  query.py
  render.py
  validation.py
  __main__.py

skills/vcc25/
  lingshu.py

adapters/vcc25/
  lingshu.py
  state.py
  perturbench.py

tests/domain_knowledge/
tests/vcc25_adapters/
```

知识量达到数百篇且确定需要语义召回后，才考虑把相同 Schema 导入 SQLite 或向量索引。

## 4. 卡片模型

三类卡共享以下字段：

- `id`、`schema_version`、`asset_type`、`title`；
- `summary_plain`：面向研究人员和 Agent 的人话摘要；
- `tags`、`layers`：用于 L1–L5 确定性检索；
- `sources`、`relations`：来源和跨卡引用；
- `review_status`、`review_basis`、`reviewed_by`、`reviewed_at`：启用状态与审核依据。

### 4.1 PaperCard

额外记录研究问题、方法机制、数据、主要结果、适用条件、局限、失败案例以及关联仓库和模型。
按本轮项目决定，首批卡片统一设为 `approved`，使 Agent 可以立即检索；同时必须写
`review_basis: provisional_initial_release`。该字段表示“获准进入首版”，不表示负责人已经完成
逐篇精读。后续人工完整阅读后更新为 `review_basis: human_full_review`，不得删除两者的区别。

### 4.2 RepositoryCard

额外记录官方 URL、固定 revision、维护状态、代码许可证、安装方式、训练/推理入口、输入输出
格式、依赖以及 VCC25 适配差异。

### 4.3 ModelCard

ModelCard 同时表示已发布 checkpoint 和需要训练的明确模型配方，包含：

- `usage_mode`：`task_checkpoint`、`foundation_checkpoint` 或 `trainable_recipe`；
- `execution_readiness`：`ready`、`adapter_required`、`finetune_required`、`train_required`、
  `reference_only`；其中 `ready` 只用于已经通过本项目冒烟推理的资产；
- `artifacts[]`：权重、配置、词表、基因嵌入和辅助文件；
- `licenses.code` 与 `licenses.weights`：代码和权重许可证分别记录；
- `io_contract`：基因空间、输入格式、扰动字段和输出类型；
- `resource_profile`：GPU、显存和时间预算；
- `vcc25_compatibility`：兼容等级、缺口和证据；
- `execution_recipe`：只引用白名单 adapter 和结构化 action，不保存可直接执行的任意 shell。

每个 artifact 包含 `role`、`uri`、`format`、`access`、可选大小，以及完整性状态。来源没有公布
checksum 时明确写 `not_published`，不得伪造校验值。

## 5. 固定评测契约

`knowledge/vcc25/evaluation_contract.json` 是不可被 Agent 选择的系统约束，记录：

- 数据集与训练/代理验证/最终测试边界；
- 18,080 个基因及其顺序来源；
- 指标名称、定义和 evaluator 版本；
- 哪些数据允许用于模型选择；
- 测试集隔离和结果回流限制。

每个实验结果必须携带 `evaluation_contract_id`。只有 ID 相同的结果才允许比较或计算
`delta`。当前远端存在 `0.30614`、`0.09268` 和内部约 `0.667` 等不同协议分数；在负责人
确认其协议前，它们均不得自动参与模型排序。

## 6. 检索与 Agent 接口

查询对象包含：

```python
KnowledgeQuery(
    task_id="vcc25",
    layer="L5",
    text="寻找能够微调的遗传扰动模型",
    asset_types=("repository", "model"),
    readiness=("ready", "adapter_required", "finetune_required"),
    max_results=5,
    approved_only=True,
)
```

查询结果包含人话摘要、适合与不适合的原因、来源、关联资产、执行准备度、adapter ID 和内容
哈希。排序规则固定为层级匹配、标签/查询词匹配、执行准备度、ID；相同输入必须返回相同结果。

- L1/L2：优先论文、研究方向和已知局限；
- L3/L4：优先机制、假设、失败案例和数据处理知识；
- L5：优先代码库、模型、资源约束和执行配方。

默认只返回 `approved` 卡片。首批卡片均可被普通 Agent 检索，但渲染结果必须同时显示
`review_basis`，避免把临时批准误写成人工完整审核。

## 7. 执行配方与安全边界

知识库只返回声明式配方，例如：

```json
{
  "adapter_id": "vcc25.state",
  "actions": ["check_assets", "prepare", "train", "predict", "convert_output"],
  "resource_profile": {"gpu_count": 1, "gpu_type": "H200", "max_minutes": 180}
}
```

实际命令必须由白名单 CandidateAdapter 构造并校验。Adapter 负责：

- 检查依赖、权重、配置和许可证声明；
- 校验输入路径、参数和资源上限；
- 拒绝隐藏测试表达、答案、`real_de` 和最终 evaluator 参考路径；
- 把输出转换为 official H1 contract；
- 记录实际执行参数、revision、权重和输出证据。

首版 adapter 不自动克隆仓库或下载大权重；缺少资产时返回结构化的 `blocked`/`not_ready`
结果。下载和部署由明确的准备步骤完成。

## 8. 首批论文、代码库和模型

### 8.1 论文及选择理由

| 论文 | 选择理由（人话） |
|---|---|
| Lingshu-Cell | 当前要超过的目标，直接对应 VCC H1，必须弄清它的模型、权重和推理技巧。 |
| STATE | Arc 官方扰动模型并提供 VCC 训练流程，是最接近“微调后直接跑”的方案。 |
| GEARS | 遗传扰动预测经典模型，训练接口成熟，适合作为图模型 baseline。 |
| Linear Baseline | 用来判断复杂模型是否真的有效，避免花很多算力却不如简单方法。 |
| PerturBench | 把多个扰动模型放进统一框架，适合 Agent 自动换模型并公平比较。 |
| scGenePT | 已在 scGPT 基础上针对遗传扰动微调，比通用 scGPT 更接近当前任务。 |

### 8.2 代码库及选择理由

| 代码库 | 选择理由（人话） |
|---|---|
| Alibaba DAMO Lingshu-Cell | 官方 VCC 推理实现，远端已有相应权重和辅助文件。 |
| ArcInstitute/state | 有标准预处理、训练、推理 CLI 和 VCC 教程。 |
| altoslabs/perturbench | 一个接口可以训练多种轻量和复杂模型，最适合 Agent 探索。 |
| snap-stanford/GEARS | 成熟的遗传扰动训练与预测实现。 |
| czi-ai/scGenePT | 公开微调权重、词表、基因嵌入和推理代码。 |
| const-ae/linear_perturbation_prediction-Paper | 简单、快、可解释，是统一实验起点。 |

### 8.3 模型及选择理由

| 模型 | 准备度 | 选择理由（人话） |
|---|---|---|
| Lingshu VCC 85M | `adapter_required` | VCC 专用权重已经存在，但需先用本项目 Adapter 完成一次冒烟推理才能标为 `ready`。 |
| STATE ST-HVG-Replogle | `finetune_required` | 已在遗传扰动数据上训练，工具链完整，适合迁移到 H1。 |
| PerturBench LatentAdditive | `train_required` | 模型轻、训练快，适合 Agent 第一轮自动实验。 |
| scGenePT GO-All | `finetune_required` | 已学习遗传扰动和 GO 基因知识。 |
| GEARS | `train_required` | 没有 H1 专用权重，但训练流程成熟，是重要图模型路线。 |
| Linear/Pseudobulk | `train_required` | 成本最低，是所有复杂模型必须超过的底线。 |

首版只优先打通 Lingshu、STATE 和 PerturBench LatentAdditive 三条执行链路；scGenePT、GEARS
和线性方法先完成知识卡和配方，其中现有 RSI 内部线性/伪批量实现继续沿用。PRiMeFlow 虽与
VCC 高度相关，但公开流程需要多 GPU 预训练且没有公开 checkpoint，首版只作为后续
`reference_only` 候选，不进入执行范围。

## 9. 兼容性验证

每个模型依次经过：

1. 静态验证：仓库、revision、资产、许可证和版本存在；
2. 加载验证：依赖可解析，checkpoint 与代码入口匹配；
3. 小数据冒烟：使用公开训练数据的小样本完成一次推理；
4. VCC 代理验证：仅用训练集内部划分完成训练、预测和指标计算；
5. 输出验证：18,080 基因、顺序正确、无 NaN/Inf、可转 official H1 contract；
6. 资源验证：满足声明的单张 H200、180 分钟预算。

只有通过实际冒烟推理的模型才能标为 `ready`。仅检查到官方文档或权重存在时，最高只能标为
`finetune_required` 或 `train_required`。

## 10. 测试集隔离

- 论文可以记录公开最终成绩，但标记为 `reported_only`，不作为本轮搜索 reward；
- Agent 优化阶段只用训练集内部划分或外部代理数据；
- Prompt 和执行配方不得包含隐藏测试表达矩阵、答案、`real_de` 或最终参考路径；
- 最终测试只能在候选冻结后由 evaluator 执行，结果不得回流到同一轮搜索；
- validator 对受限字段和路径执行拒绝，而不是仅作提示。

## 11. 实施顺序

1. 实现卡片类型、Schema、校验、确定性查询和 curator preview；
2. 加入首批论文、仓库和模型卡，按本轮决定设为 `approved`，并统一标记
   `review_basis: provisional_initial_release`；
3. 实现三个 ResearchSkill 入口并验证 L1–L5 选择逻辑；
4. 实现 Lingshu、STATE、PerturBench 三个白名单 Adapter 的静态与小数据接口；
5. 在获准的算力环境完成冒烟推理，再更新实际准备度；
6. 视首轮结果决定是否继续打通 scGenePT 和 GEARS。

## 12. 首版验收标准

1. 三类卡片均可校验、关联和检索；
2. 首批卡片默认进入 Agent，且每条结果都显示其 `review_basis`；
3. Agent 可按 L1–L5、资产类型和执行准备度检索；
4. 固定 EvaluationContract 不出现在可选择结果中；
5. Lingshu、STATE、PerturBench 至少通过静态兼容性验证；
6. 在获准环境中至少一个模型完成小数据端到端推理后才能标为 `ready`；
7. 所有输出均通过 VCC25 形状、基因顺序和安全校验；
8. 原有 memory、reward、trajectory 测试不受影响；
9. 远端源码和既有权重保持不变。

## 13. 实施前需确认的外部约束

- `0.306` 对应的准确数据划分、指标定义和 evaluator 版本；
- 单次 Agent 实验是否固定为一张 H200、180 分钟；
- 是否允许把 STATE/scGenePT 权重下载到共享存储；
- 是否允许使用 Replogle、Norman 等外部扰动数据；
- 谁负责把论文卡的 `review_basis` 从临时批准更新为人工完整阅读；
- 最终测试结果的归档和禁止回流策略。

这些未确认项不会阻止知识库、卡片和静态 Adapter 的实现，但会阻止相关模型升级为 `ready`
或把论文标记为 `human_full_review`。
