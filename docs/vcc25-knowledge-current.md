# VCC25 Knowledge Base（LoopRSI 当前实现说明）

## 1. 目标与边界

知识库只为 Agent 提供可审阅、可追踪的研究依据：

- 论文与研究方法（`PaperCard`）；
- 开源代码库（`RepositoryCard`）；
- 可运行、可微调或可重新训练的模型（`ModelCard`）。

VCC25 是固定评测任务，不作为可选择的 benchmark 知识。最终测试答案、`real_de` 和其他隐藏测试数据不得进入检索结果、模型选择或搜索奖励。

知识库提供“依据和候选方案”；memory、trajectory、reward 和 validation record 负责记录实际实验结果。Evidence record 不是第四类卡片，而是被 ModelCard 引用的验证和 provenance 记录。

## 2. 当前系统关系

```text
PaperCard / RepositoryCard / ModelCard
                 ↓
       KnowledgeStore（JSON + Schema）
       校验、过滤、确定性排序、内容哈希
                 ↓
 ResearchSkill / KnowledgeBridge
  注入有界卡片摘要和 provenance
                 ↓
 KnowledgeQuery 检索候选 + Agent 比较候选
                 ↓
  Agent 动态提出新 idea / candidate plan
                 ↓
 L1–L5：方向 → 问题 → 假设 → 机制 → 编码请求
                 ↓
 受控 VCC25 Adapter 执行
                 ↓
 固定 VCC25EvaluationContract 评测
                 ↓
 trajectory / reward / validation / Memory
```

L5 已支持多候选选择，并记录 `selected_model_id`、`selected_repository_id`、`selection_reason` 和 `rejected_alternatives`。

## 3. 卡片类型

### PaperCard

记录论文解决的问题、方法的人话说明、核心机制、主要结论、适用条件、局限、关联仓库和模型，以及审核状态。AI 整理内容默认应为 `draft`；只有完成人工阅读审核后才可标记为 `approved`。

### RepositoryCard

记录官方仓库、固定 revision/commit、许可证、维护状态、安装方式、训练和推理入口、输入输出格式、VCC25 适配差异及关联模型。

### ModelCard

记录模型类型和用途、代码/权重/配置等资产、各自许可证、输入基因空间、输出类型、资源需求、VCC25 兼容性、白名单 Adapter 和受控执行配方。

当前实现的 `execution_readiness` 包含：

- `ready`：已具备可直接推理条件，并有通过的 smoke evidence；
- `adapter_required`：资产或方法可用，但仍需适配执行接口；
- `finetune_required`：加载已有权重后需要微调；
- `train_required`：有代码或方法依据，但需要重新训练；
- `reference_only`：暂时只能作为研究参考。

`adapter_required` 是当前实现相对于旧设计新增的状态，后续应在正式 schema/文档中继续固化。

模型权重不进入 Git；卡片只记录地址、大小、版本、角色、格式、访问方式和 integrity/checksum 信息。

## 4. 固定评测 Contract

当前文件：`knowledge/vcc25/evaluation_contract.json`。

当前已固定：

- 任务 ID：`vcc25`；
- 基因数量：18,080；
- 指标：PCC；
- 模型选择只能使用训练集划分或批准的外部代理数据；
- 最终测试答案隔离到候选冻结之后；
- 公开最终分数只作记录，不混入本地搜索 reward。

当前仍需补齐：精确 evaluator revision 和最终 gene-order owner confirmation。目前 contract 中仍保留 “exact evaluator revision pending owner confirmation” 的说明。

所有实验结果最终都必须带 `evaluation_contract_id`。知识库和 CLI 已能输出该 ID，但它尚未在所有 adapter、trajectory、reward 和结果存储路径中形成强制的端到端约束，属于待对接项。

## 5. 当前卡片和 Adapter 覆盖

当前知识库已包含论文、仓库和模型卡，并登记以下主要 Adapter：

- Lingshu；
- GEARS；
- STATE；
- PerturBench；
- PRiMeFlow；
- scGenePT；
- linear baseline、PRESAGE、scLAMBDA 等参考路线。

当前验证状态并不等于全部 ready：Lingshu 和 GEARS 已有较明确的 smoke evidence；PRiMeFlow 仍在兼容性和 smoke 验证阶段；STATE、scGenePT、PerturBench 等主要仍是 finetune/train 路线或候选路线。

## 6. KnowledgeStore 与检索

后端仍使用 Git 管理的 JSON、JSON Schema 和 Python 标准库，不使用 PostgreSQL、Elasticsearch 或向量数据库。

`KnowledgeQuery` 支持：

- `task_id`；
- 层级 `L1`–`L5`；
- 文本查询；
- 资产类型过滤；
- readiness 过滤；
- 最大返回数量；
- 默认只返回 `approved` 卡片。

排序是确定性的，综合考虑：

1. 当前层级匹配；
2. 标题、标签、摘要的加权文本重叠；
3. 执行准备度；
4. 通过的 smoke evidence 数量；
5. VCC25 compatibility level；
6. 卡片 ID 稳定排序。

渲染结果包含卡片 ID、摘要、限制、Adapter ID、来源、smoke evidence 和内容哈希；超出 token budget 时只丢弃完整卡片，不截断单张卡片。

## 7. ResearchSkill / KnowledgeBridge

当前已存在 Lingshu 对齐、数据处理和模型设计相关 ResearchSkill，并可向 Agent 注入有界的研究知识和 provenance。KnowledgeBridge 负责将知识注入层与五层 Agent、Adapter 选择和执行请求连接起来。

仍待完善的部分：

- 将 `KnowledgeQuery` 作为稳定的通用多卡片检索接口，供不同任务和不同 Agent 层统一调用；
- 完成多模型、多任务 ResearchSkill 的接口对齐；
- 统一旧测试/旧调用协议与当前 `SkillContext -> SkillResult` 接口；
- 让 Agent 能基于检索到的多张卡片显式生成、比较和记录新的 idea，而不是只选择已有候选。

## 8. 受控执行接口

卡片不保存任意 shell 命令，只引用白名单 Adapter 和有限动作，例如：

```json
{
  "adapter_id": "vcc25.gears",
  "actions": ["check_assets", "prepare", "train", "predict", "convert_output"]
}
```

Adapter 负责检查资产和依赖、校验输入路径、阻止读取最终测试答案、生成受控训练/推理请求、转换为官方 H1 输出格式，并保存实际参数和产物。

## 9. 兼容性验证门槛

模型应依次通过：

1. 仓库、权重、许可证和版本检查；
2. Python 包和 checkpoint 加载检查；
3. 小数据 smoke 推理；
4. 训练集内部代理评测；
5. 18,080 基因、正确顺序、无 NaN 的输出检查；
6. 单张 H200、180 分钟资源约束检查。

没有通过实际 smoke 测试的模型不能标记为 `ready`。

## 10. 已完善内容

- 三类卡片和 JSON Schema 已能加载、校验、检索；
- 默认过滤未审核卡片；
- 有确定性多信号排序；
- 有固定 VCC25 contract 对象；
- 有测试集敏感标识符拒绝逻辑；
- 有 Adapter 白名单和资产检查；
- 有 model smoke evidence 和 compatibility 分级；
- L5 已支持多候选比较和选择结果记录；
- 有 provenance、来源和内容哈希；
- 有多条模型路线及验证记录。

## 11. 待对接与优先级

### P0：必须先完成

- 修复 ResearchSkill 新旧接口不一致导致的测试失败；
- 将 `evaluation_contract_id` 强制贯穿 adapter 输出、实验结果、trajectory、reward 和 ΔPCC 比较；
- 锁定 evaluator revision 和 gene-order 来源。

### P1：支持 Agent 动态提出新 idea

- 让 Agent 通过 `KnowledgeQuery` 返回多张相关卡片及限制条件；
- 定义 idea/candidate-plan 的结构化输出 schema；
- 记录 idea 使用了哪些 card IDs、provenance hashes 和被拒绝的替代方案；
- 将 idea 生成与 memory、validation record 解耦，避免最终测试结果反向污染搜索。

### P2：扩展执行覆盖

- 完成 GEARS 和 PRiMeFlow 的兼容性、smoke 和资源门槛验证；
- 再依次推进 STATE、scGenePT、PerturBench 等路线；
- 将 `adapter_required` 正式写入统一 readiness 文档和 schema。

## 12. 当前验收结论

知识库基础设施和卡片检索已经可用，且比旧设计增加了多候选排序、smoke evidence、Adapter 检查和 L5 选择记录。当前尚不能宣称“Agent 已经完全通过知识库自主提出并闭环验证新 idea”，因为通用 ResearchSkill 接口、contract 全链路绑定和结构化 idea 生成仍需完成。

