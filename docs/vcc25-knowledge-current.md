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

## 3. 卡片与 EvidenceRecord Schema

所有卡片都共享以下字段：

```json
{
  "id": "kb:<paper|repo|model>:<slug>",
  "schema_version": "1.0",
  "asset_type": "paper | repository | model",
  "title": "…",
  "summary_plain": "给 Agent 看的短摘要",
  "tags": ["…"],
  "layers": ["L1", "L2", "L3", "L4", "L5"],
  "sources": [{"url": "https://…"}],
  "relations": ["kb:repo:…", "kb:model:…"],
  "review_status": "draft | approved | rejected",
  "review_basis": "人工阅读或审核依据"
}
```

`id` 必须使用对应前缀；`layers` 只能包含 L1–L5；`sources` 至少包含一个 URL；`relations` 必须指向知识库中已存在的卡片。任何卡片中的敏感标识符（例如 `real_de`、`test_answers`、`hidden_test`）都会被递归拒绝。

### PaperCard

除共享字段外，PaperCard 应记录：

```json
{
  "problem": "论文解决的问题",
  "method_plain": "方法的人话说明",
  "mechanism": "核心机制",
  "conclusions": ["主要结论"],
  "applicability": ["适用条件"],
  "limitations": ["局限"],
  "related_repositories": ["kb:repo:…"],
  "related_models": ["kb:model:…"]
}
```

AI 整理内容默认应为 `draft`；只有完成人工阅读审核后才可标记为 `approved`。

### RepositoryCard

除共享字段外，RepositoryCard 应记录：

```json
{
  "revision": "固定 commit/tag",
  "license": "代码许可证",
  "maintenance": "维护状态",
  "install": ["安装方法或依赖说明"],
  "entrypoints": ["训练/推理入口"],
  "io_contract": {"input": "…", "output": "…"},
  "vcc25_fit": {"status": "…", "gaps": ["…"]},
  "related_models": ["kb:model:…"]
}
```

### ModelCard

除共享字段外，ModelCard 的当前 schema 重点字段如下：

```json
{
  "usage_mode": "预训练 / 微调 / 重新训练 / 参考",
  "execution_readiness": "ready | adapter_required | finetune_required | train_required | reference_only",
  "artifacts": [{
    "role": "code | weights | config | vocabulary | gene_embedding",
    "uri": "地址或受控存储引用",
    "format": "格式",
    "access": "访问方式",
    "integrity": {"size": "…", "checksum": "…", "revision": "…"}
  }],
  "licenses": {"code": "…", "weights": "…"},
  "io_contract": {"gene_space": "…", "output_type": "…"},
  "resource_profile": {"gpu_count": 1, "gpu_type": "H200", "max_minutes": 180},
  "vcc25_compatibility": {"level": "…", "gaps": ["…"]},
  "execution_recipe": {"adapter_id": "vcc25.…", "actions": ["check_assets", "prepare", "train", "predict", "convert_output"]},
  "smoke_evidence": ["evidence record 引用"]
}
```

当 `execution_readiness` 为 `ready` 时，`smoke_evidence` 至少要有一条通过记录。权重不进入 Git；卡片只记录地址、大小、版本、角色、格式、访问方式和 integrity/checksum 信息。

### EvidenceRecord

EvidenceRecord 已有实现：`domain_knowledge.EvidenceRecord` 定义结构，`EvidenceStore.from_directory()` 会加载 `knowledge/vcc25/evidence/*.json`，并由 `KnowledgeStore.evidence` 暴露。它不是独立可检索卡片类型，也不参与 Paper/Repository/Model 的资产类型过滤；它是实验、预检、smoke、checkpoint round-trip 或环境探针产生的结构化记录，由 ModelCard 的 `smoke_evidence` 或其他字段引用。Adapter 生命周期中的 `ReproductionEvidence` 是生成这类记录的控制器侧类型。

当前结构约定：

```json
{
  "id": "evidence:<slug>:<date-or-run>",
  "record_type": "smoke | preflight | validation | probe | reproduction",
  "status": "passed | failed | blocked | passed_with_persistence_failure",
  "authority": "smoke_only | full_validation | infrastructure_probe",
  "model_id": "kb:model:…",
  "repository_id": "kb:repo:…",
  "source_manifest": "knowledge/vcc25/evidence/…json",
  "code_commit": "…",
  "weights_revision": "…",
  "evaluation_contract_id": "vcc25-eval-v1",
  "validated_gene_count": 18080,
  "output_checks": ["finite", "shape_checked", "gene_order_consistent"],
  "metrics": {},
  "selection_feedback_allowed": false,
  "notes": ["限制、失败原因或环境说明"]
}
```

Evidence 只能证明它实际验证过的范围；`infrastructure_probe` 不等价于模型已经可用，也不能单独把模型提升为 `ready`。

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

当前 contract 已登记 evaluator revision 和 gene-order revision 字段；官方 gene-order 文件尚未进入仓库，因此 `gene_order_sha256` 仍需用 owner 提供的正式基因顺序文件计算后替换。

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

### 加载和校验

`KnowledgeStore.from_directory()` 首先加载 `evaluation_contract.json`，然后按 `papers/`、`repositories/`、`models/` 三个目录读取 JSON。每张卡片依次经过：

1. JSON 可解析性和对象类型检查；
2. 共享字段、ID 前缀、层级、来源、审核状态校验；
3. ModelCard 专属字段、资产角色、readiness、recipe 和 smoke evidence 校验；
4. 敏感标识符递归扫描；
5. 卡片 ID 唯一性和 relations 引用完整性校验。

加载后 KnowledgeCard 是不可变视图，列表和嵌套对象被冻结，避免 Agent 或调用方修改知识库内存状态。

### 查询参数

`KnowledgeQuery` 支持：

- `task_id`；
- 层级 `L1`–`L5`；
- 文本查询；
- 资产类型过滤；
- readiness 过滤；
- 最大返回数量；
- 默认只返回 `approved` 卡片。

### 候选过滤

检索前先做硬过滤：

- `task_id` 不是 `vcc25` 时不返回任何卡片；
- `approved_only=True` 时排除 `draft` 和 `rejected`；
- `asset_types`、`readiness`、`layer` 作为白名单过滤；
- 有文本查询时，标题、摘要或标签完全没有 token 命中则排除；
- 查询文本和过滤参数本身也进行敏感标识符检查。

### 确定性排序

排序使用升序 tuple key，越靠前越优先：

1. `layer_rank`：卡片包含查询层级为 0，否则为 1；
2. `weighted_overlap`：文本 token 命中分数取负值，标题命中权重 3、标签权重 2、摘要权重 1，因此分数越高越靠前；
3. `READINESS_ORDER`：`ready` → `adapter_required` → `finetune_required` → `train_required` → `reference_only`；
4. `-smoke_evidence_score`：通过的 smoke evidence 越多越靠前，但 `infrastructure_probe` 不计入；
5. `COMPATIBILITY_ORDER`：`direct_task_candidate` → `existing_internal_route` → `finetune_candidate` → `trainable_candidate` → `future_finetune_candidate` → `future_train_candidate` → `reference_only`；
6. `card.id`：最终稳定 tie-break，保证同一输入每次返回相同顺序。

这意味着 readiness 不是唯一标准：两个同为 `train_required` 的模型，会继续按有效 smoke evidence 和 VCC25 compatibility 排序。

### L1–L5 的检索方式

检索不是五层各自维护一套数据库，而是使用同一个 KnowledgeStore，通过层级、资产类型和文本意图改变查询参数：

| 层级 | 主要检索内容 | 典型方式 | 返回重点 |
|---|---|---|---|
| L1 方向 | PaperCard | `layer="L1"`, `asset_types=("paper",)`，按研究方向/问题文本匹配 | 研究问题、方法类别、适用条件 |
| L2 问题 | PaperCard，必要时补充 RepositoryCard | `layer="L2"` + 问题文本 token overlap | 论文结论、数据条件、已知限制 |
| L3 假设 | PaperCard/RepositoryCard/ModelCard | `layer="L3"`，按机制、标签、局限检索 | 机制、可检验假设、失败模式 |
| L4 机制 | PaperCard/RepositoryCard/ModelCard | `layer="L4"`，按机制和实现约束检索 | 训练/推理机制、资源和适配差异 |
| L5 编码请求 | ModelCard + RepositoryCard | `layer="L5"`，通常加 `asset_types` 和 `readiness`；可按模型名或路线文本定向 | Adapter ID、动作、资产、smoke evidence、限制 |

L5 查询优先返回可执行模型候选；对模型卡的 `relations` 会补入关联 RepositoryCard，使 Agent 同时看到“模型是什么”和“代码从哪里来”。查询结果随后交给渲染器，按 token budget 只保留完整卡片，并返回 `card_ids`、`content_hash` 和来源信息。

渲染结果包含卡片 ID、摘要、限制、Adapter ID、来源、smoke evidence 和内容哈希；超出 token budget 时只丢弃完整卡片，不截断单张卡片。

## 7. ResearchSkill / KnowledgeBridge

### ResearchSkill 的职责

ResearchSkill 不拥有另一份卡片数据库，也不直接执行训练命令。它负责把 Agent 当前层级和研究问题转换成受限的 KnowledgeQuery，再把检索结果渲染成有 provenance 的 PromptFragment/SkillResult。

当前 Lingshu skill 分为三类：

- `vcc25.lingshu.alignment`：主要服务 L1/L2，注入论文卡和方向/问题依据；
- `vcc25.lingshu.data_processing`：主要服务 L3/L4，注入数据处理机制、适用条件和限制；
- `vcc25.lingshu.model_design`：主要服务 L5，注入 RepositoryCard + ModelCard，支持模型候选比较。

标准调用过程为：

1. 接收 `SkillContext`，读取 `research_state.task_id`、问题文本、当前层级和约束；
2. 校验 task/layer 是否匹配，拒绝非 VCC25 或不支持层级；
3. 构造 `KnowledgeQuery`，设置层级、文本、资产类型、readiness 和 token budget；
4. 调用 KnowledgeStore 做硬过滤和确定性排序；
5. 渲染完整卡片，附带 `card_id`、source URL、Adapter ID、限制和 evidence 来源；
6. 返回 `SkillResult`/`PromptFragment`，包含 skill ID、内容哈希、卡片来源和下一步 gate；
7. Agent 只能基于该有界结果提出 idea 或 coding request，不能通过 Skill 读取任意路径或最终测试答案。

### KnowledgeBridge 的职责

KnowledgeBridge 是 ResearchSkill 与五层 Agent 之间的路由和完整性检查层。它会：

- 为每个 L1–L5 选择唯一激活的 skill；
- 检查 task ID、非空内容、`knowledge:vcc25:` provenance 前缀；
- 重新计算内容 hash 并和 fragment 声明值比较；
- 检查 token budget 为正数；
- 将卡片 ID、skill ID、内容、hash 和预算封装为 `KnowledgeInjection`；
- 把检索到的卡片传给 Agent 的候选比较和 L5 选择逻辑。

L5 当前支持多候选选择，Agent 需要在 coding request 中给出 `selected_model_id`、`selected_repository_id`、`selection_reason` 和 `rejected_alternatives`。

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

## 10. 待对接与优先级

### 支持 Agent 动态提出新 idea

- 让 Agent 通过 `KnowledgeQuery` 返回多张相关卡片及限制条件；
- 定义 idea/candidate-plan 的结构化输出 schema；
- 记录 idea 使用了哪些 card IDs、provenance hashes 和被拒绝的替代方案；
- 将 idea 生成与 memory、validation record 解耦，避免最终测试结果反向污染搜索。

### 扩展执行覆盖

- 完成 GEARS 和 PRiMeFlow 的兼容性、smoke 和资源门槛验证；
- 再依次推进 STATE、scGenePT、PerturBench 等路线；
- 将 `adapter_required` 正式写入统一 readiness 文档和 schema。

## 11. 当前验收结论

知识库基础设施和卡片检索已经可用，且比旧设计增加了多候选排序、EvidenceStore、Adapter 检查和 L5 选择记录。当前尚不能宣称“Agent 已经完全通过知识库自主提出并闭环验证新 idea”，主要剩余边界是结构化 idea schema、trajectory/reward 的 contract 强校验，以及官方 gene-order checksum 的确认。

配套架构图：[vcc25-knowledge-current-v3.png](vcc25-knowledge-current-v3.png)。图中以绿色标注已实现组件，以黄色虚线标注仍需继续对接的边界。

