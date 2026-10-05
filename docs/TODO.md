# LoopRSI / VCC25 TODO

## 正式实验前：知识库论文复现扫查

正式 VCC25 请求前，LoopRSI 必须在独立实验目录内读取 PaperCard、CodeCard 和 ModelCard，逐项尝试复现下列方法。当前执行方式已改为本机运行 LoopRSI 和固定版本的官方源码；只有确有可行实验时，才通过 SSH 将 GPU 计算提交到远端。源码检查、静态知识卡片和模型实际复现是不同阶段，不得混记。

每篇论文完成后必须生成一条 EvidenceCard，并记录：复现状态、代码版本或提交、模型/权重、运行命令、数据范围、验证指标、限制、未读取最终测试表达矩阵的证明，以及对应的 PaperCard、CodeCard、ModelCard ID。状态只能使用：

- `reproduced`：按当前任务契约完成可重复运行，并有结果证据。
- `partial`：部分入口或子集运行成功，但尚未完成完整复现。
- `blocked`：因代码、依赖、权重、资源或契约不兼容而无法继续，并记录阻塞原因。
- `reference_only`：只有论文/资料信息，当前没有可执行复现入口。

论文清单：

- [ ] GEARS
- [ ] Linear perturbation prediction baseline
- [ ] Lingshu-Cell
- [ ] PRESAGE
- [ ] PRiMeFlow
- [ ] scGenePT
- [ ] scLAMBDA
- [ ] STATE

完成标准：

- [ ] 每篇论文都有明确复现状态，不允许留空或只写“已收录”。
- [ ] 每篇论文都有对应 EvidenceCard；`blocked` 和 `reference_only` 也必须有原因记录。
- [ ] EvidenceCard 必须来自 LoopRSI 的真实尝试；若受网络、依赖、权重、适配或资源阻塞，记录实际命令、版本、数据范围、错误摘要和后续可行条件，不得伪造运行或指标。本机源码预检和远端 GPU 运行应分别标明执行位置。
- [ ] 分层 LoopAgent 汇总 EvidenceCard，并自主选择 VCC25 的论文、代码和模型。
- [ ] 生成唯一 `candidate_request`，其中包含实际引用的 `card_id` 和选择理由。
- [ ] 复现证据、候选请求和代码版本归档后，才进入 CUDA smoke、完整验证和最终评测门禁。
- [ ] 最终测试表达矩阵在上述阶段均不得读取，最终评测结果不得回流 Agent。

## 当前已知状态

- 知识库基础卡片已通过校验：9 PaperCard、9 CodeCard、9 ModelCard，共 27 张；其中 PerturBench 作为基础设施条目保留，不是本轮正式复现对象。
- 正式复现目标为 8 个方法。统一预检入口已覆盖 8 个目标；当前清单见 `knowledge/vcc25/evidence/paper-reproduction-status-20260930.json`：Lingshu 为 `partial`，GEARS、Linear、PRESAGE、PRiMeFlow、scGenePT、scLAMBDA 因远端 GitHub HTTPS 拉取超时而缺上游代码副本，STATE 因缺 checkpoint 暂为 `blocked`。
- 远端 LoopRSI 的 `reproduce-papers` 入口已逐项读取 8 组 PaperCard、CodeCard、ModelCard，并在独立实验目录运行受限源码探测；8 张逐篇 EvidenceCard 与汇总见 `artifacts/remote-looprsi-paper-reproduction-20260930/`。Lingshu 为 `partial`，引用远端既有完整验证原始产物；其余 7 项因各自的 GitHub `git ls-remote` 在 15 秒内超时而为 `blocked`。没有运行其他方法的训练或推理，不能把这次源码尝试称为完整论文复现。
- 用户已授权改为本机拉取官方源码并运行 LoopRSI 控制流程，仅在有可行实验时通过 SSH 使用远端 GPU。本机通过 GitHub API 和 codeload 固定取得 8 个官方仓库的提交及归档哈希；源码清单在 `work/vcc25-sources/manifest.json`（本机独立 Codex 工作目录）。本机 `reproduce-papers --source-manifest` 已核验 8 个归档与入口并生成独立 EvidenceCard：Lingshu 仍为 `partial`，其余 7 项仍为 `blocked`；7 个 Python 入口语法通过，线性基线的 R 入口因本机缺 `Rscript` 未通过。该检查不等于方法训练或推理复现，未提交 GPU job。
- 此前 `codex exec` 的模型服务超时只说明可选 Codex 决策后端不可用，不能作为 LoopRSI 启动阻塞；旧启动尝试日志保留在 `knowledge/vcc25/evidence/remote-agent-start-attempt-20260930.json` 供审计。`knowledge_vcc25` 现有五层入口仍使用 `RecordingDecisionBackend`，其固定预检决策不是逐篇复现或自主候选选择。
- Lingshu 已有 CUDA smoke 和完整训练/验证证据，但这不代表其他论文已完成复现。
- 现有验证结果可以作为 EvidenceCard 证据，不以“两个指标同时提升”作为建立 ModelCard 的前置条件。
- 下一步是在本机固定源码上接通方法专用的 H1 训练/推理执行和分层 Agent 对逐篇 EvidenceCard 的汇总选择；远端 GPU 只由本机经 SSH 向带 `_pool` 的 charged group 提交已核验可行的单卡作业。正式 `candidate_request` 尚未由 Agent 引用证据 `card_id` 生成；候选冻结、完整验证和唯一一次正式 VCC25 评测仍被门禁阻止。

## 接口接通情况与待办

当前调用链是 `KnowledgeStore.query` 检索三类卡片 → ResearchSkill 将卡片注入五层 Agent → 模型卡的 `execution_recipe.adapter_id` 找到适配器。`check_assets` 只检查资产路径；`prepare` 只生成受约束的 `CandidateExecutionRequest`，不会自己执行。`reproduce-papers` 目前执行的是源码和入口检查；`knowledge_vcc25` 运行入口仍使用返回固定决策的 `RecordingDecisionBackend`。现有 `Vcc25Worker` 可以运行固定的 VCC25 实验入口，但尚未按 Agent 选择切换到八种上游方法。

可在项目内完成的 TODO：

- [ ] 为八种方法逐项接通真实的 H1 训练或推理入口：映射固定源码、权重和输入数据，执行适配器请求，校验 18,080 基因的输出顺序，并将命令、版本、数据范围、指标与失败原因写入逐篇 EvidenceCard。`reproduction_entry.py` 当前只报告源码是否存在，不能把 `partial` 当作训练或推理成功。
- [ ] 给 `knowledge_vcc25` 接入可用的真实决策后端，让 L1-L5 读取逐篇 EvidenceCard 后生成选择理由；校验候选请求确实引用所选论文、仓库、模型及证据的 `card_id`，再交给执行器。固定的 `RecordingDecisionBackend` 只用于预检。
- [ ] 把“Agent 选择 → 对应方法适配器 → 执行器 → EvidenceCard/验证结果”串成可审计流程，并以一次小规模真实运行验证；缺少权重或依赖时记录 `blocked`，不填造指标。线性基线的本机 R 入口还需安装或配置 `Rscript` 后重试。

当前外部阻塞：

- STATE 尚缺可用 checkpoint。需取得并核验权重，或先完成其训练流程，才能进行相应推理复现；这不阻止其他方法继续。
- 正式 VCC25 的基因顺序来源和 evaluator 精确版本仍待任务负责人确认。在确认前，只能记录代理验证结果，不能声称与 `0.306` 基线同口径，也不能开启正式评测。

远端 GitHub HTTPS 超时已有本机固定源码归档作为当前替代路径，因此不再把它列为本机复现的总阻塞；每张旧证据卡中的远端失败记录仍保留。
