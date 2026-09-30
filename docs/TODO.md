# LoopRSI / VCC25 TODO

## 正式实验前：知识库论文复现扫查

正式 VCC25 请求前，必须由分层 LoopAgent 读取知识库中的 PaperCard、CodeCard 和 ModelCard，逐项尝试复现下列论文对应的代码、模型或训练/推理入口。静态知识卡片不等同于复现完成。

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
- [ ] 分层 LoopAgent 汇总 EvidenceCard，并自主选择 VCC25 的论文、代码和模型。
- [ ] 生成唯一 `candidate_request`，其中包含实际引用的 `card_id` 和选择理由。
- [ ] 复现证据、候选请求和代码版本归档后，才进入 CUDA smoke、完整验证和最终评测门禁。
- [ ] 最终测试表达矩阵在上述阶段均不得读取，最终评测结果不得回流 Agent。

## 当前已知状态

- 知识库基础卡片已通过校验：9 PaperCard、9 CodeCard、9 ModelCard，共 27 张；其中 PerturBench 作为基础设施条目保留，不是本轮正式复现对象。
- 正式复现目标为 8 个方法。第一轮逐篇入口预检已启动；当前清单见 `knowledge/vcc25/evidence/paper-reproduction-status-20260930.json`：Lingshu 为 `partial`，其余 7 个正式目标因适配器或运行资产缺口暂为 `blocked`。
- Lingshu 已有 CUDA smoke 和完整训练/验证证据，但这不代表其他论文已完成复现。
- 现有验证结果可以作为 EvidenceCard 证据，不以“两个指标同时提升”作为建立 ModelCard 的前置条件。
- 在论文复现扫查、EvidenceCard 汇总和候选请求生成完成前，不得启动唯一一次正式 VCC25 评测。
