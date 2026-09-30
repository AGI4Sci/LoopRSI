# VCC25 领域知识库使用说明

这个目录给 Agent 提供三类可检索“弹药”：论文说明为什么做，仓库说明代码从哪里来，模型卡说明当前能否直接执行。VCC25 评测规则是固定系统契约，不是第四类可选卡片。

## 最小工作流

```bash
python3 -m domain_knowledge --root knowledge/vcc25 validate
python3 -m domain_knowledge --root knowledge/vcc25 query --layer L5 --text perturbation --readiness adapter_required finetune_required --token-budget 400
python3 -m domain_knowledge --root knowledge/vcc25 check-adapter --model-id kb:model:lingshu-vcc-85m
```

`query` 只返回完整卡片，预算不够时整张省略，避免来源被截断。输出里的 `card_ids`、`content_hash` 和 `evaluation_contract_id` 应随实验轨迹一起保存。

第二版模型范围固定为：Lingshu-Cell、PRiMeFlow、STATE、GEARS、PRESAGE、scGenePT、scLAMBDA，以及 Linear/Pseudobulk baseline。PerturBench 仅作为 PRiMeFlow 的依赖基础设施记录，`research_role=dependency_infrastructure`，不作为独立研究候选排序。

PRiMeFlow、PRESAGE 和 scLAMBDA 的卡片在官方论文标识、仓库地址、固定 revision 和权重尚未完成核验前，只能用于方向检索，不能直接执行；它们的 `smoke_evidence` 保持为空。

## 新增或修改卡片

1. 在 `knowledge/vcc25/papers`、`repositories` 或 `models` 中复制同类 JSON，使用唯一的 `kb:*` ID。
2. 用人话填写 `summary_plain`、`fit_reason` 和 `limitations`，来源至少包含一个权威 HTTPS 地址。
3. 模型权重不进 Git；`artifacts` 只记录角色、URI、格式、访问条件、大小和完整性状态。
4. 更新跨卡 `relations`，再运行 `validate` 和 `tests/domain_knowledge`。

首批卡片的 `review_status: approved` 表示允许 Agent 使用；`review_basis: provisional_initial_release` 表示尚不能当作负责人逐篇精读后的结论。人工完整阅读并核对事实后，才改成 `human_full_review`，同时记录审阅人和时间。

## 固定评测与结果

`evaluation_contract.json` 固定任务、18,080 基因顺序、指标和隔离规则，不参与检索。实验结果必须记录相同的 `evaluation_contract_id` 才能比较。公开最终测试成绩只能作为 `reported_only` 背景信息，不能在同一轮搜索中充当 reward。

## 资产映射和适配器

`check-adapter` 默认只报告缺少的资产角色；可用 `--asset checkpoint=/absolute/path` 重复映射本地文件。适配器只构造白名单 `CandidateExecutionRequest`，不会克隆、下载或启动程序，并会拒绝路径穿越、未声明环境变量、超预算资源和受限评测标识。

静态资产齐全也不会把模型自动标成 `ready`。需要在批准的计算节点上，用公开训练数据小样本完成加载、推理、18,080 基因顺序及 NaN/Inf 检查，并把可审计证据写回 `smoke_evidence` 后，才能单独提升准备度。

Lingshu 独立副本的两条件推理 smoke 已通过，证据见 `knowledge/vcc25/evidence/lingshu-v2-smoke-20260930.json`。它验证了输出内部基因顺序一致、形状和数值，但尚未与正式 VCC25 基因顺序逐项比对，也没有训练或完整验证，因此模型仍为 `adapter_required`，不产生 reward。

## 五层知识实验

本地无 GPU 预检只验证 L1–L5 知识注入、结构化决策和 L5 编码请求，不训练模型，也不产生或比较分数：

```bash
python3 -m hier_loop.cli knowledge-preflight \
  --config configs/rsi_step0/hier_vcc25_knowledge.json \
  --workspace /tmp/vcc25-knowledge-preflight
```

检查每层实际使用的卡片时，只读 `layer_decisions.jsonl` 中的 `skill_id`、`card_ids`、`content_hash`、`token_budget` 和 `source`；不要把最终测试表达矩阵或内容复制进日志。`candidate_request.json` 只是编码计划，预检阶段不会调用 Codex 或 GPU。

远端 Phase A 必须先创建不位于两个只读源码目录内的新实验目录，再传入测试过的代码快照：

```bash
ssh -i /Users/edy/.ssh/id_ed25519_ailab -o IdentitiesOnly=yes \
  gzy-rsi.gaozhangyang+root.ailab-deepdivegzy.ws@h.pjlab.org.cn
python3 -m hier_loop.cli knowledge-preflight \
  --config configs/rsi_step0/hier_vcc25_knowledge.json \
  --workspace "$EXPERIMENT_DIR/artifacts/phase-a-preflight"
```

阶段标签和门禁：

- `smoke_only` 只证明小数据流程可运行，不产生 reward，也不能晋升候选。
- 只有完整的 `search_validation` 可以计算 `candidate_validation_pcc - same_contract_lingshu_validation_pcc` 并进入晋升判断。
- 候选必须先按代码哈希冻结，且验证提升、种子完整和泄漏审计全部通过，才能进入 `final_official`。
- `final_official` 整个实验最多运行一次，必须记录 `selection_feedback_allowed=false`；结果只用于最终报告，不能回流 Agent、memory 或新一轮候选选择。
