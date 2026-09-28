# OMNI-AR 统一任务契约

## 目标

框架只理解任务契约，不理解 CIFAR、VCC25、图像或单细胞等领域语义。每个 case 用一个 `task_spec.yaml` 声明任务、入口、数据、指标和搜索边界。Heuresis 输出统一 proposal，trial 执行后由 controller 把 case 原始 JSON 映射为统一 result。

```text
task_spec.yaml
     │
     ├── task / data / entrypoints ──> Heuresis proposal
     │                                  │
     └── metrics / search policy ───────┼──> Codex + trial
                                        │
case raw JSON ── metric.source mapping ─┴──> standardized result JSON
```

## 三个版本化对象

| 对象 | 版本 | Schema | 作用 |
|---|---|---|---|
| Task spec | `omni-ar-task/v1` | `schemas/task_spec.schema.json` | 定义一个 case 的固定契约 |
| Proposal | `omni-ar-proposal/v2` | `schemas/proposal_v2.schema.json` | 定义不含命令的结构化实验建议 |
| Result | `omni-ar-result/v2` | `schemas/result_v2.schema.json` | 定义跨任务可选择、可审计的 trial 结果 |
| Regression suite | `omni-ar-regression/v1` | `schemas/regression_suite.schema.json` | 冻结代表性方法、编译参数和历史结果锚点 |

完整通用示例位于 `controller_bash/examples/`。实际 case 的规范入口：

- CIFAR：`tasks/cifar/task_spec.yaml`
- VCC25：`tasks/vcc25/task_spec.yaml`

旧 case 目录内的 task spec 仅为兼容已有运行目录保留；新一轮 controller、
Heuresis 和 rjob 应使用上述 `tasks/<task>/task_spec.yaml`。

两者顶层结构和校验规则完全相同，只替换具体字段。

统一 adapter 位于 `tasks/<task>/`，每个 adapter 必须实现：

```text
prepare_data
validate_data
run_baseline
run_trial
evaluate
summarize_results
```

`task_contract.py` 会加载 task spec 指定的 adapter class，并检查六个方法都存在。BADE/AFR、CRPM/pseudo-bulk 等算法仍属于任务后端，不进入 controller。

## Task spec 约定

- `workspace.project_root` 相对 task spec 所在目录解析。
- `workspace.base=repository` 时，project root 相对 OMNI-AR 仓库根目录解析；推荐 `tasks/` 下的规范使用该模式。
- `workspace.implementation_dir` 相对 project root 解析。
- `entrypoints.train/evaluate` 必须包含 `{result_path}`。
- `adapter.trial_defaults` 固定该任务的完整、受控训练默认值；预算敏感字段不能依赖训练脚本隐式默认值。
- entrypoint 在 implementation directory 中执行。
- `data.*.path` 是 case 声明的数据位置；框架不推断数据模态。
- `metrics.*.source` 是原始结果 JSON 的点分路径，例如 `test.budget_accuracy` 或 `metrics.mean_delta_pearson`；数组索引也可使用，例如 `test.exit_accuracy.3`。
- `metrics.primary.role` 固定为 `primary`；secondary 的 `role` 是 `constraint` 或 `diagnostic`。约束指标可声明通用的 `constraint.operator/value`，controller 不理解指标名称。
- `search.editable_paths` 和 `protected_paths` 相对 implementation directory。
- 可选 `search.interface_editable_paths` 相对 repository root，并且必须仍在
  task project 内；它只用于显式开放 task-local Adapter 等接口文件，不会
  开放通用 controller、数据或任务规范。
- editable 与 protected 不得重叠；proposal 中的代码路径必须落在 editable 范围内。

## Proposal 约定

proposal 必须使用 task spec 中的 `task.name`。Heuresis 只输出结构化的
`experiment_proposals`，不能输出 shell、entrypoint、CLI flag、数据路径或 rjob
命令。单个实验对象固定为：

```json
{
  "hypothesis": "...",
  "change_scope": ["model", "loss"],
  "parameters": {},
  "expected_effect": {},
  "acceptance_criteria": {},
  "resource_request": {}
}
```

例如只改变 loss，而不重复训练预算：

```json
{
  "hypothesis": "Robust loss improves the primary metric.",
  "change_scope": ["loss"],
  "parameters": {"loss_type": "huber"},
  "expected_effect": {"quality": {"direction": "increase", "minimum_change": 0.01}},
  "acceptance_criteria": {"quality": {"operator": ">=", "value": 0.5}},
  "resource_request": {"gpu_count": 1, "cpu": 8, "memory_mb": 20000, "max_runtime_minutes": 60}
}
```

Task Adapter 将 `parameters` 覆盖到 `adapter.trial_defaults` 上，再生成唯一的
`run_trial` 调用。未知参数会在提交 rjob 前被拒绝。以 batch size 为例：即使
Heuresis 只建议 `loss_type=huber`，Adapter 仍会保留 task spec 中明确声明的
`batch_size`，不会退回训练脚本的隐式默认值。

`resource_request` 同样是结构化字段，GPU 和运行时间不能超过 task spec，CPU 和
内存不能超过 controller 配额。`acceptance_criteria` 会在结果回收后逐项计算并写入
trial record。旧 `omni-ar-proposal/v1` 仅为读取历史实验保留；新 proposal 不再允许
自定义 `command`。

## Result 约定

case 训练代码继续输出自己的原始 JSON，不要求 CIFAR 和 VCC25 修改成相同内部结构。controller 根据 task spec 的 `metrics.*.source` 生成 `trial_NNN.result.json`：

- 所有任务的外层字段固定为 `status/task/trial_id/metrics/baseline_metrics/resource_usage/protocol/artifacts`（另有版本字段 `schema_version`，失败时可有 `error`）。
- `metrics` 的 key 是任务自己的指标名；每条记录必须给出 `value`、`direction`、`role`、`baseline_value`、`beats_baseline` 和 `constraint_satisfied`。
- `role` 只能是 `primary`、`constraint` 或 `diagnostic`；`status=ok` 时恰好有一个 primary。
- `resource_usage.within_budget` 固定存在。`true/false` 表示可判定，实际资源数据不足时为 `null`，不能把未知预算伪装为满足。
- `protocol.stability` 固定存在。单 seed 必须标记 `single_seed` 且 `std=null`；只有真实聚合多个 seed 时才能标记 `multi_seed` 并填写统计量；没有科研指标时为 `unavailable`。
- 原始结果和执行命令保存在 `artifacts`，原始文件始终保留。
- 缺失主指标、非有限数值、方向/角色不一致会导致 result contract 失败。
- `skipped/simulated/failed` 的 `metrics` 为空，不能进入科研选择。

通用选择器 `scripts/select_results.py` 只读取上述元数据：状态、角色、方向、baseline 比较、约束判定、预算判定和稳定性；代码中没有领域指标名或任务阈值。默认要求主指标超过 baseline、所有约束满足且预算满足。可用 `--require-multiseed` 将多 seed 稳定性设为硬门槛：

```bash
python controller_bash/scripts/select_results.py result_a.json result_b.json \
  --require-multiseed --out selection.json
```

历史 `omni-ar-result/v1` 仍可由 `validate-result` 读取，但新 trial 一律产出 v2，且 v1 不进入通用选择器。

## 统一执行后端

`scripts/execution_backends.py` 是 rjob 基础设施的唯一实现位置。Task Adapter
只把 proposal 编译成 payload 命令；它不知道 namespace、镜像、共享盘、rjob
状态和日志接口。controller 的固定生命周期为：

```text
payload command
  -> 生成 worker
  -> rsync 受控代码路径到共享盘
  -> 按 resource_request 提交 rjob
  -> 轮询状态
  -> 保存 submit/get/worker 日志
  -> 回收机器可读原始 JSON
  -> 归一化为 omni-ar-result/v2
  -> 失败/超时记录（超时默认取消任务）
```

同步由 `RJOB_SYNC_SOURCE`、`RJOB_SYNC_PATHS` 和 `RJOB_SHARED_FOLDER` 控制。
`RJOB_SYNC_PATHS` 只列 worker 所需目录，避免上传无关数据；同步使用增量 rsync，
不使用 `--delete`。`rjob_dry_run` 只输出同步计划和提交命令，不写共享盘、不提交任务。
任务目录中的历史 submit 脚本已经弃用，任务代码不得再出现完整 rjob 调度流程。

## 命令

验证 task spec：

```bash
python controller_bash/scripts/task_contract.py validate-task \
  --task-spec /path/to/task_spec.yaml
```

验证 proposal：

```bash
python controller_bash/scripts/task_contract.py validate-proposal \
  --task-spec /path/to/task_spec.yaml \
  --proposal /path/to/proposal.json
```

把已有原始结果转换成统一 result：

```bash
python controller_bash/scripts/task_contract.py normalize-result \
  --task-spec /path/to/task_spec.yaml \
  --raw-result /path/to/raw.json \
  --proposal-id imported-result \
  --trial-name historical-trial \
  --out /path/to/result.json
```

controller case env 必须设置：

```bash
TASK_SPEC=/absolute/path/to/task_spec.yaml
SUGGESTION_SCHEMA=/absolute/path/to/controller_bash/schemas/proposal_v2.schema.json
```

## 统一回归入口

`run_research` 读取 `task_spec.yaml` 旁边的 `regression_suite.yaml`，把其中不含命令的
proposal 交给 Task Adapter 编译，再统一调用 `scripts/run_trials.py`。默认仅生成 rjob
计划；真实 GPU 回归显式添加 `--execution-mode rjob`：

```bash
run_research --task tasks/cifar/task_spec.yaml
run_research --task tasks/vcc25/task_spec.yaml

run_research --task tasks/cifar/task_spec.yaml --execution-mode rjob
run_research --task tasks/vcc25/task_spec.yaml --execution-mode rjob
```

回归入口强制 `policy.allow_search=false`，因此不会把“验证旧方法仍可运行”误当成继续
调参。输出同时检查 proposal 编译后的关键参数、Result-v2、历史结果锚点，以及运行
前后通用 prompt/controller 的 SHA-256 指纹。任务切换只允许改变 task spec、adapter、
context 和 regression suite，不要求也不允许重写通用 prompt。
