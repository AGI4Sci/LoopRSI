# Final Repository Migration Plan

Date: 2026-09-25

## source directories

Primary source:

```text
/data/zhangzhicheng/omni-ar_heuresis-skill-decouple
```

Supplemental sources:

```text
/data/zhangzhicheng/omni-ar_heuresis-skill
/data/zhangzhicheng/omni-ar_heuresis
```

The final repository must not be a direct clone or full merge of any old repository. The decouple repository is used as the current architecture and VCC pipeline authority. Supplemental sources are used only for selected research history, missing docs, and potentially missing non-VCC task material.

## target directory

```text
/data/zhangzhicheng/omni-ar_heuresis-final
```

## modules to copy

From `omni-ar_heuresis-skill-decouple`:

```text
ai4ai/
compatibility/legacy/
controller_bash/examples/
controller_bash/prompts/
controller_bash/schemas/
controller_bash/scripts/
controller_bash/tests/
datasets/
docs/
omni_ar/
skills/
stability/
tasks/
tests/
```

VCC required files:

```text
tasks/vcc25/task_plugin.yaml
tasks/vcc25/task_spec*.yaml
tasks/vcc25/official_h1_adapter.py
tasks/vcc25/native_evaluator.py
tasks/vcc25/implementation/
```

Top-level selected metadata:

```text
historical_score_provenance.json
vcc_research_only_manifest.json
VCC_RESEARCH_ONLY_PAYLOAD_AUDIT.md
CHALLENGE_STATE_MODULARITY_SELECTION.md
CHALLENGE_STATE_S1_PAYLOAD_REVIEW.md
S2_TRANSPORT_RECOVERY.md
```

## documents to merge

From `omni-ar_heuresis-skill-decouple`:

```text
docs/FINAL_ARCHITECTURE.md
docs/CURRENT_27_STEP_RESEARCH_PIPELINE.md
docs/TASK_PLUGIN_ARCHITECTURE.md
docs/ARCHITECTURE_MIGRATION_REPORT.md
docs/PHASE_*_REPORT.md
docs/PHASE_*_AUDIT.md
docs/REGRESSION_REPORT.md
docs/VCC25_ITERATION_LOG.md
Agent报告/49-*.md
Agent报告/50-*.md
Agent报告/51-*.md
Agent报告/52-*.md
```

From `omni-ar_heuresis-skill`:

```text
research_history/*/README.md
HIPSCI_*.md
runs/vcc25_lingshu_iter_v67_20260925/*.md
```

From `omni-ar_heuresis-skill-decouple` selected runs and evaluations:

```text
runs/vcc25_lingshu_iter_v67_20260925/*.md
runs/vcc25_lingshu_iter_v68_20260925/*.md
runs/vcc25_decouple_v59_20260923/*proposal.json
runs/vcc25_decouple_v59_20260923/validated_suggestion.json
runs/vcc25_decouple_v59_20260923/trial_results*.json
historical_evaluation/*/README.md
historical_evaluation/*/*REPORT*.md
historical_evaluation/*/*FINAL*.md
historical_evaluation/*/*manifest.json
historical_evaluation/*/*comparison*.json
historical_evaluation/*/final_verification.json
historical_evaluation/*/adapter_result.json
```

These selected run/evaluation files will be placed under:

```text
research_history/
archive_index/
```

## directories excluded

Never copy wholesale:

```text
runs/
rjob_packages/
.runtime/
runtime/
rjob_runtime/
logs/
controller_bash/logs/
__pycache__/
.pytest_cache/
historical_evaluation/*/cell_eval/
historical_evaluation/*/benchmark/
release_verification/
reproduction_runs/
_legacy_archive/
modularity_paired_trials/
challenge_state_*_trials/
```

Never copy large artifact classes:

```text
*.h5ad
*.pt
*.pth
*.ckpt
*.npz
*.npy
*.pkl
*.pickle
large *.csv
prediction artifacts
model checkpoints
cache files
```

## dependencies

External dependencies remain external and must not be copied:

```text
Official H1 assets:
/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1

Cell-Eval runtime and Python environment
rjob cluster image and mount configuration
Heuresis_PJLAB-boyue, if kept as vendored source or converted to setup dependency
ResearchStudio-main, if kept as vendored source or converted to setup dependency
```

## path migration required

Scan file types:

```text
*.py
*.yaml
*.yml
*.json
*.sh
```

Old paths to report, not rewrite automatically:

```text
/data/zhangzhicheng/omni-ar_heuresis
/data/zhangzhicheng/omni-ar_heuresis-skill
/data/zhangzhicheng/omni-ar_heuresis-skill-decouple
```

Expected new root:

```text
/data/zhangzhicheng/omni-ar_heuresis-final
```

Known likely migration targets:

```text
controller_bash/configs/*.env
tasks/vcc25/task_plugin.yaml runtime support notes
selected run manifests archived under archive_index/
```

## uncertain files

Do not copy these until reviewed:

```text
Heuresis_PJLAB-boyue/
ResearchStudio-main/
experimental/
acceptance_inputs/
baseline_snapshot/
sanitized_*.json
challenge_state_*.json
modularity_paired_trials/
historical_evaluation/*/logs and CSV outputs
old non-VCC NatureBench/MPP task trees from omni-ar_heuresis
```

See `uncertain_files.md` for a running list.
