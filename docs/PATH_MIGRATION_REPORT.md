# Path Migration Report

Date: 2026-09-25

## Scope

Scanned migrated `*.py`, `*.yaml`, `*.yml`, `*.json`, and `*.sh` files for:

```text
/data/zhangzhicheng/omni-ar_heuresis
/data/zhangzhicheng/omni-ar_heuresis-skill
/data/zhangzhicheng/omni-ar_heuresis-skill-decouple
```

No path was blindly replaced. Historical paths in manifests and frozen acceptance evidence are preserved as provenance until an owner decides whether they are executable configuration or historical metadata.

## Findings

| file/category | old path | new path | need migration | reason |
| --- | --- | --- | --- | --- |
| `controller_bash/configs/*.env` | `/data/zhangzhicheng/omni-ar_heuresis-skill` | `/data/zhangzhicheng/omni-ar_heuresis-final` | yes | These are executable controller/rjob configs and still point at the old source root. |
| `datasets/*/dataset.yaml` acceptance metadata | `/data/zhangzhicheng/omni-ar_heuresis/...` | external dataset root or final-relative logical reference | review | These fields identify historical dataset provenance; replacing them may destroy reproducibility evidence. |
| `tasks/*/implementation/frozen/*.json` | `/data/zhangzhicheng/omni-ar_heuresis/...` | final-relative path or archive provenance field | review | Frozen acceptance outputs are evidence, not necessarily runtime config. |
| `docs/supplemental/historical_score_provenance.json` | `/data/zhangzhicheng/omni-ar_heuresis-skill-decouple/...` | final path or external archive URI | review | Historical evaluator and artifact paths are intentionally preserved as provenance. |
| `research_history/selected_runs/v59/*.json` | `/data/zhangzhicheng/omni-ar_heuresis-skill-decouple/...` | final path only if used operationally | no automatic rewrite | These are historical run records and reference files outside the final repo. |
| `archive_index/historical_evaluation/**/*.json` | `/data/zhangzhicheng/omni-ar_heuresis-skill-decouple/...` | archive URI or external evidence root | no automatic rewrite | Archive manifests must retain original execution identity. |

## Required follow-up

1. Normalize executable `.env` configs to use the final repository root or environment variables.
2. Separate historical provenance keys from executable path keys in dataset and frozen-result JSON.
3. Add a documented external asset root variable for official H1 assets.
4. Re-run this scan after config normalization.

## Important limitation

The presence of an old path does not prove the referenced artifact should be copied. The migration policy deliberately keeps source code separate from historical execution environments and large data.
