# Final Repository Audit

Date: 2026-09-25

## source repository

Primary:

```text
/data/zhangzhicheng/omni-ar_heuresis-skill-decouple
```

Supplemental:

```text
/data/zhangzhicheng/omni-ar_heuresis-skill
/data/zhangzhicheng/omni-ar_heuresis
```

## final repository

```text
/data/zhangzhicheng/omni-ar_heuresis-final
```

## files copied

The final repository contains:

```text
ai4ai/
compatibility/legacy/
controller_bash/
datasets/
docs/
omni_ar/
skills/
stability/
tasks/
tests/
research_history/
archive_index/
README.md
FINAL_REPO_MIGRATION_PLAN.md
uncertain_files.md
PATH_MIGRATION_REPORT.md
```

The current VCC implementation is present:

```text
tasks/vcc25/task_plugin.yaml
tasks/vcc25/task_spec_official_h1.yaml
tasks/vcc25/official_h1_adapter.py
tasks/vcc25/native_evaluator.py
tasks/vcc25/implementation/
```

Selected history copied:

```text
V59 proposal/result JSON
V67 research notes
V68 design/audit notes
research_history README files
historical evaluation reports/manifests
```

## files excluded

Excluded by policy:

```text
runs/
rjob_packages/
rjob_runtime/
.runtime/
runtime/
logs/
__pycache__/
.pytest_cache/
release_verification/
reproduction_runs/
_legacy_archive/
Cell-Eval output directories
*.h5ad
*.pt
*.pth
*.ckpt
*.npz
*.npy
large prediction artifacts
```

No excluded large file was found in the final repository during the final scan.

## git status

The final directory was initialized as a new git repository after file consolidation. No source repository was changed.

Observed status on 2026-09-25:

```text
## No commits yet on main
All consolidated files are untracked pending maintainer review.
```

## missing dependencies

- Python dependencies for controller, task adapters, and Cell-Eval.
- rjob command/runtime and cluster permissions.
- Optional upstream/vendor dependency policy for Heuresis_PJLAB-boyue and ResearchStudio-main.
- External official H1 assets.

## external assets

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1
```

The final repository contains no copy of these assets.

## known limitations

- Several executable configs and historical manifests still contain old absolute paths. See `PATH_MIGRATION_REPORT.md`.
- The final scan found 62 files with old-root references. Most are frozen acceptance metadata or historical provenance; executable config references require follow-up.
- The final repository is a consolidation baseline, not a validated relocatable release.
- Broad NatureBench/MPP material from `omni-ar_heuresis` was not merged because it is outside the current VCC maintenance scope.
- No training, prediction generation, evaluator rerun, or scientific execution was performed.
- Historical JSON records may point to files that were deliberately excluded from the final repository.
