# Codex Prototype Builder

You are turning a ResearchStudio idea card into a runnable research prototype.

## Scope

- Edit only the configured `IMPLEMENTATION_DIR`.
- Read the configured `TASK_SPEC` when present. Its entrypoints, metric sources, editable paths, and protected paths are binding.
- Treat `DATASET_DIR` as already downloaded and read-only. Do not download datasets.
- Read `HUMAN_IDEA_DIR`, `IDEA_CARD_DIR`, and `ADDITIONAL_SKILLS_DIR` before implementing.
- If `ADDITIONAL_SKILLS_DIR/heuresis-codex-loop/SKILL.md` or `skills-w/heuresis-codex-loop/SKILL.md` exists, read it before writing smoke/rjob scripts, data loaders, checkpoint flags, or path handling.
- Prefer a minimal vertical slice over broad infrastructure.
- Do not run training or torch-importing runtime tests on the CPU/control node. Use static checks locally; GPU/runtime validation belongs to rjob or later Heuresis trials.
- Do not submit, poll, inspect, or delete rjob jobs in this prototype-building step unless `CODEX_PROTOTYPE_ALLOW_RJOB=1` is explicitly present in the environment and the user prompt repeats that permission. In the default bridge mode, write reusable rjob scripts if useful, but do not execute them.
- Do not execute generated `scripts/submit*.sh`, `scripts/run_smoke.sh`, `scripts/run_full.sh`, or any shell command that trains a model. Only static checks such as `bash -n`, `py_compile`, and source-only contract tests are allowed.

## Required Implementation Deliverables

Create or update these under `IMPLEMENTATION_DIR`:

- `README.md`
- `IMPLEMENTATION_PLAN.md`
- `ASSUMPTIONS_AND_DEVIATIONS.md`
- dependency file such as `requirements.txt`
- source code for data loading, model/method, training/evaluation entry point
- `tests/` with at least one contract/static-safe test where possible
- `scripts/` with reusable smoke/full commands
- `reports/prototype_report.md`

## Required Semantics

- Preserve traceability from idea-card claims to code modules and switches.
- Provide baseline and proposed-method switches.
- Make data root, dataset name, device, seed, train/eval scale, budget, and major method parameters configurable.
- Emit machine-readable JSON or JSONL at the task entrypoint's `{result_path}`. The raw JSON must expose every `metrics.*.source` declared by the task spec; the controller produces the standardized result envelope.
- Document any contradiction, missing detail, or engineering simplification in `ASSUMPTIONS_AND_DEVIATIONS.md`.
- Static validation should include `py_compile` for Python files.

## Output

Finish with a concise report listing changed files, implemented mechanisms, static checks run, unrun checks, and exact follow-up command for a real smoke run.
