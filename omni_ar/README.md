# Omni-AR Loop Service

This package exposes the reusable workflow as bounded function calls. Dataset
Adapters remain prebuilt inside Dataset Packs; users select a reviewed usage
mode and do not write adapter code.

```python
from omni_ar import loop

datasets = loop.datasets()
data = loop.dataset_loading("vcc25@2", mode="heldout_target")
plan = loop.design_code("tasks/vcc25/task_spec.yaml", proposal)
```

`design_code` and `run_trial` with their defaults never launch training.
Execution requires `run_trial(..., execute=True, output=...)`; distributed
execution remains the controller/rjob backend's responsibility.

CLI equivalents:

```bash
./loopctl datasets
./loopctl dataset-loading --dataset vcc25@2 --mode heldout_target
./loopctl design-code --task tasks/vcc25/task_spec.yaml --proposal proposal.json
```

The facade returns `omni-ar-loop/v1` JSON envelopes so ResearchStudio,
Heuresis, CLI clients, and a future HTTP service can share one implementation.

## Rough idea initialization

Run the question flow before ResearchStudio:

```bash
./run_research init
```

Interactive initialization now starts by asking the user for a one- or
two-sentence rough idea. Boyue determines whether it matches a registered task,
is ambiguous, or describes a new task. Ambiguities and conflicts with an
optional `--task` hint are clarified before any Adapter-specific choice is
shown. Boyue then generates 3–5 short follow-up questions from domain-general
research dimensions. Registered tasks use reviewed Adapter choices; new tasks
receive a planning-only task profile and are not forced into the nearest
existing Adapter. The generated plan is contract-validated and both
`question_plan.json` and the answers are saved before the confirmed
`rough_idea.yaml` is passed to ResearchStudio.

An unregistered task can proceed through ResearchStudio literature retrieval
and Idea Card generation. Its Rough Idea explicitly records an unbound dataset,
metric, protocol, and Adapter; Heuresis execution and training remain disabled
until those contracts are supplied. The system never fabricates executable
bindings for a new task.

For a supported unregistered dataset, the initializer can now create those
contracts through the reviewed generic adapters:

```bash
./run_research init \
  --dataset /path/to/data.npz \
  --dataset-id my-image-task
```

Before asking research questions, this command profiles the data, creates a
versioned Dataset Pack, generates and validates `task_spec.yaml`, validates the
data through the shared generic Task Adapter, and runs a lightweight CPU
baseline. Text classification supports CSV/TSV/JSONL with recognizable text
and label fields. Image classification supports NPZ feature/label arrays and
class-directory layouts. Unrecognized fields, unsupported task types, invalid
splits, or non-finite baseline metrics stop the run with a concrete error.

Use `--question-count 3..5` or `--question-model MODEL` to control this stage.
`--task` is now an optional hint, not an unconditional binding. Supplying
`--answers` remains the offline/batch path and does not call an LLM; that mode
still requires `--task` because there is no interactive resolution step.

For a reviewable non-interactive run:

```bash
./run_research init \
  --task tasks/vcc25/task_spec.yaml \
  --answers omni_ar/initialization/vcc25_answers.example.yaml \
  --output-dir research_initializations/vcc25-example \
  --yes
./run_research preview \
  --rough-idea research_initializations/vcc25-example/rough_idea.yaml
./run_research dry-run \
  --rough-idea research_initializations/vcc25-example/rough_idea.yaml \
  --output-dir research_initializations/vcc25-example/dry_run
```

For a registered task, dry-run invokes ResearchStudio, validates a standard
Heuresis proposal, and compiles it through the configured Task Adapter. For an
unregistered task it stops after the ResearchStudio boundary with
`researchstudio_ready_adapter_required`. Neither path starts training or
submits an rjob.

# Unified multi-round run

After a rough idea is confirmed, the complete orchestrator can be started with:

```bash
./run_research auto-run \
  --rough-idea research_initializations/<id>/rough_idea.yaml \
  --output-dir research_runs/<id> \
  --rounds 5 \
  --strategy omni_epic \
  --planning-mode live \
  --execution-mode rjob \
  --coding-mode plan
```

Alternatively, add the same controls to `run_research init --auto-run`; after
the final QA confirmation no further user action is required. Interactive QA
also asks whether to submit real rjobs, whether the Coding Agent may apply
task-local changes, and whether configured external services may receive the
research brief. The confirmed choices and limits are stored in
`rough_idea.yaml` under `execution_policy`. `coding-mode`
is conditional: registered parameters and mechanisms go directly through the
Adapter, while `implementation_requests` are routed to the sandboxed Coding
Agent. In `apply` mode each candidate receives an isolated Git worktree plus a
filtered snapshot of current local source changes. Data, credentials and
generated research outputs are excluded, so users do not need to create a
temporary commit before starting the loop.

Startup checks are written to `records/preflight.json`, and
`records/stages.json` records each completed or failed stage. A failed project
can continue without repeating completed rounds or ResearchStudio planning:

```bash
./run_research resume --project-dir research_runs/<id>
```

Planning failures use the retry count confirmed during QA. The rjob backend
keeps bounded submission and log retries.

Live planning writes `planning_progress.json` and prints immediate stage
updates for ResearchStudio, literature retrieval checkpoints, Boyue proposal
generation, and Adapter compilation. Long-running stages print a heartbeat
every 30 seconds. The defaults can be adjusted without changing task code:

```text
OMNI_AR_PLANNING_HEARTBEAT_SEC=30
OMNI_AR_RESEARCHSTUDIO_TIMEOUT_SEC=600
OMNI_AR_HEURESIS_STAGE_TIMEOUT_SEC=420
IDEASPARK_BOYUE_TIMEOUT_SEC=180
HEURESIS_SUGGESTION_TIMEOUT_SEC=180
```

`planning_cache.json` binds cached ResearchStudio and Heuresis artifacts to the
rough-idea hash, Task Spec hash, model, context hash, and artifact hashes. A
retry or `resume` reuses only matching completed stages. Partial ResearchStudio
phase outputs remain available to its existing `--resume` checkpoints.

Every project writes `records/project.json`, `records/archive.json`,
`records/lineage.json`, per-round records and an append-only `events.jsonl`.
Live OMNI-EPIC runs also use its Boyue MoI reviewer after the native seed phase;
rejections and reviewer failures are recorded and never reach execution.
Termination creates `final_package/`, which contains the best accepted real
candidate, input/config/result hashes and a reproduction command. Simulation
is useful for testing orchestration but is never eligible as a scientific
winner.

Executed trials use separate result categories:

- `failed_train / training_failed`: training did not complete;
- `invalid / result_invalid`: the standard result is missing or malformed;
- `invalid / method_inactive`: required method diagnostics are missing or inactive;
- `verification / activation_verified`: the registered feature or mechanism was
  proven active by a real run, but the protocol is diagnostic-only;
- `rejected / below_acceptance`: training and validation completed, but the
  acceptance threshold was not met.

The last category retains its observed score and cannot be selected as a
winner. Legacy project records can be updated with
`controller_bash/scripts/migrate_archive_categories.py`.

Every newly normalized result also carries `features`. For each selected
feature it records the active flag, Builder id/version, immutable input hashes,
feature-file hashes, fit scope, train/evaluation usage scope, required
diagnostics and observed diagnostics. A selected feature without successful
Builder validation or an explicit runtime active flag is classified as
`method_inactive` and cannot enter the accepted archive.
