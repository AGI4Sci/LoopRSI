# VCC25 Knowledge-Driven Hierarchical Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an isolated five-layer VCC25 research loop that retrieves approved knowledge cards, generates a bounded candidate change, scores search candidates only on train/validation data, and performs at most one final official evaluation to determine whether the frozen candidate exceeds `0.306` Pearson delta.

**Architecture:** Add a knowledge bridge and evaluation-authority gate around the existing `HierarchicalLoop`, then introduce a knowledge-driven worker with a pluggable JSON decision backend. L1-L4 use read-only Codex decisions; L5 hands a bounded request to the existing isolated coding-agent runner. Remote execution is staged into a new experiment directory; the two existing remote source directories remain read-only.

**Tech Stack:** Python 3.10+, stdlib `unittest`, existing `ai4ai` plugin interfaces, Codex CLI JSON mode, existing OMNI-AR coding-agent runner, existing VCC25 validation-only runner, rjob for GPU execution.

**Spec:** `docs/superpowers/specs/2026-09-29-vcc25-knowledge-hier-loop-experiment-design.md`

## Global Constraints

- Never modify `/mnt/shared-storage-user/gaozhangyang/RSI` or `/mnt/shared-storage-gpfs2/beam-gpfs02/huangwenxuan/lingshu-cell-agent`.
- Stage all remote code and outputs under a newly validated experiment directory.
- Only `search_validation` results from the full validation contract may produce reward.
- `smoke_only`, synthetic, plan-proxy, historical, and `final_official` results are never reward eligible.
- Final-test expression data must not be visible to planning, code generation, training, validation, or candidate selection.
- Permit one candidate and at most one coding repair in the first experiment.
- Permit one `final_official` evaluation for the entire experiment, only after the candidate hash is frozen.
- Every layer decision must record skill ID, card IDs, content hash, and evaluation authority.

## Review Focus

- A skill implementing `skill_id` without `.manifest` must load and inject successfully; legacy manifest-backed skills must continue to work.
- Empty or over-budget knowledge retrieval must fail preflight rather than silently running a knowledge-free loop.
- Results with missing, misspelled, or disallowed authority must be promotion- and reward-ineligible.
- Candidate paths containing symlinks or `..` must not escape the isolated experiment workspace.
- Final-test result files must never be accepted by the validation reward or memory path.

---

### Task 1: Skill Compatibility and Knowledge Bridge

**Files:**
- Create: `hier_loop/knowledge_bridge.py`
- Modify: `controller_bash/scripts/apply_skill_context.py`
- Test: `tests/hier_loop/test_knowledge_bridge.py`

**Interfaces:**
- Consumes: `TaskPluginManifest`, `ResearchSkill`, and layer context mappings.
- Produces: `resolve_skill_id(skill: object) -> str`, `KnowledgeInjection` and `KnowledgeBridge.inject(layer: str, context: Mapping[str, Any]) -> KnowledgeInjection`.

- [ ] **Step 1: Write the failing compatibility and provenance tests**

  Add tests named `test_new_research_skill_id_is_supported`, `test_legacy_manifest_skill_id_is_supported`, `test_each_layer_receives_nonempty_vcc25_knowledge`, `test_injection_records_card_ids_and_content_hash`, and `test_empty_fragment_fails_preflight`. Assert the exact L1-L5 skill routing already defined by `skills/vcc25/lingshu.py`.

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `python3 -m unittest tests.hier_loop.test_knowledge_bridge -v`

  Expected: import failure for `hier_loop.knowledge_bridge` or missing compatibility behavior.

- [ ] **Step 3: Implement the bridge and update the old hook**

  Implement immutable injection records containing `layer`, `skill_id`, `card_ids`, `content`, `content_hash`, `token_budget`, and `source`. Update `apply_skill_context.py` to call `resolve_skill_id` instead of directly reading `skill.manifest.skill_id`.

- [ ] **Step 4: Run focused and existing skill tests**

  Run: `python3 -m unittest tests.hier_loop.test_knowledge_bridge tests.domain_knowledge.test_lingshu_skill -v`

  Expected: all tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add hier_loop/knowledge_bridge.py controller_bash/scripts/apply_skill_context.py tests/hier_loop/test_knowledge_bridge.py
  git commit -m "feat: bridge VCC25 knowledge into hierarchy contexts"
  ```

### Task 2: Evaluation Authority and Reward Gate

**Files:**
- Create: `hier_loop/evaluation_authority.py`
- Test: `tests/hier_loop/test_evaluation_authority.py`

**Interfaces:**
- Consumes: experiment result mappings and the same-contract validation baseline.
- Produces: `EvaluationRecord.from_mapping(value: Mapping[str, Any])`, `promotion_eligible(record: EvaluationRecord) -> bool`, and `validation_reward(record: EvaluationRecord, baseline: float) -> float | None`.

- [ ] **Step 1: Write failing authority tests**

  Cover the two allowed authorities, reject missing/unknown authority, require `full_validation=true` for promotion, return `candidate_pcc - baseline_pcc` only for `search_validation`, and return `None` for `smoke_only`, synthetic, plan-proxy, historical, and `final_official` inputs.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python3 -m unittest tests.hier_loop.test_evaluation_authority -v`

  Expected: import failure for `hier_loop.evaluation_authority`.

- [ ] **Step 3: Implement strict result parsing and reward gating**

  Reject non-finite metrics, missing candidate hashes, and final records without `selection_feedback_allowed=false`. Keep this module independent of the existing permissive `_reward_for` method.

- [ ] **Step 4: Run focused tests**

  Run: `python3 -m unittest tests.hier_loop.test_evaluation_authority -v`

  Expected: all tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add hier_loop/evaluation_authority.py tests/hier_loop/test_evaluation_authority.py
  git commit -m "feat: gate hierarchy rewards by evaluation authority"
  ```

### Task 3: Structured Layer Agent and Knowledge-Driven Worker

**Files:**
- Create: `hier_loop/layer_agent.py`
- Create: `hier_loop/knowledge_worker.py`
- Create: `hier_loop/schemas/layer_decision.schema.json`
- Test: `tests/hier_loop/test_knowledge_worker.py`

**Interfaces:**
- Consumes: `KnowledgeBridge`, current loop context, prior structured layer decisions, and a `LayerDecisionBackend`.
- Produces: `LayerDecisionBackend.decide(request: LayerDecisionRequest) -> Mapping[str, Any]`, `RecordingDecisionBackend`, `CodexJsonDecisionBackend`, and `KnowledgeDrivenVcc25Worker.step(...) -> StepResult`.

- [ ] **Step 1: Write failing worker tests**

  Assert that L1 produces a direction, L2 a problem, L3 a falsifiable hypothesis, L4 a mechanism, and L5 a coding request. Assert every action contains knowledge provenance, prior-layer decisions flow forward, no layer fabricates a numeric score, and malformed backend output fails the step.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python3 -m unittest tests.hier_loop.test_knowledge_worker -v`

  Expected: imports fail for the new worker modules.

- [ ] **Step 3: Implement the backend protocol and recording backend**

  `RecordingDecisionBackend` returns schema-valid deterministic decisions for no-GPU preflight and archives the full prompt. `CodexJsonDecisionBackend` invokes `codex exec --ephemeral --sandbox read-only --output-schema <schema> -`, parses only the final JSON object, and never grants workspace write access for L1-L4.

- [ ] **Step 4: Implement `KnowledgeDrivenVcc25Worker`**

  Inject knowledge before every decision, validate the backend result, and return `score=None` for planning layers. L5 emits a bounded coding request with allowed paths, hypothesis, activation diagnostics, train command, validation command, and expected artifacts; it does not execute code in this task.

- [ ] **Step 5: Run worker, bridge, and loop regression tests**

  Run: `python3 -m unittest tests.hier_loop.test_knowledge_worker tests.hier_loop.test_knowledge_bridge tests.hier_loop.test_memory_embedding -v`

  Expected: all tests pass.

- [ ] **Step 6: Commit**

  ```bash
  git add hier_loop/layer_agent.py hier_loop/knowledge_worker.py hier_loop/schemas/layer_decision.schema.json tests/hier_loop/test_knowledge_worker.py
  git commit -m "feat: add knowledge-driven VCC25 hierarchy worker"
  ```

### Task 4: Bounded Candidate Execution and Validation-Only Contract

**Files:**
- Create: `hier_loop/candidate_executor.py`
- Create: `hier_loop/split_audit.py`
- Test: `tests/hier_loop/test_candidate_executor.py`
- Test: `tests/hier_loop/test_split_audit.py`

**Interfaces:**
- Consumes: the L5 coding request, experiment workspace, VCC25 target-list metadata, and existing `controller_bash/scripts/run_coding_agent.py`.
- Produces: `CandidateExecutor.prepare(request, workspace)`, `CandidateExecutor.run_preflight(...)`, `SplitAudit`, and a validation request that targets `official_h1_autonomous_research_loop.py` rather than canonical `run_trial`.

- [ ] **Step 1: Write failing boundary and leakage tests**

  Assert candidate paths remain below the resolved workspace, symlink escapes fail, only declared implementation paths are writable, final-test expression paths and canonical `run_trial` are rejected during search, target-list overlap fails, and a validation-only command is accepted.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python3 -m unittest tests.hier_loop.test_candidate_executor tests.hier_loop.test_split_audit -v`

  Expected: imports fail for the new modules.

- [ ] **Step 3: Implement preparation and dry preflight**

  Reuse the existing coding-agent runner's worktree and path-policy behavior. The preflight materializes the request, verifies tools and assets, but does not invoke Codex or GPU execution.

- [ ] **Step 4: Implement the split audit and validation-only command guard**

  Store only target identifiers/hashes and counts in `split_audit.json`; do not copy expression matrices or final-test contents into logs.

- [ ] **Step 5: Run focused and controller regression tests**

  Run: `python3 -m unittest tests.hier_loop.test_candidate_executor tests.hier_loop.test_split_audit controller_bash.tests.test_coding_snapshot -v`

  Expected: all tests pass.

- [ ] **Step 6: Commit**

  ```bash
  git add hier_loop/candidate_executor.py hier_loop/split_audit.py tests/hier_loop/test_candidate_executor.py tests/hier_loop/test_split_audit.py
  git commit -m "feat: isolate VCC25 candidate and validation execution"
  ```

### Task 5: Experiment CLI, Configuration, and Audit Artifacts

**Files:**
- Create: `hier_loop/knowledge_experiment.py`
- Create: `configs/rsi_step0/hier_vcc25_knowledge.json`
- Modify: `hier_loop/cli.py`
- Test: `tests/hier_loop/test_knowledge_experiment.py`
- Modify: `docs/domain-knowledge.md`

**Interfaces:**
- Consumes: experiment phase (`preflight`, `smoke`, `validation`, or `final`), workspace, decision backend, and fixed configuration.
- Produces: `run_manifest.json`, `layer_decisions.jsonl`, `candidate_request.json`, `candidate.patch`, `candidate_code_sha256.json`, `split_audit.json`, phase result, and `summary.json`.

- [ ] **Step 1: Write failing CLI and artifact tests**

  Verify `knowledge-preflight` runs all five layers with `RecordingDecisionBackend`, produces no GPU command, writes every required provenance field, refuses a destination equal to either source checkout, and refuses `final` without a frozen candidate and promotion record.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `python3 -m unittest tests.hier_loop.test_knowledge_experiment -v`

  Expected: command or module is missing.

- [ ] **Step 3: Implement the phase runner and CLI command**

  Add `hier_loop knowledge-preflight --config ... --workspace ...`. Keep `smoke`, `validation`, and `final` behind explicit phase methods whose prerequisites are checked from immutable artifacts.

- [ ] **Step 4: Document exact local and remote commands**

  Explain the authority labels, promotion gate, single-use final budget, and how to inspect card provenance without exposing restricted data.

- [ ] **Step 5: Run the full local suite**

  Run: `python3 -m unittest discover -s tests -v`

  Expected: all non-optional tests pass; only previously documented dependency-based skips remain.

- [ ] **Step 6: Commit**

  ```bash
  git add hier_loop/knowledge_experiment.py configs/rsi_step0/hier_vcc25_knowledge.json hier_loop/cli.py tests/hier_loop/test_knowledge_experiment.py docs/domain-knowledge.md
  git commit -m "feat: add auditable VCC25 knowledge experiment CLI"
  ```

### Task 6: Stage the Independent Remote Experiment and Run Phase A

**Files:**
- Remote create: a new timestamped experiment directory outside both source checkouts.
- Remote create: `artifacts/phase-a-preflight/` inside that experiment directory.
- Never modify: the two source checkouts listed in Global Constraints.

**Interfaces:**
- Consumes: the tested local commit and immutable remote data/model paths.
- Produces: a remote experiment copy plus Phase A audit artifacts.

- [ ] **Step 1: Record source fingerprints and validate destination**

  Capture local commit, remote RSI file hashes needed by the experiment, Lingshu code revision, weight revision, and destination realpath. Abort if the destination equals or is contained within either source checkout.

- [ ] **Step 2: Stage tracked code into the new directory**

  Transfer the tested local tree without `.git`, caches, outputs, credentials, final-test data, or large model assets. Bind large immutable assets by read-only path in the experiment config.

- [ ] **Step 3: Run compile, unit, plugin-load, and asset preflights remotely**

  Run Python compilation, the focused hierarchy/knowledge tests, task-plugin loading, Codex/rjob availability checks, and adapter asset checks. Do not invoke Codex decisions or submit a GPU job yet.

- [ ] **Step 4: Run five-layer no-GPU preflight**

  Execute `knowledge-preflight` with `RecordingDecisionBackend`. Verify L1-L5 each contain non-empty card provenance and L5 emits a schema-valid candidate request.

- [ ] **Step 5: Run read-only Codex planning probe for L1-L4**

  Execute one ephemeral read-only decision per planning layer. Verify JSON-schema compliance and that no workspace file hash changes.

- [ ] **Step 6: Archive and report Phase A evidence**

  Preserve the manifest and hashes in the experiment directory. If any check fails, stop before GPU work and report the exact failing gate.

### Task 7: Run Phase B/C and Conditionally Phase D

**Files:**
- Remote create: new append-only artifact directories per attempt inside the isolated experiment workspace.
- Remote create: candidate worktree produced by the existing coding-agent runner.

**Interfaces:**
- Consumes: the Phase A-approved L5 request, immutable train/validation assets, and promoted candidate hash.
- Produces: smoke evidence, same-contract validation baseline, candidate validation reward, and at most one final official result.

- [ ] **Step 1: Generate one candidate with at most one repair**

  Invoke the coding-agent runner only inside its candidate worktree. Confirm changed paths, activation diagnostics, static checks, patch hash, and absence of source-checkout modifications.

- [ ] **Step 2: Run small-data train/validation smoke**

  Submit the bounded GPU job using the validation-only entry point. Label its result `smoke_only`, archive resource usage and split audit, and verify that no reward or promotion is calculated.

- [ ] **Step 3: Establish the same-contract full-validation Lingshu baseline**

  Evaluate Lingshu through the identical validation preprocessing, genes, targets, seeds, and metric implementation that the candidate will use. Stop if this baseline cannot be established; the historical final `0.3056` is not a substitute.

- [ ] **Step 4: Run the candidate's full validation experiment**

  Submit the frozen seed set, produce `search_validation` evidence, and calculate reward only through `validation_reward`. Feed the result back to the hierarchy experience store.

- [ ] **Step 5: Apply the promotion gate**

  Promote only if the candidate improves the same-contract validation baseline, all seeds complete, code and data hashes match, and leakage/resource gates pass. Otherwise stop and report that the first experiment did not produce a final candidate.

- [ ] **Step 6: Freeze and run the one-shot final evaluation only if promoted**

  Record the candidate hash and irreversible final-evaluation budget before submission. Run canonical VCC25 evaluation once, set `selection_feedback_allowed=false`, and report whether `pearson_delta > 0.306` without feeding the result into reward, memory, or another candidate round.

- [ ] **Step 7: Run completion verification**

  Recheck remote source hashes, local test results, artifact completeness, authority labels, and the absence of a second final submission. Report measured results and limitations whether or not the threshold is exceeded.
