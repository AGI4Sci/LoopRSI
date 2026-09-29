# VCC25 Executable Domain Knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free VCC25 knowledge plugin whose provisionally approved paper, repository, and model cards can guide RSI decisions and safely describe runnable Lingshu, STATE, and PerturBench workflows.

**Architecture:** Git-managed JSON is loaded into immutable Python card types, validated, and queried deterministically. Existing `ResearchSkill` entrypoints render bounded evidence; separate allowlisted adapters inspect assets and construct typed execution requests without downloading weights or executing arbitrary commands. VCC25 evaluation is a fixed contract, never an Agent-selectable card.

**Tech Stack:** Python 3 standard library, JSON, JSON Schema documents, `unittest`, existing `ai4ai.plugin_protocols` and `CandidateExecutionRequest`.

**Spec:** `docs/superpowers/specs/2026-09-29-vcc25-domain-knowledge-design.md`

## Global Constraints

- Add no PyYAML, jsonschema, database, or vector-store runtime dependency.
- Seed cards use `review_status: approved` and `review_basis: provisional_initial_release`; rendered evidence must show the basis.
- `VCC25EvaluationContract` is fixed configuration and must never appear in Agent search results.
- Model weights remain outside Git; cards store URI, size when known, integrity status, licenses, and adapter reference.
- Published final-test results are `reported_only` evidence and cannot drive the same search loop.
- Reject `real_de`, hidden test-expression/answer fields, and final-reference paths from prompts and execution requests.
- Do not change existing reward, memory, or trajectory semantics.
- Do not mutate the remote RSI source or its existing weights.

## Review Focus

- Duplicate IDs, malformed asset arrays, broken relations, or an unknown card type must fail with file and field context; Task 1 tests each.
- A model cannot claim `ready` without recorded smoke evidence; Tasks 1 and 4 test downgrade/rejection behavior.
- A provisional card must be searchable but its review basis must remain visible in every rendered hit; Tasks 2 and 3 test this.
- Near a token boundary, rendering must omit a whole card rather than truncate its provenance; Task 3 tests this.
- Restricted test identifiers embedded in nested card/proposal/adapter data must be rejected recursively; Tasks 1, 3, and 4 test all three boundaries.

---

## File Structure

- `domain_knowledge/cards.py`: immutable cards, query, evaluation-contract, and validation error types.
- `domain_knowledge/store.py`: JSON loading, recursive safety validation, relation checks, and deterministic query.
- `domain_knowledge/render.py`: bounded human-readable rendering and content hashes.
- `domain_knowledge/__main__.py`: validate/query CLI.
- `knowledge/vcc25/`: schemas, fixed evaluation contract, and seed cards.
- `skills/vcc25/lingshu.py`: three manifest-declared `ResearchSkill` implementations.
- `adapters/vcc25/base.py`: allowlisted adapter protocol and structured asset-check result.
- `adapters/vcc25/{lingshu,state,perturbench}.py`: static asset validation and typed request construction.
- `tests/domain_knowledge/` and `tests/vcc25_adapters/`: focused and end-to-end tests.
- `docs/domain-knowledge.md`: curator and execution workflow.

### Task 1: Card Store and Fixed Evaluation Contract

**Files:**
- Create: `domain_knowledge/__init__.py`
- Create: `domain_knowledge/cards.py`
- Create: `domain_knowledge/store.py`
- Create: `tests/domain_knowledge/__init__.py`
- Create: `tests/domain_knowledge/test_store.py`

**Interfaces:**
- Produces: `KnowledgeCard`, `KnowledgeQuery`, `VCC25EvaluationContract`, `KnowledgeValidationError`.
- Produces: `KnowledgeStore.from_directory(root: Path) -> KnowledgeStore`.
- Produces: `KnowledgeStore.query(query: KnowledgeQuery) -> tuple[KnowledgeCard, ...]`.
- Produces: `KnowledgeStore.evaluation_contract -> VCC25EvaluationContract` separately from searchable cards.
- Produces: `VCC25EvaluationContract.assert_comparable(contract_ids: Sequence[str]) -> None`, raising
  `KnowledgeValidationError` when any result references a different contract.
- Produces: `KnowledgeStore.validate() -> tuple[str, ...]`, returning no warnings for a valid tree and raising `KnowledgeValidationError` for invalid data.

- [ ] **Step 1: Write failing store and contract tests**

Use temporary real JSON trees to test: all three card types; provisional approved retrieval; asset-type/layer/readiness filtering; stable ID tie-break; unknown L1–L5 rejection; duplicate IDs; missing shared/model fields; malformed `artifacts`; broken relations; evaluation contract exclusion from query; mismatched result contract IDs; recursive restricted identifiers; and `ready` without smoke evidence.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.domain_knowledge.test_store -v`
Expected: import failure for the missing `domain_knowledge` package.

- [ ] **Step 3: Implement immutable types, validation, and query**

Load only `papers`, `repositories`, and `models`; require namespaced IDs `kb:paper:*`, `kb:repo:*`, and `kb:model:*`. Load `evaluation_contract.json` through its dedicated type. Rank by layer match, token/tag overlap, readiness order, then ID. Traverse nested dict/list/string values for restricted identifiers.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `python3 -m unittest tests.domain_knowledge.test_store -v`
Run: `python3 -m unittest discover -s tests/hier_loop -t . -p 'test_*.py' && python3 -m unittest discover -s tests/rsi_step0 -t . -p 'test_*.py'`
Expected: all pass; existing skip count remains 3.

- [ ] **Step 5: Commit**

Commit: `feat: add VCC25 knowledge store and evaluation contract`

### Task 2: Provisionally Approved Seed Knowledge

**Files:**
- Create: `knowledge/vcc25/evaluation_contract.json`
- Create: `knowledge/vcc25/schemas/{card,paper,repository,model,evaluation-contract}.schema.json`
- Create: six files under `knowledge/vcc25/papers/`
- Create: six files under `knowledge/vcc25/repositories/`
- Create: six files under `knowledge/vcc25/models/`
- Create: `tests/domain_knowledge/test_seed_knowledge.py`

**Interfaces:**
- Consumes: `KnowledgeStore` from Task 1.
- Produces paper IDs: `kb:paper:lingshu-cell`, `kb:paper:state`, `kb:paper:gears`, `kb:paper:linear-baseline`, `kb:paper:perturbench`, `kb:paper:scgenept`.
- Produces matching repository IDs under `kb:repo:*` and model IDs for Lingshu VCC 85M, STATE ST-HVG-Replogle, PerturBench LatentAdditive, scGenePT GO-All, GEARS, and Linear/Pseudobulk.
- Produces all seed cards with `review_status: approved`, `review_basis: provisional_initial_release`, and explicit source/relation provenance.

- [ ] **Step 1: Write the failing real-tree test**

Assert exact seed counts (6/6/6), unique IDs, valid relations, at least one authoritative HTTPS source per public card, visible provisional review basis, no benchmark card, fixed evaluation contract exclusion, no committed weight-like files, and model-specific readiness/artifact/license/IO/adapter fields. Assert Lingshu is `adapter_required`, STATE and scGenePT are `finetune_required`, PerturBench/GEARS/Linear are `train_required`.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.domain_knowledge.test_seed_knowledge -v`
Expected: FAIL because the knowledge tree is absent.

- [ ] **Step 3: Add schemas, contract, and source-checked cards**

Write concise Chinese `summary_plain`, `fit_reason`, and `limitations` fields. Represent unknown sizes/checksums honestly with integrity status `not_published`. Record the local `0.30614` claim only as a source note tied to an unresolved contract, never as a sortable model score.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m unittest discover -s tests/domain_knowledge -t . -p 'test_*.py'`
Expected: all pass.

- [ ] **Step 5: Commit**

Commit: `data: seed executable VCC25 knowledge assets`

### Task 3: Bounded Rendering and ResearchSkill Integration

**Files:**
- Create: `domain_knowledge/render.py`
- Create: `skills/__init__.py`
- Create: `skills/vcc25/__init__.py`
- Create: `skills/vcc25/lingshu.py`
- Create: `tests/domain_knowledge/test_lingshu_skill.py`

**Interfaces:**
- Consumes: `KnowledgeStore.query` and real seed cards.
- Produces: `RenderedKnowledge(content, card_ids, estimated_tokens, content_hash)`.
- Produces: `render_cards(cards: Sequence[KnowledgeCard], token_budget: int) -> RenderedKnowledge`.
- Produces: `LingshuAlignmentSkill`, `LingshuDataProcessingInsightSkill`, and `LingshuModelDesignInsightSkill` implementing existing `ResearchSkill`.

- [ ] **Step 1: Write failing renderer and plugin tests**

Load the real manifest and seed tree. Assert all entrypoints instantiate; non-VCC25 contexts do not activate; L1/L2 returns paper evidence, L3/L4 includes mechanisms/limitations, and L5 returns repository/model readiness and adapter IDs. Assert provisional basis, card IDs, source, SHA-256 hash, whole-card budget behavior, deterministic output, and recursive rejection of restricted proposals.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.domain_knowledge.test_lingshu_skill -v`
Expected: import failure for `skills.vcc25.lingshu`.

- [ ] **Step 3: Implement renderer and the three skills**

Use one private base configured by layers and asset types. Resolve the default knowledge root relative to the repository and allow a constructor-injected root for tests. Return `None` when no safe whole card fits the budget.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `python3 -m unittest discover -s tests/domain_knowledge -t . -p 'test_*.py'`
Run the existing hier_loop and rsi_step0 commands from Task 1.
Expected: all pass; existing skip count remains 3.

- [ ] **Step 5: Commit**

Commit: `feat: expose VCC25 knowledge through research skills`

### Task 4: Allowlisted Model Adapters

**Files:**
- Create: `adapters/__init__.py`
- Create: `adapters/vcc25/__init__.py`
- Create: `adapters/vcc25/base.py`
- Create: `adapters/vcc25/lingshu.py`
- Create: `adapters/vcc25/state.py`
- Create: `adapters/vcc25/perturbench.py`
- Create: `tests/vcc25_adapters/__init__.py`
- Create: `tests/vcc25_adapters/test_adapters.py`

**Interfaces:**
- Produces: `AssetCheck(status: str, missing: tuple[str, ...], evidence: Mapping[str, Any])`.
- Produces: `VCC25ModelAdapter.check_assets(model: KnowledgeCard, locations: Mapping[str, str]) -> AssetCheck`.
- Produces: `VCC25ModelAdapter.prepare(action: str, model: KnowledgeCard, context: Mapping[str, Any]) -> CandidateExecutionRequest`.
- Produces: registry function `get_adapter(adapter_id: str) -> VCC25ModelAdapter` for exactly `vcc25.lingshu`, `vcc25.state`, and `vcc25.perturbench`.

- [ ] **Step 1: Write failing adapter tests**

Using temporary harmless files, assert each adapter reports exact missing roles; accepts only declared actions; constructs `CandidateExecutionRequest` with an allowlisted executable/module, explicit working/output directories, required inputs, resource metadata, and no shell interpolation; rejects unknown adapter IDs, path traversal, undeclared environment keys, final-test identifiers, over-budget resources, and a model/adapter mismatch. Assert static checks do not mark any model `ready` or execute subprocesses.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.vcc25_adapters.test_adapters -v`
Expected: import failure for `adapters.vcc25`.

- [ ] **Step 3: Implement protocol, registry, and minimal adapters**

Use tuple commands only. Lingshu references its declared inference script and multi-file assets; STATE uses `state tx` actions; PerturBench uses installed `train`/`predict` entrypoints. Constructors accept explicit repository/executable locations and never clone or download.

- [ ] **Step 4: Verify GREEN and all suites**

Run: `python3 -m unittest discover -s tests/vcc25_adapters -t . -p 'test_*.py'`
Run domain_knowledge plus existing hier_loop/rsi_step0 suites.
Expected: all pass; no external process or network access occurs.

- [ ] **Step 5: Commit**

Commit: `feat: add safe VCC25 model adapters`

### Task 5: CLI, Documentation, and End-to-End Contract

**Files:**
- Create: `domain_knowledge/__main__.py`
- Create: `docs/domain-knowledge.md`
- Modify: `README.md`
- Create: `tests/domain_knowledge/test_end_to_end.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `python3 -m domain_knowledge --root knowledge/vcc25 validate`.
- Produces: `python3 -m domain_knowledge --root knowledge/vcc25 query --layer L5 --text perturbation --readiness adapter_required finetune_required --token-budget 400`.
- Produces: `python3 -m domain_knowledge --root knowledge/vcc25 check-adapter --model-id kb:model:lingshu-vcc-85m`.

- [ ] **Step 1: Write failing CLI and end-to-end tests**

Run CLI functions in-process against the real tree. Assert validation succeeds; evaluation contract is reported separately; L1 and L5 return appropriate provisionally approved cards; adapter checking reports missing or present roles without execution; JSON output contains stable IDs and hashes; restricted input exits nonzero; and no benchmark choice appears.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.domain_knowledge.test_end_to_end -v`
Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement CLI and concise maintainer documentation**

Document card editing, provisional versus full-review basis, fixed evaluation contract, asset mapping, safe adapter checks, and the separate later step required for real smoke inference on an approved compute node.

- [ ] **Step 4: Final verification**

Run all three CLI commands above.
Run: `python3 -m unittest discover -s tests/domain_knowledge -t . -p 'test_*.py'`
Run: `python3 -m unittest discover -s tests/vcc25_adapters -t . -p 'test_*.py'`
Run existing hier_loop and rsi_step0 suites.
Expected: every suite passes, existing skip count remains 3, Git worktree is clean, and no external process/network call occurs.

- [ ] **Step 5: Commit**

Commit: `docs: document executable VCC25 knowledge workflow`
