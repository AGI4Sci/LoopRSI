# VCC25 Task Context

This adapter owns the VCC25 execution boundary. The generic controller must not
contain guide, target, batch, gene-expression, CRPM, or pseudo-bulk logic.

Backends remain task-local under:

- `tasks/vcc25/implementation/crpm/`
- `tasks/vcc25/implementation/train.py`
- `tasks/vcc25/implementation/scripts/`

Adapter method registry:

- Baselines: `global_mean`, `batch_control_mean`
- Trials: `crpm`, `guide_crpm`, `target_prototype`, `pseudobulk`
- `pseudobulk` maps to `guide_crpm` with aggregate training enabled.
- `pseudobulk_hierarchy` maps to `hierarchy_crpm`, but is exposed to Heuresis
  only when the activation verification marker matches the current training and
  model implementation hashes.

The verified hierarchy is `root -> target -> guide`. It learns ancestor
embeddings, combines them with a weighted sum, and adds the projected hierarchy
contribution to CRPM. Adapter parameters are `hierarchy_embedding_dim`,
`hierarchy_aggregation=weighted_sum`, `hierarchy_alpha_learnable`, and the
negative-control-only `hierarchy_shuffle`. This is not the complete GHCT in the
Idea Card: `vcc25@2` has no pathway/gene-program parent annotations and the
current backend is CRPM rather than a flow-matching transport network.

Every hierarchy run must emit contract
`vcc25.guide_hierarchy_activation.v1`. The Adapter rejects a run unless it has
positive activation count, affected rows, transformed dimension, gradient norm,
parameter update, contribution sum, and embedding norm. It also requires a
perfect guide-parent match for the real mapping and a changed mapping for the
shuffle control.

Scientific comparisons should use held-out-guide or another explicitly declared
held-out perturbation protocol. `vcc25_tenth_512g.npz` is suitable for fast
plumbing checks; `vcc25_full_512g.npz` is the task spec's full-data input.

The full-data adapter chain was verified on an H200 through rjob as
`zzc-vcc-task-adapter-full-13729568`: all six actions completed and the job
exited successfully. That run intentionally used only two CRPM optimizer steps,
so it proves execution compatibility, not model quality.

The canonical source is `/data/zhangzhicheng/omni-ar/competition_train.h5`
(SHA-256 `3dc850b767553648b91bf0b21a0e753be3010f36cf0cf3e98f6ef616c75aa847`,
221273 cells by 18080 genes). The task adapter trains through the verified
512-gene artifact derived from that H5; all cells are retained and only the gene
dimension is reduced. The prepared artifact SHA-256 is
`267f6f58014d3ee56fae280172602dcd2b5281a39f26bfdfb16d8a0e4d0eb4f1`.

The current preserved full-data configuration is count-weighted pseudo-bulk
with frequency-adaptive guide shrinkage. Three 250-epoch H200 trials on seeds
20250805/06/07 produced mean delta Pearson `0.6671778121` (population standard
deviation `0.0905980035`). The previous constant-shrinkage mean was
`0.6656623662`. This is a small, consistent parameter-level gain, not a new
architecture. The frozen guide-CRPM regression case explicitly retains
`guide_shrinkage_type=constant` so the default change does not alter that
historical method.

## Biological hierarchy and flow-matching verification (2026-08-18)

The versioned dataset annotation `vcc25-go-bp-20260818` adds an official GO
Biological Process DAG between the root and target nodes. It covers 147 of 150
non-control target genes (98%), then links all 151 targets and 189 guides. The
asset contains 3,430 nodes and 7,653 edges; its files, source URLs, source
hashes, transformation parameters, validation checks, and unmapped genes are
recorded under `datasets/vcc25/annotations/go-bp-20260818/`. This is a GO DAG,
not a curated pathway/gene-program hierarchy, so it must not be described as
the complete pathway GHCT from the Idea Card.

The Dataset Adapter provides bounded `load_hierarchy`, `validate_hierarchy`,
`hierarchy_statistics`, and `build_negative_control` actions. Supported
negative controls are target-annotation shuffle and degree-preserving target
swap. Heuresis receives these capabilities through the Task Adapter; the
generic controller does not interpret biological nodes or edges.

Two hierarchy experiments were rejected and remain hidden from Heuresis:

- The original `root -> target -> guide` CRPM hierarchy failed its three-seed
  stability comparison against both no hierarchy and shuffled parents.
- GO-DAG CRPM activated correctly, but the full real hierarchy did not exceed
  both the no-hierarchy and shuffled controls in the fixed seed-20250805 run.

The task-local conditional-flow backend predicts pseudo-bulk perturbation
transport from target, guide, and batch conditions and uses a flow-matching
velocity objective with Euler inference. The hierarchy-conditioned variant
adds multi-ancestor GO-DAG conditioning. Both methods emit fail-closed
activation diagnostics for optimizer events, affected rows, gradient norm,
parameter update, velocity contribution, and trajectory displacement; the GO
variant additionally verifies hierarchy gradients, updates, contribution,
layer activation, annotation hash, and negative-control identity.

At 250 epochs under the unchanged full-data held-out-guide protocol, seeds
20250805/06/07 produced:

- no-hierarchy flow: 0.567855, 0.409704, 0.324210 (mean 0.433923);
- real GO-hierarchy flow: 0.573335, 0.476612, 0.449190 (mean 0.499713);
- degree-preserving control: 0.568225, 0.465109, 0.374886 (mean 0.469407).

The real hierarchy effect was positive on all three seeds, averaging +0.065790
against no hierarchy and +0.030306 against the degree-preserving control.
Therefore `pseudobulk_flow_matching` and
`pseudobulk_go_hierarchy_flow` are eligible for bounded Heuresis proposals.
They are research mechanisms, not new best VCC25 results: their absolute scores
remain below the established CRPM configuration. Verification-only method
names are never exposed to Heuresis, and any missing/stale marker, changed code
hash, changed annotation hash, missing diagnostic, inactive phenotype, protocol
change, or budget violation fails closed before archive acceptance.

On 2026-08-19, a task-local prototype-residual flow composition was added. The
existing low-rank target/batch prototype supplies the initial state and the
conditional flow learns only the remaining transport. Boyue Heuresis proposed
three multi-parameter combinations under fixed `vcc25@2`, `full_512g`,
held-out-guide, seed 20250805, 250 epochs, batch size 256, learning rate 0.001,
and one-GPU budget. Their full screening scores were 0.549219, 0.565659, and
0.587296. Only the third exceeded the previous same-seed no-hierarchy flow
reference, although none met Heuresis's 0.65 screen criterion.

The selected depth-4, condition-64, 16-step, rank-8, blend-0.3 candidate was
then evaluated as a strict three-way pair. No hierarchy scored 0.610605, the
real GO hierarchy scored 0.587296, and the degree-preserving annotation control
scored 0.674056. Consequently the real GO prototype-flow method is rejected
and hidden from Heuresis. The high negative-control score is not an accepted
biological result; it is evidence that structural randomization or
regularization, rather than true GO semantics, caused the apparent gain. The
no-hierarchy prototype-flow method remains eligible for bounded research but
is not a new task best.
