# VCC25 L5 validation selection

`hier_loop.cli run --worker knowledge_vcc25` shares one memory manager between
the hierarchy, `HistoricalDecisionBackend`, and `Vcc25Worker`. L5 selects only
variants currently supported by the official H1 adapter. A failed variant is
avoided on the next round. A successful variant can be selected from its real
validation score. If every supported variant has failed, the decision stops.

Every real L5 attempt writes a `vcc25.validation-record/v1` JSON file under the
configured `worker.result_root`, including status, method, variant, seed,
authority, metric, elapsed wall time, requested GPU count, and error. Passed
records require a finite metric and explicit train/validation split isolation.
Failed records have no metric value and cannot feed selection. Both passed and
failed attempts are persisted to L5 memory with cost; only passed attempts get
a numeric reward. Other method adapters can emit the same record through
`hier_loop.validation_record`, but they remain unavailable for L5 selection
until their execution path and validation contract are integrated.

The recording backend remains available for no-execution preflight. Historical
selection is an execution policy, not evidence of improved VCC25 performance;
that requires completed train/validation runs and a controlled comparison.
