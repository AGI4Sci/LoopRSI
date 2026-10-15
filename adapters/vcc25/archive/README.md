# Archive — `adapters/vcc25/`

This directory holds files that were historically part of the
`adapters/vcc25/` workspace but are **not part of the live adapter
contract** today. They are preserved here for audit and reproducibility
rather than executed from a controller.

If a file is referenced by production code, by an `if __name__ ==
"__main__"` rjob-side script wired into a runner, or by a unit test
under `tests/vcc25_adapters/`, it does **not** belong here — it stays
one level up.

## Layout

| Subfolder | What lives here | Why it is archived |
| --- | --- | --- |
| `probes/` | Six one-off diagnostic scripts (`probe_gears_*`, `probe_l5_container_assets`, `probe_log_artifact_transfer`, `probe_presage_environment`, `probe_gpfs_write`). | Each was a write-and-forget check used while bringing an rjob container up. Three probe scripts (`probe_gears_import`, `probe_primeflow_environment`, `probe_primeflow_offline_install`) remain live at the package root because rjob runners and unit tests still import them. |
| `smoke/` | `run_l5_validation_smoke.py` and `preflight_l5_remote_assets.py`. | Zero references in production, tests, or other scripts. Superseded by `run_presage_h1_smoke.py` / `run_primeflow_h1_smoke.py`. Some paths hard-code another user's GPFS prefix, which is why they are not safe to re-run. |
| `h1_data_prep/` | `build_h1_gears_dataset.py`, `build_h1_gears_shards.py`, `build_h1_gears_smoke.py`, `plan_h1_graph_cache.py`, `verify_h1_gears_shards.py`, `verify_gears_checkpoint.py`, `compact_gears_go.py`. | One-shot H1 data-engineering scripts that materialized the 150/50 train/validation split, GEARS shard pickles, GO-edge compaction, and checkpoint verification. They are not imported; they were invoked by shell pipelines during the bounded H1 GPU smoke campaign. Kept here as program-of-record so the same inputs can be regenerated. |
| `assets/sources/` | `PRESAGE-2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb.tar.gz`. | Pinned source archive whose SHA-256 is recorded in `PRESAGE_ASSET_MANIFEST.json`. The manifest's `relative_path` points into this folder. |
| `assets/wheelhouse/` | Reserved for the pinned PRESAGE Python wheels. Currently empty. | The manifest declares the path as `pending_packaging`; wheels are populated out-of-band. |
| `assets/patches/` | `gears-smoke-single-de.patch`, `gears-scipy-series-index.patch`. | Patches recorded inside the `knowledge/vcc25/evidence/gears-h1-gpu-smoke-20261005.json` evidence card. They are not applied at runtime; they are part of the audit trail. |
| `gears_smoke_split.json` | A two-condition `{ACAT2, ACLY}` split. | Never imported; superseded by the fixed 150/50 H1 split that lives in the gear H1 preflight input layer. |

## How paths are interpreted

`PRESAGE_ASSET_MANIFEST.json` is the only file in the live tree that
references any of the archived assets. After the move, its
`source_archive.relative_path` is `archive/assets/sources/PRESAGE-...`
and its `wheelhouse.relative_path` is `archive/assets/wheelhouse/`. The
manifest itself remains at `adapters/vcc25/PRESAGE_ASSET_MANIFEST.json`
and is still loaded by `run_presage_h1_smoke.py`'s resolver.

## What is **not** archived

The following files are intentionally kept at the package root because
they are still part of the live adapter contract:

- `__init__.py`, `base.py`, `protocol.py`, `external.py`
- The six paper adapters: `gears.py`, `lingshu.py`, `perturbench.py`,
  `primeflow.py`, `scgenept.py`, `state.py`
- `gears_h1.py`, `scgenept_h1.py`, `gears_shard_loader.py`,
  `reproduction_entry.py`
- The four live rjob runners: `run_gears_smoke_bundle.py`,
  `run_gears_checkpoint_smoke.py`, `run_presage_h1_smoke.py`,
  `run_primeflow_h1_smoke.py`
- The three live probes: `probe_gears_import.py`,
  `probe_primeflow_environment.py`, `probe_primeflow_offline_install.py`
- `PRESAGE_ASSET_MANIFEST.json`

## Restoring a file

If a future adapter needs one of the archived scripts back, move it
back with `git mv archive/<bucket>/<file>.py .` and update any
`sys.path.insert` callsites or rjob shell scripts accordingly. The
`PRESAGE_ASSET_MANIFEST.json` will need its `relative_path` reverted in
the same change.
