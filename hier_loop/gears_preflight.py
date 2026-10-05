"""Pinned GEARS H1 execution preflight with a separate evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from adapters.vcc25.gears_h1 import preflight

from .paper_reproduction import _inside, _read_card


def run_gears_preflight(
    experiment_dir: Path, source_manifest: Path, output_dir: Path,
    processed_dataset: Path, split_json: Path, gene2go: Path, gene_names: Path,
) -> dict:
    experiment_dir = experiment_dir.resolve()
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    if not _inside(source_manifest, experiment_dir) or not _inside(output_dir, experiment_dir):
        raise ValueError("manifest and output must stay inside the independent experiment")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("GEARS preflight output must be empty")
    records = json.loads(source_manifest.read_text(encoding="utf-8"))["records"]
    source = next((row for row in records if row.get("method") == "gears"), None)
    if source is None or source.get("status") != "fetched":
        raise ValueError("pinned GEARS source is absent")
    repository = _read_card(Path(__file__).resolve().parents[1], "repositories", "gears")
    model = _read_card(Path(__file__).resolve().parents[1], "models", "gears")
    paper = _read_card(Path(__file__).resolve().parents[1], "papers", "gears")
    source_dir = Path(source["source_dir"]).resolve()
    archive = Path(source["archive"]).resolve()
    if (source.get("official_url") != repository["official_url"]
            or not _inside(source_dir, experiment_dir) or not _inside(archive, experiment_dir)
            or not archive.is_file()
            or hashlib.sha256(archive.read_bytes()).hexdigest() != source["archive_sha256"]):
        raise ValueError("GEARS source manifest does not match the pinned archive")
    command = [
        sys.executable, "-m", "hier_loop.cli", "gears-preflight",
        "--experiment-dir", str(experiment_dir), "--source-manifest", str(source_manifest),
        "--output", str(output_dir), "--processed-dataset", str(processed_dataset),
        "--split-json", str(split_json), "--gene2go", str(gene2go),
        "--gene-names", str(gene_names),
    ]
    result = preflight(argparse.Namespace(
        source_repository=str(source_dir), processed_dataset=str(processed_dataset),
        split_json=str(split_json), gene2go=str(gene2go), gene_names=str(gene_names),
        output=str(output_dir),
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "schema_version": "vcc25.paper-evidence/v1",
        "card_id": f"kb:evidence:gears:{source['revision'][:12]}:h1-preflight",
        "authority": "local_looprsi_h1_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "gears", "paper_card_id": paper["id"],
        "code_card_id": repository["id"], "model_card_id": model["id"],
        "observed_source_revision": source["revision"],
        "source_archive_sha256": source["archive_sha256"],
        "commands_attempted": [{"command": command, "exit_code": 0}],
        "data_scope": result["data_scope"], "status": "blocked",
        "blocker": "H1 inputs or dependencies missing" if result["status"] == "blocked"
                   else "preflight ready; training and validation have not run",
        "missing_assets": result["missing_assets"],
        "missing_modules": result["missing_modules"],
        "validation_metrics": None,
        "limitations": ["No GEARS training or inference was run"],
        "test_expression_read": False, "gpu_used": False,
        "final_evaluation_run": False,
    }
    (output_dir / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evidence_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card
