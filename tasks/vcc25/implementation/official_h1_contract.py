from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_GENES = 18080
EXPECTED_TEST_TARGETS = 100
PERT_KEYS = ("target_gene", "perturbation", "target", "gene")


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_gene_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_condition_targets(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty condition CSV: {path}")
    key = next((name for name in PERT_KEYS if name in rows[0]), None)
    if key is None:
        raise ValueError(f"no perturbation column in {path}; columns={list(rows[0])}")
    return {str(row[key]).strip() for row in rows if str(row[key]).strip()}


def _h5ad_index(group: Any) -> list[str]:
    index_key = _decode(group.attrs.get("_index", "_index"))
    return [_decode(item) for item in group[index_key][...]]


def _h5ad_column(group: Any, key: str) -> list[str]:
    node = group[key]
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        categories = [_decode(item) for item in node["categories"][...]]
        return [categories[int(code)] for code in node["codes"][...] if int(code) >= 0]
    return [_decode(item) for item in node[...]]


def inspect_h5ad(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        x = handle["X"]
        shape = tuple(x.shape) if hasattr(x, "shape") else tuple(x.attrs["shape"])
        var_names = _h5ad_index(handle["var"])
        obs_names = _h5ad_index(handle["obs"])
        pert_key = next((key for key in PERT_KEYS if key in handle["obs"]), None)
        perturbations = _h5ad_column(handle["obs"], pert_key) if pert_key else []
    return {
        "path": str(path),
        "shape": [int(shape[0]), int(shape[1])],
        "var_names": var_names,
        "obs_names": obs_names,
        "perturbation_column": pert_key,
        "perturbations": perturbations,
    }


def validate_contract(
    prediction_h5ad: Path,
    reference_test_h5ad: Path,
    condition_csv: Path,
    gene_names_csv: Path,
) -> dict[str, Any]:
    genes = read_gene_names(gene_names_csv)
    condition_targets = read_condition_targets(condition_csv)
    prediction = inspect_h5ad(prediction_h5ad)
    reference = inspect_h5ad(reference_test_h5ad)
    expected_targets = condition_targets | {"non-targeting"}
    prediction_targets = set(prediction["perturbations"])
    reference_targets = set(reference["perturbations"])
    gates = {
        "gene_count_18080": len(genes) == EXPECTED_GENES
        and prediction["shape"][1] == EXPECTED_GENES
        and reference["shape"][1] == EXPECTED_GENES,
        "gene_order_matches_official": prediction["var_names"] == genes
        and reference["var_names"] == genes,
        "obs_count_matches_reference_test": prediction["shape"][0] == reference["shape"][0],
        "obs_order_matches_reference_test": prediction["obs_names"] == reference["obs_names"],
        "target_column_present": bool(prediction["perturbation_column"])
        and bool(reference["perturbation_column"]),
        "condition_targets_100": len(condition_targets) == EXPECTED_TEST_TARGETS,
        "prediction_targets_match_condition_csv": prediction_targets == expected_targets,
        "reference_targets_match_condition_csv": reference_targets == expected_targets,
    }
    return {
        "protocol_id": "vcc-h1-official-prediction-contract-v1",
        "status": "ready" if all(gates.values()) else "blocked",
        "directly_comparable_input": all(gates.values()),
        "gates": gates,
        "inputs": {
            "prediction_h5ad": str(prediction_h5ad.resolve()),
            "reference_test_h5ad": str(reference_test_h5ad.resolve()),
            "condition_csv": str(condition_csv.resolve()),
            "gene_names_csv": str(gene_names_csv.resolve()),
        },
        "evidence": {
            "prediction_shape": prediction["shape"],
            "reference_shape": reference["shape"],
            "condition_target_count": len(condition_targets),
            "prediction_target_count": len(prediction_targets - {"non-targeting"}),
            "reference_target_count": len(reference_targets - {"non-targeting"}),
            "prediction_perturbation_column": prediction["perturbation_column"],
            "reference_perturbation_column": reference["perturbation_column"],
        },
        "note": (
            "This validates only the official H1 prediction H5AD contract. "
            "It does not compute Cell-Eval metrics and does not make an official leaderboard claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-h5ad", required=True)
    parser.add_argument("--reference-test-h5ad", required=True)
    parser.add_argument("--condition-csv", required=True)
    parser.add_argument("--gene-names", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_contract(
        Path(args.prediction_h5ad),
        Path(args.reference_test_h5ad),
        Path(args.condition_csv),
        Path(args.gene_names),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
