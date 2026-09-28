#!/usr/bin/env python3
"""Memory-safe native evaluator for the vcc25 official H1 protocol.

Reuses the cell-eval metric algorithms (``PerturbationAnndataPair``,
``pearson_delta``, ``mae``) but never materializes the 12 GB dense prediction
matrix: per-perturbation bulk means are accumulated in bounded row chunks from
the h5ad on disk.  Real bulks are read directly from the precomputed
``real_de.csv`` (``target_mean`` / ``ref_mean`` == cell-eval's perturbation /
control pseudobulks).

Writes ``official_h1_result.json`` in exactly the layout produced by
``run_official_h1_candidate_g.sh`` so the adapter's read path is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _sha256_stream(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_real_de(real_de_csv: Path, pred_genes: np.ndarray, control_pert: str):
    """Real pseudobulks (101 x G) from real_de.csv, aligned to pred gene order.

    Returns (real_bulks, pert_names) where rows are sorted non-control
    perturbations followed by the control perturbation (cell-eval sort order).
    """
    import polars as pl

    de = pl.read_csv(real_de_csv)
    need = {"target", "feature", "target_mean", "ref_mean"}
    missing = need - set(de.columns)
    if missing:
        raise SystemExit(f"real_de.csv missing columns: {sorted(missing)}")
    # Control pseudobulk = ref_mean (non-targeting pool) per gene.
    ref_stats = de.group_by("feature").agg(
        pl.col("ref_mean").mean().alias("ref_mean"),
        pl.col("ref_mean").std().alias("ref_std"),
    )
    if float(ref_stats["ref_std"].drop_nulls().max()) > 1e-6:
        raise SystemExit("ref_mean is not constant across targets in real_de.csv")
    ref = ref_stats.select(["feature", "ref_mean"]).to_numpy()
    targets = sorted({str(t) for t in de["target"].unique().to_list()})
    if control_pert in targets:
        targets.remove(control_pert)
    pert_names = np.array(targets + [control_pert], dtype=object)
    n_pert = len(pert_names)
    n_genes = len(pred_genes)
    gene_col = {g: i for i, g in enumerate(pred_genes)}
    bulks = np.zeros((n_pert, n_genes), dtype=np.float64)
    for tgt_i, tgt in enumerate(targets):
        sub = de.filter(pl.col("target") == tgt)
        feats = sub["feature"].to_numpy()
        means = sub["target_mean"].to_numpy()
        try:
            col = np.array([gene_col[f] for f in feats], dtype=np.intp)
        except KeyError as exc:
            raise SystemExit(f"gene {exc.args[0]!r} in real_de.csv missing from prediction var")
        if np.any(col < 0):
            bad = feats[col < 0][:5]
            raise SystemExit(f"genes in real_de.csv missing from prediction var: {bad}")
        bulks[tgt_i, col] = means
    # Control row: map ref_mean per gene into the control row.
    ref_feats = ref[:, 0]
    ref_means = ref[:, 1].astype(np.float64)
    try:
        col = np.array([gene_col[f] for f in ref_feats], dtype=np.intp)
    except KeyError as exc:
        raise SystemExit(f"ref feature {exc.args[0]!r} missing from prediction var")
    if np.any(col < 0):
        raise SystemExit("ref_mean features missing from prediction var")
    bulks[-1, col] = ref_means
    return bulks, pert_names


def _pred_bulks_from_h5ad(pred_h5ad: Path, control_pert: str, chunk_rows: int = 4096):
    """Predicted pseudobulks via bounded row-chunk accumulation (h5py).

    Returns (bulks, pert_names) with the same row order as ``_load_real_de``.
    """
    import h5py

    with h5py.File(pred_h5ad, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group):
            raise SystemExit("prediction h5ad X must be a dense dataset")
        n_rows, n_genes = X.shape
        tg = f["obs"]["target_gene"]
        names = np.array(tg[()], dtype=str)
        uniq, counts = np.unique(names, return_counts=True)
        if control_pert not in set(uniq):
            raise SystemExit(
                f"control perturbation {control_pert!r} not found in prediction obs"
            )
        targets = sorted({u for u in uniq.tolist() if u != control_pert})
        order = {p: i for i, p in enumerate(targets + [control_pert])}
        sums = np.zeros((len(order), n_genes), dtype=np.float64)
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            block = np.asarray(X[start:stop], dtype=np.float64)
            labels = names[start:stop]
            for lab in np.unique(labels):
                idx = np.flatnonzero(labels == lab)
                sums[order[lab]] += block[idx].sum(axis=0)
        bulks = sums / np.maximum(counts[[order[u] for u in uniq]], 1)[:, None]
        return bulks, np.array(targets + [control_pert], dtype=object)


def _metrics_from_bulks(real_bulks, pred_bulks, pert_names, genes, control_pert):
    """Reuse cell-eval's PerturbationAnndataPair + pearson_delta/mae."""
    import anndata as ad
    import pandas as pd
    from cell_eval._types import PerturbationAnndataPair
    from cell_eval.metrics import pearson_delta, mae

    var = pd.DataFrame(index=pd.Index(genes.astype(str), name="gene"))
    real = ad.AnnData(X=real_bulks, obs=pd.DataFrame({"_p": pert_names}), var=var)
    pred = ad.AnnData(X=pred_bulks, obs=pd.DataFrame({"_p": pert_names}), var=var)
    pair = PerturbationAnndataPair(
        real=real, pred=pred, pert_col="_p", control_pert=control_pert
    )
    pd_map = pearson_delta(pair)
    mae_map = mae(pair)
    keys = [p for p in pert_names if p != control_pert]
    return {
        "pearson_delta": float(np.mean([pd_map[k] for k in keys])),
        "mae": float(np.mean([mae_map[k] for k in keys])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-h5ad", required=True)
    ap.add_argument("--real-de-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--control-pert", default="non-targeting")
    ap.add_argument("--candidate-id", default="candidate_g_promoted_delta_model")
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--skip-sha256", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    pred_path = Path(args.pred_h5ad)
    if not pred_path.is_file():
        raise SystemExit(f"prediction h5ad missing: {pred_path}")
    if not Path(args.real_de_csv).is_file():
        raise SystemExit(f"real_de.csv missing: {args.real_de_csv}")

    print(f"[native-eval] reading prediction bulks from {pred_path}", flush=True)
    with __import__("h5py").File(pred_path, "r") as f:
        genes = np.array(f["var"]["_index"][()], dtype=str)
        n_genes = len(genes)
    if n_genes != 18080:
        raise SystemExit(f"unexpected gene count {n_genes} (expected 18080)")

    pred_bulks, pert_names = _pred_bulks_from_h5ad(
        pred_path, args.control_pert, args.chunk_rows
    )
    print("[native-eval] reading real DE bulks", flush=True)
    real_bulks, real_names = _load_real_de(
        Path(args.real_de_csv), genes, args.control_pert
    )
    if not np.array_equal(pert_names, real_names):
        raise SystemExit(
            f"perturbation order mismatch:\n pred {pert_names}\n real {real_names}"
        )
    n_pert_nonctrl = len(pert_names) - 1
    if n_pert_nonctrl != 100:
        raise SystemExit(f"expected 100 non-control perturbations, got {n_pert_nonctrl}")

    print("[native-eval] computing cell-eval metrics on bulks", flush=True)
    metrics = _metrics_from_bulks(
        real_bulks, pred_bulks, pert_names, genes, args.control_pert
    )
    print("[native-eval] metrics:", json.dumps(metrics, sort_keys=True), flush=True)

    contract_path = out_dir / "official_h1_contract.json"
    if not contract_path.is_file():
        raise SystemExit(f"contract missing: {contract_path}")
    print("[native-eval] hashing artifacts", flush=True)
    pred_sha = None if args.skip_sha256 else _sha256_stream(pred_path)
    contract_sha = _sha256_stream(contract_path)

    # Mirror the runner's agg_results.csv layout (statistic rows) so the
    # aggregate artifact stays inspectable.
    agg_dir = out_dir / "cell_eval"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_path = agg_dir / "native_agg_results.csv"
    metric_names = sorted(metrics)
    with agg_path.open("w", encoding="utf-8") as fh:
        fh.write("statistic," + ",".join(metric_names) + "\n")
        fh.write("mean," + ",".join(f"{metrics[m]:.10g}" for m in metric_names) + "\n")
        fh.write("count," + ",".join("100" for _ in metric_names) + "\n")
        fh.write("null_count," + ",".join("0" for _ in metric_names) + "\n")

    summary = {
        "protocol_id": "vcc-h1-cell-eval-v1",
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "status": "pass",
        "targets": 100,
        "genes": 18080,
        "metrics": metrics,
        "references": {
            "lingshu_local_pearson_delta": 0.24808131580837134,
            "lingshu_paper_pearson_delta": 0.306,
        },
        "artifacts": {
            "prediction_h5ad": str(pred_path.resolve()),
            "prediction_sha256": pred_sha,
            "contract_sha256": contract_sha,
            "aggregate_sha256": _sha256_stream(agg_path),
        },
    }
    result_path = out_dir / "official_h1_result.json"
    result_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[native-eval] wrote " + str(result_path), flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
