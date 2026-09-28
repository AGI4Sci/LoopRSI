from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from crpm.cold_start import TargetFeatureCRPM
from crpm.target_features import hashed_gene_symbol_features, projected_gene_embedding_features
from official_h1_cold_start_probe import (
    CONTROL,
    choose_targets,
    control_baselines,
    encode_rows,
    load_sample,
    mean_pairwise_l2,
    read_symbol_to_gene_id,
    mean_profile_correlation,
    mse,
)
from official_h1_contract import read_condition_targets, read_gene_names

DEFAULT_ROOT = Path('/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1')
DEFAULT_ASSETS = DEFAULT_ROOT / 'assets/official_2025'
PROTOCOL_ID = 'vcc-h1-nonnegative-parameterization-validation-v1'


def normalize_log1p(x: np.ndarray, target_sum: float | None = None) -> tuple[np.ndarray, float]:
    lib = x.sum(axis=1, dtype=np.float64)
    positive = lib > 0
    if target_sum is None:
        target_sum = float(np.median(lib[positive])) if positive.any() else 1.0
    scale = np.ones_like(lib, dtype=np.float64)
    scale[positive] = target_sum / lib[positive]
    return np.log1p(x.astype(np.float64) * scale[:, None]).astype(np.float32), target_sum


class PositiveHead(torch.nn.Module):
    def __init__(self, base: TargetFeatureCRPM, beta: float = 1.0, anchored: bool = False) -> None:
        super().__init__()
        self.base = base
        self.softplus = torch.nn.Softplus(beta=beta)
        self.anchored = anchored

    def condition(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.base.condition(*args, **kwargs)

    @staticmethod
    def _inv_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    def forward(self, features: torch.Tensor, batch_id: torch.Tensor, target_id: torch.Tensor | None = None, allow_id: bool = False) -> torch.Tensor:
        if not self.anchored:
            return self.softplus(self.base(features, batch_id, target_id, allow_id=allow_id))
        weights = self.base.condition(features, target_id, allow_id=allow_id) + self.base.batch_weights(batch_id)
        residual = weights @ self.base.gene_programs
        anchor = self._inv_softplus(self.base.baselines[batch_id])
        return self.softplus(anchor + residual)


def corr_delta(y_true: np.ndarray, y_pred: np.ndarray, base: np.ndarray, targets: list[str]) -> float:
    values = []
    tarr = np.asarray(targets, dtype=object)
    for target in sorted(set(targets) - {CONTROL}):
        mask = tarr == target
        if not mask.any():
            continue
        truth = (y_true[mask] - base[mask]).mean(axis=0, dtype=np.float64)
        pred = (y_pred[mask] - base[mask]).mean(axis=0, dtype=np.float64)
        truth -= truth.mean(); pred -= pred.mean()
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        if denom > 0:
            values.append(float(np.dot(truth, pred) / denom))
    return float(np.mean(values)) if values else float('nan')


def run_one(args: argparse.Namespace, mode: str, train_x: np.ndarray, val_x: np.ndarray, train: Any, val: Any, features: dict[str, np.ndarray], train_target_order: list[str], batch_order: list[str]) -> dict[str, Any]:
    train_features, train_target_ids, train_batch_ids, _ = encode_rows(train, train_target_order, batch_order, features, False)
    val_features, _, val_batch_ids, unknown_val_batches = encode_rows(val, train_target_order, batch_order, features, True)
    baselines = control_baselines(train_x, train.target_names_per_row, train.batch_names_per_row, batch_order)
    baseline_pred = baselines[val_batch_ids]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    base = TargetFeatureCRPM(torch.from_numpy(baselines), int(train_features.shape[1]), args.rank, len(train_target_order), len(batch_order), False)
    model: torch.nn.Module = PositiveHead(base, args.softplus_beta, anchored=mode.startswith('anchored')) if (mode.startswith('softplus') or mode.startswith('anchored')) else base
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_features), torch.from_numpy(train_target_ids), torch.from_numpy(train_batch_ids))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    losses=[]; steps=0
    while steps < args.max_steps:
        for y, feat, tid, bid in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(feat, bid, tid, allow_id=False)
            loss = torch.mean((pred - y) ** 2)
            loss.backward(); opt.step()
            losses.append(float(loss.detach())); steps += 1
            if steps >= args.max_steps:
                break
    with torch.no_grad():
        pred = model(torch.from_numpy(val_features), torch.from_numpy(val_batch_ids), None, allow_id=False).numpy().astype(np.float32)
        cond = model.condition(torch.from_numpy(np.asarray([features[t] for t in sorted(set(val.target_names_per_row) - {CONTROL})], dtype=np.float32)), None, allow_id=False).numpy()
    target_means=[]
    for target in sorted(set(val.target_names_per_row) - {CONTROL}):
        mask = np.asarray([t == target for t in val.target_names_per_row], dtype=bool)
        target_means.append(pred[mask].mean(axis=0))
    target_means=np.asarray(target_means, dtype=np.float32)
    return {
        'mode': mode,
        'prediction_finite': bool(np.isfinite(pred).all()),
        'prediction_min': float(np.min(pred)),
        'prediction_max': float(np.max(pred)),
        'prediction_nonnegative': bool(np.min(pred) >= 0),
        'baseline_mse': mse(val_x, baseline_pred),
        'prediction_mse': mse(val_x, pred),
        'delta_mse_vs_baseline': float(mse(val_x, baseline_pred) - mse(val_x, pred)),
        'baseline_mean_profile_correlation': mean_profile_correlation(val_x, baseline_pred, val.target_names_per_row),
        'prediction_mean_profile_correlation': mean_profile_correlation(val_x, pred, val.target_names_per_row),
        'delta_mean_profile_correlation_vs_baseline': float(mean_profile_correlation(val_x, pred, val.target_names_per_row) - mean_profile_correlation(val_x, baseline_pred, val.target_names_per_row)),
        'delta_correlation': corr_delta(val_x, pred, baseline_pred, val.target_names_per_row),
        'conditioning_pairwise_l2_mean': mean_pairwise_l2(cond),
        'prediction_target_mean_pairwise_l2_mean': mean_pairwise_l2(target_means),
        'loss_first': losses[0] if losses else None,
        'loss_last': losses[-1] if losses else None,
        'optimizer_steps': steps,
        'unknown_validation_batches_mapped_to_global_control': unknown_val_batches,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    genes = read_gene_names(args.gene_names)
    train_targets_all = read_condition_targets(args.train_targets)
    val_targets_all = read_condition_targets(args.validation_targets)
    test_targets_all = read_condition_targets(args.test_targets)
    if train_targets_all & val_targets_all or train_targets_all & test_targets_all or val_targets_all & test_targets_all:
        raise ValueError('official target splits are not disjoint')
    train_targets = choose_targets(train_targets_all, args.n_train_targets, args.seed)
    val_targets = choose_targets(val_targets_all, args.n_val_targets, args.seed + 1)
    train = load_sample(args.train_h5ad, genes, train_targets, args.max_rows_per_target, args.seed + 10)
    val = load_sample(args.validation_h5ad, genes, val_targets, args.max_rows_per_target, args.seed + 20)
    all_targets = sorted({CONTROL, *train_targets_all, *val_targets_all, *test_targets_all})
    hash_features = hashed_gene_symbol_features(all_targets, genes, args.feature_dim)
    symbol_to_gene_id = read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding_features, embedding_metadata = projected_gene_embedding_features(
        all_targets, symbol_to_gene_id, str(args.gene_embedding_npz), args.feature_dim, args.embedding_projection_seed
    )
    if args.representation == 'hash':
        feature_mat = hash_features
        representation_metadata = {
            'type': 'deterministic_gene_symbol_hash_features',
            'feature_dim': args.feature_dim,
            'coverage_insufficient_for_official_h1': False,
            'uses_expression': False,
            'test_time_available': True,
        }
    elif args.representation == 'embedding':
        feature_mat = embedding_features
        official_targets = [*train_targets_all, *val_targets_all, *test_targets_all]
        representation_metadata = {
            **embedding_metadata,
            'coverage_train_selected': int(np.count_nonzero(np.linalg.norm(embedding_features[[all_targets.index(t) for t in train_targets]], axis=1) > 0)),
            'coverage_validation_selected': int(np.count_nonzero(np.linalg.norm(embedding_features[[all_targets.index(t) for t in val_targets]], axis=1) > 0)),
            'coverage_test_panel': int(np.count_nonzero(np.linalg.norm(embedding_features[[all_targets.index(t) for t in test_targets_all]], axis=1) > 0)),
            'coverage_insufficient_for_official_h1': any(t in embedding_metadata['unmapped_sample'] for t in official_targets),
        }
    else:
        feature_mat = np.concatenate([hash_features, embedding_features], axis=1)
        official_targets = [*train_targets_all, *val_targets_all, *test_targets_all]
        representation_metadata = {
            'type': 'hash_plus_projected_fixed_gene_embedding_features',
            'feature_dim': int(feature_mat.shape[1]),
            'hash_dim': args.feature_dim,
            'embedding_dim': args.feature_dim,
            'embedding_metadata': embedding_metadata,
            'coverage_insufficient_for_official_h1': any(t in embedding_metadata['unmapped_sample'] for t in official_targets),
            'uses_expression': False,
            'test_time_available': True,
        }
    fmap = {t: feature_mat[i] for i,t in enumerate(all_targets)}
    train_target_order = [CONTROL] + train_targets
    batch_order = ['__global__'] + sorted(set(train.batch_names_per_row))
    train_log, target_sum = normalize_log1p(train.x)
    val_log, _ = normalize_log1p(val.x, target_sum)
    modes = []
    modes.append(run_one(args, 'linear_count_current', train.x, val.x, train, val, fmap, train_target_order, batch_order))
    modes.append(run_one(args, 'softplus_count', train.x, val.x, train, val, fmap, train_target_order, batch_order))
    modes.append(run_one(args, 'anchored_softplus_count', train.x, val.x, train, val, fmap, train_target_order, batch_order))
    modes.append(run_one(args, 'softplus_log1p', train_log, val_log, train, val, fmap, train_target_order, batch_order))
    modes.append(run_one(args, 'anchored_softplus_log1p', train_log, val_log, train, val, fmap, train_target_order, batch_order))
    selected = max([m for m in modes if m['prediction_nonnegative'] and m['prediction_finite']], key=lambda m: (m['delta_mean_profile_correlation_vs_baseline'], m['delta_mse_vs_baseline']), default=None)
    return {
        'protocol_id': PROTOCOL_ID,
        'status': 'pass' if selected else 'failed',
        'data_contract': {
            'train_targets_total': len(train_targets_all), 'validation_targets_total': len(val_targets_all), 'test_targets_total': len(test_targets_all),
            'split_disjointness': True, 'gene_count': len(genes), 'gene_order_preserved': True, 'test_expression_read': False,
        },
        'representation': representation_metadata,
        'probe_config': vars(args) | {'train_rows': len(train.x), 'validation_rows': len(val.x), 'log1p_target_sum_from_train_sample': target_sum},
        'modes': modes,
        'selected_mode': selected['mode'] if selected else None,
        'official_score_claim': False,
        'official_test_prediction_generated': False,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-h5ad', type=Path, default=DEFAULT_ASSETS/'train/adata_Training.h5ad')
    ap.add_argument('--validation-h5ad', type=Path, default=DEFAULT_ASSETS/'validation/adata_Validation.h5ad')
    ap.add_argument('--test-targets', type=Path, default=DEFAULT_ASSETS/'test/pert_counts_Test.csv')
    ap.add_argument('--train-targets', type=Path, default=DEFAULT_ASSETS/'train/pert_counts_Training.csv')
    ap.add_argument('--validation-targets', type=Path, default=DEFAULT_ASSETS/'validation/pert_counts_Validation.csv')
    ap.add_argument('--gene-names', type=Path, default=DEFAULT_ASSETS/'gene_names.csv')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--n-train-targets', type=int, default=150)
    ap.add_argument('--n-val-targets', type=int, default=50)
    ap.add_argument('--max-rows-per-target', type=int, default=4)
    ap.add_argument('--feature-dim', type=int, default=128)
    ap.add_argument('--representation', choices=['hash', 'embedding', 'hash_embedding'], default='hash')
    ap.add_argument('--gene-embedding-npz', type=Path, default=DEFAULT_ROOT/'assets/lingshu_hf_b77f980/gene_embeddings.npz')
    ap.add_argument('--embedding-projection-seed', type=int, default=20260908)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--softplus-beta', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=20260907)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result['status'] == 'pass' else 2

if __name__ == '__main__':
    raise SystemExit(main())
