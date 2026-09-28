from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
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
    mean_profile_correlation,
    mse,
    read_symbol_to_gene_id,
)
from official_h1_contract import read_condition_targets, read_gene_names
from official_h1_nonnegative_probe import normalize_log1p

DEFAULT_ROOT = Path('/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1')
DEFAULT_ASSETS = DEFAULT_ROOT / 'assets/official_2025'
PROTOCOL_ID = 'vcc-h1-autonomous-validation-research-loop-v1'


class AnchoredSoftplusHead(torch.nn.Module):
    def __init__(self, base: TargetFeatureCRPM) -> None:
        super().__init__()
        self.base = base
        self.softplus = torch.nn.Softplus()

    @staticmethod
    def inv_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    def condition(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.base.condition(*args, **kwargs)

    def residual(self, features: torch.Tensor, batch_id: torch.Tensor, target_id: torch.Tensor | None = None, allow_id: bool = False) -> torch.Tensor:
        weights = self.base.condition(features, target_id, allow_id=allow_id) + self.base.batch_weights(batch_id)
        return weights @ self.base.gene_programs

    def forward(self, features: torch.Tensor, batch_id: torch.Tensor, target_id: torch.Tensor | None = None, allow_id: bool = False) -> torch.Tensor:
        anchor = self.inv_softplus(self.base.baselines[batch_id])
        return self.softplus(anchor + self.residual(features, batch_id, target_id, allow_id=allow_id))


class AmplitudeAwareHead(AnchoredSoftplusHead):
    def __init__(self, base: TargetFeatureCRPM, rank: int) -> None:
        super().__init__(base)
        self.gain = torch.nn.Sequential(torch.nn.Linear(rank, 1), torch.nn.Softplus())
        with torch.no_grad():
            self.gain[0].weight.zero_()
            self.gain[0].bias.fill_(math.log(math.expm1(1.0)))

    def residual(self, features: torch.Tensor, batch_id: torch.Tensor, target_id: torch.Tensor | None = None, allow_id: bool = False) -> torch.Tensor:
        cond = self.base.condition(features, target_id, allow_id=allow_id)
        weights = cond + self.base.batch_weights(batch_id)
        return self.gain(cond) * (weights @ self.base.gene_programs)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent: str | None
    hypothesis: str
    representation: str
    head: str
    loss: str
    decision_rule: str


def delta_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray, targets: list[str]) -> dict[str, float]:
    corrs=[]; direction=[]; true_norms=[]; pred_norms=[]; sparsity=[]
    tarr=np.asarray(targets, dtype=object)
    for target in sorted(set(targets)-{CONTROL}):
        mask=tarr==target
        if not mask.any():
            continue
        truth=(y_true[mask]-baseline[mask]).mean(axis=0, dtype=np.float64)
        pred=(y_pred[mask]-baseline[mask]).mean(axis=0, dtype=np.float64)
        true_norms.append(float(np.linalg.norm(truth)))
        pred_norms.append(float(np.linalg.norm(pred)))
        threshold=np.quantile(np.abs(truth), 0.95) if np.any(truth) else 0.0
        top=np.abs(truth) >= threshold
        if top.any():
            direction.append(float(np.mean(np.sign(truth[top]) == np.sign(pred[top]))))
        sparsity.append(float(np.mean(np.abs(pred) > 0.05)))
        tc=truth-truth.mean(); pc=pred-pred.mean()
        denom=float(np.linalg.norm(tc)*np.linalg.norm(pc))
        if denom>0:
            corrs.append(float(np.dot(tc,pc)/denom))
    true_mean=float(np.mean(true_norms)) if true_norms else float('nan')
    pred_mean=float(np.mean(pred_norms)) if pred_norms else float('nan')
    return {
        'delta_correlation': float(np.mean(corrs)) if corrs else float('nan'),
        'de_direction_agreement_top5pct': float(np.mean(direction)) if direction else float('nan'),
        'true_delta_l2_mean': true_mean,
        'predicted_delta_l2_mean': pred_mean,
        'predicted_to_true_delta_l2_ratio': float(pred_mean/true_mean) if true_mean>0 else float('nan'),
        'predicted_delta_sparsity_abs_gt_0_05': float(np.mean(sparsity)) if sparsity else float('nan'),
    }


def build_features(args: argparse.Namespace, genes: list[str], all_targets: list[str], train_h5ad: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hash_features = hashed_gene_symbol_features(all_targets, genes, args.feature_dim)
    symbol_to_gene_id = read_symbol_to_gene_id(train_h5ad, genes)
    emb_features, emb_meta = projected_gene_embedding_features(
        all_targets, symbol_to_gene_id, str(args.gene_embedding_npz), args.feature_dim, args.embedding_projection_seed
    )
    features = np.concatenate([hash_features, emb_features], axis=1)
    return {t: features[i] for i,t in enumerate(all_targets)}, {
        'type':'hash_plus_projected_fixed_gene_embedding_features',
        'feature_dim': int(features.shape[1]),
        'hash_dim': args.feature_dim,
        'embedding_dim': args.feature_dim,
        'embedding_metadata': emb_meta,
        'uses_expression': False,
        'test_time_available': True,
    }


def train_and_eval(args: argparse.Namespace, candidate: Candidate, seed: int, train: Any, val: Any, fmap: dict[str, np.ndarray], train_targets: list[str], batch_order: list[str], de_weights: np.ndarray | None) -> dict[str, Any]:
    t0=time.time()
    train_log, target_sum=normalize_log1p(train.x)
    val_log,_=normalize_log1p(val.x, target_sum)
    train_features, train_target_ids, train_batch_ids,_=encode_rows(train, [CONTROL]+train_targets, batch_order, fmap, False)
    val_features,_,val_batch_ids,unknown_batches=encode_rows(val, [CONTROL]+train_targets, batch_order, fmap, True)
    baselines=control_baselines(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order)
    baseline_pred=baselines[val_batch_ids]
    torch.manual_seed(seed); np.random.seed(seed)
    base=TargetFeatureCRPM(torch.from_numpy(baselines), int(train_features.shape[1]), args.rank, len(train_targets)+1, len(batch_order), False)
    model = AmplitudeAwareHead(base, args.rank) if candidate.head == 'amplitude_aware_anchored_softplus' else AnchoredSoftplusHead(base)
    opt=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ds=TensorDataset(torch.from_numpy(train_log), torch.from_numpy(train_features), torch.from_numpy(train_target_ids), torch.from_numpy(train_batch_ids))
    loader=DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    weight_tensor=torch.from_numpy(de_weights.astype(np.float32)) if de_weights is not None and candidate.loss == 'train_delta_de_weighted_mse' else None
    losses=[]; steps=0
    while steps < args.max_steps:
        for y, feat, tid, bid in loader:
            opt.zero_grad(set_to_none=True)
            pred=model(feat,bid,tid,allow_id=False)
            err=(pred-y)**2
            loss=torch.mean(err * weight_tensor) if weight_tensor is not None else torch.mean(err)
            loss.backward(); opt.step()
            losses.append(float(loss.detach())); steps+=1
            if steps >= args.max_steps:
                break
    with torch.no_grad():
        pred=model(torch.from_numpy(val_features), torch.from_numpy(val_batch_ids), None, allow_id=False).numpy().astype(np.float32)
        cond=model.condition(torch.from_numpy(np.asarray([fmap[t] for t in sorted(set(val.target_names_per_row)-{CONTROL})], dtype=np.float32)), None, allow_id=False).numpy()
    target_means=[]
    for target in sorted(set(val.target_names_per_row)-{CONTROL}):
        mask=np.asarray([t==target for t in val.target_names_per_row], dtype=bool)
        target_means.append(pred[mask].mean(axis=0))
    diag=delta_diagnostics(val_log, pred, baseline_pred, val.target_names_per_row)
    return {
        'candidate_id': candidate.candidate_id,
        'seed': seed,
        'runtime_seconds': time.time()-t0,
        'status':'pass' if np.isfinite(pred).all() and float(pred.min()) >= 0 else 'failed',
        'prediction_finite': bool(np.isfinite(pred).all()),
        'prediction_nonnegative': bool(float(pred.min()) >= 0),
        'prediction_min': float(pred.min()),
        'prediction_max': float(pred.max()),
        'baseline_mse': mse(val_log, baseline_pred),
        'prediction_mse': mse(val_log, pred),
        'delta_mse_vs_baseline': float(mse(val_log, baseline_pred)-mse(val_log, pred)),
        'baseline_mean_profile_correlation': mean_profile_correlation(val_log, baseline_pred, val.target_names_per_row),
        'prediction_mean_profile_correlation': mean_profile_correlation(val_log, pred, val.target_names_per_row),
        'delta_mean_profile_correlation_vs_baseline': float(mean_profile_correlation(val_log, pred, val.target_names_per_row)-mean_profile_correlation(val_log, baseline_pred, val.target_names_per_row)),
        'conditioning_pairwise_l2_mean': mean_pairwise_l2(cond),
        'prediction_target_mean_pairwise_l2_mean': mean_pairwise_l2(np.asarray(target_means, dtype=np.float32)),
        'unknown_validation_batches_mapped_to_global_control': unknown_batches,
        'loss_first': losses[0] if losses else None,
        'loss_last': losses[-1] if losses else None,
        'optimizer_steps': steps,
        **diag,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys=['delta_mse_vs_baseline','delta_mean_profile_correlation_vs_baseline','delta_correlation','de_direction_agreement_top5pct','predicted_to_true_delta_l2_ratio','predicted_delta_sparsity_abs_gt_0_05','runtime_seconds']
    out={}
    for key in keys:
        vals=[float(r[key]) for r in rows if not math.isnan(float(r[key]))]
        out[key+'_mean']=float(np.mean(vals)) if vals else float('nan')
        out[key+'_min']=float(np.min(vals)) if vals else float('nan')
        out[key+'_max']=float(np.max(vals)) if vals else float('nan')
    return out


def train_delta_weights(train_log: np.ndarray, targets: list[str], batches: list[str], batch_order: list[str]) -> np.ndarray:
    baselines=control_baselines(train_log, targets, batches, batch_order)
    batch_index={b:i for i,b in enumerate(batch_order)}
    deltas=[]
    for row_target,row_batch,x in zip(targets,batches,train_log):
        if row_target == CONTROL:
            continue
        bid=batch_index.get(row_batch,0)
        deltas.append(np.abs(x-baselines[bid]))
    if not deltas:
        return np.ones(train_log.shape[1], dtype=np.float32)
    score=np.mean(np.asarray(deltas), axis=0)
    score=score/(np.mean(score)+1e-6)
    return np.clip(1.0 + score, 1.0, 5.0).astype(np.float32)


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    genes=read_gene_names(args.gene_names)
    train_all=read_condition_targets(args.train_targets)
    val_all=read_condition_targets(args.validation_targets)
    test_all=read_condition_targets(args.test_targets)
    if train_all & val_all or train_all & test_all or val_all & test_all:
        raise ValueError('official target splits are not disjoint')
    seeds=[int(s) for s in args.seeds.split(',')]
    first_seed=seeds[0]
    train_targets=choose_targets(train_all, args.n_train_targets, first_seed)
    val_targets=choose_targets(val_all, args.n_val_targets, first_seed+1)
    train=load_sample(args.train_h5ad, genes, train_targets, args.max_rows_per_target, first_seed+10)
    val=load_sample(args.validation_h5ad, genes, val_targets, args.max_rows_per_target, first_seed+20)
    all_targets=sorted({CONTROL,*train_all,*val_all,*test_all})
    fmap, feature_meta=build_features(args, genes, all_targets, args.train_h5ad)
    batch_order=['__global__']+sorted(set(train.batch_names_per_row))
    train_log,_=normalize_log1p(train.x)
    de_weights=train_delta_weights(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order)
    candidates=[
        Candidate('candidate_a_hash_embedding_anchor', None, 'Fixed hash+gene embedding target prior should improve unseen target residual direction over hash-only.', 'hash_embedding', 'anchored_softplus', 'mse', 'baseline representation candidate'),
        Candidate('candidate_b_amplitude_aware_hash_embedding', 'candidate_a_hash_embedding_anchor', 'Weak predicted delta amplitude suggests a feature-conditioned positive residual gain may improve DE direction without changing objective.', 'hash_embedding', 'amplitude_aware_anchored_softplus', 'mse', 'promote if delta correlation and DE direction improve across seeds'),
        Candidate('candidate_c_de_weighted_hash_embedding', 'candidate_a_hash_embedding_anchor', 'MSE may underweight perturbation-relevant genes; train-only delta weights may improve DE direction agreement.', 'hash_embedding', 'anchored_softplus', 'train_delta_de_weighted_mse', 'promote if DE direction improves without nonnegative failure'),
        Candidate('candidate_d_amplitude_de_weighted_hash_embedding', 'candidate_b_amplitude_aware_hash_embedding', 'Combine amplitude-aware residual scaling with train-only DE weighting to target both amplitude and direction.', 'hash_embedding', 'amplitude_aware_anchored_softplus', 'train_delta_de_weighted_mse', 'promote only if both delta correlation and DE direction improve'),
    ]
    lineage=[]; results={}
    for cand in candidates:
        rows=[]
        cand_dir=args.output_dir/cand.candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        proposal={'candidate_id':cand.candidate_id,'parent':cand.parent,'hypothesis':cand.hypothesis,'implementation':{'representation':cand.representation,'head':cand.head,'loss':cand.loss},'decision_rule':cand.decision_rule,'test_expression_used':False,'official_score_claim':False}
        (cand_dir/'proposal.json').write_text(json.dumps(proposal, indent=2, sort_keys=True)+'\n')
        for seed in seeds:
            row=train_and_eval(args,cand,seed,train,val,fmap,train_targets,batch_order,de_weights)
            rows.append(row)
            (cand_dir/f'result_seed{seed}.json').write_text(json.dumps(row, indent=2, sort_keys=True)+'\n')
        results[cand.candidate_id]={'proposal':proposal,'runs':rows,'aggregate':aggregate(rows),'status':'pass' if all(r['status']=='pass' for r in rows) else 'failed'}
        lineage.append({'candidate_id':cand.candidate_id,'parent':cand.parent,'result_ref':str(cand_dir)})
    ranking=sorted(results.items(), key=lambda kv: (kv[1]['aggregate']['delta_correlation_mean'], kv[1]['aggregate']['de_direction_agreement_top5pct_mean'], kv[1]['aggregate']['delta_mean_profile_correlation_vs_baseline_mean']), reverse=True)
    best_id=ranking[0][0]
    base=results['candidate_a_hash_embedding_anchor']['aggregate']
    best=results[best_id]['aggregate']
    summary={
        'protocol_id':PROTOCOL_ID,
        'status':'pass',
        'data_contract':{'train_targets_total':len(train_all),'validation_targets_total':len(val_all),'test_targets_total':len(test_all),'split_disjointness':True,'gene_count':len(genes),'gene_order_preserved':True,'test_expression_read':False},
        'feature_metadata':feature_meta,
        'training_config':vars(args) | {'seeds':seeds,'train_rows':len(train.x),'validation_rows':len(val.x)},
        'candidate_lineage':lineage,
        'results':results,
        'best_candidate':best_id,
        'improvement_vs_candidate_a':{k:best[k]-base[k] for k in best if k.endswith('_mean') and k in base},
        'scientific_interpretation':'Validation-only autonomous loop found whether residual amplitude and train-only DE weighting improve perturbation-specific signal over hash+embedding anchored softplus baseline.',
        'official_100_target_prediction_generated':False,
        'official_score_claim':False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'lineage_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=str)+'\n')
    return summary


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-h5ad', type=Path, default=DEFAULT_ASSETS/'train/adata_Training.h5ad')
    ap.add_argument('--validation-h5ad', type=Path, default=DEFAULT_ASSETS/'validation/adata_Validation.h5ad')
    ap.add_argument('--train-targets', type=Path, default=DEFAULT_ASSETS/'train/pert_counts_Training.csv')
    ap.add_argument('--validation-targets', type=Path, default=DEFAULT_ASSETS/'validation/pert_counts_Validation.csv')
    ap.add_argument('--test-targets', type=Path, default=DEFAULT_ASSETS/'test/pert_counts_Test.csv')
    ap.add_argument('--gene-names', type=Path, default=DEFAULT_ASSETS/'gene_names.csv')
    ap.add_argument('--gene-embedding-npz', type=Path, default=DEFAULT_ROOT/'assets/lingshu_hf_b77f980/gene_embeddings.npz')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--n-train-targets', type=int, default=150)
    ap.add_argument('--n-val-targets', type=int, default=50)
    ap.add_argument('--max-rows-per-target', type=int, default=8)
    ap.add_argument('--feature-dim', type=int, default=128)
    ap.add_argument('--embedding-projection-seed', type=int, default=20260908)
    ap.add_argument('--rank', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=192)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seeds', default='20260907,20260908,20260909')
    return ap.parse_args()


def main() -> int:
    result=run_loop(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
