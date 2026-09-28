from __future__ import annotations

import argparse, json, time, math
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from crpm.cold_start import TargetFeatureCRPM
from crpm.target_features import hashed_gene_symbol_features, projected_gene_embedding_features
from official_h1_autonomous_research_loop import AnchoredSoftplusHead, aggregate, delta_diagnostics
from official_h1_cold_start_probe import CONTROL, choose_targets, control_baselines, encode_rows, load_sample, mean_pairwise_l2, mean_profile_correlation, mse, read_symbol_to_gene_id
from official_h1_contract import read_condition_targets, read_gene_names
from official_h1_nonnegative_probe import normalize_log1p

DEFAULT_ROOT=Path('/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1')
DEFAULT_ASSETS=DEFAULT_ROOT/'assets/official_2025'
PROTOCOL_ID='vcc-h1-neighborhood-direction-prior-validation-v1'


def load_embedding_matrix(genes, symbol_to_gene_id, embedding_npz):
    mat=np.zeros((len(genes),0), dtype=np.float32); mapped=np.zeros(len(genes), dtype=bool)
    with np.load(embedding_npz, allow_pickle=False) as z:
        dim=None
        for i,g in enumerate(genes):
            gid=symbol_to_gene_id.get(g)
            if gid in z.files:
                v=np.asarray(z[gid], dtype=np.float32)
                if dim is None:
                    dim=v.shape[0]; mat=np.zeros((len(genes),dim), dtype=np.float32)
                mat[i]=v; mapped[i]=True
    norms=np.linalg.norm(mat, axis=1, keepdims=True)
    mat=np.divide(mat, norms, out=np.zeros_like(mat), where=norms>0)
    return mat, mapped


def build_hash_embedding(args, genes, all_targets, train_h5ad):
    h=hashed_gene_symbol_features(all_targets, genes, args.feature_dim)
    sym=read_symbol_to_gene_id(train_h5ad, genes)
    e,meta=projected_gene_embedding_features(all_targets, sym, str(args.gene_embedding_npz), args.feature_dim, args.embedding_projection_seed)
    feat=np.concatenate([h,e], axis=1)
    return {t:feat[i] for i,t in enumerate(all_targets)}, sym, meta


def train_base(args, seed, train, val, fmap, train_targets, batch_order):
    train_log,target_sum=normalize_log1p(train.x); val_log,_=normalize_log1p(val.x,target_sum)
    trf,trtid,trbid,_=encode_rows(train,[CONTROL]+train_targets,batch_order,fmap,False)
    vf,_,vbid,unknown=encode_rows(val,[CONTROL]+train_targets,batch_order,fmap,True)
    baselines=control_baselines(train_log,train.target_names_per_row,train.batch_names_per_row,batch_order)
    baseline_pred=baselines[vbid]
    torch.manual_seed(seed); np.random.seed(seed)
    base=TargetFeatureCRPM(torch.from_numpy(baselines), int(trf.shape[1]), args.rank, len(train_targets)+1, len(batch_order), False)
    model=AnchoredSoftplusHead(base)
    opt=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ds=TensorDataset(torch.from_numpy(train_log),torch.from_numpy(trf),torch.from_numpy(trtid),torch.from_numpy(trbid))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    losses=[]; steps=0
    while steps<args.max_steps:
        for y,feat,tid,bid in loader:
            opt.zero_grad(set_to_none=True); pred=model(feat,bid,tid,allow_id=False)
            loss=torch.mean((pred-y)**2); loss.backward(); opt.step(); losses.append(float(loss.detach())); steps+=1
            if steps>=args.max_steps: break
    with torch.no_grad():
        pred=model(torch.from_numpy(vf), torch.from_numpy(vbid), None, allow_id=False).numpy().astype(np.float32)
        cond=model.condition(torch.from_numpy(np.asarray([fmap[t] for t in sorted(set(val.target_names_per_row)-{CONTROL})], dtype=np.float32)), None, allow_id=False).numpy()
    return train_log,val_log,pred,baseline_pred,cond,unknown,{'optimizer_steps':steps,'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None}


def train_direction_stats(train_log, train, genes, batch_order):
    gene_index={g:i for i,g in enumerate(genes)}; batch_index={b:i for i,b in enumerate(batch_order)}
    baselines=control_baselines(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order)
    signed=[]; abs_delta=[]
    by_target={}
    for x,t,b in zip(train_log, train.target_names_per_row, train.batch_names_per_row):
        if t==CONTROL: continue
        delta=x-baselines[batch_index.get(b,0)]
        abs_delta.append(np.abs(delta)); signed.append(delta)
        by_target.setdefault(t,[]).append(delta)
    global_signed=np.mean(np.asarray(signed), axis=0).astype(np.float32)
    global_abs=np.mean(np.asarray(abs_delta), axis=0).astype(np.float32)
    target_delta={t:np.mean(np.asarray(v), axis=0).astype(np.float32) for t,v in by_target.items()}
    return global_signed, global_abs, target_delta


def nearest_train_targets(target, train_targets, emb_mat, gene_index, k):
    if target not in gene_index: return []
    i=gene_index[target]; sims=[]
    tv=emb_mat[i]
    if not np.any(tv): return []
    for tr in train_targets:
        j=gene_index.get(tr)
        if j is None or not np.any(emb_mat[j]): continue
        sims.append((float(np.dot(tv, emb_mat[j])), tr))
    sims.sort(reverse=True)
    return sims[:k]


def prior_matrix(kind, val_targets_per_row, genes, train_targets, emb_mat, target_delta, global_signed, global_abs, self_effect, scale, k):
    out=np.zeros((len(val_targets_per_row), len(genes)), dtype=np.float32)
    gene_index={g:i for i,g in enumerate(genes)}
    for r,t in enumerate(val_targets_per_row):
        if t==CONTROL: continue
        if kind in {'self','self_neighbor','self_module'}:
            j=gene_index.get(t)
            if j is not None:
                out[r,j]+=self_effect
        if kind in {'embedding_neighbor','self_neighbor'}:
            neigh=nearest_train_targets(t, train_targets, emb_mat, gene_index, k)
            if neigh:
                sims=np.asarray([max(0.0,s) for s,_ in neigh], dtype=np.float32)
                if float(sims.sum())<=0: sims=np.ones_like(sims)
                sims=sims/(sims.sum()+1e-8)
                for w,(_,nt) in zip(sims,neigh):
                    out[r]+=float(w)*target_delta[nt]
        if kind in {'module','self_module'}:
            j=gene_index.get(t)
            if j is not None and np.any(emb_mat[j]):
                sims=emb_mat @ emb_mat[j]
                mask=sims >= np.quantile(sims[sims>0], 0.995) if np.any(sims>0) else np.zeros_like(sims,dtype=bool)
                direction=np.sign(self_effect) if self_effect !=0 else -1.0
                out[r,mask]+=direction*global_abs[mask]
    return scale*out


def eval_pred(name, pred, val_log, baseline_pred, val, cond, scale):
    target_means=[]
    vt=sorted(set(val.target_names_per_row)-{CONTROL})
    for t in vt:
        mask=np.asarray([x==t for x in val.target_names_per_row], dtype=bool)
        target_means.append(pred[mask].mean(axis=0))
    return {
        'arm':name,'scale':scale,'status':'pass' if np.isfinite(pred).all() and float(pred.min())>=0 else 'failed',
        'prediction_finite':bool(np.isfinite(pred).all()),'prediction_nonnegative':bool(float(pred.min())>=0),'prediction_min':float(pred.min()),'prediction_max':float(pred.max()),
        'baseline_mse':mse(val_log, baseline_pred),'prediction_mse':mse(val_log,pred),'delta_mse_vs_baseline':float(mse(val_log,baseline_pred)-mse(val_log,pred)),
        'baseline_mean_profile_correlation':mean_profile_correlation(val_log,baseline_pred,val.target_names_per_row),'prediction_mean_profile_correlation':mean_profile_correlation(val_log,pred,val.target_names_per_row),'delta_mean_profile_correlation_vs_baseline':float(mean_profile_correlation(val_log,pred,val.target_names_per_row)-mean_profile_correlation(val_log,baseline_pred,val.target_names_per_row)),
        'conditioning_pairwise_l2_mean':mean_pairwise_l2(cond),'prediction_target_mean_pairwise_l2_mean':mean_pairwise_l2(np.asarray(target_means,dtype=np.float32)),
        **delta_diagnostics(val_log,pred,baseline_pred,val.target_names_per_row),
    }


def run_seed(args, seed):
    t0=time.time(); genes=read_gene_names(args.gene_names)
    train_all=read_condition_targets(args.train_targets); val_all=read_condition_targets(args.validation_targets); test_all=read_condition_targets(args.test_targets)
    if train_all & val_all or train_all & test_all or val_all & test_all: raise ValueError('split leakage')
    train_targets=choose_targets(train_all,args.n_train_targets,seed); val_targets=choose_targets(val_all,args.n_val_targets,seed+1)
    train=load_sample(args.train_h5ad,genes,train_targets,args.max_rows_per_target,seed+10); val=load_sample(args.validation_h5ad,genes,val_targets,args.max_rows_per_target,seed+20)
    all_targets=sorted({CONTROL,*train_all,*val_all,*test_all}); fmap,sym,emb_meta=build_hash_embedding(args,genes,all_targets,args.train_h5ad)
    emb_mat,mapped=load_embedding_matrix(genes,sym,str(args.gene_embedding_npz)); batch_order=['__global__']+sorted(set(train.batch_names_per_row))
    train_log,val_log,base_pred,baseline_pred,cond,unknown,train_meta=train_base(args,seed,train,val,fmap,train_targets,batch_order)
    global_signed,global_abs,target_delta=train_direction_stats(train_log,train,genes,batch_order)
    self_vals=[]; gi={g:i for i,g in enumerate(genes)}
    for t,d in target_delta.items():
        if t in gi: self_vals.append(float(d[gi[t]]))
    self_effect=float(np.median(self_vals)) if self_vals else 0.0
    arms={'base_hash_embedding_anchor':eval_pred('base_hash_embedding_anchor',base_pred,val_log,baseline_pred,val,cond,0.0)}
    for kind in args.variants:
        for scale in args.scales:
            prior=prior_matrix(kind,val.target_names_per_row,genes,train_targets,emb_mat,target_delta,global_signed,global_abs,self_effect,scale,args.neighbor_k)
            pred=np.maximum(0.0, base_pred+prior).astype(np.float32)
            arms[f'{kind}_{scale:g}x']=eval_pred(f'{kind}_{scale:g}x',pred,val_log,baseline_pred,val,cond,scale)
    return {'seed':seed,'runtime_seconds':time.time()-t0,'embedding_gene_panel_coverage':float(mapped.mean()),'embedding_target_metadata':emb_meta,'train_self_effect':self_effect,'unknown_validation_batches':unknown,**train_meta,'arms':arms}


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proposal={'candidate_id':'candidate_f_neighborhood_direction_prior','parent':'candidate_e_self_gene_direction_prior','hypothesis':'Self-gene prior is too local; fixed gene-embedding nearest-neighbor and module priors may spread train-learned perturbation direction to unseen targets.','implementation':{'base':'hash_embedding_anchor','variants':['embedding_neighbor','module','self_neighbor','self_module'],'source':'fixed gene embeddings + train-only perturbation deltas'},'test_expression_used':False,'official_score_claim':False}
    (args.output_dir/'proposal.json').write_text(json.dumps(proposal,indent=2,sort_keys=True)+'\n')
    seeds=[int(s) for s in args.seeds.split(',')]; by_arm={}; refs=[]
    for seed in seeds:
        res=run_seed(args,seed); p=args.output_dir/f'result_seed{seed}.json'; p.write_text(json.dumps(res,indent=2,sort_keys=True,default=str)+'\n'); refs.append(str(p))
        for arm,row in res['arms'].items(): by_arm.setdefault(arm,[]).append({**row,'seed':seed,'runtime_seconds':res['runtime_seconds']})
    arms={arm:{'runs':rows,'aggregate':aggregate(rows),'status':'pass' if all(r['status']=='pass' for r in rows) else 'failed'} for arm,rows in by_arm.items()}
    ranked=sorted(arms.items(), key=lambda kv:(kv[1]['aggregate']['delta_correlation_mean'],kv[1]['aggregate']['de_direction_agreement_top5pct_mean'],kv[1]['aggregate']['delta_mse_vs_baseline_mean']), reverse=True)
    summary={'protocol_id':PROTOCOL_ID,'proposal':proposal,'data_contract':{'train_targets':args.n_train_targets,'validation_targets':args.n_val_targets,'test_expression_read':False,'official_100_target_prediction_generated':False},'arms':arms,'best_arm':ranked[0][0],'seed_results_ref':refs,'finding':'Neighborhood prior variants are promoted only if they beat candidate_e/base on validation delta correlation and DE-direction across 3 seeds.'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,default=str)+'\n')
    return summary


def parse_args():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-h5ad',type=Path,default=DEFAULT_ASSETS/'train/adata_Training.h5ad'); ap.add_argument('--validation-h5ad',type=Path,default=DEFAULT_ASSETS/'validation/adata_Validation.h5ad')
    ap.add_argument('--train-targets',type=Path,default=DEFAULT_ASSETS/'train/pert_counts_Training.csv'); ap.add_argument('--validation-targets',type=Path,default=DEFAULT_ASSETS/'validation/pert_counts_Validation.csv'); ap.add_argument('--test-targets',type=Path,default=DEFAULT_ASSETS/'test/pert_counts_Test.csv'); ap.add_argument('--gene-names',type=Path,default=DEFAULT_ASSETS/'gene_names.csv')
    ap.add_argument('--gene-embedding-npz',type=Path,default=DEFAULT_ROOT/'assets/lingshu_hf_b77f980/gene_embeddings.npz'); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--n-train-targets',type=int,default=150); ap.add_argument('--n-val-targets',type=int,default=50); ap.add_argument('--max-rows-per-target',type=int,default=32); ap.add_argument('--max-steps',type=int,default=2048); ap.add_argument('--rank',type=int,default=24); ap.add_argument('--feature-dim',type=int,default=128); ap.add_argument('--embedding-projection-seed',type=int,default=20260908); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--learning-rate',type=float,default=8e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--neighbor-k',type=int,default=8); ap.add_argument('--seeds',default='20260907,20260908,20260909'); ap.add_argument('--scales',type=float,nargs='+',default=[0.25,0.5,1.0]); ap.add_argument('--variants',nargs='+',default=['embedding_neighbor','module','self_neighbor','self_module'])
    return ap.parse_args()

if __name__=='__main__': print(json.dumps(run(parse_args()),indent=2,sort_keys=True,default=str))
