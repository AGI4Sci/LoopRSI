from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from crpm.cold_start import TargetFeatureCRPM
from official_h1_neighborhood_prior_probe import (
    DEFAULT_ASSETS, DEFAULT_ROOT, CONTROL, AnchoredSoftplusHead,
    build_hash_embedding, load_embedding_matrix, nearest_train_targets,
    train_direction_stats, eval_pred, aggregate,
)
from official_h1_cold_start_probe import _h5ad_column, _h5ad_index, choose_targets, control_baselines, encode_rows, load_sample
from official_h1_contract import read_condition_targets, read_gene_names
from official_h1_nonnegative_probe import normalize_log1p
from official_h1_autonomous_research_loop import AmplitudeAwareHead, train_delta_weights

CANDIDATE_ID='candidate_g_promoted_delta_model'
PROTOCOL_ID='vcc-h1-candidate-g-promotion-v1'
FIXED_NEIGHBOR_K=8
FIXED_PRIOR_SCALE=0.75


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, 0.0).astype(np.float32)


def inv_softplus_np(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.maximum(y.astype(np.float32), eps)
    return (y + np.log(-np.expm1(-y))).astype(np.float32)


def build_self_neighbor_prior_rows(targets: list[str], genes: list[str], state: dict[str, Any]) -> np.ndarray:
    gi={g:i for i,g in enumerate(genes)}
    prior=np.zeros((len(targets), len(genes)), dtype=np.float32)
    for r,t in enumerate(targets):
        if t==CONTROL: continue
        neigh=nearest_train_targets(t,state['train_targets'],state['emb_mat'],gi,FIXED_NEIGHBOR_K)
        if neigh:
            sims=np.asarray([max(0.0,s) for s,_ in neigh],dtype=np.float32)
            if float(sims.sum())<=0: sims=np.ones_like(sims)
            sims=sims/(sims.sum()+1e-8)
            for w,(_,nt) in zip(sims,neigh): prior[r]+=float(w)*state['target_delta'][nt]
        j=gi.get(t)
        if j is not None: prior[r,j]+=state['self_effect']
    return prior


def apply_candidate_f_prior(base_pred: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return softplus_np(inv_softplus_np(base_pred) + FIXED_PRIOR_SCALE * prior)


def read_obs_schema(path: Path, genes: list[str]) -> dict[str, Any]:
    with h5py.File(path,'r') as h:
        var=_h5ad_index(h['var'])
        if var != genes: raise ValueError('reference var_names do not match official gene order')
        return {'shape':[int(v) for v in h['X'].attrs['shape']], 'obs_names':_h5ad_index(h['obs']), 'target_gene':_h5ad_column(h['obs'],'target_gene'), 'guide_id':_h5ad_column(h['obs'],'guide_id'), 'batch':_h5ad_column(h['obs'],'batch')}


def write_h5ad(output: Path, obs_schema: dict[str, Any], genes: list[str], pred_iter, metadata: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True); sdt=h5py.string_dtype('utf-8')
    n_obs,n_genes=obs_schema['shape']; mins=[]; maxs=[]; finite=True
    with h5py.File(output,'w') as h:
        x=h.create_dataset('X', shape=(n_obs,n_genes), dtype='float32', chunks=(min(chunk_size,n_obs), n_genes))
        x.attrs['encoding-type']='array'; x.attrs['encoding-version']='0.2.0'
        obs=h.create_group('obs'); obs.attrs['_index']='_index'; obs.attrs['encoding-type']='dataframe'; obs.attrs['encoding-version']='0.2.0'; obs.attrs.create('column-order',['target_gene','guide_id','batch'],dtype=sdt)
        obs.create_dataset('_index',data=np.asarray(obs_schema['obs_names'],dtype=object),dtype=sdt)
        for c in ['target_gene','guide_id','batch']: obs.create_dataset(c,data=np.asarray(obs_schema[c],dtype=object),dtype=sdt)
        var=h.create_group('var'); var.attrs['_index']='_index'; var.attrs['encoding-type']='dataframe'; var.attrs['encoding-version']='0.2.0'; var.attrs.create('column-order',[],dtype=sdt); var.create_dataset('_index',data=np.asarray(genes,dtype=object),dtype=sdt)
        uns=h.create_group('uns'); uns.attrs['candidate_f_metadata']=json.dumps(metadata,sort_keys=True)
        for g in ['layers','obsm','obsp','varm','varp']: h.create_group(g)
        for start,end,pred in pred_iter:
            x[start:end]=pred.astype(np.float32); mins.append(float(pred.min())); maxs.append(float(pred.max())); finite=finite and bool(np.isfinite(pred).all())
    return {'prediction_min':min(mins),'prediction_max':max(maxs),'prediction_finite':finite,'prediction_nonnegative':min(mins)>=0,'shape':[n_obs,n_genes]}


def train_model(args, seed: int):
    genes=read_gene_names(args.gene_names); train_all=read_condition_targets(args.train_targets); val_all=read_condition_targets(args.validation_targets); test_all=read_condition_targets(args.test_targets)
    if train_all & val_all or train_all & test_all or val_all & test_all: raise ValueError('split leakage')
    train_targets=choose_targets(train_all,args.n_train_targets,seed); train=load_sample(args.train_h5ad,genes,train_targets,args.max_rows_per_target,seed+10)
    all_targets=sorted({CONTROL,*train_all,*val_all,*test_all}); fmap,sym,feature_meta=build_hash_embedding(args,genes,all_targets,args.train_h5ad)
    batch_order=['__global__']+sorted(set(train.batch_names_per_row)); train_log,target_sum=normalize_log1p(train.x)
    trf,trtid,trbid,_=encode_rows(train,[CONTROL]+train_targets,batch_order,fmap,False)
    baselines=control_baselines(train_log,train.target_names_per_row,train.batch_names_per_row,batch_order)
    torch.manual_seed(seed); np.random.seed(seed)
    base=TargetFeatureCRPM(torch.from_numpy(baselines), int(trf.shape[1]), args.rank, len(train_targets)+1, len(batch_order), False); model=AmplitudeAwareHead(base, args.rank)
    opt=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ds=TensorDataset(torch.from_numpy(train_log),torch.from_numpy(trf),torch.from_numpy(trtid),torch.from_numpy(trbid)); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    gene_weights=torch.from_numpy(train_delta_weights(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order))
    losses=[]; steps=0
    while steps<args.max_steps:
        for y,feat,tid,bid in loader:
            opt.zero_grad(set_to_none=True); pred=model(feat,bid,tid,allow_id=False); loss=torch.mean((pred-y)**2 * gene_weights); loss.backward(); opt.step(); losses.append(float(loss.detach())); steps+=1
            if steps>=args.max_steps: break
    emb_mat,mapped=load_embedding_matrix(genes,sym,str(args.gene_embedding_npz)); _,_,target_delta=train_direction_stats(train_log,train,genes,batch_order)
    self_vals=[]; gi={g:i for i,g in enumerate(genes)}
    for t,d in target_delta.items():
        if t in gi: self_vals.append(float(d[gi[t]]))
    self_effect=float(np.median(self_vals)) if self_vals else 0.0
    return {'genes':genes,'train_targets':train_targets,'fmap':fmap,'batch_order':batch_order,'baselines':baselines,'model':model,'target_delta':target_delta,'emb_mat':emb_mat,'self_effect':self_effect,'feature_meta':feature_meta,'target_sum':target_sum,'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,'optimizer_steps':steps,'embedding_gene_panel_coverage':float(mapped.mean())}


def chunk_predictions(state, obs_schema, chunk_size):
    genes=state['genes']; gi={g:i for i,g in enumerate(genes)}; batch_index={b:i for i,b in enumerate(state['batch_order'])}; global_batch=batch_index['__global__']
    model=state['model']; model.eval()
    with torch.no_grad():
        for start in range(0, obs_schema['shape'][0], chunk_size):
            end=min(start+chunk_size, obs_schema['shape'][0]); targets=obs_schema['target_gene'][start:end]; batches=obs_schema['batch'][start:end]
            feats=np.asarray([state['fmap'][t] for t in targets],dtype=np.float32); bids=np.asarray([batch_index.get(b,global_batch) for b in batches],dtype=np.int64)
            pred=model(torch.from_numpy(feats), torch.from_numpy(bids), None, allow_id=False).numpy().astype(np.float32)
            prior=build_self_neighbor_prior_rows(list(targets), genes, state)
            yield start,end,apply_candidate_f_prior(pred, prior)


def validate_prediction_schema(pred_h5ad: Path, ref_h5ad: Path, condition_csv: Path, gene_names: Path, expected_targets: int) -> dict[str, Any]:
    genes=read_gene_names(gene_names); cond=read_condition_targets(condition_csv)
    with h5py.File(pred_h5ad,'r') as p, h5py.File(ref_h5ad,'r') as r:
        p_obs=_h5ad_index(p['obs']); r_obs=_h5ad_index(r['obs']); p_var=_h5ad_index(p['var']); r_var=_h5ad_index(r['var']); p_t=set(_h5ad_column(p['obs'],'target_gene')); r_t=set(_h5ad_column(r['obs'],'target_gene'))
        sample=p['X'][:min(64,p['X'].shape[0])]
        gates={'gene_count':len(genes)==p['X'].shape[1]==r['X'].attrs['shape'][1], 'gene_order':p_var==r_var==genes, 'obs_count':p['X'].shape[0]==r['X'].attrs['shape'][0], 'obs_order':p_obs==r_obs, 'target_count':len(cond)==expected_targets, 'prediction_targets':p_t==cond|{CONTROL}, 'reference_targets':r_t==cond|{CONTROL}, 'nonnegative_sample':float(sample.min())>=0, 'finite_sample':bool(np.isfinite(sample).all())}
    return {'status':'ready' if all(gates.values()) else 'blocked','gates':gates,'expected_targets':expected_targets,'prediction_h5ad':str(pred_h5ad),'reference_h5ad':str(ref_h5ad)}


def run(args):
    t0=time.time(); state=train_model(args,args.seed); obs=read_obs_schema(args.reference_h5ad,state['genes'])
    metadata={'protocol_id':PROTOCOL_ID,'candidate_id':CANDIDATE_ID,'configuration':'promoted_amplitude_de_weighted_self_neighbor','training_head':'amplitude_aware_anchored_softplus','training_loss':'train_only_delta_weighted_mse','neighbor_k':FIXED_NEIGHBOR_K,'prior_scale':FIXED_PRIOR_SCALE,'output_parameterization':'anchored_softplus_log1p with train-only self-neighbor prior applied in softplus-logit space','output_scale':'log1p_normalized_expression','uses_test_expression':False,'official_score_claim':False,'seed':args.seed,'rank':args.rank,'max_steps':args.max_steps,'max_rows_per_target':args.max_rows_per_target,**{k:state[k] for k in ['feature_meta','target_sum','loss_first','loss_last','optimizer_steps','embedding_gene_panel_coverage']}}
    write_info=write_h5ad(args.output_h5ad,obs,state['genes'],chunk_predictions(state,obs,args.chunk_size),metadata,args.chunk_size)
    contract=validate_prediction_schema(args.output_h5ad,args.reference_h5ad,args.condition_csv,args.gene_names,args.expected_targets)
    result={**metadata,**write_info,'contract':contract,'runtime_seconds':time.time()-t0,'status':'ready' if contract['status']=='ready' and write_info['prediction_nonnegative'] and write_info['prediction_finite'] else 'blocked'}
    args.manifest.parent.mkdir(parents=True,exist_ok=True); args.manifest.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n')
    return result


def parse_args():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-h5ad',type=Path,default=DEFAULT_ASSETS/'train/adata_Training.h5ad'); ap.add_argument('--reference-h5ad',type=Path,required=True); ap.add_argument('--condition-csv',type=Path,required=True); ap.add_argument('--train-targets',type=Path,default=DEFAULT_ASSETS/'train/pert_counts_Training.csv'); ap.add_argument('--validation-targets',type=Path,default=DEFAULT_ASSETS/'validation/pert_counts_Validation.csv'); ap.add_argument('--test-targets',type=Path,default=DEFAULT_ASSETS/'test/pert_counts_Test.csv'); ap.add_argument('--gene-names',type=Path,default=DEFAULT_ASSETS/'gene_names.csv'); ap.add_argument('--gene-embedding-npz',type=Path,default=DEFAULT_ROOT/'assets/lingshu_hf_b77f980/gene_embeddings.npz'); ap.add_argument('--output-h5ad',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--expected-targets',type=int,required=True); ap.add_argument('--n-train-targets',type=int,default=150); ap.add_argument('--max-rows-per-target',type=int,default=32); ap.add_argument('--max-steps',type=int,default=2048); ap.add_argument('--rank',type=int,default=24); ap.add_argument('--feature-dim',type=int,default=128); ap.add_argument('--embedding-projection-seed',type=int,default=20260908); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--learning-rate',type=float,default=8e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--chunk-size',type=int,default=128); ap.add_argument('--seed',type=int,default=20260907); return ap.parse_args()

if __name__=='__main__': print(json.dumps(run(parse_args()),indent=2,sort_keys=True,default=str))
