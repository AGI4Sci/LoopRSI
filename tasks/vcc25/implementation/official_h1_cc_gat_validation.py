from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from crpm.cc_gat import CCGAT,_topk_overlap
from official_h1_autonomous_research_loop import delta_diagnostics
from official_h1_cold_start_probe import CONTROL,choose_targets,control_baselines,encode_rows,load_sample
from official_h1_contract import read_condition_targets,read_gene_names
from official_h1_nonnegative_probe import normalize_log1p
ROOT=Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1"); ASSETS=ROOT/"assets/official_2025"; REF=.608694

def sig(x,names,base,bids,order):
 d=x-base[bids]; a=np.asarray(names,dtype=object); return np.stack([d[a==t].mean(0) for t in order]).astype(np.float32)
def run(a):
 start=time.time(); genes=read_gene_names(a.gene_names); gi={g:i for i,g in enumerate(genes)}
 ta=read_condition_targets(a.train_targets); va=read_condition_targets(a.validation_targets); xa=read_condition_targets(a.test_targets)
 if ta&va or ta&xa or va&xa: raise ValueError("official target splits overlap")
 missing=sorted((ta|va|xa)-gi.keys())
 if missing: raise ValueError(f"official targets absent from full gene panel: {missing[:5]}")
 trt=choose_targets(ta,a.n_train_targets,a.seed); vat=choose_targets(va,a.n_validation_targets,a.seed+1)
 tr=load_sample(a.train_h5ad,genes,trt,a.max_rows_per_target,a.seed+10); val=load_sample(a.validation_h5ad,genes,vat,a.max_rows_per_target,a.seed+20)
 tx,total=normalize_log1p(tr.x); vx,_=normalize_log1p(val.x,total); batches=["__global__"]+sorted(set(tr.batch_names_per_row)); fmap={x:np.zeros(1,dtype=np.float32) for x in {CONTROL,*trt,*vat}}
 _,_,tb,_=encode_rows(tr,[CONTROL]+trt,batches,fmap,False); _,_,vb,unknown=encode_rows(val,[CONTROL]+trt,batches,fmap,True); base=control_baselines(tx,tr.target_names_per_row,tr.batch_names_per_row,batches)
 y=sig(tx,tr.target_names_per_row,base,tb,trt); tn=np.asarray([gi[x] for x in trt]); vn=np.asarray([gi[x] for x in vat])
 model=CCGAT.fit(y,tn,genes,a.context_dim,a.gat_heads,a.gat_layers,a.grn_neighbors,a.grn_threshold,a.ridge,a.seed)
 pred,signal,events=model.predict_delta(vn); order=np.random.default_rng(a.seed+1).permutation(len(vn)); shuffled,shuf_signal,_=model.predict_delta(vn,order)
 lookup={x:i for i,x in enumerate(vat)}; rows=lambda z:np.stack([np.zeros(len(genes),dtype=np.float32) if x==CONTROL else z[lookup[x]] for x in val.target_names_per_row])
 baseline=base[vb]; pm=delta_diagnostics(vx,baseline+rows(pred),baseline,val.target_names_per_row); sm=delta_diagnostics(vx,baseline+rows(shuffled),baseline,val.target_names_per_row)
 truth=np.stack([(vx[np.asarray(val.target_names_per_row,dtype=object)==x]-baseline[np.asarray(val.target_names_per_row,dtype=object)==x]).mean(0) for x in vat])
 score=float(pm['delta_correlation']); control=float(sm['delta_correlation']); gates={'full_gene_target_coverage':True,'prediction_finite':bool(np.isfinite(pred).all()),'context_conditioning_beats_shuffle':score>control,'beats_promotion_reference':score>REF,'test_expression_used':False}
 ok=all(v for k,v in gates.items() if k!='test_expression_used') and not gates['test_expression_used']
 out={'protocol_id':'vcc-h1-cc-gat-validation-v1','candidate_variant':'candidate_cc_gat','status':'promoted' if ok else 'rejected','data_contract':{'train_targets':len(trt),'validation_targets':len(vat),'test_targets_known_by_identity_only':len(xa),'genes':len(genes),'max_rows_per_target':a.max_rows_per_target,'test_expression_read':False,'unknown_validation_batches':unknown},'configuration':{'seed':a.seed,'context_dim':a.context_dim,'gat_heads':a.gat_heads,'gat_layers':a.gat_layers,'grn_neighbors':a.grn_neighbors,'grn_threshold':a.grn_threshold,'ridge':a.ridge},'validation_metrics':pm,'shuffled_context_metrics':sm,'diagnostics':{'context_vector_norm_positive':float(np.min(np.linalg.norm(model.contexts,axis=1))),'attention_conditioning_events_nonzero':events,'propagated_signal_l2_variance_across_targets':float(np.var(np.linalg.norm(signal,axis=1))),'shuffled_context_performance_delta':score-control,'pre_decoder_de_overlap_at_100':_topk_overlap(truth,signal),'shuffled_pre_decoder_de_overlap_at_100':_topk_overlap(truth,shuf_signal)},'promotion_reference':REF,'gates':gates,'runtime_seconds':time.time()-start}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); return out
def args():
 p=argparse.ArgumentParser()
 for n,d in [('train_h5ad',ASSETS/'train/adata_Training.h5ad'),('validation_h5ad',ASSETS/'validation/adata_Validation.h5ad'),('train_targets',ASSETS/'train/pert_counts_Training.csv'),('validation_targets',ASSETS/'validation/pert_counts_Validation.csv'),('test_targets',ASSETS/'test/pert_counts_Test.csv'),('gene_names',ASSETS/'gene_names.csv')]: p.add_argument('--'+n.replace('_','-'),type=Path,default=d)
 p.add_argument('--output',type=Path,required=True); p.add_argument('--n-train-targets',type=int,default=150); p.add_argument('--n-validation-targets',type=int,default=50); p.add_argument('--max-rows-per-target',type=int,default=32); p.add_argument('--context-dim',type=int,default=128); p.add_argument('--gat-heads',type=int,default=4); p.add_argument('--gat-layers',type=int,default=2); p.add_argument('--grn-neighbors',type=int,default=32); p.add_argument('--grn-threshold',type=float,default=.3); p.add_argument('--ridge',type=float,default=.001); p.add_argument('--seed',type=int,default=20260913); return p.parse_args()
if __name__=='__main__':
 x=run(args()); print(json.dumps(x,indent=2,sort_keys=True)); raise SystemExit(0 if x['status']=='promoted' else 3)
