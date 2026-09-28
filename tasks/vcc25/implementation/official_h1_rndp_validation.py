from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from crpm.rndp import build_coexpression_graph, degree_matched_random_graph, pagerank_torch
from official_h1_autonomous_research_loop import delta_diagnostics
from official_h1_cold_start_probe import CONTROL, choose_targets, control_baselines, encode_rows, load_sample
from official_h1_contract import read_condition_targets, read_gene_names
from official_h1_nonnegative_probe import normalize_log1p

ROOT=Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1")
ASSETS=ROOT/"assets/official_2025"
PROMOTION_REFERENCE=0.608694

def signatures(x, names, baselines, batches, order):
    d=x-baselines[batches]; a=np.asarray(names,dtype=object)
    return np.stack([d[a==t].mean(0) for t in order]).astype(np.float32)

def fit_predict(y,tr_nodes,q_nodes,ind,w,a):
    td=torch.from_numpy(pagerank_torch(tr_nodes,ind,w,a.pagerank_alpha,a.pagerank_iterations,a.device)).to(a.device)
    qd=torch.from_numpy(pagerank_torch(q_nodes,ind,w,a.pagerank_alpha,a.pagerank_iterations,a.device)).to(a.device)
    yt=torch.from_numpy(y).to(a.device); _,_,vh=torch.linalg.svd(td,full_matrices=False); basis=vh[:min(a.calibration_rank,len(vh))]
    feat=td@basis.T; cal=torch.linalg.lstsq(feat,yt).solution; residual=yt-feat@cal
    sparse=torch.sign(residual)*torch.relu(torch.abs(residual)-a.lasso_alpha)
    unit=lambda x:x/torch.clamp(torch.linalg.vector_norm(x,dim=1,keepdim=True),min=1e-8)
    sim=torch.relu(unit(qd)@unit(td).T); sim=sim/torch.clamp(sim.sum(1,keepdim=True),min=1e-8)
    return ((qd@basis.T)@cal+sim@sparse).cpu().numpy().astype(np.float32)

def run(a):
    t=time.time(); genes=read_gene_names(a.gene_names); gi={g:i for i,g in enumerate(genes)}
    ta=read_condition_targets(a.train_targets); va=read_condition_targets(a.validation_targets); xa=read_condition_targets(a.test_targets)
    if ta&va or ta&xa or va&xa: raise ValueError("official target splits are not disjoint")
    missing=sorted((ta|va|xa)-gi.keys())
    if missing: raise ValueError(f"official targets absent from full gene panel: {missing[:5]}")
    trt=choose_targets(ta,a.n_train_targets,a.seed); vat=choose_targets(va,a.n_validation_targets,a.seed+1)
    tr=load_sample(a.train_h5ad,genes,trt,a.max_rows_per_target,a.seed+10); val=load_sample(a.validation_h5ad,genes,vat,a.max_rows_per_target,a.seed+20)
    tx,total=normalize_log1p(tr.x); vx,_=normalize_log1p(val.x,total); batches=["__global__"]+sorted(set(tr.batch_names_per_row))
    fmap={x:np.zeros(1,dtype=np.float32) for x in {CONTROL,*trt,*vat}}
    _,_,tb,_=encode_rows(tr,[CONTROL]+trt,batches,fmap,False); _,_,vb,unknown=encode_rows(val,[CONTROL]+trt,batches,fmap,True)
    base=control_baselines(tx,tr.target_names_per_row,tr.batch_names_per_row,batches); y=signatures(tx,tr.target_names_per_row,base,tb,trt)
    ind,w=build_coexpression_graph(y,a.coexpression_threshold,a.neighbors,a.graph_block_size)
    tn=np.asarray([gi[x] for x in trt]); vn=np.asarray([gi[x] for x in vat]); pred=fit_predict(y,tn,vn,ind,w,a)
    ri,rw=degree_matched_random_graph(ind,w,a.seed); random=fit_predict(y,tn,vn,ri,rw,a); lookup={x:i for i,x in enumerate(vat)}
    rows=lambda z:np.stack([np.zeros(len(genes),dtype=np.float32) if x==CONTROL else z[lookup[x]] for x in val.target_names_per_row])
    baseline=base[vb]; pm=delta_diagnostics(vx,baseline+rows(pred),baseline,val.target_names_per_row); rm=delta_diagnostics(vx,baseline+rows(random),baseline,val.target_names_per_row)
    score=float(pm["delta_correlation"]); rscore=float(rm["delta_correlation"])
    gates={"full_gene_target_coverage":True,"prediction_finite":bool(np.isfinite(pred).all()),"structured_graph_beats_random_control":score>rscore,"beats_validation_promotion_reference":score>PROMOTION_REFERENCE,"test_expression_used":False}
    ok=all(v for k,v in gates.items() if k!="test_expression_used") and not gates["test_expression_used"]
    out={"protocol_id":"vcc-h1-rndp-validation-v1","candidate_variant":"candidate_rndp","status":"promoted" if ok else "rejected","data_contract":{"train_targets":len(trt),"validation_targets":len(vat),"test_targets_known_by_identity_only":len(xa),"genes":len(genes),"max_rows_per_target":a.max_rows_per_target,"test_expression_read":False,"unknown_validation_batches":unknown},"configuration":{"seed":a.seed,"pagerank_alpha":a.pagerank_alpha,"pagerank_iterations":a.pagerank_iterations,"calibration_rank":a.calibration_rank,"lasso_alpha":a.lasso_alpha,"coexpression_threshold":a.coexpression_threshold,"neighbors":a.neighbors},"validation_metrics":pm,"random_graph_control_metrics":rm,"promotion_reference":PROMOTION_REFERENCE,"gates":gates,"runtime_seconds":time.time()-t}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); return out

def args():
    p=argparse.ArgumentParser()
    for n,d in [("train_h5ad",ASSETS/"train/adata_Training.h5ad"),("validation_h5ad",ASSETS/"validation/adata_Validation.h5ad"),("train_targets",ASSETS/"train/pert_counts_Training.csv"),("validation_targets",ASSETS/"validation/pert_counts_Validation.csv"),("test_targets",ASSETS/"test/pert_counts_Test.csv"),("gene_names",ASSETS/"gene_names.csv")]: p.add_argument("--"+n.replace("_","-"),type=Path,default=d)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--n-train-targets",type=int,default=150); p.add_argument("--n-validation-targets",type=int,default=50); p.add_argument("--max-rows-per-target",type=int,default=32); p.add_argument("--pagerank-alpha",type=float,default=.15); p.add_argument("--pagerank-iterations",type=int,default=30); p.add_argument("--calibration-rank",type=int,default=64); p.add_argument("--lasso-alpha",type=float,default=.001); p.add_argument("--coexpression-threshold",type=float,default=.3); p.add_argument("--neighbors",type=int,default=32); p.add_argument("--graph-block-size",type=int,default=256); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=20260912); return p.parse_args()
if __name__=="__main__":
    x=run(args()); print(json.dumps(x,indent=2,sort_keys=True)); raise SystemExit(0 if x["status"]=="promoted" else 3)
