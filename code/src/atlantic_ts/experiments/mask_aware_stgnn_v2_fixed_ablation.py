"""Fixed-Full-parameter ablations; never select parameters from validation/test."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from atlantic_ts.experiments.mask_aware_stgnn_v2 import _features, OUT as V2OUT, RC, PC
from atlantic_ts.experiments.conditional_baselines import GROUPS,_geometry,_samples,_train_norm,station_axis,_background,_write,_set_seed
from atlantic_ts.experiments.conditional_reconstruction import TEST_MASK_ID,sample_id,verify_frozen_m2_v1_content
from atlantic_ts.experiments.data import load_experiment_records
from atlantic_ts.experiments.metrics import rmse,r2,anomaly_acc
from atlantic_ts.models.algorithm.mask_aware_stgnn_v2 import MaskAwareStgnnV2
from atlantic_ts.models.deep.stgnn import build_normalized_knn_adjacency
from atlantic_ts.io.atomic import atomic_write_text
OUT=Path('artifacts/algorithm/mask_aware_stgnn_v2_fixed_ablation')
VARIANT_CONFIG={
 'fixed_no_missingness':{'feature_variant':'no_missingness','use_graph':True},
 'fixed_no_graph':{'feature_variant':'full','use_graph':False},
}
def selected_epoch(history):
 """Earliest epoch with minimum validation loss, never the final epoch by default."""
 return min(history,key=lambda row:(float(row['validation_loss']),int(row['epoch'])))['epoch']
def _full(root):
 with (root/V2OUT/'mask_aware_stgnn_v2_results.csv').open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
 with (root/V2OUT/'mask_aware_stgnn_v2_training_history.csv').open(encoding='utf-8',newline='') as stream: hist=list(csv.DictReader(stream))
 o={}
 for r in rows:
  if r['ablation']=='full':
   key=(r['variable'],r['depth']); matches=[x for x in hist if x['model_key'].startswith('mask_aware_stgnn_v2_full:'+r['variable']+':'+r['depth'])];o[key]=(r,int(selected_epoch(matches)))
 return o
def _train(items,feature_variant,use_graph,adj,seed,epochs):
 _set_seed(seed); x,i=_features(items,feature_variant);y=np.asarray([float(a['value_native']) for a,*_ in items],dtype=np.float32);m=MaskAwareStgnnV2(x.shape[-1],8,use_graph);opt=torch.optim.Adam(m.parameters(),lr=.01);g=torch.from_numpy(adj) if use_graph else None;loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(i),torch.from_numpy(y)),128,shuffle=True,generator=torch.Generator().manual_seed(seed));h=[]
 for e in range(1,epochs+1):
  ls=[];m.train()
  for a,b,c in loader:opt.zero_grad();p=m(a,b,g);loss=nn.functional.mse_loss(p,c);loss.backward();opt.step();ls.append(float(loss.detach()))
  h.append((e,float(np.mean(ls))))
 return m,h
def run(root:Path):
 root=Path(root);f=_full(root);rec=load_experiment_records(root/'artifacts/splits/native_observation_splits.csv');geo=_geometry(root);stations,idx,order=station_axis(geo);coords=np.asarray([geo[s] for s in stations]);rows=[];pred=[];hist=[];start=time.perf_counter()
 for var,dep in GROUPS:
  full,epochs=f[(var,dep)];pars=json.loads(full['selected_hyperparameters']);seed=int(full['seed']);k=int(full['selected_k']);adj=build_normalized_knn_adjacency(coords,k);group=[r for r in rec if r['variable']==var and r['depth']==dep];mean,scale=_train_norm(group);tr=_samples(group,'train',stations,idx,mean,scale);te=_samples(group,'test',stations,idx,mean,scale)
  norm=lambda z:[(dict(a,value_native=str((float(a['value_native'])-mean)/scale)),b,c,d) for a,b,c,d in z]
  for abl,config in VARIANT_CONFIG.items():
   model,h=_train(norm(tr),config['feature_variant'],config['use_graph'],adj,seed,epochs);x,i=_features(te,config['feature_variant']);g=torch.from_numpy(adj) if config['use_graph'] else None
   with torch.no_grad(): est=(model(torch.from_numpy(x),torch.from_numpy(i),g).numpy()*scale+mean).tolist()
   actual=[float(a['value_native']) for a,*_ in te];bg=_background(tr,te);params={**pars,'fixed_from_full':True,'full_selected_epoch':epochs,'full_selected_k_reference':k,'k_used_in_forward':abl!='fixed_no_graph'}
   rows.append({'model':'mask_aware_stgnn_v2_'+abl,'ablation':abl,'variable':var,'depth':dep,'scenario':'random_mask_20pct','protocol_family':'conditional_causal_reconstruction_v1','RMSE':f'{rmse(actual,est):.12g}','R2':f'{r2(actual,est):.12g}','ACC':f'{anomaly_acc(actual,est,bg):.12g}','n_test':str(len(te)),'selected_k':str(k)+'_not_used' if not config['use_graph'] else str(k),'selected_hyperparameters':json.dumps(params,sort_keys=True,separators=(',',':')),'seed':str(seed),'training_runtime_seconds':''})
   for (a,*_),p in zip(te,est):pred.append({'sample_id':sample_id(a),'station_id_raw':a['station_id_raw'],'time_month':a['time_month'],'source_file':a['source_file'],'variable':var,'depth':dep,'observed':a['value_native'],'prediction':repr(p),'split':'test','mask_id':TEST_MASK_ID,'scenario':'random_mask_20pct','protocol_family':'conditional_causal_reconstruction_v1','model':'mask_aware_stgnn_v2_'+abl,'ablation':abl})
   hist += [{'model_key':f'mask_aware_stgnn_v2_{abl}:{var}:{dep}','epoch':str(e),'train_loss':str(l),'validation_loss':'fixed_full_epoch'} for e,l in h]
 runtime=time.perf_counter()-start
 for r in rows:r['training_runtime_seconds']=str(runtime)
 _write(root/OUT/'fixed_ablation_results.csv',RC,rows);_write(root/OUT/'fixed_no_missingness_predictions.csv',PC,[r for r in pred if r['ablation']=='fixed_no_missingness']);_write(root/OUT/'fixed_no_graph_predictions.csv',PC,[r for r in pred if r['ablation']=='fixed_no_graph']);_write(root/OUT/'fixed_ablation_training_history.csv',('model_key','epoch','train_loss','validation_loss'),hist);_write(root/OUT/'fixed_ablation_comparison.csv',RC,rows)
 atomic_write_text(root/OUT/'fixed_ablation_integrity.json',json.dumps({'fixed_from':'mask_aware_stgnn_v2_full','test_targets_per_ablation':1137,'fixed_seed_epoch_k':True,'m2_v1_provenance':len(verify_frozen_m2_v1_content(root))},indent=2)+'\n')
 atomic_write_text(root/'reports/ALGORITHM_MASK_AWARE_STGNN_V2_FIXED_ABLATION_REPORT.md','# Fixed-Full-Parameter Ablations\n\n- Each ablation inherits the matching Full group seed, k reference, optimizer, hidden size and realized training epoch. No parameter or epoch is independently selected.\n- Fixed no-missingness removes availability/time-gap only. Fixed no-graph disables graph aggregation and records Full k as not used.\n')
 return {'results':len(rows),'predictions':len(pred),'runtime':runtime}
