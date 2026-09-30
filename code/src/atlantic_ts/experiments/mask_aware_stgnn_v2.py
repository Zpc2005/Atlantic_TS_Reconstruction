"""Train v2 mask-aware STGNN variants under the frozen conditional protocol."""
from __future__ import annotations
import csv, hashlib, io, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from atlantic_ts.experiments.conditional_baselines import GROUPS, OUT as V1OUT, _geometry, _month, _samples, _train_norm, _write, _set_seed, station_axis, _background
from atlantic_ts.experiments.conditional_reconstruction import TEST_MASK_ID, sample_id, verify_frozen_m2_v1_content
from atlantic_ts.experiments.data import load_experiment_records
from atlantic_ts.experiments.metrics import rmse, r2, anomaly_acc
from atlantic_ts.models.deep.stgnn import build_normalized_knn_adjacency
from atlantic_ts.models.algorithm.mask_aware_stgnn_v2 import MaskAwareStgnnV2
from atlantic_ts.io.atomic import atomic_write_text

OUT=Path('artifacts/algorithm/mask_aware_stgnn_v2'); PROTO='conditional_causal_reconstruction_v1'; CAP=6
RC=('model','ablation','variable','depth','scenario','protocol_family','RMSE','R2','ACC','n_test','selected_k','selected_hyperparameters','seed','training_runtime_seconds')
PC=('sample_id','station_id_raw','time_month','source_file','variable','depth','observed','prediction','split','mask_id','scenario','protocol_family','model','ablation')

def _gap(av):
 out=np.zeros_like(av,dtype=np.float32)
 for t in range(av.shape[0]): out[t]=0 if t==0 else np.minimum(CAP,out[t-1]+1); out[t][av[t].astype(bool)]=0
 return out/CAP
def _features(items, ablation):
 values=np.asarray([x[1] for x in items],dtype=np.float32); av=np.asarray([x[2] for x in items],dtype=np.float32); gaps=np.asarray([_gap(x[2]) for x in items],dtype=np.float32)
 months=np.asarray([int(x[0]['month_raw']) for x in items],dtype=np.float32); angles=2*np.pi*months/12
 sin=np.repeat(np.sin(angles)[:,None,None],7,axis=1); sin=np.repeat(sin,20,axis=2); cos=np.repeat(np.cos(angles)[:,None,None],7,axis=1); cos=np.repeat(cos,20,axis=2)
 parts=[values, sin, cos] if ablation=='no_missingness' else [values,av,gaps,sin,cos]
 return np.stack(parts,axis=-1),np.asarray([x[3]['target_station_index'] for x in items],dtype=np.int64)
def _fit(train,val,ablation,adj,seed):
 _set_seed(seed); x,i=_features(train,ablation); y=np.asarray([float(x[0]['value_native']) for x in train],dtype=np.float32)
 # targets arrive already normalized by caller.
 model=MaskAwareStgnnV2(x.shape[-1],8,ablation!='no_graph'); opt=torch.optim.Adam(model.parameters(),lr=.01); loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(i),torch.from_numpy(y)),128,shuffle=True,generator=torch.Generator().manual_seed(seed)); vx,vi=_features(val,ablation); vy=np.asarray([float(x[0]['value_native']) for x in val],dtype=np.float32); graph=None if ablation=='no_graph' else torch.from_numpy(adj); best=(1e99,None); hist=[]; wait=3
 for epoch in range(1,16):
  model.train(); ls=[]
  for a,b,c in loader: opt.zero_grad(); p=model(a,b,graph); loss=nn.functional.mse_loss(p,c); loss.backward();opt.step();ls.append(float(loss.detach()))
  with torch.no_grad(): vl=float(nn.functional.mse_loss(model(torch.from_numpy(vx),torch.from_numpy(vi),graph),torch.from_numpy(vy)))
  hist.append((epoch,float(np.mean(ls)),vl))
  if vl<best[0]:best=(vl,{k:v.clone() for k,v in model.state_dict().items()});wait=3
  else:
   wait-=1
   if not wait:break
 model.load_state_dict(best[1]);return model,hist
def run(root:Path):
 root=Path(root); rec=load_experiment_records(root/'artifacts/splits/native_observation_splits.csv'); geo=_geometry(root); stations,index,orderhash=station_axis(geo); coords=np.asarray([geo[s] for s in stations]); rows=[];pred=[];hist=[];start=time.perf_counter(); variants=(('full',True),('no_missingness',True),('no_graph',False))
 for gi,(var,dep) in enumerate(GROUPS):
  group=[r for r in rec if r['variable']==var and r['depth']==dep]; mean,scale=_train_norm(group); tr=_samples(group,'train',stations,index,mean,scale); va=_samples(group,'validation',stations,index,mean,scale); te=_samples(group,'test',stations,index,mean,scale)
  for vi,(abl,graphuse) in enumerate(variants):
   runs=[]
   for ci,k in enumerate((3,5)):
    adj=build_normalized_knn_adjacency(coords,k); seed=20260729+gi*30+vi*2+ci
    # convert labels to normalized temporary tuples
    norm=lambda z: [(dict(a, value_native=str((float(a['value_native']) - mean) / scale)), b, c, d) for a, b, c, d in z]
    model,h=_fit(norm(tr),norm(va),abl,adj,seed); runs.append((h[-1][2],k,model,adj,h,seed))
   _,k,model,adj,h,seed=min(runs,key=lambda z:(z[0],z[1])); tx,ti=_features(te,abl); graph=None if abl=='no_graph' else torch.from_numpy(adj)
   with torch.no_grad(): est=(model(torch.from_numpy(tx),torch.from_numpy(ti),graph).numpy()*scale+mean).tolist()
   actual=[float(a['value_native']) for a,*_ in te]; bg=_background(tr,te); name='mask_aware_stgnn_v2_'+abl; params={'hidden_size':8,'learning_rate':.01,'k_candidates':[3,5],'sequence_length':7,'time_gap_cap_months':CAP,'station_order_sha256':orderhash,'adjacency_sha256':hashlib.sha256(adj.tobytes()).hexdigest(),'max_epochs':15,'patience':3}
   rows.append({'model':name,'ablation':abl,'variable':var,'depth':dep,'scenario':'random_mask_20pct','protocol_family':PROTO,'RMSE':f'{rmse(actual,est):.12g}','R2':f'{r2(actual,est):.12g}','ACC':f'{anomaly_acc(actual,est,bg):.12g}','n_test':str(len(te)),'selected_k':str(k),'selected_hyperparameters':json.dumps(params,sort_keys=True,separators=(',',':')),'seed':str(seed),'training_runtime_seconds':''})
   for (a,b,c,d),p in zip(te,est): pred.append({'sample_id':sample_id(a),'station_id_raw':a['station_id_raw'],'time_month':a['time_month'],'source_file':a['source_file'],'variable':var,'depth':dep,'observed':a['value_native'],'prediction':repr(p),'split':'test','mask_id':TEST_MASK_ID,'scenario':'random_mask_20pct','protocol_family':PROTO,'model':name,'ablation':abl})
   hist += [{'model_key':f'{name}:{var}:{dep}','epoch':str(e),'train_loss':str(tl),'validation_loss':str(vl)} for e,tl,vl in h]
 runtime=time.perf_counter()-start
 for r in rows:r['training_runtime_seconds']=f'{runtime:.6f}'
 _write(root/OUT/'mask_aware_stgnn_v2_results.csv',RC,rows);_write(root/OUT/'mask_aware_stgnn_v2_predictions.csv',PC,pred);_write(root/OUT/'mask_aware_stgnn_v2_training_history.csv',('model_key','epoch','train_loss','validation_loss'),hist)
 _write(root/OUT/'mask_aware_stgnn_v2_ablation_comparison.csv',RC,rows)
 integ={'canonical_station_order_hash':orderhash,'mask_id':TEST_MASK_ID,'protocol_family':PROTO,'time_gap_cap_months':CAP,'test_targets_per_ablation':1137,'availability_definition':'1=legal visible native observation; 0=missing or masked','frozen_m2_v1_provenance_files':len(verify_frozen_m2_v1_content(root))};atomic_write_text(root/OUT/'mask_aware_stgnn_v2_input_integrity.json',json.dumps(integ,indent=2)+'\n')
 v1={r['sample_id']:r for r in csv.DictReader((root/V1OUT/'conditional_stgnn_v1_predictions.csv').open())}; paired=[]
 for r in pred:
  if r['ablation']=='full':
   b=v1[r['sample_id']];fe=abs(float(r['observed'])-float(r['prediction']));be=abs(float(r['observed'])-float(b['prediction']));paired.append({'sample_id':r['sample_id'],'variable':r['variable'],'depth':r['depth'],'full_absolute_error':repr(fe),'baseline_absolute_error':repr(be),'error_difference':repr(fe-be),'full_improved':str(fe<be).lower()})
 _write(root/OUT/'mask_aware_stgnn_v2_paired_error_comparison.csv',('sample_id','variable','depth','full_absolute_error','baseline_absolute_error','error_difference','full_improved'),paired)
 report='# Mask-aware STGNN v2\n\n- Full receives value, availability, capped causal time-gap, and month features; no-missingness removes availability/time-gap; no-graph removes graph aggregation.\n- All variants use frozen conditional protocol and 1,137 test targets. Geometry remains encoded metadata, not verified physical coordinates.\n- No comparison to DINEOF as a common protocol ranking is made.\n';atomic_write_text(root/'reports/ALGORITHM_MASK_AWARE_STGNN_V2_REPORT.md',report)
 return {'results':len(rows),'predictions':len(pred),'runtime':runtime}
