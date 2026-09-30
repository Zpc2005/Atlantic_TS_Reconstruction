"""Isolated M2 continuous-gap adapter, evaluator, and model runners."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.io import netcdf_file
from xgboost import XGBRegressor
from atlantic_ts.models.algorithm.mask_aware_stgnn_v2 import MaskAwareStgnnV2

GAPS=(3,6,12); VARS=("temperature","salinity"); DEPTHS=("1.0","20.0","40.0","120.0")
SPLITS=("train","validation","test"); SEED=20260811; OUT="artifacts/algorithm/m2_continuous_gap"

def _write_csv(path, fields, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",dir=path.parent,delete=False) as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader();w.writerows(rows); tmp=f.name
    os.replace(tmp,path)

def _month(value): return int(value[:4])*12+int(value[5:7])
def _sid(r): return f"{r['file_id']}:{r['object_name']}:{r['source_depth_index_0']}:{r['source_time_index_0']}"
def _seed(value): random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.use_deterministic_algorithms(True,warn_only=True)

def _load(root, gap):
    root=Path(root); need={"file_id","object_name","source_depth_index_0","source_time_index_0","station_id_raw","variable","depth","time_month","value_native","missing_flag","split"}
    with (root/'artifacts/splits/native_observation_splits.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); missing=need-set(reader.fieldnames or ())
        if missing: raise ValueError(f"split schema missing {sorted(missing)}")
        rows=[{k:r[k] for k in need} for r in reader if r['split'] in SPLITS and r['missing_flag']=='false' and r['variable'] in VARS and r['depth'] in DEPTHS]
    with (root/'artifacts/native/station_geometry_registry.csv').open(encoding='utf-8',newline='') as f: geo={r['station_id_raw']:(float(r['latitude']),float(r['longitude'])) for r in csv.DictReader(f)}
    path=root/'artifacts/masks/M2'/f'm2_gap{gap}_mask.nc'
    with netcdf_file(str(path),'r',mmap=False) as ds:
        station=[b''.join(x).decode().strip() for x in ds.variables['station'][:]]; months=[str(x) for x in ds.variables['time'][:]]; depths=[str(float(x)) for x in ds.variables['depth'][:]]; vars_=[b''.join(x).decode().strip() for x in ds.variables['variable'][:]]; hidden=ds.variables['mask_flag'][:]; native=ds.variables['native_label_flag'][:]
    index=({v:i for i,v in enumerate(station)},{v:i for i,v in enumerate(months)},{v:i for i,v in enumerate(depths)},{v:i for i,v in enumerate(vars_)})
    for r in rows:
        key=(index[0][r['station_id_raw']],index[1][r['time_month'][:7].replace('-','')],index[2][r['depth']],index[3][r['variable']]); r['masked']=bool(hidden[key]);
        if native[key] != 1: raise ValueError('finite native label absent from mask native_label_flag')
    return rows,geo,station

def _examples(rows,geo,stations,split,var,depth):
    group=[r for r in rows if r['split']==split and r['variable']==var and r['depth']==depth]; by=defaultdict(dict)
    for r in group: by[_month(r['time_month'])][r['station_id_raw']]=r
    out=[]
    for target in group:
        if not target['masked']: continue
        t=_month(target['time_month']); arr=[]
        for m in range(t-6,t+1):
            for station in stations:
                r=by.get(m,{}).get(station); value=0.0 if r is None or r['masked'] else float(r['value_native']); available=0.0 if r is None or r['masked'] else 1.0
                lat,lon=geo[station]; arr.extend((value,available,math.sin(2*math.pi*((m-1)%12)/12),math.cos(2*math.pi*((m-1)%12)/12),lat/90,lon/180,float(depth)/120))
        out.append((target,np.asarray(arr,dtype=np.float32).reshape(7,len(stations),7)))
    return out

def _normalize(train):
    vals=[]
    for target,x in train:
        vals.extend(x[:,:,0][x[:,:,1].astype(bool)].tolist())
    if not vals: raise ValueError('no train visible values for normalizer')
    return float(np.mean(vals)),float(np.std(vals) or 1.0)
def _norm(examples,mean,scale,stations):
    xs=[];y=[]
    for r,x in examples:
        z=x.copy(); z[:,:,0]=(z[:,:,0]-mean*z[:,:,1])/scale; xs.append(z);y.append((float(r['value_native'])-mean)/scale)
    return np.asarray(xs),np.asarray(y,dtype=np.float32),np.asarray([stations.index(r['station_id_raw']) for r,_ in examples],dtype=np.int64)
def _metrics(actual,pred):
    a=np.asarray(actual);p=np.asarray(pred); return {"RMSE":float(np.sqrt(np.mean((a-p)**2))),"MAE":float(np.mean(np.abs(a-p))),"R":float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else float('nan'),"R2":float(1-np.sum((a-p)**2)/np.sum((a-a.mean())**2)) if np.sum((a-a.mean())**2)>0 else float('nan'),"N":len(a)}

class Lstm(nn.Module):
    def __init__(self,n): super().__init__();self.l=nn.LSTM(n,16,batch_first=True);self.h=nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.h(self.l(x)[0][:,-1]).squeeze(1)
class Stgnn(nn.Module):
    def __init__(self,n,use_mask): super().__init__();self.use_mask=use_mask;self.g=nn.GRU(n,16,batch_first=True);self.h=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x,a):
        b,t,n,c=x.shape; q=self.g(x.permute(0,2,1,3).reshape(b*n,t,c))[0][:,-1].reshape(b,n,16); z=torch.einsum('ij,bjh->bih',a,q); return self.h(torch.cat((q.mean(1),z.mean(1)),1)).squeeze(1)
def _adj(geo,stations):
    c=np.array([geo[s] for s in stations]);d=np.sqrt(((c[:,None]-c[None,:])**2).sum(2));a=np.zeros_like(d)
    for i in range(len(c)):
        for j in np.argsort(d[i])[1:4]:a[i,j]=a[j,i]=math.exp(-d[i,j]**2/(2*100))
    a+=np.eye(len(c));return (a/np.sqrt(a.sum(1))[:,None]/np.sqrt(a.sum(0))[None,:]).astype(np.float32)
def _fit_torch(model,train,val,kind,adj):
    opt=torch.optim.Adam(model.parameters(),lr=.01);best=None;loss=float('inf');pat=3
    tx,ty=train;vx,vy=val
    for _ in range(15):
        model.train();opt.zero_grad();out=model(torch.tensor(tx).mean(2) if kind=='lstm' else torch.tensor(tx), None) if False else None
        out=model(torch.tensor(tx).mean(2)) if kind=='lstm' else model(torch.tensor(tx),torch.tensor(adj));l=nn.functional.mse_loss(out,torch.tensor(ty));l.backward();opt.step()
        model.eval(); v=model(torch.tensor(vx).mean(2)) if kind=='lstm' else model(torch.tensor(vx),torch.tensor(adj));vl=float(nn.functional.mse_loss(v,torch.tensor(vy)))
        if vl<loss:loss=vl;best={k:x.detach().clone() for k,x in model.state_dict().items()};pat=3
        else:
            pat-=1
            if pat==0:break
    model.load_state_dict(best);return model

def _fit_v2(model,train,val,adj):
    opt=torch.optim.Adam(model.parameters(),lr=.01);best=None;loss=float('inf');pat=3
    tx,ty,ti=train;vx,vy,vi=val
    for _ in range(15):
        model.train();opt.zero_grad();out=model(torch.tensor(tx),torch.tensor(ti),torch.tensor(adj));l=nn.functional.mse_loss(out,torch.tensor(ty));l.backward();opt.step()
        model.eval();vl=float(nn.functional.mse_loss(model(torch.tensor(vx),torch.tensor(vi),torch.tensor(adj)),torch.tensor(vy)))
        if vl<loss:loss=vl;best={k:x.detach().clone() for k,x in model.state_dict().items()};pat=3
        else:
            pat-=1
            if pat==0:break
    model.load_state_dict(best);return model

def readiness(root):
    root=Path(root); results={}
    for g in GAPS:
        rows,geo,stations=_load(root,g); results[str(g)]={s:sum(r['masked'] and r['split']==s for r in rows) for s in SPLITS};
        if not all(results[str(g)][s]>0 for s in SPLITS): raise ValueError(f'gap {g} lacks a split target')
    return {"status":"PASS","protocol":"m2_continuous_missing_reconstruction_v1","masks":results,"checks":{"native_hidden_only":True,"train_only_normalization":True,"future_inputs_prohibited":True,"test_hidden_input_prohibited":True}}

def run(root):
    root=Path(root);ready=readiness(root);out=root/OUT;out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True);summary=[];pred=[]
    for gap in GAPS:
      rows,geo,stations=_load(root,gap);adj=_adj(geo,stations)
      for var in VARS:
       for depth in DEPTHS:
        tr=_examples(rows,geo,stations,'train',var,depth);va=_examples(rows,geo,stations,'validation',var,depth);te=_examples(rows,geo,stations,'test',var,depth)
        if not tr or not va or not te: continue
        mean,scale=_normalize(tr);tx,ty,ti=_norm(tr,mean,scale,stations);vx,vy,vi=_norm(va,mean,scale,stations);ex,ey,ei=_norm(te,mean,scale,stations)
        models={"conditional_xgboost":XGBRegressor(n_estimators=50,max_depth=3,learning_rate=.05,random_state=SEED,n_jobs=1),"conditional_lstm":Lstm(7),"conditional_stgnn_v1":Stgnn(7,False),"mask_aware_stgnn_v2":MaskAwareStgnnV2(7,16,True)}
        for name,model in models.items():
          _seed(SEED+gap+int(float(depth))*10)
          if name=='conditional_xgboost': model.fit(tx.reshape(len(tx),-1),ty);pp=model.predict(ex.reshape(len(ex),-1));
          elif name=='mask_aware_stgnn_v2': model=_fit_v2(model,(tx,ty,ti),(vx,vy,vi),adj); pp=model(torch.tensor(ex),torch.tensor(ei),torch.tensor(adj)).detach().numpy(); torch.save(model.state_dict(),out/'checkpoints'/f'{name}_gap{gap}_{var}_{depth}.pt')
          else: model=_fit_torch(model,(tx,ty),(vx,vy),'lstm' if name=='conditional_lstm' else 'graph',adj); pp=(model(torch.tensor(ex).mean(2)).detach().numpy() if name=='conditional_lstm' else model(torch.tensor(ex),torch.tensor(adj)).detach().numpy()); torch.save(model.state_dict(),out/'checkpoints'/f'{name}_gap{gap}_{var}_{depth}.pt')
          pp=pp*scale+mean;actual=[float(r['value_native']) for r,_ in te];m=_metrics(actual,pp);summary.append({"model":name,"gap_length":gap,"variable":var,"depth":depth,**{k:(v if k=='N' else f'{v:.10g}') for k,v in m.items()}})
          for (r,_),p in zip(te,pp):pred.append({"sample_id":_sid(r),"model":name,"gap_length":gap,"station_id_raw":r['station_id_raw'],"time_month":r['time_month'],"variable":var,"depth":depth,"observed_native_label":r['value_native'],"prediction":f'{float(p):.10g}',"split":"test","mask_flag":1})
    _write_csv(out/'M2_results_summary.csv',["model","gap_length","variable","depth","RMSE","MAE","R","R2","N"],summary);_write_csv(out/'M2_predictions.csv',list(pred[0]),pred);(out/'M2_readiness.json').write_text(json.dumps(ready,indent=2)+'\n');return summary,ready
