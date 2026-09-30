"""Prepare a derived plotting table and non-final Figure 8 layout preview."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

root=Path(__file__).resolve().parents[1]; base=root/'artifacts/algorithm/m2_continuous_gap'
with (base/'M2_results_summary.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
fields=['model','variable','depth','gap_length','RMSE','MAE','R','R2','N']
with (base/'figure8_plotting_dataset.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
with (base/'stgnn_v2_120m_salinity_gap12_review.csv').open(encoding='utf-8',newline='') as f: anomaly=[r for r in csv.DictReader(f) if r['station_id_raw']!='ALL']
models=['conditional_xgboost','conditional_lstm','conditional_stgnn_v1','mask_aware_stgnn_v2']; labels=['XGBoost','LSTM','STGNN v1','Mask-aware v2']; colors=['#4c78a8','#f58518','#54a24b','#e45756']; depths=['1.0','20.0','40.0','120.0']; gaps=['3','6','12']
fig=plt.figure(figsize=(15,10),constrained_layout=True); gs=fig.add_gridspec(2,2)
def rmse_panel(spec,var,letter):
 sub=spec.subgridspec(2,2,wspace=.15,hspace=.25)
 for i,depth in enumerate(depths):
  ax=fig.add_subplot(sub[i//2,i%2]);
  for model,label,color in zip(models,labels,colors):
   y=[float(next(r['RMSE'] for r in rows if r['model']==model and r['variable']==var and r['depth']==depth and r['gap_length']==gap)) for gap in gaps]
   ax.plot(gaps,y,marker='o',color=color,label=label)
  n=[next(r['N'] for r in rows if r['model']==models[0] and r['variable']==var and r['depth']==depth and r['gap_length']==gap) for gap in gaps]
  ax.set_title(f'{letter if i==0 else ""} {var.title()}, {depth} m',loc='left',fontsize=10);ax.set_xlabel('Gap length (months)');ax.set_ylabel('RMSE');ax.grid(alpha=.25);ax.text(.02,.04,'N='+ '/'.join(n),transform=ax.transAxes,fontsize=7)
  if i==0:ax.legend(fontsize=7,frameon=False)
rmse_panel(gs[0,0],'temperature','(a)');rmse_panel(gs[0,1],'salinity','(b)')
ax=fig.add_subplot(gs[1,0])
for model,label,color in zip(models,labels,colors):
 for depth,marker in zip(depths,['o','s','^','D']):
  x=np.arange(3)+({'conditional_xgboost':-.12,'conditional_lstm':-.04,'conditional_stgnn_v1':.04,'mask_aware_stgnn_v2':.12}[model]); y=[]
  for gap in gaps:
   sub=[float(r['R']) for r in rows if r['model']==model and r['depth']==depth and r['gap_length']==gap]
   y.append(np.mean(sub))
  ax.plot(x,y,color=color,marker=marker,alpha=.85)
ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['3','6','12']);ax.set_xlabel('Gap length (months)');ax.set_ylabel('Mean correlation R across variables');ax.set_title('(c) Correlation by model and depth');ax.grid(alpha=.25);ax.text(.02,.02,'Marker: 1 / 20 / 40 / 120 m\nLines are descriptive; no monotonicity claim.',transform=ax.transAxes,fontsize=8)
ax.legend([Line2D([0],[0],color=c,label=l) for c,l in zip(colors,labels)]+[Line2D([0],[0],color='black',marker=m,linestyle='',label=f'{d} m') for d,m in zip(depths,['o','s','^','D'])],labels+ [f'{d} m' for d in depths],fontsize=7,ncol=2,frameon=False,loc='upper left')
ax=fig.add_subplot(gs[1,1]); names=[r['station_id_raw'] for r in anomaly]; x=np.arange(len(names));ax.bar(x-.2,[float(r['observed_variance']) for r in anomaly],.4,label='Observed variance',color='#4c78a8');ax.bar(x+.2,[float(r['prediction_variance']) for r in anomaly],.4,label='Predicted variance',color='#e45756');ax.set_xticks(x,names,rotation=65,ha='right',fontsize=7);ax.set_ylabel('Variance');ax.set_title('(d) Salinity, 120 m, 12-month gap: v2 failure case');ax.legend(fontsize=8,frameon=False);ax.text(.02,.92,'Near-constant predictions are retained.',transform=ax.transAxes,fontsize=8,va='top');ax.grid(axis='y',alpha=.25)
fig.suptitle('Figure 8 layout preview — continuous-gap reconstruction; non-final',fontsize=15)
fig.savefig(base/'Figure8_manuscript_layout_preview.png',dpi=180)
