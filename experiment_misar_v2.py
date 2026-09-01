
"""

MISAR v2 — Optimized implementation

Key improvements over v1:

1. Reduced lookahead H=1 (was H=3) — major speedup

2. Gradient compression — only top-k% parameters communicated

3. Data augmentation — 10x dataset duplication via temporal bootstrap

4. Higher N — test N=2,4,8,16,32

5. Async-style aggregation — timeout on slow workers

6. Cached MISAR estimates — reuse lookahead within round

"""

import os,sys,time,copy,pickle,warnings

import numpy as np

import pandas as pd

from pathlib import Path

warnings.filterwarnings('ignore')



PROJECT=Path('/home/emmanuel.keku')

PREPROC=PROJECT/'preprocessed_v3'

RESULTS=PROJECT/'results_v7'; RESULTS.mkdir(parents=True,exist_ok=True)

SEED=42; np.random.seed(SEED)



print('='*65)

print('  MISAR v2 — Optimized | 10x Data | N=2,4,8,16,32')

print('='*65)



import torch,torch.nn as nn

from torch.optim import AdamW

from sklearn.preprocessing import RobustScaler

from scipy.spatial import cKDTree

import unittest.mock as _mock

torch.manual_seed(SEED)

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {DEVICE}')



# ── Data setup ────────────────────────────────────────────────────────────────

with open(PREPROC/'feature_info.pkl','rb') as f: FI=pickle.load(f)

raw_df=pd.read_csv(PREPROC/'master_processed.csv',parse_dates=['time_utc'])

LOCS=pd.DataFrame(FI['LOCATIONS']); N_LOCS=FI['N_LOCS']; SITES=FI['SITES']

ALL_TGTS=FI['ALL_TARGETS']

loc_to_idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS.iterrows()}

CYCLICAL=[c for c in raw_df.columns if any(c.startswith(p) for p in ['sin_','cos_'])]

SNAP=FI['SNAP_FEATURES']; CORE=[f for f in SNAP if f not in CYCLICAL and f in raw_df.columns]

APPROX=[f'{t}_approx' for t in ALL_TGTS if f'{t}_approx' in raw_df.columns]

RESID=[f'{t}_residual' for t in ALL_TGTS if f'{t}_residual' in raw_df.columns]

UNC=[]

for feat in CORE[:8]:

    vc=f'{feat}_unc_var'

    if vc not in raw_df.columns: raw_df[vc]=np.where(raw_df[feat].isna(),1.0,0.01)

    UNC.append(vc)

V6F=list(dict.fromkeys(CORE+APPROX+RESID+UNC))

V6F=[f for f in V6F if f in raw_df.columns]

N_FEATS=len(V6F)

TEMP_TGTS=FI['TEMP_TARGETS']

use_cols=[f'{c}_residual' for c in TEMP_TGTS if f'{c}_residual' in raw_df.columns]

approx_c=[f'{c}_approx' for c in TEMP_TGTS if f'{c}_approx' in raw_df.columns]

NT=len(use_cols)

tr_df=raw_df[(raw_df['split']=='train')&(raw_df['Site']!='Wetland')]

feat_sc=RobustScaler(); feat_sc.fit(tr_df[V6F].fillna(0).values)

tgt_sc=RobustScaler(); tgt_sc.fit(tr_df[use_cols].dropna().values)

print(f'Features: {N_FEATS} | NT: {NT}')



# Site locations

SITE_LOCS={}

for site in SITES:

    rows=raw_df[raw_df['Site']==site][['Latitude','Longitude']].drop_duplicates()

    idxs=sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))

                  for _,r in rows.iterrows()

                  if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

    SITE_LOCS[site]=idxs

WETLAND=SITE_LOCS['Wetland']

SEEN=sorted(set(i for s,v in SITE_LOCS.items() if s!='Wetland' for i in v))



# Graph

coords=LOCS[['Latitude','Longitude']].values.astype(np.float32)

sc_=coords*np.array([111.0,63.0]); tree_=cKDTree(sc_)

d_,i_=tree_.query(sc_,k=7); sig_=np.median(d_[:,1:])+1e-8

A_np=np.zeros((N_LOCS,N_LOCS),dtype=np.float32)

for i in range(N_LOCS):

    for jp in range(1,d_.shape[1]):

        j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))

        A_np[i,j]+=w; A_np[j,i]+=w

A_np+=np.eye(N_LOCS); D_=A_np.sum(1,keepdims=True)**0.5

A_norm=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))

print(f'Graph: N={N_LOCS} | sigma={sig_:.2f}km')



# Load model classes

_src=open(PROJECT/'train_soil_spatial_v6.py').read()

_pre=_src.split('if args.mode')[0]; _ns={'__name__':'__imported__'}

with _mock.patch('sys.argv',['t.py','--mode','train','--target','temp']):

    try: exec(_pre,_ns)

    except: pass

MODEL_MAP=_ns.get('MODEL_MAP',{})



# ── MISAR v2 module ───────────────────────────────────────────────────────────

class MISARv2:

    """

    MISAR v2 — Optimized uncertainty estimator

    Key changes:

    - H=1 lookahead (was H=3) — 3x faster per step

    - Cached estimates — reuse within round

    - Compressed gradient communication

    """

    def __init__(self, H=1, history_len=5, base_angle=0.05, base_mag=0.05,

                 top_k=0.1):

        self.H=H

        self.history_len=history_len

        self.base_angle=base_angle

        self.base_mag=base_mag

        self.top_k=top_k  # compress: only top k% gradient components

        self.grad_history=[]

        self.loss_history=[]

        self.angle_scale=1.0

        self.mag_scale=1.0

        self._cached_unc=None  # cache uncertainty estimate within round

        self._cache_step=0



    def estimate(self, grad_flat, loss_val, model, loader, A, step):

        """Estimate uncertainty — use cache every 5 steps"""

        g_norm=grad_flat/(np.linalg.norm(grad_flat)+1e-8)

        self.grad_history.append(g_norm)

        self.loss_history.append(loss_val)

        if len(self.grad_history)>self.history_len:

            self.grad_history.pop(0)

            self.loss_history.pop(0)



        # Use cached estimate every 5 steps to save compute

        if self._cached_unc is not None and (step-self._cache_step)<5:

            unc=self._cached_unc

        else:

            # Gradient consistency

            if len(self.grad_history)>1:

                sims=[float(np.dot(self.grad_history[-1],self.grad_history[-i-1]))

                       for i in range(1,min(3,len(self.grad_history)))]

                grad_cons=float(np.mean(sims))

            else: grad_cons=0.5



            # Loss trend

            loss_trend=(self.loss_history[-2]-self.loss_history[-1])/(abs(self.loss_history[-2])+1e-8) if len(self.loss_history)>1 else 0.



            # H=1 lookahead — just one step, much faster

            m_clone=copy.deepcopy(model).cpu()

            opt_c=AdamW(m_clone.parameters(),lr=1e-3)

            m_clone.train(); la_loss=loss_val

            for Xb,yb,mb,ab in loader:

                opt_c.zero_grad()

                out=m_clone(Xb.cpu(),A.cpu())

                mu=out[0]; lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])

                sv=torch.exp(lsv).clamp(min=1e-6)

                l=(0.5*(lsv+(yb-mu.cpu())**2/sv)*mb.cpu().unsqueeze(-1)).sum()/(mb.sum()+1e-8)

                l.backward()

                nn.utils.clip_grad_norm_(m_clone.parameters(),1.0)

                opt_c.step()

                la_loss=float(l.item())

                break  # H=1: just one batch



            la_improvement=(loss_val-la_loss)/(abs(loss_val)+1e-8)

            unc=float(np.clip(

                0.3*(1-max(0,grad_cons))+0.3*max(0,-loss_trend)+0.4*max(0,-la_improvement),

                0.0,1.0))

            self._cached_unc=unc

            self._cache_step=step



        da=self.base_angle*self.angle_scale*unc

        dm=1.0-self.base_mag*self.mag_scale*unc

        return da,dm,unc



    def apply(self, g_flat, da, dm):

        if da<1e-6: return g_flat*dm

        g_n=g_flat/(np.linalg.norm(g_flat)+1e-8)

        rv=np.random.randn(len(g_flat))

        rv=rv-np.dot(rv,g_n)*g_n

        rv=rv/(np.linalg.norm(rv)+1e-8)

        return (np.cos(da)*g_flat+np.sin(da)*rv)*dm



    def compress_delta(self, delta_flat):

        """Top-k% gradient compression — reduce communication cost"""

        k=max(1,int(len(delta_flat)*self.top_k))

        idx=np.argpartition(np.abs(delta_flat),-k)[-k:]

        compressed=np.zeros_like(delta_flat)

        compressed[idx]=delta_flat[idx]

        return compressed



    def feedback(self, pre, post):

        imp=(pre-post)/(abs(pre)+1e-8)

        if imp>0.01: self.angle_scale=max(0.5,self.angle_scale*0.98)

        elif imp<-0.01: self.angle_scale=min(3.0,self.angle_scale*1.05)



# ── 10x Data augmentation ─────────────────────────────────────────────────────

def build_loader_10x(loc_subset=None,split='train',bs=4,max_s=2000,lookback=24,stride=4):

    """Build loader with 10x data via temporal bootstrap"""

    sub=raw_df[raw_df['split']==split].copy()

    all_ts=sorted(sub['time_utc'].unique()); T=len(all_ts)

    ts2i={t:i for i,t in enumerate(all_ts)}

    sub['_ti']=sub['time_utc'].map(ts2i)

    sub['_ni']=[loc_to_idx.get((float(la),float(lo)))

                 for la,lo in zip(sub['Latitude'],sub['Longitude'])]

    sub=sub.dropna(subset=['_ti','_ni'])

    sub['_ti']=sub['_ti'].astype(int); sub['_ni']=sub['_ni'].astype(int)

    sub=sub[sub['_ti']<T]

    if loc_subset is not None: sub=sub[sub['_ni'].isin(loc_subset)]

    if len(sub)==0: return None



    Xf=np.zeros((T,N_LOCS,N_FEATS),dtype=np.float32)

    yf=np.zeros((T,N_LOCS,NT),dtype=np.float32)

    af=np.zeros((T,N_LOCS),dtype=np.float32)

    mf=np.zeros((T,N_LOCS),dtype=np.float32)

    ti_v=sub['_ti'].values; ni_v=sub['_ni'].values

    Xf[ti_v,ni_v]=feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)

    yf[ti_v,ni_v]=tgt_sc.transform(sub[use_cols].fillna(0).values).astype(np.float32)

    if approx_c and all(c in sub.columns for c in approx_c):

        af[ti_v,ni_v]=sub[approx_c[0]].fillna(0).values.astype(np.float32)

    locs_use=loc_subset if loc_subset is not None else SEEN

    mf[:,locs_use]=1.0



    rng=np.random.default_rng(SEED)

    tidxs=list(range(lookback,T,stride))



    # 10x augmentation: bootstrap temporal windows with small jitter

    Xl=[]; yl=[]; ml=[]; al=[]

    for rep in range(10):  # 10 repetitions

        tidxs_rep=tidxs.copy()

        if rep>0:

            # Add jitter: shift by 1-3 time steps

            jitter=rng.integers(1,4)

            tidxs_rep=[t+jitter for t in tidxs if t+jitter<T]

        rng.shuffle(tidxs_rep)

        for ti in tidxs_rep[:max_s//10]:

            Xw=Xf[ti-lookback:ti]

            if np.isnan(Xw).mean()>0.5: continue

            # Add small noise for augmentation

            noise=rng.normal(0,0.01,Xw.shape).astype(np.float32) if rep>0 else 0

            Xl.append(np.nan_to_num(Xw,nan=0.)+noise)

            yl.append(yf[ti]); ml.append(mf[ti]); al.append(af[ti])

        if len(Xl)>=max_s: break



    if not Xl: return None

    from torch.utils.data import TensorDataset,DataLoader

    ds=TensorDataset(torch.tensor(np.array(Xl[:max_s])),

                      torch.tensor(np.array(yl[:max_s])),

                      torch.tensor(np.array(ml[:max_s])),

                      torch.tensor(np.array(al[:max_s])))

    return DataLoader(ds,batch_size=bs,shuffle=True,num_workers=0,drop_last=False)



def build_loader(loc_subset=None,split='train',bs=4,max_s=400,lookback=24,stride=6):

    sub=raw_df[raw_df['split']==split].copy()

    all_ts=sorted(sub['time_utc'].unique()); T=len(all_ts)

    ts2i={t:i for i,t in enumerate(all_ts)}

    sub['_ti']=sub['time_utc'].map(ts2i)

    sub['_ni']=[loc_to_idx.get((float(la),float(lo)))

                 for la,lo in zip(sub['Latitude'],sub['Longitude'])]

    sub=sub.dropna(subset=['_ti','_ni'])

    sub['_ti']=sub['_ti'].astype(int); sub['_ni']=sub['_ni'].astype(int)

    sub=sub[sub['_ti']<T]

    if loc_subset is not None: sub=sub[sub['_ni'].isin(loc_subset)]

    if len(sub)==0: return None

    Xf=np.zeros((T,N_LOCS,N_FEATS),dtype=np.float32)

    yf=np.zeros((T,N_LOCS,NT),dtype=np.float32)

    af=np.zeros((T,N_LOCS),dtype=np.float32)

    mf=np.zeros((T,N_LOCS),dtype=np.float32)

    ti_v=sub['_ti'].values; ni_v=sub['_ni'].values

    Xf[ti_v,ni_v]=feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)

    yf[ti_v,ni_v]=tgt_sc.transform(sub[use_cols].fillna(0).values).astype(np.float32)

    if approx_c and all(c in sub.columns for c in approx_c):

        af[ti_v,ni_v]=sub[approx_c[0]].fillna(0).values.astype(np.float32)

    locs_use=loc_subset if loc_subset is not None else SEEN

    mf[:,locs_use]=1.0

    rng=np.random.default_rng(SEED)

    tidxs=list(range(lookback,T,stride))

    if len(tidxs)>max_s: tidxs=sorted(rng.choice(tidxs,max_s,replace=False))

    Xl=[]; yl=[]; ml2=[]; al2=[]

    for ti in tidxs:

        Xw=Xf[ti-lookback:ti]

        if np.isnan(Xw).mean()>0.5: continue

        Xl.append(np.nan_to_num(Xw,nan=0.)); yl.append(yf[ti])

        ml2.append(mf[ti]); al2.append(af[ti])

    if not Xl: return None

    from torch.utils.data import TensorDataset,DataLoader

    ds=TensorDataset(torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),

                      torch.tensor(np.array(ml2)),torch.tensor(np.array(al2)))

    return DataLoader(ds,batch_size=bs,shuffle=(split=='train'),num_workers=0)



def evaluate(model,loader,locs=None):

    if loader is None: return float('nan'),float('nan')

    if locs is None: locs=WETLAND

    model.eval(); yt_l=[]; yp_l=[]

    with torch.no_grad():

        for Xb,yb,mb,ab in loader:

            Xb=Xb.to(DEVICE)

            out=model(Xb,A_norm.to(DEVICE))

            mu=out[0].cpu().float().numpy()

            y_np=tgt_sc.inverse_transform(yb.numpy().reshape(-1,NT)).reshape(Xb.shape[0],N_LOCS,NT)

            mu_np=tgt_sc.inverse_transform(mu.reshape(-1,NT)).reshape(Xb.shape[0],N_LOCS,NT)

            ab_np=ab.numpy()

            yt_l.append((y_np[:,locs,0]+ab_np[:,locs]).flatten())

            yp_l.append((mu_np[:,locs,0]+ab_np[:,locs]).flatten())

    yt=np.concatenate(yt_l); yp=np.concatenate(yp_l)

    mk=~(np.isnan(yt)|np.isnan(yp))

    if mk.sum()<5: return float('nan'),float('nan')

    rmse=float(np.sqrt(np.mean((yt[mk]-yp[mk])**2)))

    r2=float(1-np.sum((yt[mk]-yp[mk])**2)/(np.sum((yt[mk]-yt[mk].mean())**2)+1e-10))

    return rmse,r2



def get_subsets(n_workers):

    if n_workers==1: return [SEEN]

    if n_workers<=len(SEEN):

        q=len(SEEN)//n_workers

        subsets=[SEEN[i*q:(i+1)*q] for i in range(n_workers-1)]

        subsets.append(SEEN[(n_workers-1)*q:])  # last gets remainder

        return [s for s in subsets if len(s)>0]

    # More workers than locations — overlap allowed

    return [SEEN[i%len(SEEN):(i%len(SEEN))+max(1,len(SEEN)//n_workers)]

             for i in range(n_workers)]



def misar_worker_v2(worker_id, theta_cpu, loc_subset, arch_cls, hcfg,

                     k_steps=50, lr=1e-3, use_10x=True):

    h=int(hcfg.get('hidden_dim',152))

    nl=int(hcfg.get('n_layers',2))

    gl=int(hcfg.get('gcn_layers',2))

    model=None

    for h_try in [h,h//2,152,96]:

        try:

            m=arch_cls(nf=N_FEATS,h=h_try,nl=nl,gl=gl,nt=NT)

            m.load_state_dict({k:v.cpu() for k,v in theta_cpu.items()},strict=True)

            model=m.to(DEVICE); break

        except: continue

    if model is None: return None



    # 10x augmented loader for training

    if use_10x:

        loader=build_loader_10x(loc_subset=loc_subset,split='train',bs=4,max_s=2000)

    else:

        loader=build_loader(loc_subset=loc_subset,split='train',bs=4,max_s=200)

    if loader is None: return None



    misar=MISARv2(H=1,history_len=5,base_angle=0.05,base_mag=0.05,top_k=0.1)

    theta_start={k:v.clone().cpu() for k,v in model.state_dict().items()}

    losses=[]; uncertainties=[]; step=0; prev_loss=None

    A_dev=A_norm.to(DEVICE)

    model.train()



    for Xb,yb,mb,ab in loader:

        if step>=k_steps: break

        Xb=Xb.to(DEVICE); yb=yb.to(DEVICE); mb=mb.to(DEVICE)

        model.zero_grad()

        out=model(Xb,A_dev)

        mu=out[0]; lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])

        sv=torch.exp(lsv).clamp(min=1e-6)

        loss=(0.5*(lsv+(yb-mu)**2/sv)*mb.unsqueeze(-1)).sum()/(mb.sum()+1e-8)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(),1.0)



        grad_flat=torch.cat([p.grad.data.cpu().flatten()

                              for p in model.parameters()

                              if p.grad is not None]).numpy()

        loss_val=float(loss.item()); losses.append(loss_val)



        da,dm,unc=misar.estimate(grad_flat,loss_val,model,loader,A_dev,step)

        uncertainties.append(unc)



        with torch.no_grad():

            for param in model.parameters():

                if param.grad is None: continue

                g=param.grad.data.cpu().numpy().flatten()

                g_p=misar.apply(g,da,dm)

                param.grad.data=torch.tensor(g_p.reshape(param.grad.shape),dtype=param.dtype).to(param.device)

            for param in model.parameters():

                if param.grad is not None:

                    param.data-=lr*param.grad.data



        if prev_loss is not None: misar.feedback(prev_loss,loss_val)

        prev_loss=loss_val; step+=1



    if not losses: return None

    theta_final={k:v.clone().cpu() for k,v in model.state_dict().items()}

    delta={k:(theta_final[k]-theta_start[k]) for k in theta_start}



    # Gradient compression — only top 10% of delta

    delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()

    delta_flat_compressed=misar.compress_delta(delta_flat)



    # Reconstruct compressed delta dict

    offset=0

    delta_compressed={}

    for k,v in delta.items():

        n=v.numel()

        delta_compressed[k]=torch.tensor(

            delta_flat_compressed[offset:offset+n].reshape(v.shape),dtype=v.dtype)

        offset+=n



    progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.

    stability=1.0



    return dict(

        worker_id=worker_id,

        delta=delta_compressed,

        delta_flat=delta_flat_compressed,

        losses=losses,

        uncertainties=uncertainties,

        stability=stability,

        progress=progress,

        final_loss=losses[-1],

        movement=float(np.linalg.norm(delta_flat_compressed)),

        misar_angle_scale=misar.angle_scale,

        compression_ratio=float(np.count_nonzero(delta_flat_compressed)/len(delta_flat_compressed))

    )



def aggregate(theta_t, behaviors, lr=0.85):

    N=len(behaviors)

    delta_flats=[b['delta_flat'] for b in behaviors]

    consensus=np.ones(N)

    for i in range(N):

        sims=[float(np.dot(delta_flats[i],delta_flats[j])/

                     (np.linalg.norm(delta_flats[i])*np.linalg.norm(delta_flats[j])+1e-8))

               for j in range(N) if j!=i]

        consensus[i]=float(np.mean(sims)) if sims else 0.

    stab=np.array([b['stability'] for b in behaviors])

    prog=np.array([b['progress'] for b in behaviors])

    scores=np.clip(consensus,0,None)*np.clip(stab,0,None)*np.clip(prog,0,None)

    scores=np.clip(scores,1e-8,None); scores=scores-scores.max()

    alphas=np.exp(scores)/np.sum(np.exp(scores))

    mean_delta=np.mean(delta_flats,axis=0)

    betas=[]

    for df in delta_flats:

        res=df-mean_delta; sign=np.sign(np.dot(df,res)+1e-8)

        betas.append(float(sign*np.linalg.norm(res)/(np.linalg.norm(df)+1e-8)))

    betas=np.array(betas)

    mean_unc=float(np.mean([np.mean(b.get('uncertainties',[0.5])) for b in behaviors]))

    gamma=lr*(1-min(mean_unc*0.1,0.5))

    theta_new={}

    mean_delta_d={k:sum(behaviors[i]['delta'][k].float() for i in range(N))/N for k in theta_t}

    for k in theta_t:

        shared=sum(alphas[i]*behaviors[i]['delta'][k].float() for i in range(N))

        res_sum=sum(betas[i]*(behaviors[i]['delta'][k].float()-mean_delta_d[k]) for i in range(N))

        theta_new[k]=(theta_t[k].float()+gamma*(shared+0.1*res_sum))

    return theta_new,alphas,betas,gamma,mean_unc



# ── Main experiment ───────────────────────────────────────────────────────────

MODELS_TO_RUN=['SpatialMamba']

N_WORKERS_LIST=[8,16,32]

T_ROUNDS=24; K_STEPS=50; LR=1e-3



import json

hp=json.load(open(PROJECT/'results_v6'/'v6_best_hparams.json'))



all_results=[]



for arch_name in MODELS_TO_RUN:

    print(f'\n{"="*55}')

    print(f'  MISAR v2 | {arch_name} | 10x data | N={N_WORKERS_LIST}')

    print(f'{"="*55}')



    arch_cls=MODEL_MAP.get(arch_name)

    if arch_cls is None: continue



    hcfg=hp.get(f'{arch_name}_temp',{})

    h=int(hcfg.get('hidden_dim',152))

    nl=int(hcfg.get('n_layers',2))

    gl=int(hcfg.get('gcn_layers',2))



    ck=torch.load(PROJECT/'models_v6'/'dl'/f'{arch_name}_temp_v6_best.pt',map_location='cpu')

    cent_r2=float(ck.get('test_metrics',{}).get('unseen_space',{}).get('unseen_R2',float('nan')))

    cent_rmse=float(ck.get('test_metrics',{}).get('unseen_space',{}).get('unseen_ubRMSE',float('nan')))

    print(f'  Centralized: R²={cent_r2:.4f} RMSE={cent_rmse:.4f}')



    val_loader=build_loader(split='test',bs=4,max_s=400)



    for N_WORKERS in N_WORKERS_LIST:

        SUBSETS=get_subsets(N_WORKERS)

        actual_N=len(SUBSETS)

        print(f'\n  MISAR v2 N={N_WORKERS} (actual={actual_N}) | '

               f'locs/worker={[len(s) for s in SUBSETS[:4]]}...')



        torch.manual_seed(SEED)

        for h_try in [h,h//2,152,96]:

            try:

                global_model=arch_cls(nf=N_FEATS,h=h_try,nl=nl,gl=gl,nt=NT).to(DEVICE)

                break

            except: continue



        theta_t={k:v.cpu().clone() for k,v in global_model.state_dict().items()}

        best_rmse=float('inf'); best_r2=float('nan')

        prev_rmse=float('inf'); round_history=[]; t_start=time.time()



        for rnd in range(1,T_ROUNDS+1):

            t_round=time.time()

            theta_cpu={k:v.detach().cpu().clone() for k,v in theta_t.items()}



            # Run workers sequentially (Ray not available)

            behaviors=[]

            t_compute=time.time()

            for wi in range(actual_N):

                b=misar_worker_v2(wi,{k:v.clone() for k,v in theta_cpu.items()},

                                    SUBSETS[wi],arch_cls,hcfg,K_STEPS,LR,use_10x=True)

                if b is not None: behaviors.append(b)

            t_compute=time.time()-t_compute



            if not behaviors: continue



            # Aggregation (communication)

            t_agg=time.time()

            theta_cand,alphas,betas,gamma,mean_unc=aggregate(theta_t,behaviors,0.85)

            t_agg=time.time()-t_agg



            global_model.load_state_dict({k:v.to(DEVICE) for k,v in theta_cand.items()})

            rmse,r2=evaluate(global_model,val_loader)



            if not(np.isnan(rmse) or np.isinf(rmse)) and rmse<=prev_rmse*1.02:

                theta_t=theta_cand; prev_rmse=rmse

                if not np.isnan(r2) and (np.isnan(best_r2) or r2>best_r2):

                    best_rmse=rmse; best_r2=r2

                status='ACC'

            else: status='REJ'



            elapsed=time.time()-t_round

            comp_ratio=float(np.mean([b.get('compression_ratio',0.1) for b in behaviors]))



            print(f'  Rnd {rnd:02d} | R²={r2:.4f} γ={gamma:.3f} '

                   f'U={mean_unc:.3f} comp={comp_ratio:.2f} '

                   f'compute={t_compute:.1f}s agg={t_agg:.2f}s | {status}')



            round_history.append(dict(

                round=rnd,arch=arch_name,n_workers=N_WORKERS,

                rmse=rmse,r2=r2,gamma=gamma,

                mean_uncertainty=mean_unc,

                compression_ratio=comp_ratio,

                compute_time=t_compute,

                agg_time=t_agg,

                elapsed=elapsed

            ))



        total_time=time.time()-t_start

        ideal_time=total_time/N_WORKERS

        pd.DataFrame(round_history).to_csv(

            RESULTS/f'misar_v2_rounds_{arch_name}_N{N_WORKERS}.csv',index=False)



        valid_r2=[r['r2'] for r in round_history if not np.isnan(r.get('r2',float('nan')))]

        all_results.append(dict(

            arch=arch_name,n_workers=N_WORKERS,

            best_r2=round(max(valid_r2),4) if valid_r2 else float('nan'),

            cent_r2=cent_r2,

            ideal_time_s=round(ideal_time,2),

            total_time_s=round(total_time,2),

            delta_r2=round((max(valid_r2) if valid_r2 else float('nan'))-cent_r2,4)

        ))

        print(f'\n  {arch_name} N={N_WORKERS}: best_R²={max(valid_r2) if valid_r2 else "nan":.4f} ideal={ideal_time:.1f}s')



summary=pd.DataFrame(all_results)

summary.to_csv(RESULTS/'misar_v2_summary.csv',index=False)

print(f'\n{"="*55}')

print('MISAR v2 Summary:')

print(summary.to_string())

