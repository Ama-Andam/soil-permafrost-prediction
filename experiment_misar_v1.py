
"""

MISAR: Model-Independent Stochastic Adaptive Routing

Uncertainty-aware gradient navigation for distributed spatial AI



Key innovation: Instead of standard gradient descent:

  θ ← θ - η∇L(θ)



MISAR wraps each gradient step with uncertainty estimation:

  μk = -η × gk                    (gradient = expected direction)

  Σk = MISAR(history, trajectory)  (uncertainty around direction)

  Δθk ~ Rotate(μk, δangle) × δmagnitude  (uncertainty-aware update)

  feedback → improve Σk estimation



MISAR lookahead:

  - Simulates H steps ahead using current gradient

  - Measures predicted trajectory variance

  - Uses variance to set angle/magnitude uncertainty

  - If lookahead predicts improvement → low uncertainty → follow gradient

  - If lookahead predicts degradation → high uncertainty → explore more

"""

import os,sys,time,copy,pickle,warnings

import numpy as np

import pandas as pd

from pathlib import Path

warnings.filterwarnings('ignore')



PROJECT=Path('/home/emmanuel.keku')

PREPROC=PROJECT/'preprocessed_v3'

RESULTS=PROJECT/'results_v7'; RESULTS.mkdir(parents=True,exist_ok=True)

FIGS=PROJECT/'figures_v7'; FIGS.mkdir(parents=True,exist_ok=True)

SEED=42; np.random.seed(SEED)



print('='*65)

print('  MISAR v1 — Uncertainty-Aware Gradient Navigation')

print('  STGCN + SpatialMamba | Soil Temperature | N=2,4 workers')

print('='*65)



import torch,torch.nn as nn

from torch.optim import AdamW

from sklearn.preprocessing import RobustScaler

from scipy.spatial import cKDTree

import unittest.mock as _mock

torch.manual_seed(SEED)

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {DEVICE}')



# ── Data setup ─────────────────────────────────────────────────────────────

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

print(f'Features: {N_FEATS} | Targets: {use_cols} | NT: {NT}')



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

print(f'Graph: N={N_LOCS} nodes | sigma={sig_:.2f}km')



# Load model classes

_src=open(PROJECT/'train_soil_spatial_v6.py').read()

_pre=_src.split('if args.mode')[0]; _ns={'__name__':'__imported__'}

with _mock.patch('sys.argv',['t.py','--mode','train','--target','temp']):

    try: exec(_pre,_ns)

    except: pass

MODEL_MAP=_ns.get('MODEL_MAP',{})



# ── MISAR module ──────────────────────────────────────────────────────────────

class MISAR:

    """

    Model-Independent Stochastic Adaptive Routing

    

    Estimates uncertainty around gradient direction using:

    1. Gradient history (consistency over past steps)

    2. Loss trajectory (is it improving?)

    3. Lookahead simulation (predict H steps ahead)

    

    Outputs:

    - delta_angle: rotation uncertainty around gradient direction

    - delta_magnitude: scaling uncertainty for step size

    """

    def __init__(self, H=3, history_len=5, base_angle=0.1, base_mag=0.1):

        self.H=H                        # lookahead horizon

        self.history_len=history_len    # gradient history length

        self.base_angle=base_angle      # base angle uncertainty (radians)

        self.base_mag=base_mag          # base magnitude uncertainty

        

        # Gradient history

        self.grad_history=[]   # list of flattened gradient vectors

        self.loss_history=[]   # list of loss values

        self.pred_history=[]   # list of predicted future losses

        

        # Learned uncertainty parameters (updated via feedback)

        self.angle_scale=1.0   # multiplier for angle uncertainty

        self.mag_scale=1.0     # multiplier for magnitude uncertainty

        self.n_updates=0

        

    def estimate_uncertainty(self, model, grad_flat, current_loss, 

                               loader, A, device, tgt_sc, N_LOCS, NT):

        """

        Core MISAR: estimate uncertainty around gradient direction

        

        Steps:

        1. Add current grad to history

        2. Compute gradient consistency (low variance = low uncertainty)

        3. Simulate lookahead: predict H steps ahead

        4. Use lookahead variance to set uncertainty

        """

        # 1. Update gradient history

        g_norm=grad_flat/(np.linalg.norm(grad_flat)+1e-8)

        self.grad_history.append(g_norm)

        self.loss_history.append(current_loss)

        if len(self.grad_history)>self.history_len:

            self.grad_history.pop(0)

            self.loss_history.pop(0)

        

        # 2. Gradient consistency — cosine similarity between recent gradients

        if len(self.grad_history)>1:

            sims=[float(np.dot(self.grad_history[-1],self.grad_history[-i-1]))

                   for i in range(1,min(3,len(self.grad_history)))]

            grad_consistency=float(np.mean(sims))  # -1 to 1

        else:

            grad_consistency=0.5

        

        # 3. Loss trend — is loss decreasing?

        if len(self.loss_history)>1:

            loss_trend=(self.loss_history[-2]-self.loss_history[-1])/(abs(self.loss_history[-2])+1e-8)

        else:

            loss_trend=0.0

        

        # 4. Lookahead simulation

        # Clone model, simulate H gradient steps, measure predicted loss variance

        lookahead_losses=[]

        m_clone=copy.deepcopy(model)

        opt_clone=AdamW(m_clone.parameters(),lr=1e-3,weight_decay=5e-4)

        m_clone.train()

        

        steps_done=0

        for Xb,yb,mb,ab in loader:

            if steps_done>=self.H: break

            Xb=Xb.to(device); yb=yb.to(device); mb=mb.to(device)

            opt_clone.zero_grad()

            out=m_clone(Xb,A)

            mu=out[0]; lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])

            sv=torch.exp(lsv).clamp(min=1e-6)

            loss=(0.5*(lsv+(yb-mu)**2/sv)*mb.unsqueeze(-1)).sum()/(mb.sum()+1e-8)

            loss.backward()

            nn.utils.clip_grad_norm_(m_clone.parameters(),1.0)

            opt_clone.step()

            lookahead_losses.append(float(loss.item()))

            steps_done+=1

        

        if len(lookahead_losses)>1:

            lookahead_var=float(np.var(lookahead_losses))

            lookahead_trend=(lookahead_losses[0]-lookahead_losses[-1])/(abs(lookahead_losses[0])+1e-8)

        else:

            lookahead_var=0.1

            lookahead_trend=0.0

        

        # 5. Compute uncertainty from all signals

        # High consistency + improving loss + positive lookahead → low uncertainty

        # Low consistency + degrading loss + negative lookahead → high uncertainty

        consistency_factor=1.0-max(0,grad_consistency)  # 0=consistent, 1=inconsistent

        loss_factor=max(0,-loss_trend)                    # 0=improving, >0=degrading

        lookahead_factor=max(0,-lookahead_trend)+lookahead_var  # higher=more uncertain

        

        # Combined uncertainty score

        uncertainty=float(np.clip(

            0.3*consistency_factor + 0.3*loss_factor + 0.4*lookahead_factor,

            0.0, 1.0

        ))

        

        # Angle and magnitude uncertainty

        delta_angle=self.base_angle*self.angle_scale*uncertainty

        delta_mag=1.0 - self.base_mag*self.mag_scale*uncertainty

        

        return delta_angle, delta_mag, uncertainty, {

            'grad_consistency':grad_consistency,

            'loss_trend':loss_trend,

            'lookahead_var':lookahead_var,

            'lookahead_trend':lookahead_trend,

            'uncertainty':uncertainty

        }

    

    def apply_uncertainty(self, delta_theta_flat, delta_angle, delta_mag):

        """

        Apply uncertainty to gradient direction:

        Δθk = Rotate(μk, δangle) × δmagnitude

        

        Rotation: add random perturbation perpendicular to gradient

        Magnitude: scale step size based on confidence

        """

        if delta_angle<1e-6:

            return delta_theta_flat*delta_mag

        

        # Create rotation in gradient space

        # Generate random unit vector perpendicular to gradient

        dim=len(delta_theta_flat)

        g_norm=delta_theta_flat/(np.linalg.norm(delta_theta_flat)+1e-8)

        

        # Random vector

        rand_vec=np.random.randn(dim)

        # Project out gradient component (make perpendicular)

        rand_vec=rand_vec-np.dot(rand_vec,g_norm)*g_norm

        rand_norm=np.linalg.norm(rand_vec)+1e-8

        rand_vec=rand_vec/rand_norm

        

        # Rotate gradient by delta_angle in direction of rand_vec

        rotated=np.cos(delta_angle)*delta_theta_flat + np.sin(delta_angle)*rand_norm*rand_vec

        

        # Apply magnitude scaling

        rotated=rotated*delta_mag

        

        return rotated

    

    def feedback(self, pre_loss, post_loss, delta_angle_used, delta_mag_used):

        """

        MISAR feedback: did the uncertainty-perturbed update help?

        Adjust angle_scale and mag_scale accordingly

        """

        self.n_updates+=1

        improvement=(pre_loss-post_loss)/(abs(pre_loss)+1e-8)

        

        if improvement>0.01:

            # Update helped — slightly reduce uncertainty next time

            self.angle_scale=max(0.5,self.angle_scale*0.98)

            self.mag_scale=min(1.5,self.mag_scale*1.01)

        elif improvement<-0.01:

            # Update hurt — increase uncertainty (explore more)

            self.angle_scale=min(3.0,self.angle_scale*1.05)

            self.mag_scale=max(0.5,self.mag_scale*0.97)

        

        return improvement



# ── Data loader ───────────────────────────────────────────────────────────────

def build_loader(loc_subset=None,split='train',bs=4,max_s=500,lookback=24,stride=6):

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

    Xl=[]; yl=[]; ml=[]; al=[]

    for ti in tidxs:

        Xw=Xf[ti-lookback:ti]

        if np.isnan(Xw).mean()>0.5: continue

        Xl.append(np.nan_to_num(Xw,nan=0.)); yl.append(yf[ti])

        ml.append(mf[ti]); al.append(af[ti])

    if not Xl: return None

    from torch.utils.data import TensorDataset,DataLoader

    ds=TensorDataset(torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),

                      torch.tensor(np.array(ml)),torch.tensor(np.array(al)))

    return DataLoader(ds,batch_size=bs,shuffle=(split=='train'),num_workers=0,drop_last=False)



# ── Evaluation ────────────────────────────────────────────────────────────────

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



# ── Subset definitions ─────────────────────────────────────────────────────────

def get_subsets(n_workers):

    if n_workers==1: return [SEEN]

    if n_workers==2:

        half=len(SEEN)//2

        return [SEEN[:half],SEEN[half:]]

    if n_workers==4:

        q=len(SEEN)//4

        return [SEEN[i*q:(i+1)*q] for i in range(4)]

    if n_workers==8:

        q=len(SEEN)//8

        return [SEEN[i*q:(i+1)*q] for i in range(8)]

    return [SEEN]



# ── MISAR worker ──────────────────────────────────────────────────────────────

def misar_worker(worker_id, theta_cpu, loc_subset, arch_cls, hcfg,

                  k_steps=50, lr=1e-3, H=3):

    """

    Single MISAR worker:

    1. Load shared weights θt

    2. For each gradient step:

       a. Compute gradient gk

       b. MISAR lookahead → estimate (δangle, δmag, Σk)

       c. Apply Δθk = Rotate(μk, δangle) × δmag

       d. Update θk

       e. Observe result → feedback to MISAR

    """

    h=int(hcfg.get('hidden_dim',152))

    nl=int(hcfg.get('n_layers',2))

    gl=int(hcfg.get('gcn_layers',2))

    

    model=arch_cls(nf=N_FEATS,h=h,nl=nl,gl=gl,nt=NT).to(DEVICE)

    # Try loading — handle hidden dim mismatch

    for h_try in [h,h//2,152,96]:

        try:

            m_tmp=arch_cls(nf=N_FEATS,h=h_try,nl=nl,gl=gl,nt=NT)

            m_tmp.load_state_dict({k:v.cpu() for k,v in theta_cpu.items()},strict=True)

            model=m_tmp.to(DEVICE); break

        except: continue

    

    loader=build_loader(loc_subset=loc_subset,split='train',bs=4,max_s=200)

    if loader is None: return None

    

    misar=MISAR(H=H,history_len=5,base_angle=0.05,base_mag=0.05)

    theta_start={k:v.clone().cpu() for k,v in model.state_dict().items()}

    

    losses=[]; uncertainties=[]; angle_mags=[]; misar_feedback=[]

    step=0; prev_loss=None

    A_dev=A_norm.to(DEVICE)

    

    model.train()

    for Xb,yb,mb,ab in loader:

        if step>=k_steps: break

        Xb=Xb.to(DEVICE); yb=yb.to(DEVICE); mb=mb.to(DEVICE)

        

        # Standard forward + backward

        model.zero_grad()

        out=model(Xb,A_dev)

        mu=out[0]; lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])

        sv=torch.exp(lsv).clamp(min=1e-6)

        loss=(0.5*(lsv+(yb-mu)**2/sv)*mb.unsqueeze(-1)).sum()/(mb.sum()+1e-8)

        loss.backward()

        

        # Get flattened gradient

        grad_flat=torch.cat([p.grad.data.cpu().flatten()

                              for p in model.parameters()

                              if p.grad is not None]).numpy()

        

        # Clip gradient norm

        nn.utils.clip_grad_norm_(model.parameters(),1.0)

        

        loss_val=float(loss.item())

        losses.append(loss_val)

        

        # MISAR: estimate uncertainty around gradient direction

        delta_angle,delta_mag,uncertainty,misar_info=misar.estimate_uncertainty(

            model,grad_flat,loss_val,loader,A_dev,DEVICE,tgt_sc,N_LOCS,NT)

        

        uncertainties.append(uncertainty)

        angle_mags.append((delta_angle,delta_mag))

        

        # Apply uncertainty to each parameter's gradient

        with torch.no_grad():

            for param in model.parameters():

                if param.grad is None: continue

                g=param.grad.data.cpu().numpy().flatten()

                # MISAR rotation + magnitude

                g_perturbed=misar.apply_uncertainty(g,delta_angle,delta_mag)
                param.grad.data=torch.tensor(g_perturbed.reshape(param.grad.shape),dtype=param.dtype).to(param.device)






        

        # Apply perturbed gradient via manual SGD-like step

        # (use lr directly since we already applied MISAR scaling)

        with torch.no_grad():

            for param in model.parameters():

                if param.grad is not None:

                    param.data-=lr*param.grad.data

        

        # MISAR feedback

        if prev_loss is not None:

            improvement=misar.feedback(prev_loss,loss_val,delta_angle,delta_mag)

            misar_feedback.append(improvement)

        prev_loss=loss_val

        step+=1

    

    if not losses: return None

    

    theta_final={k:v.clone().cpu() for k,v in model.state_dict().items()}

    delta={k:(theta_final[k]-theta_start[k]) for k in theta_start}

    delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()

    

    # Behavior metrics

    stability=float(np.mean([float(np.dot(

        delta_flat/(np.linalg.norm(delta_flat)+1e-8),

        delta_flat/(np.linalg.norm(delta_flat)+1e-8))) for _ in [1]]))

    progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.

    

    return dict(

        worker_id=worker_id,

        delta=delta,

        delta_flat=delta_flat,

        losses=losses,

        uncertainties=uncertainties,

        angle_mags=angle_mags,

        misar_feedback=misar_feedback,

        misar_angle_scale=misar.angle_scale,

        misar_mag_scale=misar.mag_scale,

        stability=stability,

        progress=progress,

        final_loss=losses[-1],

        movement=float(np.linalg.norm(delta_flat))

    )



# ── Ray parallel workers ───────────────────────────────────────────────────────

try:

    import ray

    if not ray.is_initialized():

        ray.init(ignore_reinit_error=True,num_cpus=8,num_gpus=0)

    HAS_RAY=True; print(f'Ray: {ray.__version__}')

except: HAS_RAY=False; print('Ray: not available')



if HAS_RAY:

    @ray.remote(num_cpus=2)

    def ray_misar_worker(worker_id, theta_cpu, loc_subset, arch_name, hcfg,

                          k_steps=50, lr=1e-3, H=3, seed=42):

        import torch,torch.nn as nn,numpy as np,copy,pickle,warnings,pandas as pd

        from pathlib import Path; from sklearn.preprocessing import RobustScaler

        from scipy.spatial import cKDTree; from torch.utils.data import TensorDataset,DataLoader

        import unittest.mock as _mock

        warnings.filterwarnings('ignore')

        torch.manual_seed(seed+worker_id); np.random.seed(seed+worker_id)

        

        PROJECT2=Path('/home/emmanuel.keku'); PREPROC2=PROJECT2/'preprocessed_v3'

        with open(PREPROC2/'feature_info.pkl','rb') as f: FI2=pickle.load(f)

        raw2=pd.read_csv(PREPROC2/'master_processed.csv',parse_dates=['time_utc'])

        LOCS2=pd.DataFrame(FI2['LOCATIONS']); N_LOCS2=FI2['N_LOCS']

        ALL_TGTS2=FI2['ALL_TARGETS']; SNAP2=FI2['SNAP_FEATURES']

        CYCLICAL2=[c for c in raw2.columns if any(c.startswith(p) for p in ['sin_','cos_'])]

        CORE2=[f for f in SNAP2 if f not in CYCLICAL2 and f in raw2.columns]

        APPROX2=[f'{t}_approx' for t in ALL_TGTS2 if f'{t}_approx' in raw2.columns]

        RESID2=[f'{t}_residual' for t in ALL_TGTS2 if f'{t}_residual' in raw2.columns]

        UNC2=[]

        for feat in CORE2[:8]:

            vc=f'{feat}_unc_var'

            if vc not in raw2.columns: raw2[vc]=np.where(raw2[feat].isna(),1.0,0.01)

            UNC2.append(vc)

        V6F2=list(dict.fromkeys(CORE2+APPROX2+RESID2+UNC2))

        V6F2=[f for f in V6F2 if f in raw2.columns]

        N_FEATS2=len(V6F2)

        TEMP2=FI2['TEMP_TARGETS']

        use_cols2=[f'{c}_residual' for c in TEMP2 if f'{c}_residual' in raw2.columns]

        approx2=[f'{c}_approx' for c in TEMP2 if f'{c}_approx' in raw2.columns]

        NT2=len(use_cols2)

        tr2=raw2[(raw2['split']=='train')]

        feat_sc2=RobustScaler(); feat_sc2.fit(tr2[V6F2].fillna(0).values)

        tgt_sc2=RobustScaler(); tgt_sc2.fit(tr2[use_cols2].dropna().values)

        loc2idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS2.iterrows()}

        

        coords=LOCS2[['Latitude','Longitude']].values.astype(np.float32)*np.array([111.0,63.0])

        tree_=cKDTree(coords); d_,i_=tree_.query(coords,k=7); sig_=np.median(d_[:,1:])+1e-8

        A_np=np.zeros((N_LOCS2,N_LOCS2),dtype=np.float32)

        for i in range(N_LOCS2):

            for jp in range(1,d_.shape[1]):

                j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_)); A_np[i,j]+=w; A_np[j,i]+=w

        A_np+=np.eye(N_LOCS2); D_=A_np.sum(1,keepdims=True)**0.5

        A2=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))



        _src2=open(PROJECT2/'train_soil_spatial_v6.py').read()

        _pre2=_src2.split('if args.mode')[0]; _ns2={'__name__':'__imported__'}

        with _mock.patch('sys.argv',['t.py','--mode','train','--target','temp']):

            try: exec(_pre2,_ns2)

            except: pass

        arch_cls2=_ns2.get('MODEL_MAP',{}).get(arch_name)

        if arch_cls2 is None: return None



        h=int(hcfg.get('hidden_dim',152))

        nl=int(hcfg.get('n_layers',2))

        gl=int(hcfg.get('gcn_layers',2))

        model=None

        for h_try in [h,h//2,152,96]:

            try:

                m_tmp=arch_cls2(nf=N_FEATS2,h=h_try,nl=nl,gl=gl,nt=NT2)

                m_tmp.load_state_dict({k:v.cpu() for k,v in theta_cpu.items()},strict=True)

                model=m_tmp; break

            except: continue

        if model is None: return None



        # Build loader

        sub2=raw2[raw2['split']=='train'].copy()

        all_ts=sorted(sub2['time_utc'].unique()); T2=len(all_ts)

        ts2i={t:i for i,t in enumerate(all_ts)}

        sub2['_ti']=sub2['time_utc'].map(ts2i)

        sub2['_ni']=[loc2idx.get((float(la),float(lo))) for la,lo in zip(sub2['Latitude'],sub2['Longitude'])]

        sub2=sub2.dropna(subset=['_ti','_ni']); sub2['_ti']=sub2['_ti'].astype(int); sub2['_ni']=sub2['_ni'].astype(int)

        sub2=sub2[sub2['_ti']<T2]; sub2=sub2[sub2['_ni'].isin(loc_subset)]

        if len(sub2)==0: return None

        Xf=np.zeros((T2,N_LOCS2,N_FEATS2),dtype=np.float32); yf=np.zeros((T2,N_LOCS2,NT2),dtype=np.float32)

        mf=np.zeros((T2,N_LOCS2),dtype=np.float32)

        Xf[sub2['_ti'].values,sub2['_ni'].values]=feat_sc2.transform(sub2[V6F2].fillna(0).values).astype(np.float32)

        yf[sub2['_ti'].values,sub2['_ni'].values]=tgt_sc2.transform(sub2[use_cols2].fillna(0).values).astype(np.float32)

        mf[:,loc_subset]=1.0

        tidxs=list(range(24,T2,6))[:200]

        Xl=[]; yl=[]; ml2=[]

        for ti in tidxs:

            Xw=Xf[ti-24:ti]

            if np.isnan(Xw).mean()>0.5: continue

            Xl.append(np.nan_to_num(Xw,nan=0.)); yl.append(yf[ti]); ml2.append(mf[ti])

        if not Xl: return None

        ds=TensorDataset(torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),torch.tensor(np.array(ml2)))

        loader=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0)



        # MISAR class (inline for Ray)

        class MISAR_W:

            def __init__(self,H=3,history_len=5,base_angle=0.15,base_mag=0.1):

                self.H=H; self.history_len=history_len

                self.base_angle=base_angle; self.base_mag=base_mag

                self.grad_history=[]; self.loss_history=[]

                self.angle_scale=1.0; self.mag_scale=1.0

            def estimate(self,grad_flat,loss_val,model,loader,A):

                g_norm=grad_flat/(np.linalg.norm(grad_flat)+1e-8)

                self.grad_history.append(g_norm); self.loss_history.append(loss_val)

                if len(self.grad_history)>self.history_len:

                    self.grad_history.pop(0); self.loss_history.pop(0)

                if len(self.grad_history)>1:

                    sims=[float(np.dot(self.grad_history[-1],self.grad_history[-i-1]))

                           for i in range(1,min(3,len(self.grad_history)))]

                    grad_cons=float(np.mean(sims))

                else: grad_cons=0.5

                loss_trend=(self.loss_history[-2]-self.loss_history[-1])/(abs(self.loss_history[-2])+1e-8) if len(self.loss_history)>1 else 0.

                # Lookahead

                m_clone=copy.deepcopy(model)

                opt_c=torch.optim.AdamW(m_clone.parameters(),lr=1e-3)

                m_clone.train(); la_losses=[]; steps=0

                for Xb,yb,mb in loader:

                    if steps>=H: break

                    opt_c.zero_grad(); mu_,lsv_=m_clone(Xb,A)

                    sv_=torch.exp(lsv_).clamp(min=1e-6)

                    l_=(0.5*(lsv_+(yb-mu_)**2/sv_)*mb.unsqueeze(-1)).sum()/(mb.sum()+1e-8)

                    l_.backward(); nn.utils.clip_grad_norm_(m_clone.parameters(),1.0); opt_c.step()

                    la_losses.append(float(l_.item())); steps+=1

                la_var=float(np.var(la_losses)) if len(la_losses)>1 else 0.1

                la_trend=(la_losses[0]-la_losses[-1])/(abs(la_losses[0])+1e-8) if len(la_losses)>1 else 0.

                unc=float(np.clip(0.3*(1-max(0,grad_cons))+0.3*max(0,-loss_trend)+0.4*(max(0,-la_trend)+la_var),0,1))

                da=self.base_angle*self.angle_scale*unc

                dm=1.0-self.base_mag*self.mag_scale*unc

                return da,dm,unc

            def apply(self,g_flat,da,dm):

                if da<1e-6: return g_flat*dm

                g_n=g_flat/(np.linalg.norm(g_flat)+1e-8)

                rv=np.random.randn(len(g_flat))

                rv=rv-np.dot(rv,g_n)*g_n; rv=rv/(np.linalg.norm(rv)+1e-8)

                return (np.cos(da)*g_flat+np.sin(da)*rv)*dm

            def feedback(self,pre,post):

                imp=(pre-post)/(abs(pre)+1e-8)

                if imp>0.01: self.angle_scale=max(0.5,self.angle_scale*0.98)

                elif imp<-0.01: self.angle_scale=min(3.0,self.angle_scale*1.05)



        misar_w=MISAR_W(H=H)

        theta_start={k:v.clone() for k,v in model.state_dict().items()}

        losses=[]; uncertainties=[]; step=0; prev_loss=None

        model.train()

        for Xb,yb,mb in loader:

            if step>=k_steps: break

            model.zero_grad()

            mu_,lsv_=model(Xb,A2)

            sv_=torch.exp(lsv_).clamp(min=1e-6)

            loss=(0.5*(lsv_+(yb-mu_)**2/sv_)*mb.unsqueeze(-1)).sum()/(mb.sum()+1e-8)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(),1.0)

            grad_flat=torch.cat([p.grad.data.flatten() for p in model.parameters() if p.grad is not None]).numpy()

            loss_val=float(loss.item()); losses.append(loss_val)

            da,dm,unc=misar_w.estimate(grad_flat,loss_val,model,loader,A2)

            uncertainties.append(unc)

            with torch.no_grad():

                for param in model.parameters():

                    if param.grad is None: continue

                    g=param.grad.data.numpy().flatten()

                    g_p=misar_w.apply(g,da,dm)

                    param.grad.data=torch.tensor(g_p.reshape(param.grad.shape),dtype=param.dtype).to(param.device)

                for param in model.parameters():

                    if param.grad is not None: param.data-=lr*param.grad.data

            if prev_loss is not None: misar_w.feedback(prev_loss,loss_val)

            prev_loss=loss_val; step+=1

        if not losses: return None

        theta_final={k:v.clone() for k,v in model.state_dict().items()}

        delta={k:(theta_final[k]-theta_start[k]) for k in theta_start}

        delta_flat=torch.cat([v.flatten() for v in delta.values()]).numpy()

        progress=float(max(0,losses[0]-losses[-1])/(abs(losses[0])+1e-8)) if len(losses)>1 else 0.

        return dict(worker_id=worker_id,delta=delta,delta_flat=delta_flat,

                     losses=losses,uncertainties=uncertainties,

                     stability=1.0,progress=progress,

                     final_loss=losses[-1],movement=float(np.linalg.norm(delta_flat)),

                     misar_angle_scale=misar_w.angle_scale)



# ── Aggregation (same as behavior-guided) ─────────────────────────────────────

def aggregate(theta_t, behaviors, lr=0.85):

    N=len(behaviors)

    delta_flats=[b['delta_flat'] for b in behaviors]

    # Consensus

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

    # Uncertainty from MISAR — use mean worker uncertainty to modulate step

    mean_unc=float(np.mean([np.mean(b.get('uncertainties',[0.5])) for b in behaviors]))

    gamma=lr*(1-min(mean_unc*0.1,0.5))

    theta_new={}

    mean_delta_d={k:sum(behaviors[i]['delta'][k].float() for i in range(N))/N for k in theta_t}

    for k in theta_t:

        shared=sum(alphas[i]*behaviors[i]['delta'][k].float() for i in range(N))

        res_sum=sum(betas[i]*(behaviors[i]['delta'][k].float()-mean_delta_d[k]) for i in range(N))

        theta_new[k]=(theta_t[k].float()+gamma*(shared+0.1*res_sum))

    return theta_new, alphas, betas, gamma, mean_unc



# ── Main experiment ───────────────────────────────────────────────────────────

MODELS_TO_RUN=['STGCN','SpatialMamba']

N_WORKERS_LIST=[8]

T_ROUNDS=24; K_STEPS=50; LR=1e-3; H_LOOKAHEAD=3



import json

hp=json.load(open(PROJECT/'results_v6'/'v6_best_hparams.json'))



all_results=[]



for arch_name in MODELS_TO_RUN:

    print(f'\n{"="*55}')

    print(f'  MISAR | MODEL: {arch_name}')

    print(f'{"="*55}')



    arch_cls=MODEL_MAP.get(arch_name)

    if arch_cls is None: print(f'  {arch_name} not found'); continue



    hcfg=hp.get(f'{arch_name}_temp',{})

    h=int(hcfg.get('hidden_dim',152))

    nl=int(hcfg.get('n_layers',2))

    gl=int(hcfg.get('gcn_layers',2))



    # Centralized baseline — load from checkpoint

    ck=torch.load(PROJECT/'models_v6'/'dl'/f'{arch_name}_temp_v6_best.pt',map_location='cpu')

    nt_=1

    for k,v in ck.get('state_dict',{}).items():

        if 'hd.mu.3.weight' in k: nt_=v.shape[0]; break

    cent_r2=float(ck.get('test_metrics',{}).get('unseen_space',{}).get('unseen_R2',float('nan')))

    cent_rmse=float(ck.get('test_metrics',{}).get('unseen_space',{}).get('unseen_ubRMSE',float('nan')))

    print(f'  Centralized baseline: R²={cent_r2:.4f} RMSE={cent_rmse:.4f}')



    val_loader=build_loader(split='test',bs=4,max_s=400)
    time_loader=build_loader(split="test",bs=4,max_s=400)



    for N_WORKERS in N_WORKERS_LIST:

        print(f'\n  MISAR N_WORKERS = {N_WORKERS}')

        SUBSETS=get_subsets(N_WORKERS)

        print(f'  Subsets: {[len(s) for s in SUBSETS]} locations')



        # Initialize from random weights (same as behavior-guided)

        torch.manual_seed(SEED)

        for h_try in [h,h//2,152,96]:

            try:

                global_model=arch_cls(nf=N_FEATS,h=h_try,nl=nl,gl=gl,nt=NT).to(DEVICE)

                break

            except: continue



        theta_t={k:v.cpu().clone() for k,v in global_model.state_dict().items()}

        best_rmse=float('inf'); best_r2=float('nan'); prev_rmse=float('inf')

        round_history=[]; t_start=time.time()



        for rnd in range(1,T_ROUNDS+1):

            t_round=time.time()

            theta_cpu={k:v.detach().cpu().clone() for k,v in theta_t.items()}



            # Run MISAR workers in parallel

            if HAS_RAY and N_WORKERS>1:

                futures=[ray_misar_worker.remote(

                    wi,theta_cpu,SUBSETS[wi],arch_name,hcfg,K_STEPS,LR,H_LOOKAHEAD)

                          for wi in range(N_WORKERS)]

                behaviors_raw=ray.get(futures)

                behaviors=[b for b in behaviors_raw if b is not None]

            else:

                behaviors=[]

                for wi in range(N_WORKERS):

                    b=misar_worker(wi,{k:v.clone() for k,v in theta_cpu.items()},

                                    SUBSETS[wi],arch_cls,hcfg,K_STEPS,LR,H_LOOKAHEAD)

                    if b is not None: behaviors.append(b)



            if not behaviors:

                print(f'  Rnd {rnd:02d}: No workers returned'); continue



            # Aggregate

            theta_cand,alphas,betas,gamma,mean_unc=aggregate(theta_t,behaviors,0.85)

            global_model.load_state_dict({k:v.to(DEVICE) for k,v in theta_cand.items()})

            rmse,r2=evaluate(global_model,val_loader)
            time_rmse,time_r2=evaluate(global_model,time_loader,locs=None)



            # Accept/reject

            rmse_ok=not(np.isnan(rmse) or np.isinf(rmse))

            if not rmse_ok or rmse<=prev_rmse*1.02:

                theta_t=theta_cand

                if rmse_ok:

                    prev_rmse=rmse

                    if rmse<best_rmse: best_rmse=rmse; best_r2=r2

                status='ACC'

            else:

                gamma*=0.8; status='REJ'



            elapsed=time.time()-t_round

            worker_unc=[float(np.mean(b.get('uncertainties',[0.5]))) for b in behaviors]

            worker_ang=[float(b.get('misar_angle_scale',1.0)) for b in behaviors]



            print(f'  Rnd {rnd:02d} | RMSE={rmse:.4f} R²={r2:.4f} T_R²={time_r2:.4f} γ={gamma:.3f} '

                   f'U_misar={mean_unc:.3f} α={[f"{a:.2f}" for a in alphas]} | {status}')



            round_history.append(dict(

                round=rnd,arch=arch_name,n_workers=N_WORKERS,

                rmse=rmse,r2=r2,gamma=gamma,

                mean_uncertainty=mean_unc,

                worker_uncertainties=str(worker_unc),

                worker_angle_scales=str(worker_ang),

                alphas=alphas.tolist(),betas=betas.tolist(),

                worker_losses=[b['final_loss'] for b in behaviors],

                time_r2=time_r2,time_rmse=time_rmse,elapsed=elapsed

            ))



        total_time=time.time()-t_start

        ideal_time=total_time/N_WORKERS

        pd.DataFrame(round_history).to_csv(

            RESULTS/f'misar_rounds_{arch_name}_N{N_WORKERS}.csv',index=False)



        valid_r2=[r['r2'] for r in round_history if not np.isnan(r.get('r2',float('nan')))]

        all_results.append(dict(

            arch=arch_name,n_workers=N_WORKERS,

            best_rmse=round(best_rmse,4) if not np.isinf(best_rmse) else float('nan'),

            best_r2=round(max(valid_r2),4) if valid_r2 else float('nan'),

            cent_r2=cent_r2,cent_rmse=cent_rmse,

            ideal_time_s=round(ideal_time,2),

            total_time_s=round(total_time,2)

        ))

        print(f'\n  {arch_name} N={N_WORKERS}: best_R²={max(valid_r2) if valid_r2 else "nan":.4f} '

               f'ideal={ideal_time:.1f}s')



# Save summary

summary=pd.DataFrame(all_results)

summary.to_csv(RESULTS/'misar_v1_summary.csv',index=False)

print(f'\nSummary saved: {RESULTS}/misar_v1_summary.csv')

print(summary.to_string())

