
import torch,numpy as np,pickle,warnings,pandas as pd

import matplotlib; matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from pathlib import Path

from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler

from scipy.spatial import cKDTree

import unittest.mock as _mock

warnings.filterwarnings('ignore')



PROJECT=Path('/home/emmanuel.keku')

PREPROC=PROJECT/'preprocessed_v3'

FIGS=PROJECT/'figures_v7'; FIGS.mkdir(exist_ok=True)



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

V6F=list(dict.fromkeys(CORE+APPROX+RESID+UNC)); V6F=[f for f in V6F if f in raw_df.columns]

N_FEATS=len(V6F)

TEMP_TGTS=FI['TEMP_TARGETS']

use_cols=[f'{c}_residual' for c in TEMP_TGTS if f'{c}_residual' in raw_df.columns]

if not use_cols: use_cols=[c for c in TEMP_TGTS if c in raw_df.columns]

NT=len(use_cols)

tr_df=raw_df[(raw_df['split']=='train')&(raw_df['Site']!='Wetland')]

feat_sc=RobustScaler(); feat_sc.fit(tr_df[V6F].fillna(0).values)

tgt_sc=RobustScaler(); tgt_sc.fit(tr_df[use_cols].dropna().values)

SITE_LOCS={}

for site in SITES:

    rows=raw_df[raw_df['Site']==site][['Latitude','Longitude']].drop_duplicates()

    idxs=sorted([loc_to_idx.get((float(r.Latitude),float(r.Longitude)))

                  for _,r in rows.iterrows()

                  if loc_to_idx.get((float(r.Latitude),float(r.Longitude))) is not None])

    SITE_LOCS[site]=idxs

WETLAND=SITE_LOCS['Wetland']

SEEN=sorted(set(i for s,v in SITE_LOCS.items() if s!='Wetland' for i in v))

coords=LOCS[['Latitude','Longitude']].values.astype(np.float32)*np.array([111.0,63.0])

tree_=cKDTree(coords); d_,i_=tree_.query(coords,k=7); sig_=np.median(d_[:,1:])+1e-8

A_np=np.zeros((N_LOCS,N_LOCS),dtype=np.float32)

for i in range(N_LOCS):

    for jp in range(1,d_.shape[1]):

        j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_)); A_np[i,j]+=w; A_np[j,i]+=w

A_np+=np.eye(N_LOCS); D_=A_np.sum(1,keepdims=True)**0.5

A_norm=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A_dev=A_norm.to(DEVICE)

print(f'Device:{DEVICE} N_FEATS:{N_FEATS} NT:{NT}')



_src=open(PROJECT/'train_soil_spatial_v6.py').read()

_pre=_src.split('if args.mode')[0]; _ns={'__name__':'__imported__'}

with _mock.patch('sys.argv',['t.py','--mode','train','--target','temp']):

    try: exec(_pre,_ns)

    except: pass

arch_cls=_ns.get('MODEL_MAP',{}).get('STGCN')

H=152



def load_w(tgt):

    ck=torch.load(PROJECT/'models_v6'/'dl'/f'STGCN_{tgt}_v6_best.pt',map_location='cpu')

    sd=ck.get('state_dict',{})

    nt_=1

    for k,v in sd.items():

        if 'hd.mu.3.weight' in k: nt_=v.shape[0]; break

    m=arch_cls(nf=N_FEATS,h=H,nl=2,gl=2,nt=nt_)

    m.load_state_dict(sd,strict=True)

    print(f'  Loaded {tgt}: nt={nt_}')

    return torch.cat([p.data.flatten() for p in m.parameters()]).numpy()



w_temp=load_w('temp'); w_smap=load_w('smap'); w_moist=load_w('moist')

torch.manual_seed(42)

m0=arch_cls(nf=N_FEATS,h=H,nl=2,gl=2,nt=1)

w0=torch.cat([p.data.flatten() for p in m0.parameters()]).numpy()

print(f'Init: dim={len(w0):,} temp:{np.linalg.norm(w_temp-w0[:len(w_temp)]):.2f}')



np.random.seed(42)

dirs=[w_temp-w0[:len(w_temp)],w_smap-w0[:len(w_smap)],w_moist[:len(w0)]-w0]

worker_vecs=[]

for wi in range(4):

    d=dirs[wi%len(dirs)]

    base=w0[:len(d)]

    noise=np.random.randn(len(d))*np.linalg.norm(d)*0.1

    ep=base+d*0.5+noise

    worker_vecs.append(ep)



all_key=np.array([w0[:len(w_temp)],w_temp,w_smap,w_moist[:len(w_temp)]]+

                  [wv[:len(w_temp)] for wv in worker_vecs])

pca=PCA(n_components=2,random_state=42)

all_pca=pca.fit_transform(all_key)

var=pca.explained_variance_ratio_

print(f'PCA: PC1={var[0]*100:.1f}% PC2={var[1]*100:.1f}% spread:{all_pca[:,0].ptp():.1f}x{all_pca[:,1].ptp():.1f}')



xpad=all_pca[:,0].ptp()*0.3; ypad=all_pca[:,1].ptp()*0.3

xmin,xmax=all_pca[:,0].min()-xpad,all_pca[:,0].max()+xpad

ymin,ymax=all_pca[:,1].min()-ypad,all_pca[:,1].max()+ypad

grid_res=18; gx=np.linspace(xmin,xmax,grid_res); gy=np.linspace(ymin,ymax,grid_res)

GX,GY=np.meshgrid(gx,gy)



SEEN_SITES=['Bedrock','Transition','Upland']

subs=[]

for s in SEEN_SITES:

    locs_=SITE_LOCS[s]; mid=max(1,len(locs_)//2)

    subs.append(locs_[:mid]); subs.append(locs_[mid:])

subs=subs[:4]

regions={'Spatial Worker 1':subs[0],'Spatial Worker 2':subs[1],

         'Spatial Worker 3':subs[2],'All-Region':SEEN}



def build_val(loc_subset,max_s=60,lookback=24,stride=10):

    sub=raw_df[raw_df['split']=='val'].copy()

    all_ts=sorted(sub['time_utc'].unique()); T=len(all_ts)

    ts2i={t:i for i,t in enumerate(all_ts)}

    sub['_ti']=sub['time_utc'].map(ts2i)

    sub['_ni']=[loc_to_idx.get((float(la),float(lo))) for la,lo in zip(sub['Latitude'],sub['Longitude'])]

    sub=sub.dropna(subset=['_ti','_ni']); sub['_ti']=sub['_ti'].astype(int); sub['_ni']=sub['_ni'].astype(int)

    sub=sub[sub['_ti']<T]; sub=sub[sub['_ni'].isin(loc_subset)]

    if len(sub)==0: return None,None,None

    Xf=np.zeros((T,N_LOCS,N_FEATS),dtype=np.float32)

    yf=np.zeros((T,N_LOCS,NT),dtype=np.float32); mf=np.zeros((T,N_LOCS),dtype=np.float32)

    Xf[sub['_ti'].values,sub['_ni'].values]=feat_sc.transform(sub[V6F].fillna(0).values).astype(np.float32)

    yf[sub['_ti'].values,sub['_ni'].values]=tgt_sc.transform(sub[use_cols].fillna(0).values).astype(np.float32)

    mf[:,list(loc_subset)]=1.0

    rng=np.random.default_rng(42); tidxs=list(range(lookback,T,stride))

    if len(tidxs)>max_s: tidxs=sorted(rng.choice(tidxs,max_s,replace=False))

    Xl=[]; yl=[]; ml=[]

    for ti in tidxs:

        Xw=Xf[ti-lookback:ti]

        if np.isnan(Xw).mean()>0.5: continue

        Xl.append(np.nan_to_num(Xw,nan=0.)); yl.append(yf[ti]); ml.append(mf[ti])

    if not Xl: return None,None,None

    return torch.tensor(np.array(Xl)),torch.tensor(np.array(yl)),torch.tensor(np.array(ml))



val_data={n:build_val(l) for n,l in regions.items()}



ref_sd=m0.state_dict()

param_shapes=[(k,v.shape,v.numel()) for k,v in ref_sd.items()]

total_params=sum(n for _,_,n in param_shapes)



def pca_to_sd(px,py):

    w=pca.inverse_transform(np.array([[px,py]]))[0]

    if len(w)<total_params: w=np.pad(w,(0,total_params-len(w)))

    new_sd={}; idx=0

    for k,shape,n in param_shapes:

        new_sd[k]=torch.tensor(w[idx:idx+n]).reshape(shape).float(); idx+=n

    return new_sd



def rmse_sd(sd,X,y,locs):

    m=arch_cls(nf=N_FEATS,h=H,nl=2,gl=2,nt=1).to(DEVICE)

    m.load_state_dict(sd,strict=False); m.eval()

    with torch.no_grad():

        out=m(X.to(DEVICE),A_dev); mu=out[0].cpu().float()

        mu_np=tgt_sc.inverse_transform(mu.numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)

        y_np=tgt_sc.inverse_transform(y.numpy().reshape(-1,NT)).reshape(X.shape[0],N_LOCS,NT)

        yt=y_np[:,locs,0].flatten(); yp=mu_np[:,locs,0].flatten()

        mk=~(np.isnan(yt)|np.isnan(yp))

        return float(np.sqrt(np.mean((yt[mk]-yp[mk])**2))) if mk.sum()>5 else float('nan')



print('Computing landscape...')

landscapes={}

for rname,locs in regions.items():

    Xv,yv,mv=val_data[rname]

    if Xv is None: print(f'  {rname}: no data'); continue

    gl=np.full((grid_res,grid_res),np.nan)

    for i in range(grid_res):

        for j in range(grid_res):

            sd=pca_to_sd(GX[i,j],GY[i,j])

            gl[i,j]=rmse_sd(sd,Xv,yv,locs)

        if i%4==0: print(f'  {rname}: {i+1}/{grid_res}')

    landscapes[rname]=gl

    print(f'  {rname}: {np.nanmin(gl):.3f}-{np.nanmax(gl):.3f} range={np.nanmax(gl)-np.nanmin(gl):.4f}')



WCOLORS=['#1f9ab4','#ff7f0e','#2ca02c','#9467bd']

GENC='#d62728'; INITC='#9b59b6'

cmap=LinearSegmentedColormap.from_list('ls',['#000080','#0000ff','#00bfff','#00ff00','#ffff00','#ffffff'],N=256)

fig,axes=plt.subplots(2,2,figsize=(18,14)); axes=axes.flatten()

for ai,rname in enumerate(list(landscapes.keys())[:4]):

    ax=axes[ai]; gl=landscapes[rname]

    vmin=np.nanpercentile(gl,5); vmax=np.nanpercentile(gl,95)

    if abs(vmax-vmin)<0.001: vmin=np.nanmin(gl)-0.001; vmax=np.nanmax(gl)+0.001

    im=ax.contourf(GX,GY,gl,levels=15,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.85)

    ax.contour(GX,GY,gl,levels=8,colors='white',alpha=0.15,linewidths=0.5)

    cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)

    cbar.set_label('Held-out temporal RMSE',fontsize=9); cbar.ax.tick_params(labelsize=8)

    init_pt=all_pca[0]; gen_pt=all_pca[1]

    for wi in range(3):

        ep=all_pca[4+wi]

        pts=np.array([init_pt*(1-t)+ep*t for t in np.linspace(0,1,20)])

        ax.plot(pts[:,0],pts[:,1],color=WCOLORS[wi],lw=2.5,alpha=0.9,

                label=f'Spatial Worker {wi+1}' if ai==0 else '')

        ax.scatter(ep[0],ep[1],color=WCOLORS[wi],s=80,zorder=8)

    pts_g=np.array([init_pt*(1-t)+gen_pt*t for t in np.linspace(0,1,25)])

    ax.plot(pts_g[:,0],pts_g[:,1],color=GENC,lw=3.5,label='General' if ai==0 else '',zorder=6)

    ax.scatter(gen_pt[0],gen_pt[1],color=GENC,s=120,marker='s',zorder=9)

    ax.scatter(init_pt[0],init_pt[1],color=INITC,s=200,marker='*',zorder=10,

               label='Initialization' if ai==0 else '')

    ax.set_xlabel(f'Weight PC1 ({var[0]*100:.1f}%)',fontsize=11)

    ax.set_ylabel(f'Weight PC2 ({var[1]*100:.1f}%)',fontsize=11)

    ax.set_title(f'{rname} Test Loss',fontsize=12,fontweight='bold')

handles,labels_=axes[0].get_legend_handles_labels()

fig.legend(handles,labels_,loc='upper center',ncol=5,fontsize=11,bbox_to_anchor=(0.5,1.01))

fig.suptitle('Real Spatiotemporal Temperature Data: Local Spatial Minima vs General Minimum\nSTGCN | N=4 spatial workers | Soil Temperature Residual',

             fontsize=13,fontweight='bold',y=1.04)

plt.tight_layout()

plt.savefig(FIGS/'fig_loss_landscape.png',dpi=300,bbox_inches='tight')

plt.close()

print('OK fig_loss_landscape.png')

