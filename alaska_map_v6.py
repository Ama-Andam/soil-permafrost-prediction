"""
alaska_map_v6.py — Full Alaska Spatial Field Maps
256 study stations + 12 Alaska city anchors from Alaska_2000-2025.csv
Cartopy LambertConformal projection — proper Alaska state shape
IDW interpolation over full Alaska domain
"""
import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

PROJECT = Path("/home/emmanuel.keku")
PREPROC = PROJECT / "preprocessed_v3"
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6" / "manuscript"
FIGS.mkdir(parents=True, exist_ok=True)

print("="*65)
print("  ALASKA SPATIAL FIELD MAPS v6")
print("="*65)

PROJ = ccrs.LambertConformal(central_longitude=-154.0, central_latitude=62.0,
                               standard_parallels=(55.0, 65.0))
PC   = ccrs.PlateCarree()
LON_MIN,LON_MAX,LAT_MIN,LAT_MAX = -170.0,-129.0,54.0,72.0

with open(PREPROC/"feature_info.pkl","rb") as f: FI=pickle.load(f)
LOCS   = pd.DataFrame(FI["LOCATIONS"])
N_LOCS = FI["N_LOCS"]
SITES  = FI["SITES"]
ALL_TGTS=FI["ALL_TARGETS"]
raw_df = pd.read_csv(PREPROC/"master_processed.csv", parse_dates=["time_utc"])
res_df = pd.read_csv(RESULTS/"v6_results_corrected.csv")
ak_df  = pd.read_csv(PROJECT/"Alaska_2000-2025.csv", parse_dates=["date"])
print(f"  Study stations: {N_LOCS} | Alaska cities: {ak_df['city'].nunique()}")

CITY_COORDS = {
    'Anchorage, Alaska':          (61.2181,-149.9003),
    'Barrow (Utqiaġvik), Alaska': (71.2906,-156.7887),
    'Bethel, Alaska':             (60.7922,-161.7558),
    'Fairbanks, Alaska':          (64.8378,-147.7164),
    'Homer, Alaska':              (59.6425,-151.5483),
    'Juneau, Alaska':             (58.3005,-134.4197),
    'Kenai, Alaska':              (60.5544,-151.2583),
    'Ketchikan, Alaska':          (55.3422,-131.6461),
    'Kodiak, Alaska':             (57.7900,-152.4072),
    'Palmer, Alaska':             (61.5992,-149.1147),
    'Sitka, Alaska':              (57.0531,-135.3300),
    'Wasilla, Alaska':            (61.5814,-149.4394),
}
SITE_COLORS={"Bedrock":"#3498DB","Transition":"#E67E22",
              "Upland":"#27AE60","Wetland":"#E74C3C"}
TGT_CMAPS ={"temp":"RdYlBu_r","smap":"plasma","moist":"YlGnBu"}
TGT_LABELS={"temp":"Soil Temp (0-7cm) [C]","smap":"SMAP Temp L1 [K]",
             "moist":"Soil Moisture (0-7cm) [m3/m3]"}
TGT_UNITS ={"temp":"C","smap":"K","moist":"m3/m3"}

loc_site_map={}
for site in SITES:
    rows=raw_df[raw_df["Site"]==site][["Latitude","Longitude"]].drop_duplicates()
    for _,r in rows.iterrows():
        loc_site_map[(round(float(r.Latitude),4),round(float(r.Longitude),4))]=site
LOCS["Site"]=[loc_site_map.get((round(float(r.Latitude),4),
               round(float(r.Longitude),4)),"?") for _,r in LOCS.iterrows()]

GRID_RES=250
grid_lons,grid_lats=np.meshgrid(
    np.linspace(LON_MIN,LON_MAX,GRID_RES),
    np.linspace(LAT_MIN,LAT_MAX,GRID_RES))

def idw(lats,lons,values,power=2,k=8):
    valid=~np.isnan(values)
    if valid.sum()<3: return np.full(grid_lats.shape,np.nan)
    pts=np.column_stack([lats[valid]*111.0,lons[valid]*63.0])
    gpts=np.column_stack([grid_lats.ravel()*111.0,grid_lons.ravel()*63.0])
    tree=cKDTree(pts)
    dists,idxs=tree.query(gpts,k=min(k,len(pts)))
    dists=np.maximum(dists,1e-8)
    w=1.0/dists**power; w/=w.sum(axis=1,keepdims=True)
    return (w*values[valid][idxs]).sum(axis=1).reshape(grid_lats.shape)

def make_ak_ax(fig,rows,cols,idx,title=""):
    ax=fig.add_subplot(rows,cols,idx,projection=PROJ)
    ax.set_extent([LON_MIN,LON_MAX,LAT_MIN,LAT_MAX],crs=PC)
    ax.add_feature(cfeature.LAND,     facecolor="#F0EDE8",edgecolor="none")
    ax.add_feature(cfeature.OCEAN,    facecolor="#D6EAF8",edgecolor="none")
    ax.add_feature(cfeature.LAKES,    facecolor="#D6EAF8",edgecolor="none",alpha=0.5)
    ax.add_feature(cfeature.RIVERS,   edgecolor="#AED6F1",linewidth=0.3,alpha=0.4)
    ax.add_feature(cfeature.STATES,   edgecolor="#AAAAAA",linewidth=0.5,facecolor="none")
    ax.add_feature(cfeature.COASTLINE,edgecolor="#666666",linewidth=0.7)
    ax.gridlines(crs=PC,draw_labels=False,linewidth=0.3,
                  color="grey",alpha=0.3,linestyle="--")
    if title: ax.set_title(title,fontsize=10,fontweight="bold",pad=5)
    return ax

def plot_field(ax,all_lats,all_lons,all_vals,cmap,vmin,vmax):
    field=idw(all_lats,all_lons,all_vals)
    field=np.clip(field,vmin,vmax)
    im=ax.pcolormesh(grid_lons,grid_lats,field,cmap=cmap,vmin=vmin,vmax=vmax,
                      transform=PC,shading="gouraud",rasterized=True,zorder=2)
    ax.scatter(LOCS["Longitude"].values,LOCS["Latitude"].values,
                c="white",s=3,alpha=0.5,zorder=4,transform=PC,edgecolors="none")
    for city,(lat,lon) in CITY_COORDS.items():
        ax.scatter(lon,lat,c="white",s=20,zorder=5,transform=PC,
                    edgecolors="#333",linewidths=0.5,marker="^")
    return im

import torch
from sklearn.preprocessing import RobustScaler

CYCLICAL=[c for c in raw_df.columns if any(c.startswith(p) for p in ["sin_","cos_"])]
SNAP=FI["SNAP_FEATURES"]; CORE=[f for f in SNAP if f not in CYCLICAL and f in raw_df.columns]
APPROX=[f"{t}_approx" for t in ALL_TGTS if f"{t}_approx" in raw_df.columns]
RESIDUAL=[f"{t}_residual" for t in ALL_TGTS if f"{t}_residual" in raw_df.columns]
UNC_VARS=[]
for feat in CORE[:8]:
    vc=f"{feat}_unc_var"
    if vc not in raw_df.columns: raw_df[vc]=np.where(raw_df[feat].isna(),1.0,0.01)
    UNC_VARS.append(vc)
V6F=list(dict.fromkeys(CORE+APPROX+RESIDUAL+UNC_VARS))
V6F=[f for f in V6F if f in raw_df.columns]
tr_df=raw_df[raw_df["split"]=="train"]
feat_sc=RobustScaler(); feat_sc.fit(tr_df[V6F].fillna(0).values)

coords=LOCS[["Latitude","Longitude"]].values.astype(np.float32)
sc_=coords*np.array([111.0,63.0]); tree_=cKDTree(sc_)
d_,i_=tree_.query(sc_,k=7); sig_=np.median(d_[:,1:])+1e-8
A_np=np.zeros((N_LOCS,N_LOCS),dtype=np.float32)
for i in range(N_LOCS):
    for jp in range(1,d_.shape[1]):
        j=i_[i,jp]; w=float(np.exp(-d_[i,jp]/sig_))
        A_np[i,j]+=w; A_np[j,i]+=w
A_np+=np.eye(N_LOCS); D_=A_np.sum(1,keepdims=True)**0.5
A_norm=torch.tensor((A_np/(D_*D_.T+1e-8)).astype(np.float32))
loc_to_idx={(float(r.Latitude),float(r.Longitude)):i for i,r in LOCS.iterrows()}
TGT_RAW={"temp":FI["TEMP_TARGETS"],"smap":FI["SMAP_TARGETS"],"moist":FI["MOIST_TARGETS"]}

def get_preds(arch,tgt,month=7,year=2025):
    raw_cols=TGT_RAW[tgt]
    res_cols=[f"{c}_residual" for c in raw_cols if f"{c}_residual" in raw_df.columns]
    use_cols=res_cols if res_cols else [c for c in raw_cols if c in raw_df.columns]
    approx_c=[f"{c}_approx" for c in raw_cols if f"{c}_approx" in raw_df.columns]
    if not use_cols: return None
    tgt_sc=RobustScaler(); tgt_sc.fit(tr_df[use_cols].dropna().values)
    test_df2=raw_df[raw_df["split"]=="test"].copy()
    all_ts2=sorted(test_df2["time_utc"].unique()); T2=len(all_ts2)
    ts_to_i={t:i for i,t in enumerate(all_ts2)}
    ti_target=None
    for i,t in enumerate(all_ts2):
        pt=pd.Timestamp(t)
        if pt.month==month and pt.year==year: ti_target=i; break
    if ti_target is None or ti_target<24: ti_target=T2//2
    ts_label=pd.Timestamp(all_ts2[ti_target]).strftime("%B %Y")
    test_df2["_ti"]=test_df2["time_utc"].map(ts_to_i)
    test_df2["_ni"]=[loc_to_idx.get((float(la),float(lo)))
                      for la,lo in zip(test_df2["Latitude"],test_df2["Longitude"])]
    test_df2=test_df2.dropna(subset=["_ti","_ni"])
    test_df2["_ti"]=test_df2["_ti"].astype(int); test_df2["_ni"]=test_df2["_ni"].astype(int)
    test_df2=test_df2[test_df2["_ti"]<T2]
    Xf=np.zeros((T2,N_LOCS,len(V6F)),dtype=np.float32)
    yf=np.zeros((T2,N_LOCS,len(use_cols)),dtype=np.float32)
    af=np.zeros((T2,N_LOCS,max(len(approx_c),1)),dtype=np.float32)
    Xf[test_df2["_ti"].values,test_df2["_ni"].values]=\
        feat_sc.transform(test_df2[V6F].fillna(0).values).astype(np.float32)
    yf[test_df2["_ti"].values,test_df2["_ni"].values]=\
        tgt_sc.transform(test_df2[use_cols].fillna(0).values).astype(np.float32)
    if approx_c:
        af[test_df2["_ti"].values,test_df2["_ni"].values]=\
            test_df2[approx_c].fillna(0).values.astype(np.float32)
    ckpt_p=PROJECT/"models_v6"/"dl"/f"{arch}_{tgt}_v6_best.pt"
    if not ckpt_p.exists(): return None
    ckpt=torch.load(ckpt_p,map_location="cpu")
    exec_ns={}
    try:
        exec(open(PROJECT/"train_soil_spatial_v6.py").read()
             .split("if args.mode")[0],exec_ns)
        arch_cls=exec_ns.get("MODEL_MAP",{}).get(arch)
    except: arch_cls=None
    if arch_cls is None: return None
    sd=ckpt.get("state_dict",ckpt.get("model_state_dict",{}))
    _h=96
    for k,v in sd.items():
        if "hd.mu.0.weight" in k: _h=int(v.shape[1]); break
    _hcfg=ckpt.get("hcfg",{}); _nl=int(_hcfg.get("n_layers",2)); _gl=int(_hcfg.get("gcn_layers",2))
    _nf=ckpt.get("n_feats",len(V6F)); loaded=False
    for h_try in [_h,96,128,64,152,192]:
        try:
            model=arch_cls(nf=_nf,h=h_try,nl=_nl,gl=_gl,nt=len(use_cols))
            model.load_state_dict(sd,strict=True); loaded=True; break
        except:
            try:
                model=arch_cls(nf=_nf,h=h_try,nl=_nl,gl=_gl,nt=len(use_cols))
                model.load_state_dict(sd,strict=False); loaded=True; break
            except: continue
    if not loaded: return None
    model.eval(); LB=24
    with torch.no_grad():
        Xw=torch.tensor(Xf[ti_target-LB:ti_target]).unsqueeze(0)
        out=model(Xw,A_norm.unsqueeze(0)); mu=out[0]
        lsv=out[1] if len(out)>1 else torch.zeros_like(out[0])
        mu_np=tgt_sc.inverse_transform(
            mu[0].float().numpy().reshape(-1,len(use_cols))).reshape(N_LOCS,len(use_cols))
        y_np=tgt_sc.inverse_transform(
            yf[ti_target].reshape(-1,len(use_cols))).reshape(N_LOCS,len(use_cols))
        sig_np=np.exp(0.5*lsv[0].float().numpy().reshape(N_LOCS,len(use_cols)))
    av=af[ti_target,:,0] if approx_c else np.zeros(N_LOCS)
    return dict(lats=LOCS["Latitude"].values,lons=LOCS["Longitude"].values,
                preds=mu_np[:,0]+av,trues=y_np[:,0]+av,
                sigma=sig_np[:,0],ts_label=ts_label)

def get_city_vals(tgt,month=7,year=2025):
    col_map={"temp":"soil_temperature_0_to_7cm",
              "smap":"soil_temperature_0_to_7cm",
              "moist":"soil_moisture_0_to_7cm"}
    col=col_map[tgt]
    if col not in ak_df.columns: return None,None,None
    sub=ak_df.copy()
    sub["_m"]=pd.to_datetime(sub["date"]).dt.month
    sub["_y"]=pd.to_datetime(sub["date"]).dt.year
    sub2=sub[(sub["_m"]==month)&(sub["_y"]==year)]
    if sub2.empty: sub2=sub[sub["_m"]==month]
    city_vals=sub2.groupby("city")[col].mean()
    lats=[]; lons=[]; vals=[]
    for city,(lat,lon) in CITY_COORDS.items():
        if city in city_vals.index:
            lats.append(lat); lons.append(lon); vals.append(float(city_vals[city]))
    return np.array(lats),np.array(lons),np.array(vals)

def combine(data,c_lats,c_lons,c_vals,use_preds=True):
    v=data["preds"] if use_preds else data["trues"]
    return (np.concatenate([data["lats"],c_lats]),
            np.concatenate([data["lons"],c_lons]),
            np.concatenate([v,c_vals]))

plt.rcParams.update({"figure.dpi":300,"font.family":"DejaVu Sans","font.size":11,
                      "axes.titlesize":11})

# MAP_01: Summer all targets
print("\n  MAP_01: Summer...")
fig=plt.figure(figsize=(24,9)); fig.patch.set_facecolor("white")
for ti,tgt in enumerate(["temp","smap","moist"]):
    sub=res_df[res_df["Target"]==tgt]
    if sub.empty: make_ak_ax(fig,1,3,ti+1); continue
    ba=sub.loc[sub["Space_R2"].idxmax(),"Model"]
    print(f"    [{tgt}] {ba}")
    data=get_preds(ba,tgt,7,2025)
    cl,clo,cv=get_city_vals(tgt,7,2025)
    ax=make_ak_ax(fig,1,3,ti+1)
    if data is not None and cl is not None:
        al,alo,av=combine(data,cl,clo,cv)
        vmin=np.nanpercentile(av,2); vmax=np.nanpercentile(av,98)
        im=plot_field(ax,al,alo,av,TGT_CMAPS[tgt],vmin,vmax)
        cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=0.04,shrink=0.85,aspect=28)
        cb.set_label(TGT_LABELS[tgt],fontsize=9); cb.ax.tick_params(labelsize=8)
        ax.set_title(f"{TGT_LABELS[tgt]}\n[{ba}] | {data['ts_label']}",
                      fontsize=10,fontweight="bold")
    else: print(f"    X {tgt}")
fig.suptitle("Predicted soil variables | Alaska | v6 Distributed Spatial AI\n"
              "256 study stations + 12 Alaska city anchors | IDW interpolation | Summer 2025",
              fontsize=13,fontweight="bold",y=1.01)
plt.tight_layout()
plt.savefig(FIGS/"MAP_01_predicted_summer.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print("    OK MAP_01_predicted_summer.png")

# MAP_02: Winter all targets
print("\n  MAP_02: Winter...")
fig=plt.figure(figsize=(24,9)); fig.patch.set_facecolor("white")
for ti,tgt in enumerate(["temp","smap","moist"]):
    sub=res_df[res_df["Target"]==tgt]
    if sub.empty: continue
    ba=sub.loc[sub["Space_R2"].idxmax(),"Model"]
    data=get_preds(ba,tgt,1,2025); cl,clo,cv=get_city_vals(tgt,1,2025)
    ax=make_ak_ax(fig,1,3,ti+1)
    if data is not None and cl is not None:
        al,alo,av=combine(data,cl,clo,cv)
        vmin=np.nanpercentile(av,2); vmax=np.nanpercentile(av,98)
        im=plot_field(ax,al,alo,av,TGT_CMAPS[tgt],vmin,vmax)
        cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=0.04,shrink=0.85,aspect=28)
        cb.set_label(TGT_LABELS[tgt],fontsize=9); cb.ax.tick_params(labelsize=8)
        ax.set_title(f"{TGT_LABELS[tgt]}\n[{ba}] | {data['ts_label']}",fontsize=10,fontweight="bold")
fig.suptitle("Predicted soil variables | Alaska | v6 Distributed Spatial AI\n"
              "256 study stations + 12 Alaska city anchors | IDW interpolation | Winter 2025",
              fontsize=13,fontweight="bold",y=1.01)
plt.tight_layout()
plt.savefig(FIGS/"MAP_02_predicted_winter.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print("    OK MAP_02_predicted_winter.png")

# MAP_03: Seasonal 2x3
print("\n  MAP_03: Seasonal 2x3...")
fig=plt.figure(figsize=(24,16)); fig.patch.set_facecolor("white")
for ri,(month,year,slbl) in enumerate([(7,2025,"Summer 2025"),(1,2025,"Winter 2025")]):
    for ti,tgt in enumerate(["temp","smap","moist"]):
        sub=res_df[res_df["Target"]==tgt]
        if sub.empty: continue
        ba=sub.loc[sub["Space_R2"].idxmax(),"Model"]
        data=get_preds(ba,tgt,month,year); cl,clo,cv=get_city_vals(tgt,month,year)
        ax=make_ak_ax(fig,2,3,ri*3+ti+1)
        if data is not None and cl is not None:
            al,alo,av=combine(data,cl,clo,cv)
            vmin=np.nanpercentile(av,2); vmax=np.nanpercentile(av,98)
            im=plot_field(ax,al,alo,av,TGT_CMAPS[tgt],vmin,vmax)
            cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=0.04,shrink=0.85,aspect=22)
            cb.ax.tick_params(labelsize=8)
            if ri==0: ax.set_title(TGT_LABELS[tgt],fontsize=11,fontweight="bold")
        if ti==0:
            ax.text(-0.08,0.5,slbl,transform=ax.transAxes,fontsize=11,
                     fontweight="bold",va="center",rotation=90)
fig.suptitle("Mean predicted soil variables | Alaska | v6 Distributed Spatial AI\n"
              "256 study stations + 12 city anchors | IDW spatial interpolation",
              fontsize=14,fontweight="bold",y=1.01)
plt.tight_layout()
plt.savefig(FIGS/"MAP_03_seasonal_comparison.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print("    OK MAP_03_seasonal_comparison.png")

# MAP_04: Site locations context
print("\n  MAP_04: Site locations...")
fig=plt.figure(figsize=(16,10)); fig.patch.set_facecolor("white")
ax=make_ak_ax(fig,1,1,1,"Study area locations | Alaska Permafrost | v6 Distributed Spatial AI")
for city,(lat,lon) in CITY_COORDS.items():
    ax.scatter(lon,lat,c="#555555",s=40,zorder=4,transform=PC,
                marker="^",edgecolors="white",linewidths=0.5)
    short=city.replace(", Alaska","").split(" (")[0]
    ax.text(lon+0.4,lat+0.2,short,transform=PC,fontsize=7,color="#333333",zorder=5)
for site in SITES:
    mask=LOCS["Site"]==site
    is_wl=site=="Wetland"
    ax.scatter(LOCS.loc[mask,"Longitude"].values,
                LOCS.loc[mask,"Latitude"].values,
                c=SITE_COLORS[site],s=80 if is_wl else 60,
                marker="D" if is_wl else "o",alpha=0.9,zorder=5,transform=PC,
                edgecolors="black" if is_wl else "white",
                linewidths=1.5 if is_wl else 0.5,
                label=f"{site} ({'unseen' if is_wl else 'seen'}, n={mask.sum()})")
ax.legend(loc="lower left",fontsize=9,framealpha=0.9,
           title="Ecological sites",title_fontsize=9)
ax.plot([-152.5,-148.0,-148.0,-152.5,-152.5],[65.5,65.5,69.5,69.5,65.5],
         transform=PC,color="red",linewidth=2,linestyle="--",zorder=6)
ax.text(-152.4,65.2,"Study area",transform=PC,fontsize=9,color="red",fontweight="bold",zorder=7)
plt.tight_layout()
plt.savefig(FIGS/"MAP_04_site_locations.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print("    OK MAP_04_site_locations.png")

# MAP_05: Observed vs Predicted
print("\n  MAP_05: Obs vs Pred...")
tgt="temp"
sub=res_df[res_df["Target"]==tgt]
if not sub.empty:
    ba=sub.loc[sub["Space_R2"].idxmax(),"Model"]
    data=get_preds(ba,tgt,7,2025); cl,clo,cv=get_city_vals(tgt,7,2025)
    if data is not None and cl is not None:
        fig=plt.figure(figsize=(20,9)); fig.patch.set_facecolor("white")
        all_v=np.concatenate([data["preds"],data["trues"],cv])
        vmin=np.nanpercentile(all_v,2); vmax=np.nanpercentile(all_v,98)
        ax1=make_ak_ax(fig,1,2,1)
        al,alo,av=combine(data,cl,clo,cv,use_preds=False)
        im1=plot_field(ax1,al,alo,av,TGT_CMAPS[tgt],vmin,vmax)
        ax1.set_title(f"Observed | {TGT_LABELS[tgt]} | {data['ts_label']}",fontsize=10,fontweight="bold")
        fig.colorbar(im1,ax=ax1,orientation="horizontal",pad=0.04,shrink=0.85,aspect=28,label=TGT_UNITS[tgt])
        ax2=make_ak_ax(fig,1,2,2)
        al,alo,av=combine(data,cl,clo,cv,use_preds=True)
        im2=plot_field(ax2,al,alo,av,TGT_CMAPS[tgt],vmin,vmax)
        ax2.set_title(f"Predicted [{ba}] | {TGT_LABELS[tgt]} | {data['ts_label']}",fontsize=10,fontweight="bold")
        fig.colorbar(im2,ax=ax2,orientation="horizontal",pad=0.04,shrink=0.85,aspect=28,label=TGT_UNITS[tgt])
        fig.suptitle("Observed vs Predicted | Alaska | Shared colour scale\nv6 Distributed Spatial AI | Summer 2025",
                      fontsize=13,fontweight="bold",y=1.01)
        plt.tight_layout()
        plt.savefig(FIGS/"MAP_05_obs_vs_pred.png",dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(); print("    OK MAP_05_obs_vs_pred.png")

maps=sorted(FIGS.glob("MAP_*.png"))
print(f"\n{'='*65}\n  DONE: {len(maps)} map figures")
for f in maps: print(f"  {f.name} ({f.stat().st_size//1024} KB)")