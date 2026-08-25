import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT = Path("/home/emmanuel.keku")
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6" / "manuscript"
FIGS.mkdir(parents=True, exist_ok=True)

TIER_COLORS  = {"RESERVOIR":"#9B59B6","GRAPH":"#27AE60","ATTENTION":"#E67E22","SSM":"#2980B9"}
TIER_MARKERS = {"RESERVOIR":"^","GRAPH":"D","ATTENTION":"P","SSM":"o"}
ARCH_TIERS = {"DeepESN":"RESERVOIR","SpatialESN":"RESERVOIR","GraphSAGE":"GRAPH","GAT":"GRAPH","STGCN":"GRAPH","SpatialTransformer":"ATTENTION","SpatialInformer":"ATTENTION","SpatialBiGRU":"SSM","SpatialMamba":"SSM","SpatialS4":"SSM","SpatialFuseMoE":"SSM"}
BEST_2 = {"RESERVOIR":["SpatialESN","DeepESN"],"GRAPH":["STGCN","GAT"],"ATTENTION":["SpatialTransformer","SpatialInformer"],"SSM":["SpatialMamba","SpatialBiGRU"]}
BEST_ALL = [m for ms in BEST_2.values() for m in ms]
TGT_LABEL = {"temp":"Weather Temp (C)","smap":"SMAP Temp L1 (K)","moist":"Soil Moisture"}
def tc(a): return TIER_COLORS.get(ARCH_TIERS.get(a,"?"),"grey")
def tm(a): return TIER_MARKERS.get(ARCH_TIERS.get(a,"?"),"o")
def lp(): return [mpatches.Patch(color=c,label=t) for t,c in TIER_COLORS.items()]

# Load and fix data
df = pd.read_csv(RESULTS/"v6_scaling_results.csv")
df = df[~df["arch"].isin(["BiGRU_NoGCN","GCN_NoTemporal"])].copy()
df["tier"] = df["arch"].map(ARCH_TIERS)
df = df.dropna(subset=["tier"])
for col in ["speedup","efficiency","drop_ratio","time_1gpu","r2_1gpu"]:
    if col in df.columns: df=df.drop(columns=[col])
baseline = df[df["n_gpus"]==1][["arch","target","elapsed_s","val_r2"]].rename(columns={"elapsed_s":"time_1gpu","val_r2":"r2_1gpu"})
df = df.merge(baseline,on=["arch","target"],how="left")
df["speedup"]    = df["time_1gpu"]/(df["elapsed_s"]+1e-8)
df["efficiency"] = df["speedup"]/df["n_gpus"]*100
df["drop_ratio"] = df["r2_1gpu"]-df["val_r2"]
GPU_CFGS = sorted(df["n_gpus"].unique())
print("Models:", sorted(df["arch"].unique()))

N_TUNE=50; TE=15; TR=30
tune_records=[]
for _,row in df.iterrows():
    bl=df[(df["arch"]==row["arch"])&(df["n_gpus"]==1)&(df["target"]==row["target"])]
    t1=float(bl["time_1gpu"].iloc[0]) if len(bl)>0 else 0
    sp=float(row["speedup"]) if float(row["n_gpus"])>1 else 1.0
    tune_1=(t1/TR)*TE*N_TUNE
    tune_records.append(dict(arch=row["arch"],tier=row["tier"],target=row["target"],
        n_gpus=row["n_gpus"],tune_time_min=tune_1/max(sp,0.1)/60,
        train_time_s=row["elapsed_s"],speedup=sp,drop_ratio=row["drop_ratio"]))
tune_df=pd.DataFrame(tune_records)

# Helper: bar chart per GPU config
def bar_gpu(sub, col, xlabel, title, fname, unit, models=None):
    mlist = sorted(models if models else sub["arch"].unique())
    sub2  = sub[sub["arch"].isin(mlist)]
    fig   = plt.figure(figsize=(6*len(GPU_CFGS), max(9,len(mlist)*0.65+3)))
    gs    = gridspec.GridSpec(1,len(GPU_CFGS),figure=fig,wspace=0.4)
    vmax  = sub2[col].max()*1.15
    for gi,ng in enumerate(GPU_CFGS):
        ax   = fig.add_subplot(gs[gi])
        sg   = sub2[sub2["n_gpus"]==ng].set_index("arch").reindex(mlist)
        cols = [tc(m) for m in mlist]
        bars = ax.barh(mlist, sg[col].values, color=cols, alpha=0.85, edgecolor="black", lw=0.5)
        for bar,v in zip(bars, sg[col].values):
            if not np.isnan(v):
                ax.text(v+vmax*0.01, bar.get_y()+bar.get_height()/2, f"{v:.0f}{unit}", va="center", fontsize=8.5, fontweight="bold")
        if ng>1 and "tune" not in fname:
            bm = sub2[sub2["n_gpus"]==1][col].mean()
            ax.axvline(bm,color="grey",ls="--",lw=1.5,alpha=0.6,label="1 GPU mean")
            ax.legend(fontsize=8)
        ax.set_xlim(0,vmax); ax.set_xlabel(xlabel,fontsize=10)
        ax.set_title(f"{ng} GPU" + ("s" if ng>1 else ""), fontweight="bold", fontsize=11)
        if models:
            for lbl in ax.get_yticklabels(): lbl.set_color(tc(lbl.get_text())); lbl.set_fontweight("bold")
    fig.legend(handles=lp(),loc="lower center",ncol=4,fontsize=10,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(FIGS/fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  check {fname}")

# Generate all figures
for tgt in ["temp","smap","moist"]:
    sub = df[df["target"]==tgt]
    ts  = tune_df[tune_df["target"]==tgt]
    lbl = TGT_LABEL[tgt]
    bar_gpu(sub,"elapsed_s","Training Time (s)",f"Training Time per GPU | {lbl} | nn.DataParallel 30 epochs",f"SCALE_01_train_all_{tgt}.png","s")
    bar_gpu(sub,"elapsed_s","Training Time (s)",f"Training Time | Best 2 per Tier | {lbl}",f"SCALE_02_train_best_{tgt}.png","s",models=BEST_ALL)
    bar_gpu(ts,"tune_time_min","Tuning Time (min)",f"Tuning Time per GPU | {lbl} | 50 trials estimated",f"SCALE_03_tune_all_{tgt}.png","m")
    bar_gpu(ts,"tune_time_min","Tuning Time (min)",f"Tuning Time | Best 2 per Tier | {lbl}",f"SCALE_04_tune_best_{tgt}.png","m",models=BEST_ALL)
    # SCALE_05 Train vs Tune
    tr1=sub[sub["n_gpus"]==1][["arch","elapsed_s"]].set_index("arch")
    tu1=ts[ts["n_gpus"]==1][["arch","tune_time_min"]].set_index("arch")
    cm=tr1.join(tu1).dropna(); cm["train_min"]=cm["elapsed_s"]/60
    cm=cm.sort_values("tune_time_min")
    fig,ax=plt.subplots(figsize=(16,max(8,len(cm)*0.65+2)))
    x=np.arange(len(cm)); w=0.35
    cols2=[tc(m) for m in cm.index]
    b1=ax.barh(x-w/2,cm["train_min"],w,color=cols2,alpha=0.9,edgecolor="black",lw=0.5,label="Training (30 epochs)")
    b2=ax.barh(x+w/2,cm["tune_time_min"],w,color=cols2,alpha=0.4,edgecolor="black",lw=0.5,hatch="///",label="Tuning (50 trials, estimated)")
    for bar,v in zip(b1,cm["train_min"]): ax.text(v+0.05,bar.get_y()+bar.get_height()/2,f"{v:.1f}m",va="center",fontsize=8)
    for bar,v in zip(b2,cm["tune_time_min"]): ax.text(v+0.05,bar.get_y()+bar.get_height()/2,f"{v:.1f}m",va="center",fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels([f"[{ARCH_TIERS.get(m,chr(63))[:3]}] {m}" for m in cm.index],fontsize=10)
    for i,lb in enumerate(ax.get_yticklabels()): lb.set_color(tc(cm.index[i])); lb.set_fontweight("bold")
    ax.set_xlabel("Time (minutes)",fontsize=12)
    ax.set_title(f"Training vs Tuning Time | {lbl}",fontweight="bold",fontsize=13)
    tl=[mpatches.Patch(facecolor="grey",alpha=0.9,label="Training (30 epochs)"),mpatches.Patch(facecolor="grey",alpha=0.4,hatch="///",label="Tuning (50 trials, estimated)")]
    l1=ax.legend(handles=tl,loc="lower right",fontsize=10); ax.add_artist(l1)
    ax.legend(handles=lp(),loc="upper right",fontsize=9,title="Tier")
    plt.tight_layout(); plt.savefig(FIGS/f"SCALE_05_train_vs_tune_{tgt}.png",dpi=300,bbox_inches="tight"); plt.close()
    print(f"  check SCALE_05_train_vs_tune_{tgt}.png")
    # SCALE_06 Drop ratio all
    fig,ax=plt.subplots(figsize=(16,9))
    for arch in sorted(sub["arch"].unique()):
        m=sub[sub["arch"]==arch].sort_values("n_gpus")
        if m.empty: continue
        ax.plot(m["n_gpus"],m["drop_ratio"],color=tc(arch),marker=tm(arch),lw=2,ms=7,alpha=0.75,label=f"[{ARCH_TIERS.get(arch,chr(63))[:3]}] {arch}")
    ax.axhline(0,color="black",lw=1.5,alpha=0.5); ax.axhline(0.05,color="orange",ls="--",lw=1.5,label="5% threshold")
    ax.axhline(0.10,color="red",ls=":",lw=1.5,alpha=0.5,label="10% threshold")
    ax.fill_between(GPU_CFGS,[0]*len(GPU_CFGS),[0.05]*len(GPU_CFGS),alpha=0.06,color="green")
    ax.set_xlabel("Number of GPUs",fontsize=12); ax.set_ylabel("Drop Ratio",fontsize=12)
    ax.set_xticks(GPU_CFGS); ax.set_title(f"Quality Degradation | {lbl}",fontweight="bold",fontsize=13)
    ax.legend(fontsize=8,ncol=2); plt.tight_layout()
    plt.savefig(FIGS/f"SCALE_06_drop_all_{tgt}.png",dpi=300,bbox_inches="tight"); plt.close()
    print(f"  check SCALE_06_drop_all_{tgt}.png")
    # SCALE_07 Drop ratio best 2
    fig,ax=plt.subplots(figsize=(14,8))
    for tier,mods in BEST_2.items():
        for mi,arch in enumerate(mods):
            m=sub[sub["arch"]==arch].sort_values("n_gpus")
            if m.empty: continue
            ax.plot(m["n_gpus"],m["drop_ratio"],color=TIER_COLORS[tier],marker=TIER_MARKERS[tier],lw=3 if mi==0 else 2,ls="-" if mi==0 else "--",ms=10,alpha=1 if mi==0 else 0.75,label=f"[{tier[:3]}] {arch}")
    ax.axhline(0,color="black",lw=1.5,alpha=0.5); ax.axhline(0.05,color="orange",ls="--",lw=2,alpha=0.8,label="5% threshold")
    ax.fill_between(GPU_CFGS,[0]*len(GPU_CFGS),[0.05]*len(GPU_CFGS),alpha=0.08,color="green")
    ax.set_xlabel("Number of GPUs",fontsize=12); ax.set_ylabel("Drop Ratio",fontsize=12)
    ax.set_xticks(GPU_CFGS); ax.set_title(f"Quality Degradation | Best 2 | {lbl}",fontweight="bold",fontsize=13)
    ax.legend(fontsize=9,ncol=2); plt.tight_layout()
    plt.savefig(FIGS/f"SCALE_07_drop_best_{tgt}.png",dpi=300,bbox_inches="tight"); plt.close()
    print(f"  check SCALE_07_drop_best_{tgt}.png")

# SCALE_08 Speedup+Efficiency
fig,axes=plt.subplots(2,3,figsize=(24,16))
for ti,tgt in enumerate(["temp","smap","moist"]):
    sub=df[df["target"]==tgt]
    for tier in TIER_COLORS:
        tg=sub[sub["tier"]==tier].groupby("n_gpus")
        sp=tg["speedup"].agg(["mean","std"]); ef=tg["efficiency"].agg(["mean","std"])
        if sp.empty: continue
        c=TIER_COLORS[tier]; mk=TIER_MARKERS[tier]
        axes[0,ti].plot(sp.index,sp["mean"],color=c,marker=mk,lw=2.5,ms=10,label=tier)
        axes[0,ti].fill_between(sp.index,(sp["mean"]-sp["std"]).clip(0),sp["mean"]+sp["std"],color=c,alpha=0.12)
        axes[1,ti].plot(ef.index,ef["mean"],color=c,marker=mk,lw=2.5,ms=10,label=tier)
    axes[0,ti].plot(GPU_CFGS,[g/GPU_CFGS[0] for g in GPU_CFGS],"k--",lw=1.5,alpha=0.4,label="Ideal")
    axes[0,ti].set_xlabel("GPUs",fontsize=11); axes[0,ti].set_ylabel("Speedup (x)",fontsize=11)
    axes[0,ti].set_title(TGT_LABEL[tgt],fontweight="bold",fontsize=12)
    axes[0,ti].set_xticks(GPU_CFGS); axes[0,ti].legend(fontsize=9)
    axes[1,ti].axhline(80,color="green",ls="--",lw=1.5,alpha=0.7,label="80% threshold")
    axes[1,ti].set_ylim(0,110)
    axes[1,ti].set_xlabel("GPUs",fontsize=11); axes[1,ti].set_ylabel("Efficiency (%)",fontsize=11)
    axes[1,ti].set_xticks(GPU_CFGS); axes[1,ti].legend(fontsize=9)
fig.suptitle("GPU Scaling Speedup and Efficiency | nn.DataParallel",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"SCALE_08_speedup_efficiency.png",dpi=300,bbox_inches="tight"); plt.close()
print("  check SCALE_08_speedup_efficiency.png")

# SCALE_09 Training time lines
fig,axes=plt.subplots(1,3,figsize=(24,9))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]; sub=df[df["target"]==tgt]
    for tier,mods in BEST_2.items():
        for mi,arch in enumerate(mods):
            m=sub[sub["arch"]==arch].sort_values("n_gpus")
            if m.empty: continue
            ax.plot(m["n_gpus"],m["elapsed_s"],color=TIER_COLORS[tier],marker=TIER_MARKERS[tier],lw=3 if mi==0 else 2,ls="-" if mi==0 else "--",ms=10,alpha=1 if mi==0 else 0.75,label=f"[{tier[:3]}] {arch}")
    ax.set_xlabel("Number of GPUs",fontsize=12); ax.set_ylabel("Training Time (s)",fontsize=12)
    ax.set_title(TGT_LABEL[tgt],fontweight="bold",fontsize=12)
    ax.set_xticks(GPU_CFGS); ax.legend(fontsize=8,ncol=2)
fig.suptitle("Training Time vs GPU Count | Best 2 per Tier | nn.DataParallel",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"SCALE_09_train_time_lines.png",dpi=300,bbox_inches="tight"); plt.close()
print("  check SCALE_09_train_time_lines.png")

figs=sorted(FIGS.glob("SCALE_*.png"))
print(f"Total SCALE figures: {len(figs)}")
for f in figs: print(f"  {f.name}")