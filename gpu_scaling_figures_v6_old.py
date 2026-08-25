"""
gpu_scaling_figures_v6.py
Generates publication-quality GPU scaling figures from v6_scaling_results.csv
Shows times in SECONDS (not minutes) for clear visual differences
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

PROJECT = Path("/home/emmanuel.keku")
RESULTS = PROJECT / "results_v6"
FIGS    = PROJECT / "figures_v6" / "manuscript"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":300,"font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":13,"axes.labelsize":12,"axes.linewidth":1.2,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"lines.linewidth":2.0,
})

TIER_COLORS = {
    "ABLATION":"#E74C3C","RESERVOIR":"#9B59B6",
    "GRAPH":"#27AE60","ATTENTION":"#E67E22","SSM":"#2980B9",
}
TIER_MARKERS = {
    "ABLATION":"s","RESERVOIR":"^","GRAPH":"D","ATTENTION":"P","SSM":"o",
}

df = pd.read_csv(RESULTS/"v6_scaling_results.csv")
# Fix speedup computation for all targets
for col in ['speedup','efficiency','drop_ratio','time_1gpu','r2_1gpu']:
    if col in df.columns: df=df.drop(columns=[col])
baseline = df[df['n_gpus']==1][['arch','target','elapsed_s','val_r2']].copy()
baseline = baseline.rename(columns={'elapsed_s':'time_1gpu','val_r2':'r2_1gpu'})
df = df.merge(baseline,on=['arch','target'],how='left')
df['speedup']   = df['time_1gpu']/(df['elapsed_s']+1e-8)
df['efficiency']= df['speedup']/df['n_gpus']*100
df['drop_ratio']= df['r2_1gpu']-df['val_r2']
df['tier']      = df['tier'].fillna('SSM')

GPU_CFGS = sorted(df['n_gpus'].unique())
valid = df.dropna(subset=['speedup','efficiency'])
print(f"Records: {len(df)} | Valid: {len(valid)}")

# ── SCALE_01: Wall time in SECONDS per GPU (all models, best per tier) ────────
print("SCALE_01: Wall time...")
fig,axes = plt.subplots(1,len(GPU_CFGS),figsize=(7*len(GPU_CFGS),11))
tgt = "temp"
for ax,ng in zip(axes,GPU_CFGS):
    sg = df[(df['n_gpus']==ng)&(df['target']==tgt)].sort_values('elapsed_s',ascending=True)
    colors=[TIER_COLORS.get(t,"grey") for t in sg['tier']]
    bars=ax.barh(sg['arch'],sg['elapsed_s'],color=colors,alpha=0.85,
                  edgecolor="black",lw=0.5)
    for bar,v in zip(bars,sg['elapsed_s']):
        if not np.isnan(v):
            ax.text(v+1,bar.get_y()+bar.get_height()/2,
                    f"{v:.0f}s",va="center",fontsize=8.5,fontweight="bold")
    ax.set_xlabel("Training Time (seconds)",fontsize=11)
    ax.set_title(f"{ng} GPU{'s' if ng>1 else ''} | batch={ng*4}",
                  fontweight="bold",fontsize=12)
    # Add 1 GPU baseline reference line
    if ng > 1:
        sg_1 = df[(df['n_gpus']==1)&(df['target']==tgt)]['elapsed_s'].mean()
        ax.axvline(sg_1,color='grey',ls='--',lw=1.5,alpha=0.6,label='1 GPU baseline')
        ax.legend(fontsize=9)

handles=[mpatches.Patch(color=c,label=t) for t,c in TIER_COLORS.items()]
fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=10,
           bbox_to_anchor=(0.5,-0.04))
fig.suptitle("Training Wall Time | nn.DataParallel + GraphAwareWrapper\n"
              "Weather Temp target | talon32 V100 GPUs | 30 epochs",
              fontsize=14,fontweight="bold")
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig(FIGS/"SCALE_01_walltime_seconds.png",dpi=300,bbox_inches="tight")
plt.close()
print("  ✓ SCALE_01_walltime_seconds.png")

# ── SCALE_02: Speedup curves (all models per tier) ────────────────────────────
print("SCALE_02: Speedup curves...")
fig,axes = plt.subplots(1,3,figsize=(24,9))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]
    sub=valid[valid['target']==tgt]
    for tier in TIER_COLORS:
        t_sub=sub[sub['tier']==tier].groupby('n_gpus')['speedup'].agg(['mean','std'])
        if t_sub.empty: continue
        ax.plot(t_sub.index,t_sub['mean'],
                 color=TIER_COLORS[tier],marker=TIER_MARKERS[tier],
                 lw=2.5,ms=10,label=tier)
        ax.fill_between(t_sub.index,
                          t_sub['mean']-t_sub['std'],
                          t_sub['mean']+t_sub['std'],
                          color=TIER_COLORS[tier],alpha=0.12)
    # Ideal linear
    ax.plot(GPU_CFGS,[float(g)/GPU_CFGS[0] for g in GPU_CFGS],
             "k--",lw=1.5,alpha=0.4,label="Ideal (linear)")
    ax.set_xlabel("Number of GPUs",fontsize=12)
    ax.set_ylabel("Speedup (×)",fontsize=12)
    ax.set_title(["Weather Temp","SMAP Temp L1","Soil Moisture"][ti],
                  fontweight="bold",fontsize=12)
    ax.set_xticks(GPU_CFGS)
    ax.legend(fontsize=9)
    # Annotate best speedup
    best_tier = sub.groupby(['tier','n_gpus'])['speedup'].mean().unstack()['n_gpus' if 8 in sub['n_gpus'].values else GPU_CFGS[-1]].idxmax() if not sub.empty else None

fig.suptitle("GPU Scaling Speedup | nn.DataParallel + GraphAwareWrapper\n"
              "Shaded band = ±1 std across models | Dashed = ideal linear",
              fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"SCALE_02_speedup_curves.png",dpi=300,bbox_inches="tight")
plt.close()
print("  ✓ SCALE_02_speedup_curves.png")

# ── SCALE_03: Efficiency % ────────────────────────────────────────────────────
print("SCALE_03: Parallel efficiency...")
fig,axes = plt.subplots(1,3,figsize=(24,9))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]
    sub=valid[valid['target']==tgt]
    for tier in TIER_COLORS:
        t_sub=sub[sub['tier']==tier].groupby('n_gpus')['efficiency'].mean()
        if t_sub.empty: continue
        ax.plot(t_sub.index,t_sub.values,
                 color=TIER_COLORS[tier],marker=TIER_MARKERS[tier],
                 lw=2.5,ms=10,label=tier)
    ax.axhline(80,color="green",ls="--",lw=1.5,alpha=0.7,label="80% threshold")
    ax.axhline(100,color="black",ls="--",lw=1,alpha=0.3)
    ax.fill_between(GPU_CFGS,[80]*len(GPU_CFGS),[100]*len(GPU_CFGS),
                     alpha=0.05,color="green")
    ax.set_ylim(0,110); ax.set_xticks(GPU_CFGS)
    ax.set_xlabel("Number of GPUs",fontsize=12)
    ax.set_ylabel("Parallel Efficiency (%)",fontsize=12)
    ax.set_title(["Weather Temp","SMAP Temp L1","Soil Moisture"][ti],
                  fontweight="bold",fontsize=12)
    ax.legend(fontsize=9)

fig.suptitle("Parallel Efficiency | nn.DataParallel + GraphAwareWrapper\n"
              "100% = perfect linear scaling | >80% = acceptable",
              fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"SCALE_03_efficiency.png",dpi=300,bbox_inches="tight")
plt.close()
print("  ✓ SCALE_03_efficiency.png")

# ── SCALE_04: Drop ratio per tier ─────────────────────────────────────────────
print("SCALE_04: Drop ratio...")
fig,axes = plt.subplots(1,3,figsize=(24,9))
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]
    sub=valid[valid['target']==tgt]
    for tier in TIER_COLORS:
        t_sub=sub[sub['tier']==tier].groupby('n_gpus')['drop_ratio'].mean()
        if t_sub.empty: continue
        ax.plot(t_sub.index,t_sub.values,
                 color=TIER_COLORS[tier],marker=TIER_MARKERS[tier],
                 lw=2.5,ms=10,label=tier)
    ax.axhline(0,color="black",lw=1.5,alpha=0.5)
    ax.axhline(0.05,color="orange",ls="--",lw=1.5,alpha=0.7,label="5% threshold")
    ax.fill_between(GPU_CFGS,[0]*len(GPU_CFGS),[0.05]*len(GPU_CFGS),
                     alpha=0.08,color="green",label="Acceptable zone")
    ax.set_xticks(GPU_CFGS)
    ax.set_xlabel("Number of GPUs",fontsize=12)
    ax.set_ylabel("Drop Ratio (R²₁GPU − R²ₙGPU)",fontsize=12)
    ax.set_title(["Weather Temp","SMAP Temp L1","Soil Moisture"][ti],
                  fontweight="bold",fontsize=12)
    ax.legend(fontsize=9)

fig.suptitle("Quality Degradation vs GPU Count | Drop Ratio = R² Loss from Parallelism\n"
              "Green zone = acceptable | Moisture degrades more (high spatial variability)",
              fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"SCALE_04_drop_ratio.png",dpi=300,bbox_inches="tight")
plt.close()
print("  ✓ SCALE_04_drop_ratio.png")

# ── SCALE_05: Best model per tier — time vs R² trade-off ─────────────────────
print("SCALE_05: Time vs R² trade-off...")
fig,axes = plt.subplots(1,3,figsize=(24,9))
BEST_PER_TIER = {"ABLATION":"GCN_NoTemporal","RESERVOIR":"SpatialESN",
                  "GRAPH":"STGCN","ATTENTION":"SpatialTransformer","SSM":"SpatialMamba"}
for ti,tgt in enumerate(["temp","smap","moist"]):
    ax=axes[ti]
    sub=valid[valid['target']==tgt]
    for tier,arch in BEST_PER_TIER.items():
        m_sub=sub[sub['arch']==arch].sort_values('n_gpus')
        if m_sub.empty: continue
        color=TIER_COLORS[tier]
        ax.plot(m_sub['elapsed_s'],m_sub['val_r2'],
                 color=color,marker=TIER_MARKERS[tier],
                 lw=2.5,ms=10,label=f"[{tier[:3]}] {arch}")
        for _,r in m_sub.iterrows():
            ax.annotate(f"{int(r['n_gpus'])}GPU",
                         (r['elapsed_s'],r['val_r2']),
                         textcoords="offset points",xytext=(4,4),
                         fontsize=7,color=color)
    ax.set_xlabel("Training Time (seconds)",fontsize=12)
    ax.set_ylabel("Validation R²",fontsize=12)
    ax.set_title(["Weather Temp","SMAP Temp L1","Soil Moisture"][ti],
                  fontweight="bold",fontsize=12)
    ax.legend(fontsize=8)

fig.suptitle("Pareto Front: Training Time vs Model Quality | Best Model per Tier\n"
              "Left+Up = ideal | Numbers show GPU count at each point",
              fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS/"SCALE_05_time_vs_r2.png",dpi=300,bbox_inches="tight")
plt.close()
print("  ✓ SCALE_05_time_vs_r2.png")

# ── SCALE_06: Summary heatmap (speedup per model per GPU) ─────────────────────
print("SCALE_06: Speedup heatmap...")
for tgt in ["temp","smap","moist"]:
    sub = valid[valid['target']==tgt]
    pv  = sub.pivot_table(index='arch',columns='n_gpus',
                            values='speedup',aggfunc='mean')
    if pv.empty: continue
    pv.index = [f"[{df[df['arch']==m]['tier'].iloc[0][:3] if not df[df['arch']==m].empty else '?'}] {m}"
                  for m in pv.index]
    fig,ax = plt.subplots(figsize=(14,max(8,len(pv)*0.6+2)))
    sns.heatmap(pv,ax=ax,cmap="YlOrRd",annot=True,fmt=".2f",
                 linewidths=0.5,linecolor="white",
                 annot_kws={"size":11,"weight":"bold"},
                 cbar_kws={"label":"Speedup (×)","shrink":0.8})
    ax.set_xlabel("Number of GPUs",fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=10)
    ax.set_title(f"GPU Speedup Heatmap | {['Weather Temp','SMAP Temp L1','Soil Moisture'][['temp','smap','moist'].index(tgt)]}\n"
                  f"Values = speedup vs 1 GPU baseline | Darker = faster",
                  fontweight="bold",fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS/f"SCALE_06_speedup_heatmap_{tgt}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"  ✓ SCALE_06_speedup_heatmap_{tgt}.png")

figs = sorted(FIGS.glob("SCALE_*.png"))
print(f"\n{'='*55}")
print(f"  SCALE FIGURES: {len(figs)}")
for f in figs: print(f"  {f.name}")
