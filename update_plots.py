
import pandas as pd, ast, numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from pathlib import Path

import shutil



RESULTS = Path('/home/emmanuel.keku/results_v7')

FIGS    = Path('/home/emmanuel.keku/figures_v7')

WORKER_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728',

                '#9467bd','#8c564b','#e377c2','#7f7f7f']



for arch in ['STGCN','SpatialMamba']:

    df = pd.read_csv(RESULTS/f'rounds_v3_{arch}_N4.csv')

    rounds = df['round'].tolist()

    Us     = df['U'].tolist()

    gammas = df['gamma'].tolist()



    # 1. Uncertainty controls step size

    fig,ax = plt.subplots(figsize=(12,5))

    ax.plot(rounds, Us,     color='#1f77b4', lw=2.5, label='Population uncertainty U')

    ax.plot(rounds, gammas, color='#ff7f0e', lw=2.5, label='Jump scale γ')

    ax.set_xlabel('Global round', fontsize=12)

    ax.set_ylabel('Value', fontsize=12)

    ax.set_title(f'Uncertainty Controls Step Size | {arch} | N=4\nTrain: SEEN subregions | Val: Wetland (unseen)', fontsize=12)

    ax.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig(FIGS/f'uncertainty_step_size_{arch}.png', dpi=300, bbox_inches='tight')

    plt.close()

    print(f'OK uncertainty_step_size_{arch}.png')



    # 2. Signed worker coefficients (alphas) with dynamic Y-axis zoom

    alphas_pr = []

    for _,row in df.iterrows():

        try: al = ast.literal_eval(str(row['alphas']))

        except: al = []

        alphas_pr.append(al)

    n_workers = 4

    subset_names = ['Bedrock-A','Bedrock-B','Transition-A','Transition-B']

    

    fig,ax = plt.subplots(figsize=(12,6))

    all_alphas = []

    for wi in range(n_workers):

        vals = [alphas_pr[ri][wi] if wi<len(alphas_pr[ri]) else float('nan')

                for ri in range(len(rounds))]

        all_alphas.extend(vals)

        ax.plot(rounds, vals, color=WORKER_COLORS[wi], lw=2,

                label=f'Worker {wi+1} ({subset_names[wi]})')

        

    valid_alphas = [v for v in all_alphas if not np.isnan(v)]

    if valid_alphas:

        a_min, a_max = min(valid_alphas), max(valid_alphas)

        margin = (a_max - a_min) * 0.15 if a_max != a_min else 0.02

        ax.set_ylim(a_min - margin, a_max + margin)



    ax.axhline(0, color='grey', lw=1, alpha=0.5)

    ax.set_xlabel('Global round', fontsize=12)

    ax.set_ylabel('Signed worker coefficient α', fontsize=12)

    ax.set_title(f'Uncertainty-Aware Dynamic Signed Coefficients | {arch} | N=4\nTrain: SEEN subregions | Val: Wetland (unseen)', fontsize=12)

    ax.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig(FIGS/f'uncertainty_signed_coefficients_{arch}.png', dpi=300, bbox_inches='tight')

    plt.close()

    print(f'OK uncertainty_signed_coefficients_{arch}.png')



    # 3. Beta corrections with dynamic Y-axis zoom

    betas_pr = []

    for _,row in df.iterrows():

        try: bl = ast.literal_eval(str(row['betas']))

        except: bl = []

        betas_pr.append(bl)

        

    fig,ax = plt.subplots(figsize=(12,6))

    all_betas = []

    for wi in range(n_workers):

        vals = [betas_pr[ri][wi] if wi<len(betas_pr[ri]) else float('nan')

                for ri in range(len(rounds))]

        all_betas.extend(vals)

        ax.plot(rounds, vals, color=WORKER_COLORS[wi], lw=2,

                label=f'Worker {wi+1} ({subset_names[wi]})')

        

    valid_betas = [v for v in all_betas if not np.isnan(v)]

    if valid_betas:

        b_min, b_max = min(valid_betas), max(valid_betas)

        margin = (b_max - b_min) * 0.15 if b_max != b_min else 0.1

        ax.set_ylim(b_min - margin, b_max + margin)



    ax.axhline(0, color='grey', lw=1, alpha=0.5)

    ax.set_xlabel('Global round', fontsize=12)

    ax.set_ylabel('Signed residual coefficient β', fontsize=12)

    ax.set_title(f'Dynamic Unique-Direction Corrections | {arch} | N=4\nTrain: SEEN subregions | Val: Wetland (unseen)', fontsize=12)

    ax.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig(FIGS/f'beta_corrections_{arch}.png', dpi=300, bbox_inches='tight')

    plt.close()

    print(f'OK beta_corrections_{arch}.png')



    # 4. Performance bar — all N configs

    summary = pd.read_csv(RESULTS/'behavior_guided_v3_summary.csv')

    sub = summary[summary['arch']==arch]

    fig,axes = plt.subplots(1,2,figsize=(14,6))

    ax1,ax2 = axes

    labels = [f'N={n}' for n in sub['n_workers']] + ['Centralized']

    rmses  = sub['best_rmse'].tolist() + [sub['cent_rmse'].iloc[0]]

    r2s    = sub['best_r2'].tolist()   + [sub['cent_r2'].iloc[0]]

    colors = ['#1f77b4']*len(sub) + ['#d62728']

    

    bars = ax1.bar(labels, rmses, color=colors, width=0.5)

    for bar,v in zip(bars,rmses):

        ax1.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}',

                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Wetland Val RMSE (residual)', fontsize=11)

    ax1.set_title(f'Final RMSE | {arch}', fontsize=12)

    ax1.set_ylim(0, max(rmses)*1.15)

    

    bars2 = ax2.bar(labels, r2s, color=colors, width=0.5)

    for bar,v in zip(bars2,r2s):

        ax2.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.3f}',

                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Wetland Val R² (residual)', fontsize=11)

    ax2.set_title(f'Final R² | {arch}', fontsize=12)

    ax2.set_ylim(0, 1.05)

    

    fig.suptitle(f'{arch} | Behavior-Guided Distributed v3\nTrain: SEEN subregions | Val: Wetland (unseen) | Residual-only',

                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    plt.savefig(FIGS/f'performance_{arch}.png', dpi=300, bbox_inches='tight')

    plt.close()

    print(f'OK performance_{arch}.png')



print('All static figures done')



for f in Path('/home/emmanuel.keku/figures_v7').glob('*_STGCN*.png'):

    shutil.copy(f, '/home/emmanuel.keku/for_PI_v3/')

for f in Path('/home/emmanuel.keku/figures_v7').glob('*_SpatialMamba*.png'):

    shutil.copy(f, '/home/emmanuel.keku/for_PI_v3/')

for f in Path('/home/emmanuel.keku/figures_v7').glob('final_comparison_v3.png'):

    shutil.copy(f, '/home/emmanuel.keku/for_PI_v3/')

    

print('Copied to for_PI_v3')

