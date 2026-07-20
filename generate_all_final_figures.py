import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 200, 'font.size': 12, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.titlesize': 16, 'font.family': 'sans-serif',
})

PAL = sns.color_palette("deep")
FIG_DIR = "figures_final"
os.makedirs(FIG_DIR, exist_ok=True)

df_cora = pd.read_csv("results/all_results.csv")
df_ogbn = pd.read_csv("results_ogbn/all_results.csv")
df_all = pd.concat([df_cora, df_ogbn], ignore_index=True)

df_all["dataset"] = df_all["dataset"].astype(str).str.strip()
df_all["model"] = df_all["model"].astype(str).str.strip()
df_all["defense"] = df_all["defense"].astype(str).str.strip()

DEFENSE_NAMES = {
    'none': 'None (Baseline)',
    'dropedge': 'DropEdge',
    'label_smoothing': 'Label Smooth',
    'early_stopping': 'Early Stop',
    'confidence_masking': 'Conf Mask',
    'edge_sparsification': 'Edge Spars',
    'epsd': 'EPSD (Ours)',
    'dp_sgd': 'DP-GCN'
}

# Add label-only if not present
if 'label_only_auc' not in df_all.columns:
    df_all['label_only_auc'] = np.nan
if 'lira_attack_auc' not in df_all.columns:
    df_all['lira_attack_auc'] = np.nan

def savefig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{FIG_DIR}/{name}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {name}")

# ================================================================
# Fig 1: Utility-Privacy Tradeoff (Cora & Citeseer)
# ================================================================
def plot_tradeoff():
    dsets = ["Cora", "Citeseer"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    markers = {'none': 'o', 'dropedge': 's', 'label_smoothing': '^',
               'early_stopping': 'D', 'confidence_masking': 'v', 
               'edge_sparsification': 'P', 'epsd': '*', 'dp_sgd': 'X'}
    model_colors = {'GCN': PAL[0], 'GraphSAGE': PAL[1], 'LogReg': PAL[2], 'MLP': PAL[3]}
    
    for i, ds in enumerate(dsets):
        ax = axes[i]
        sub = df_all[(df_all['dataset'] == ds) & (df_all['model'].isin(['GCN', 'GraphSAGE']))]
        if sub.empty: continue
        
        agg = sub.groupby(['model', 'defense']).agg(
            acc=('test_accuracy', 'mean'), auc=('conf_attack_auc', 'mean'),
            acc_std=('test_accuracy', 'std'), auc_std=('conf_attack_auc', 'std')
        ).reset_index()
        
        texts = []
        for _, row in agg.iterrows():
            ax.errorbar(row['acc'], row['auc'], xerr=row['acc_std'], yerr=row['auc_std'],
                        color=model_colors[row['model']], marker=markers.get(row['defense'], 'o'),
                        markersize=12 if row['defense'] == 'epsd' else 9, 
                        capsize=4, linewidth=0, elinewidth=1.5)
            
            label = DEFENSE_NAMES.get(row['defense'], row['defense'])
            if row['defense'] == 'epsd':
                texts.append(ax.text(row['acc'], row['auc'], f" {label}", fontsize=11, fontweight='bold', color='darkred'))
            else:
                texts.append(ax.text(row['acc'], row['auc'], f" {label}", fontsize=9))
        
        ax.set_title(f"{ds} - Privacy/Utility Tradeoff")
        ax.set_xlabel("Utility (Test Accuracy ↑)")
        ax.set_ylabel("Privacy Vulnerability (Conf Attack AUC ↓)")
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PAL[0], lw=4, label='GCN'),
        Line2D([0], [0], color=PAL[1], lw=4, label='GraphSAGE'),
    ]
    axes[1].legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    savefig(fig, "fig2_tradeoff_cora_citeseer")

plot_tradeoff()

# ================================================================
# Fig 2: Synthetic Datasets EPSD Effectiveness
# ================================================================
def plot_synthetic():
    syn = df_all[df_all['dataset'].str.startswith('synthetic_')]
    if syn.empty: return
    
    # We compare none vs epsd vs dropedge
    syn = syn[syn['defense'].isin(['none', 'epsd', 'dropedge'])]
    syn['homophily'] = syn['dataset'].apply(lambda x: 'High' if 'high' in x else 'Low')
    syn['density'] = syn['dataset'].apply(lambda x: x.split('_')[-1].capitalize())
    
    agg = syn.groupby(['homophily', 'density', 'model', 'defense'])[['test_accuracy', 'conf_attack_auc']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot test accuracy
    sns.barplot(data=agg[agg['model']=='GCN'], x='density', y='test_accuracy', hue='defense', 
                palette='muted', ax=axes[0], order=['Sparse', 'Medium', 'Dense'])
    axes[0].set_title("Utility on Synthetic Data (GCN)")
    axes[0].set_ylabel("Test Accuracy ↑")
    axes[0].set_ylim(0, 1)
    
    # Plot attack auc
    sns.barplot(data=agg[agg['model']=='GCN'], x='density', y='conf_attack_auc', hue='defense', 
                palette='muted', ax=axes[1], order=['Sparse', 'Medium', 'Dense'])
    axes[1].set_title("Privacy on Synthetic Data (GCN)")
    axes[1].set_ylabel("Conf Attack AUC ↓")
    axes[1].axhline(0.5, ls='--', color='black', alpha=0.5)
    
    plt.tight_layout()
    savefig(fig, "fig3_synthetic_epsd")

plot_synthetic()

# ================================================================
# Fig 3: Large-Scale ogbn-arxiv
# ================================================================
def plot_ogbn():
    ogb = df_all[df_all['dataset'] == 'ogbn-arxiv']
    if ogb.empty: return
    
    agg = ogb.groupby(['model', 'defense'])[['test_accuracy', 'conf_attack_auc']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=agg, x='model', y='test_accuracy', hue='defense', palette='Set2', ax=axes[0])
    axes[0].set_title("Utility on ogbn-arxiv")
    axes[0].set_ylabel("Test Accuracy ↑")
    
    sns.barplot(data=agg, x='model', y='conf_attack_auc', hue='defense', palette='Set2', ax=axes[1])
    axes[1].set_title("Privacy on ogbn-arxiv")
    axes[1].set_ylabel("Conf Attack AUC ↓")
    axes[1].axhline(0.5, ls='--', color='black', alpha=0.5)
    
    plt.tight_layout()
    savefig(fig, "fig4_ogbn_arxiv")

plot_ogbn()

