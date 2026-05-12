"""
Fix Figure 2 (privacy-utility tradeoff) and Figure 4 (defense effectiveness).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 200, 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.titlesize': 14, 'font.family': 'sans-serif',
})

PAL = ['#20808D', '#A84B2F', '#1B474D', '#BCE2E7', '#944454', '#FFC553']
try:
    from config import load_config
    _config = load_config()
    FIG_DIR = _config["figures_dir"]
    _results_path = os.path.join(_config["results_dir"], "all_results.csv")
except Exception:
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(_ROOT, "figures")
    _results_path = os.path.join(_ROOT, "results", "all_results.csv")
os.makedirs(FIG_DIR, exist_ok=True)
df = pd.read_csv(_results_path)

def savefig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{FIG_DIR}/{name}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {name}")

# ================================================================
# Figure 2: Utility-Privacy Tradeoff (FIXED - use adjustText)
# ================================================================
print("Figure 2: Utility-Privacy Tradeoff Curves (fixed)")
cora_gnns = df[(df['dataset'] == 'Cora') & (df['model'].isin(['GCN', 'GraphSAGE']))]
agg2 = cora_gnns.groupby(['model', 'defense']).agg(
    acc=('test_accuracy', 'mean'), auc=('conf_attack_auc', 'mean'),
    acc_std=('test_accuracy', 'std'), auc_std=('conf_attack_auc', 'std'),
).reset_index()

fig, ax = plt.subplots(figsize=(9, 6))
markers = {'none': 'o', 'dropedge': 's', 'label_smoothing': '^',
           'early_stopping': 'D', 'confidence_masking': 'v', 'edge_sparsification': 'P'}
model_colors = {'GCN': PAL[0], 'GraphSAGE': PAL[1]}

for _, row in agg2.iterrows():
    ax.errorbar(row['acc'], row['auc'], xerr=row['acc_std'], yerr=row['auc_std'],
                color=model_colors[row['model']], marker=markers[row['defense']],
                markersize=10, capsize=3, linewidth=0, elinewidth=1, markeredgecolor='white',
                markeredgewidth=0.8, zorder=3)

# Use adjustText to avoid overlapping labels
texts = []
for _, row in agg2.iterrows():
    label = row['defense'].replace('_', ' ').title()
    if label == 'None': label = 'No Defense'
    model_abbr = row['model'][:3]
    texts.append(ax.text(row['acc'], row['auc'], f"{label}", fontsize=6.5, alpha=0.85))

adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
            expand_points=(2.0, 2.0), expand_text=(1.5, 1.5), force_text=(0.8, 0.8))

ax.set_xlabel('Test Accuracy (Utility)')
ax.set_ylabel('Attack AUC (Privacy Leakage)')
ax.set_title('Privacy-Utility Tradeoff on Cora', fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Random Guess')

leg1 = [Line2D([0],[0], color=PAL[0], marker='o', linestyle='', markersize=8, label='GCN'),
        Line2D([0],[0], color=PAL[1], marker='o', linestyle='', markersize=8, label='GraphSAGE')]
leg2 = [Line2D([0],[0], color='gray', marker=v, linestyle='', markersize=7, label=k.replace('_',' ').title())
        for k, v in markers.items()]
l1 = ax.legend(handles=leg1, title='Model', loc='upper left', fontsize=7)
ax.add_artist(l1)
ax.legend(handles=leg2, title='Defense', loc='lower right', fontsize=6.5, ncol=2)
fig.tight_layout()
savefig(fig, 'fig2_privacy_utility_tradeoff')


# ================================================================
# Figure 4: Defense Effectiveness (FIXED - remove overlapping labels)
# ================================================================
print("Figure 4: Defense Effectiveness Comparison (fixed)")
cora_all = df[(df['dataset'] == 'Cora') & (df['model'].isin(['GCN', 'GraphSAGE']))]
cora_def = cora_all.groupby(['model', 'defense']).agg(
    auc=('conf_attack_auc', 'mean'), auc_std=('conf_attack_auc', 'std'),
    acc=('test_accuracy', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for idx, model in enumerate(['GCN', 'GraphSAGE']):
    ax = axes[idx]
    sub = cora_def[cora_def['model'] == model].sort_values('auc')
    colors_def = []
    for d in sub['defense']:
        if d == 'none': colors_def.append('#888888')
        elif d == 'edge_sparsification': colors_def.append(PAL[0])
        elif d == 'confidence_masking': colors_def.append(PAL[1])
        elif d == 'early_stopping': colors_def.append(PAL[2])
        elif d == 'dropedge': colors_def.append(PAL[4])
        else: colors_def.append(PAL[5])
    
    labels = [d.replace('_', ' ').title() for d in sub['defense']]
    labels = [l if l != 'None' else 'No Defense' for l in labels]
    
    bars = ax.barh(labels, sub['auc'], xerr=sub['auc_std'], capsize=3,
                   color=colors_def, edgecolor='white', height=0.6)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Attack AUC')
    ax.set_title(model, fontweight='bold')
    
    # Place value labels with more offset to avoid error bar overlap
    max_err = sub['auc_std'].max()
    for bar, val, err in zip(bars, sub['auc'].values, sub['auc_std'].values):
        x_pos = bar.get_width() + err + 0.008  # offset past the error bar cap
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Set x-axis limit to accommodate labels
    ax.set_xlim(0.4, max(sub['auc'].values + sub['auc_std'].values) + 0.06)

fig.suptitle('Defense Effectiveness on Cora (Lower = Better Privacy)', fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig4_defense_effectiveness')

print("\nFixed figures generated successfully!")
