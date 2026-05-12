"""
Generate publication-quality figures for the PrivacyGNN paper.
6 figures total as specified in the project deliverables.
Writes PNG only (300 dpi) under figures_dir from config.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

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
# Avoid silent mismatches from CSV exports / manual edits
df["dataset"] = df["dataset"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()
df["defense"] = df["defense"].astype(str).str.strip()

# --- Sanity check for citation benchmarks (Cora / Citeseer) ---
for ds in ("Cora", "Citeseer"):
    sub = df[df["dataset"] == ds]
    if sub.empty:
        print(f"  [WARN] No rows for dataset '{ds}' in {_results_path} — citation figures will be empty or partial.")
    else:
        gnn = sub[sub["model"].isin(["GCN", "GraphSAGE"])]
        for m in ("GCN", "GraphSAGE"):
            msub = gnn[gnn["model"] == m]
            if msub.empty:
                print(f"  [WARN] {ds}: no results for model {m} — defense-ranking panel for {m} will be empty.")
            defs = set(msub["defense"].unique())
            if len(defs) < 2:
                print(f"  [WARN] {ds} × {m}: only defenses {sorted(defs)} — run full defense grid for publication plots.")

# Helper
def savefig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {name}")

# ================================================================
# Figure 1: Attack AUC vs Model (Graph vs Non-Graph Comparison)
# ================================================================
print("Figure 1: Attack AUC vs Model")
citation = df[df['dataset'].isin(['Cora', 'Citeseer']) & (df['defense'] == 'none')]
agg = citation.groupby(['dataset', 'model']).agg(
    auc_mean=('conf_attack_auc', 'mean'), auc_std=('conf_attack_auc', 'std'),
    acc_mean=('test_accuracy', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for idx, ds in enumerate(['Cora', 'Citeseer']):
    ax = axes[idx]
    sub = agg[agg['dataset'] == ds].sort_values('model')
    if len(sub) == 0:
        ax.text(0.5, 0.5, 'No data\n(run with network for Cora/Citeseer)', ha='center', va='center',
                fontsize=10, transform=ax.transAxes)
        ax.set_title(ds, fontweight='bold')
        ax.set_ylabel('Attack AUC' if idx == 0 else '')
        ax.set_ylim(0.4, 0.9)
    else:
        models = sub['model'].values
        colors = [PAL[0] if m in ['LogReg', 'MLP'] else PAL[1] for m in models]
        bars = ax.bar(models, sub['auc_mean'], yerr=sub['auc_std'], capsize=4,
                      color=colors, edgecolor='white', linewidth=0.5, width=0.6)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(ds, fontweight='bold')
        ax.set_ylabel('Attack AUC' if idx == 0 else '')
        ax.set_ylim(0.4, 0.9)
        for bar, val in zip(bars, sub['auc_mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

legend_elements = [Patch(facecolor=PAL[0], label='Non-Graph (Baseline)'),
                   Patch(facecolor=PAL[1], label='Graph Neural Network')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
fig.suptitle('MIA Vulnerability: Graph vs Non-Graph Models', y=1.08, fontweight='bold')
fig.tight_layout()
savefig(fig, 'fig1_attack_auc_vs_model')

# ================================================================
# Figure 2: Utility-Privacy Tradeoff Curves (Pareto)
# ================================================================
print("Figure 2: Utility-Privacy Tradeoff Curves")
cora_gnns = df[(df['dataset'] == 'Cora') & (df['model'].isin(['GCN', 'GraphSAGE']))]
if cora_gnns.empty:
    # Fallback: use first available dataset with GCN/GraphSAGE (e.g. synthetic)
    gnns = df[df['model'].isin(['GCN', 'GraphSAGE'])]
    fallback_ds = gnns['dataset'].iloc[0] if len(gnns) > 0 else None
    if fallback_ds is not None:
        cora_gnns = df[(df['dataset'] == fallback_ds) & (df['model'].isin(['GCN', 'GraphSAGE']))]
        fig2_title = f'Privacy-Utility Tradeoff ({fallback_ds})'
    else:
        fig2_title = 'Privacy-Utility Tradeoff'
else:
    fig2_title = 'Privacy-Utility Tradeoff on Cora'

agg2 = cora_gnns.groupby(['model', 'defense']).agg(
    acc=('test_accuracy', 'mean'), auc=('conf_attack_auc', 'mean'),
    acc_std=('test_accuracy', 'std'), auc_std=('conf_attack_auc', 'std'),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5.5))
if agg2.empty:
    ax.text(0.5, 0.5, 'No data\n(run experiments for GNN + defenses)', ha='center', va='center',
            fontsize=10, transform=ax.transAxes)
    ax.set_title(fig2_title, fontweight='bold')
    ax.set_xlabel('Test Accuracy (Utility)')
    ax.set_ylabel('Attack AUC (Privacy Leakage)')
else:
    markers = {'none': 'o', 'dropedge': 's', 'label_smoothing': '^',
               'early_stopping': 'D', 'confidence_masking': 'v', 'edge_sparsification': 'P'}
    model_colors = {'GCN': PAL[0], 'GraphSAGE': PAL[1]}

    for _, row in agg2.iterrows():
        ax.errorbar(row['acc'], row['auc'], xerr=row['acc_std'], yerr=row['auc_std'],
                    color=model_colors[row['model']], marker=markers[row['defense']],
                    markersize=9, capsize=3, linewidth=0, elinewidth=1, markeredgecolor='white',
                    markeredgewidth=0.5)

    # Labels
    for _, row in agg2.iterrows():
        offset_x = 0.002 if row['defense'] != 'label_smoothing' else -0.01
        offset_y = 0.005
        label = row['defense'].replace('_', ' ').title()
        if label == 'None': label = 'No Defense'
        ax.annotate(label, (row['acc'] + offset_x, row['auc'] + offset_y),
                    fontsize=6.5, alpha=0.8)

    ax.set_xlabel('Test Accuracy (Utility)')
    ax.set_ylabel('Attack AUC (Privacy Leakage)')
    ax.set_title(fig2_title, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Random Guess')

    # Legend
    from matplotlib.lines import Line2D
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
# Figure 3: Leakage vs Homophily
# ================================================================
print("Figure 3: Leakage vs Homophily")
syn = df[(df['dataset'].str.startswith('synthetic')) & (df['defense'] == 'none')]
syn_agg = syn.groupby(['dataset', 'model']).agg(
    homo=('homophily', 'mean'), auc=('conf_attack_auc', 'mean'),
    auc_std=('conf_attack_auc', 'std'),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
model_markers = {'LogReg': 'o', 'MLP': 's', 'GCN': '^', 'GraphSAGE': 'D'}
model_colors2 = {'LogReg': PAL[0], 'MLP': PAL[1], 'GCN': PAL[4], 'GraphSAGE': PAL[5]}

for model in ['LogReg', 'MLP', 'GCN', 'GraphSAGE']:
    sub = syn_agg[syn_agg['model'] == model].sort_values('homo')
    ax.errorbar(sub['homo'], sub['auc'], yerr=sub['auc_std'],
                marker=model_markers[model], color=model_colors2[model],
                label=model, linewidth=1.5, markersize=7, capsize=3,
                markeredgecolor='white', markeredgewidth=0.5)

ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Graph Homophily')
ax.set_ylabel('Attack AUC')
ax.set_title('MIA Vulnerability vs Graph Homophily (Synthetic Graphs)', fontweight='bold')
ax.legend(title='Model', loc='upper left')
fig.tight_layout()
savefig(fig, 'fig3_leakage_vs_homophily')

# ================================================================
# Figure 4: Defense Effectiveness Comparison (Grouped Bar)
# ================================================================
print("Figure 4: Defense Effectiveness Comparison")
cora_all = df[(df['dataset'] == 'Cora') & (df['model'].isin(['GCN', 'GraphSAGE']))]
cora_def = cora_all.groupby(['model', 'defense']).agg(
    auc=('conf_attack_auc', 'mean'), auc_std=('conf_attack_auc', 'std'),
    acc=('test_accuracy', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for idx, model in enumerate(['GCN', 'GraphSAGE']):
    ax = axes[idx]
    sub = cora_def[cora_def['model'] == model].sort_values('auc')
    if sub.empty:
        ax.text(
            0.5, 0.5,
            f'No results for Cora × {model}.\n'
            'Run `python run_final.py` with Cora in `experiment_config.yaml` datasets,\n'
            'and ensure Planetoid download succeeds for all models × defenses × seeds.',
            ha='center', va='center', fontsize=9, transform=ax.transAxes,
        )
        ax.set_title(model, fontweight='bold')
        ax.set_xlabel('Attack AUC')
        continue
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
    
    for bar, val in zip(bars, sub['auc']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=8)

fig.suptitle('Defense Effectiveness on Cora (Lower = Better Privacy)', fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig4_defense_effectiveness')

# ================================================================
# Figure 5: Leakage vs Graph Density
# ================================================================
print("Figure 5: Leakage vs Density")
syn_dens = syn.copy()
syn_dens['density_cat'] = syn_dens['dataset'].apply(
    lambda x: x.split('_')[2] if 'synthetic' in x else 'N/A')
syn_dens['homo_cat'] = syn_dens['dataset'].apply(
    lambda x: x.split('_')[1] if 'synthetic' in x else 'N/A')

dens_agg = syn_dens.groupby(['density_cat', 'homo_cat', 'model']).agg(
    auc=('conf_attack_auc', 'mean'), auc_std=('conf_attack_auc', 'std'),
    dens=('density', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for idx, homo in enumerate(['high', 'low']):
    ax = axes[idx]
    sub = dens_agg[dens_agg['homo_cat'] == homo]
    for model in ['LogReg', 'MLP', 'GCN', 'GraphSAGE']:
        msub = sub[sub['model'] == model].sort_values('dens')
        ax.errorbar(msub['dens'] * 1000, msub['auc'], yerr=msub['auc_std'],
                    marker=model_markers[model], color=model_colors2[model],
                    label=model, linewidth=1.5, markersize=7, capsize=3,
                    markeredgecolor='white', markeredgewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Graph Density (x10^-3)')
    ax.set_ylabel('Attack AUC' if idx == 0 else '')
    ax.set_title(f'{"High" if homo == "high" else "Low"} Homophily', fontweight='bold')
    if idx == 0: ax.legend(title='Model', fontsize=7)

fig.suptitle('MIA Vulnerability vs Graph Density', fontweight='bold', y=1.03)
fig.tight_layout()
savefig(fig, 'fig5_leakage_vs_density')

# ================================================================
# Figure 6: Comprehensive Heatmap of Attack AUC
# ================================================================
print("Figure 6: Comprehensive Heatmap")
all_agg = df.groupby(['dataset', 'model', 'defense']).agg(
    auc=('conf_attack_auc', 'mean'),
).reset_index()

# Focus on GCN and GraphSAGE with all defenses, plus baselines
gnns = all_agg[all_agg['model'].isin(['GCN', 'GraphSAGE'])]
baselines = all_agg[(all_agg['model'].isin(['LogReg', 'MLP'])) & (all_agg['defense'] == 'none')]
# For baselines, set defense to "none" (they don't have defenses)

# Create a combined label
gnns = gnns.copy()
gnns['config'] = gnns['model'] + ' + ' + gnns['defense'].replace({'none': 'No Defense'}).str.replace('_', ' ').str.title()
baselines = baselines.copy()
baselines['config'] = baselines['model'] + ' (Baseline)'

combined = pd.concat([baselines[['dataset', 'config', 'auc']], gnns[['dataset', 'config', 'auc']]])

# Shorten dataset names
combined['dataset'] = combined['dataset'].replace({
    'synthetic_high_sparse': 'Syn-High-Sparse',
    'synthetic_high_medium': 'Syn-High-Med',
    'synthetic_high_dense': 'Syn-High-Dense',
    'synthetic_low_sparse': 'Syn-Low-Sparse',
    'synthetic_low_medium': 'Syn-Low-Med',
    'synthetic_low_dense': 'Syn-Low-Dense',
})

pivot = combined.pivot_table(index='config', columns='dataset', values='auc')
# Order columns
col_order = ['Cora', 'Citeseer', 'Syn-High-Sparse', 'Syn-High-Med', 'Syn-High-Dense',
             'Syn-Low-Sparse', 'Syn-Low-Med', 'Syn-Low-Dense']
col_order = [c for c in col_order if c in pivot.columns]
pivot = pivot[col_order]

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5,
            ax=ax, vmin=0.45, vmax=0.85, cbar_kws={'label': 'Attack AUC'})
ax.set_title('Membership Inference Attack AUC Across All Configurations', fontweight='bold', pad=15)
ax.set_xlabel('Dataset')
ax.set_ylabel('Model + Defense Configuration')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
savefig(fig, 'fig6_comprehensive_heatmap')

print("\nAll 6 figures generated successfully!")
