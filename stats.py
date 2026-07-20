import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as multitest

def analyze_results(results_csv="results/all_results.csv", output_csv="results/statistical_analysis.csv"):
    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        print(f"Results file {results_csv} not found.")
        return

    # Extract relevant metrics
    metrics = ['test_accuracy', 'test_auroc', 'conf_attack_auc', 'threshold_attack_auc', 'shadow_attack_auc', 'label_only_attack_auc']
    metrics = [m for m in metrics if m in df.columns]
    
    # We want to do paired tests for each (dataset, model, defense) vs (dataset, model, none)
    groups = df.groupby(['dataset', 'model'])
    
    analysis_records = []
    
    for (dataset, model), group_df in groups:
        none_df = group_df[group_df['defense'] == 'none'].sort_values('seed')
        if none_df.empty:
            continue
            
        defenses = group_df['defense'].unique()
        defenses = [d for d in defenses if d != 'none']
        
        for defense in defenses:
            def_df = group_df[group_df['defense'] == defense].sort_values('seed')
            
            # Match seeds
            common_seeds = np.intersect1d(none_df['seed'], def_df['seed'])
            if len(common_seeds) < 2:
                continue
                
            n_none = none_df[none_df['seed'].isin(common_seeds)]
            n_def = def_df[def_df['seed'].isin(common_seeds)]
            
            for metric in metrics:
                val_none = n_none[metric].values
                val_def = n_def[metric].values
                
                if np.isnan(val_none).all() or np.isnan(val_def).all():
                    continue
                
                # Paired t-test
                diffs = val_def - val_none
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
                
                if std_diff > 0:
                    t_stat, p_val = stats.ttest_rel(val_def, val_none)
                    
                    # Cohen's d (effect size)
                    pooled_std = np.sqrt((np.std(val_none, ddof=1)**2 + np.std(val_def, ddof=1)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # 95% CI of the difference
                    ci_hw = stats.t.ppf(0.975, len(diffs)-1) * (std_diff / np.sqrt(len(diffs)))
                    ci_low, ci_high = mean_diff - ci_hw, mean_diff + ci_hw
                else:
                    p_val = 1.0 if mean_diff == 0 else 0.0
                    cohens_d = 0.0
                    ci_low, ci_high = mean_diff, mean_diff
                
                analysis_records.append({
                    'dataset': dataset,
                    'model': model,
                    'defense': defense,
                    'metric': metric,
                    'mean_none': np.mean(val_none),
                    'mean_def': np.mean(val_def),
                    'mean_diff': mean_diff,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'cohens_d': cohens_d,
                    'p_val': p_val,
                    'n_seeds': len(common_seeds)
                })
                
    if not analysis_records:
        print("No paired data found to analyze.")
        return
        
    res_df = pd.DataFrame(analysis_records)
    
    # Apply Benjamini-Hochberg FDR correction across all p-values
    reject, pvals_corrected, _, _ = multitest.multipletests(res_df['p_val'], alpha=0.05, method='fdr_bh')
    res_df['p_val_corrected'] = pvals_corrected
    res_df['significant'] = reject
    
    res_df.to_csv(output_csv, index=False)
    print(f"Statistical analysis saved to {output_csv}")

if __name__ == "__main__":
    analyze_results()
