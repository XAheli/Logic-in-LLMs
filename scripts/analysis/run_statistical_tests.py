#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for Syllogistic Reasoning Paper
Runs all statistical tests on the corrected metrics from raw_responses data
"""

import json
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, pearsonr, spearmanr
from itertools import combinations

# Change to project root directory for relative paths
os.chdir(Path(__file__).parent.parent.parent)

# Define the 14 models (excluding mixtral-8x22b-instruct due to API errors)
models_14 = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gpt-oss-20b',
    'glm-4.6',
    'kimi-k2-instruct',
    'deepseek-v3.1',
    'gemini-2.5-flash-lite',
    'qwen3-next-80b-a3b-instruct',
    'qwen3-next-80b-a3b-thinking',
    'llama-3.3-70b-instruct',
    'gemma-3-27b-it',
    'llama-3.1-8b-instruct',
    'llama-3.2-3b-instruct',
    'llama-3.2-1b-instruct'
]

strategies = ['zero_shot', 'one_shot', 'few_shot', 'zero_shot_cot']
temperatures = ['0.0', '0.5', '1.0']
results_dir = Path('results/raw_responses')

print("="*120)
print("LOADING DATA FROM RAW RESPONSES")
print("="*120)

# Collect per-configuration accuracy
config_accuracy = []

for model in models_14:
    for temp in temperatures:
        for strategy in strategies:
            filepath = results_dir / f"temperature_{temp}" / f"{model}_{strategy}.json"
            
            if not filepath.exists():
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                if not results:
                    continue
                
                # Calculate accuracy
                correct = sum(1 for r in results 
                            if r.get('predicted', '').lower() == 'correct' 
                            and r.get('ground_truth_syntax', '').lower() == 'valid'
                            or r.get('predicted', '').lower() == 'incorrect' 
                            and r.get('ground_truth_syntax', '').lower() == 'invalid')
                
                total = len(results)
                accuracy = (correct / total * 100) if total > 0 else 0
                
                config_accuracy.append({
                    'model': model,
                    'temperature': temp,
                    'strategy': strategy,
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                })
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

df = pd.DataFrame(config_accuracy)
print(f"✅ Loaded {len(df)} configurations")
print(f"   Models: {df['model'].nunique()}")
print(f"   Strategies: {df['strategy'].nunique()}")
print(f"   Temperatures: {df['temperature'].nunique()}")

# ============================================================================
# TEST 1: Strategy Comparisons (Paired t-tests)
# ============================================================================
print("\n" + "="*120)
print("TEST 1: PAIRED T-TESTS - Strategy Comparisons")
print("="*120)

strategy_tests = []
baseline = 'zero_shot'

for strategy in ['one_shot', 'few_shot', 'zero_shot_cot']:
    # Get paired data (same model + temp)
    baseline_data = df[df['strategy'] == baseline].set_index(['model', 'temperature'])['accuracy']
    comparison_data = df[df['strategy'] == strategy].set_index(['model', 'temperature'])['accuracy']
    
    # Align indices
    paired_baseline = baseline_data.reindex(comparison_data.index)
    paired_comparison = comparison_data
    
    # Remove NaN pairs
    valid_mask = paired_baseline.notna() & paired_comparison.notna()
    paired_baseline = paired_baseline[valid_mask]
    paired_comparison = paired_comparison[valid_mask]
    
    if len(paired_baseline) > 1:
        t_stat, p_value = ttest_rel(paired_baseline, paired_comparison)
        mean_diff = paired_comparison.mean() - paired_baseline.mean()
        cohens_d = mean_diff / np.std(paired_baseline - paired_comparison)
        
        strategy_tests.append({
            'Baseline': baseline,
            'Comparison': strategy,
            'N_pairs': len(paired_baseline),
            'Mean_Baseline': paired_baseline.mean(),
            'Mean_Comparison': paired_comparison.mean(),
            'Mean_Diff': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'Cohens_d': cohens_d,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })

# Apply Holm-Bonferroni correction (sequential method)
n_comparisons = len(strategy_tests)
strategy_tests_sorted = sorted(strategy_tests, key=lambda x: x['p_value'])
for rank, test in enumerate(strategy_tests_sorted, start=1):
    test['p_value_holm'] = min(test['p_value'] * (n_comparisons - rank + 1), 1.0)
    test['Significant_holm'] = '***' if test['p_value_holm'] < 0.001 else '**' if test['p_value_holm'] < 0.01 else '*' if test['p_value_holm'] < 0.05 else 'ns'
    # Also keep Bonferroni for comparison
    test['p_value_bonferroni'] = min(test['p_value'] * n_comparisons, 1.0)
    test['Significant_bonferroni'] = '***' if test['p_value_bonferroni'] < 0.001 else '**' if test['p_value_bonferroni'] < 0.01 else '*' if test['p_value_bonferroni'] < 0.05 else 'ns'

strategy_df = pd.DataFrame(strategy_tests_sorted)
print(strategy_df.to_string(index=False, float_format='%.4f'))

# Save
strategy_df.to_csv('results/analysis/tables/stats_strategy_ttests.csv', index=False, float_format='%.4f')
print("\n✅ Saved to: results/analysis/tables/stats_strategy_ttests.csv")

# ============================================================================
# TEST 2: Wilcoxon Signed-Rank Tests (Non-parametric)
# ============================================================================
print("\n" + "="*120)
print("TEST 2: WILCOXON SIGNED-RANK TESTS - Strategy Comparisons (Non-parametric)")
print("="*120)

wilcoxon_tests = []

for strategy in ['one_shot', 'few_shot', 'zero_shot_cot']:
    baseline_data = df[df['strategy'] == baseline].set_index(['model', 'temperature'])['accuracy']
    comparison_data = df[df['strategy'] == strategy].set_index(['model', 'temperature'])['accuracy']
    
    paired_baseline = baseline_data.reindex(comparison_data.index)
    paired_comparison = comparison_data
    
    valid_mask = paired_baseline.notna() & paired_comparison.notna()
    paired_baseline = paired_baseline[valid_mask]
    paired_comparison = paired_comparison[valid_mask]
    
    if len(paired_baseline) > 1:
        w_stat, p_value = wilcoxon(paired_baseline, paired_comparison)
        median_diff = paired_comparison.median() - paired_baseline.median()
        
        wilcoxon_tests.append({
            'Baseline': baseline,
            'Comparison': strategy,
            'N_pairs': len(paired_baseline),
            'Median_Baseline': paired_baseline.median(),
            'Median_Comparison': paired_comparison.median(),
            'Median_Diff': median_diff,
            'W_statistic': w_stat,
            'p_value': p_value,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })

# Apply Holm-Bonferroni correction (sequential method)
n_comparisons = len(wilcoxon_tests)
wilcoxon_tests_sorted = sorted(wilcoxon_tests, key=lambda x: x['p_value'])
for rank, test in enumerate(wilcoxon_tests_sorted, start=1):
    test['p_value_holm'] = min(test['p_value'] * (n_comparisons - rank + 1), 1.0)
    test['Significant_holm'] = '***' if test['p_value_holm'] < 0.001 else '**' if test['p_value_holm'] < 0.01 else '*' if test['p_value_holm'] < 0.05 else 'ns'
    # Also keep Bonferroni for comparison
    test['p_value_bonferroni'] = min(test['p_value'] * n_comparisons, 1.0)
    test['Significant_bonferroni'] = '***' if test['p_value_bonferroni'] < 0.001 else '**' if test['p_value_bonferroni'] < 0.01 else '*' if test['p_value_bonferroni'] < 0.05 else 'ns'

wilcoxon_df = pd.DataFrame(wilcoxon_tests_sorted)
print(wilcoxon_df.to_string(index=False, float_format='%.4f'))

wilcoxon_df.to_csv('results/analysis/tables/stats_strategy_wilcoxon.csv', index=False, float_format='%.4f')
print("\n✅ Saved to: results/analysis/tables/stats_strategy_wilcoxon.csv")

# ============================================================================
# TEST 3: McNemar's Test for Strategy Pairs (INSTANCE-LEVEL)
# ============================================================================
print("\n" + "="*120)
print("TEST 3: McNEMAR'S TEST - Strategy Comparisons (Instance-Level)")
print("="*120)

mcnemar_tests = []

for strategy in ['one_shot', 'few_shot', 'zero_shot_cot']:
    print(f"\nComparing {baseline} vs {strategy} at instance level...")
    
    # Create instance-level contingency table by loading raw data
    both_correct = 0
    baseline_only = 0
    comparison_only = 0
    both_incorrect = 0
    
    # Iterate through all models and temperatures
    for model in models_14:
        for temp in temperatures:
            baseline_file = results_dir / f"temperature_{temp}" / f"{model}_{baseline}.json"
            comparison_file = results_dir / f"temperature_{temp}" / f"{model}_{strategy}.json"
            
            if not baseline_file.exists() or not comparison_file.exists():
                continue
            
            try:
                with open(baseline_file, 'r') as f:
                    baseline_results = json.load(f).get('results', [])
                with open(comparison_file, 'r') as f:
                    comparison_results = json.load(f).get('results', [])
                
                # Match instances by syllogism ID
                for b_res, c_res in zip(baseline_results, comparison_results):
                    # Check if predictions are correct
                    b_pred = b_res.get('predicted', '').lower()
                    c_pred = c_res.get('predicted', '').lower()
                    gt = b_res.get('ground_truth_syntax', '').lower()
                    
                    b_correct = (b_pred == 'correct' and gt == 'valid') or (b_pred == 'incorrect' and gt == 'invalid')
                    c_correct = (c_pred == 'correct' and gt == 'valid') or (c_pred == 'incorrect' and gt == 'invalid')
                    
                    if b_correct and c_correct:
                        both_correct += 1
                    elif b_correct and not c_correct:
                        baseline_only += 1
                    elif not b_correct and c_correct:
                        comparison_only += 1
                    else:
                        both_incorrect += 1
                        
            except Exception as e:
                print(f"  Warning: Error processing {model}_{temp}: {e}")
                continue
    
    # McNemar's test uses b and c (discordant pairs)
    total_instances = both_correct + baseline_only + comparison_only + both_incorrect
    print(f"  Total instances: {total_instances}")
    print(f"  Both correct: {both_correct}, Baseline only: {baseline_only}, Comparison only: {comparison_only}, Both incorrect: {both_incorrect}")
    
    if baseline_only + comparison_only > 0:
        chi2_stat = ((abs(baseline_only - comparison_only) - 1) ** 2) / (baseline_only + comparison_only)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
        
        mcnemar_tests.append({
            'Baseline': baseline,
            'Comparison': strategy,
            'N_instances': total_instances,
            'Both_Correct': both_correct,
            'Baseline_Only': baseline_only,
            'Comparison_Only': comparison_only,
            'Both_Incorrect': both_incorrect,
            'Chi2_statistic': chi2_stat,
            'p_value': p_value,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    else:
        print(f"  No discordant pairs found!")
        mcnemar_tests.append({
            'Baseline': baseline,
            'Comparison': strategy,
            'N_instances': total_instances,
            'Both_Correct': both_correct,
            'Baseline_Only': baseline_only,
            'Comparison_Only': comparison_only,
            'Both_Incorrect': both_incorrect,
            'Chi2_statistic': 0.0,
            'p_value': 1.0,
            'Significant': 'ns'
        })

mcnemar_df = pd.DataFrame(mcnemar_tests)
print(mcnemar_df.to_string(index=False, float_format='%.4f'))

mcnemar_df.to_csv('results/analysis/tables/stats_strategy_mcnemar.csv', index=False, float_format='%.4f')
print("\n✅ Saved to: results/analysis/tables/stats_strategy_mcnemar.csv")

# ============================================================================
# TEST 4a: Friedman Test for Strategy Effects
# ============================================================================
print("\n" + "="*120)
print("TEST 4a: FRIEDMAN TEST - Strategy Effects")
print("="*120)

# For each model+temperature, get accuracy across all four strategies
strategy_groups = []
for strategy in strategies:
    strategy_data = df[df['strategy'] == strategy].set_index(['model', 'temperature'])['accuracy']
    strategy_groups.append(strategy_data)

# Align all four
aligned_strategy_data = pd.concat(strategy_groups, axis=1, keys=strategies)
aligned_strategy_data = aligned_strategy_data.dropna()

if len(aligned_strategy_data) > 0:
    chi2_stat_strat, p_value_strat = friedmanchisquare(*[aligned_strategy_data[strat] for strat in strategies])
    
    strategy_friedman_results = {
        'Test': 'Friedman (Strategy)',
        'N_observations': len(aligned_strategy_data),
        'Chi2_statistic': chi2_stat_strat,
        'df': len(strategies) - 1,
        'p_value': p_value_strat,
        'Mean_zero_shot': aligned_strategy_data['zero_shot'].mean(),
        'Mean_one_shot': aligned_strategy_data['one_shot'].mean(),
        'Mean_few_shot': aligned_strategy_data['few_shot'].mean(),
        'Mean_zero_shot_cot': aligned_strategy_data['zero_shot_cot'].mean(),
        'Significant': '***' if p_value_strat < 0.001 else '**' if p_value_strat < 0.01 else '*' if p_value_strat < 0.05 else 'ns'
    }
    
    strategy_friedman_df = pd.DataFrame([strategy_friedman_results])
    print(strategy_friedman_df.to_string(index=False, float_format='%.4f'))
    
    strategy_friedman_df.to_csv('results/analysis/tables/stats_strategy_friedman.csv', index=False, float_format='%.4f')
    print("\n✅ Saved to: results/analysis/tables/stats_strategy_friedman.csv")

# ============================================================================
# TEST 4b: Friedman Test for Temperature Effects
# ============================================================================
print("\n" + "="*120)
print("TEST 4b: FRIEDMAN TEST - Temperature Effects")
print("="*120)

# For each model+strategy, get accuracy at three temperatures
temp_groups = []
for temp in temperatures:
    temp_data = df[df['temperature'] == temp].set_index(['model', 'strategy'])['accuracy']
    temp_groups.append(temp_data)

# Align all three
aligned_data = pd.concat(temp_groups, axis=1, keys=temperatures)
aligned_data = aligned_data.dropna()

if len(aligned_data) > 0:
    chi2_stat, p_value = friedmanchisquare(*[aligned_data[temp] for temp in temperatures])
    
    temp_results = {
        'Test': 'Friedman (Temperature)',
        'N_observations': len(aligned_data),
        'Chi2_statistic': chi2_stat,
        'df': len(temperatures) - 1,
        'p_value': p_value,
        'Mean_temp_0.0': aligned_data['0.0'].mean(),
        'Mean_temp_0.5': aligned_data['0.5'].mean(),
        'Mean_temp_1.0': aligned_data['1.0'].mean(),
        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    }
    
    temp_df = pd.DataFrame([temp_results])
    print(temp_df.to_string(index=False, float_format='%.4f'))
    
    temp_df.to_csv('results/analysis/tables/stats_temperature_friedman.csv', index=False, float_format='%.4f')
    print("\n✅ Saved to: results/analysis/tables/stats_temperature_friedman.csv")

# ============================================================================
# TEST 5: Belief Bias Significance (Paired t-test)
# ============================================================================
print("\n" + "="*120)
print("TEST 5: PAIRED T-TEST - Belief Bias (Congruent vs Incongruent)")
print("="*120)

# Load belief bias data
bias_df = pd.read_csv('results/analysis/tables/paper_table3_belief_bias.csv')

t_stat, p_value = ttest_rel(bias_df['Congruent_Acc'], bias_df['Incongruent_Acc'])
mean_diff = bias_df['Bias_Effect'].mean()
cohens_d = mean_diff / bias_df['Bias_Effect'].std()

bias_test = {
    'Test': 'Belief Bias (Congruent vs Incongruent)',
    'N_models': len(bias_df),
    'Mean_Congruent': bias_df['Congruent_Acc'].mean(),
    'Mean_Incongruent': bias_df['Incongruent_Acc'].mean(),
    'Mean_Bias_Effect': mean_diff,
    't_statistic': t_stat,
    'p_value': p_value,
    'Cohens_d': cohens_d,
    'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
}

bias_test_df = pd.DataFrame([bias_test])
print(bias_test_df.to_string(index=False, float_format='%.4f'))

bias_test_df.to_csv('results/analysis/tables/stats_belief_bias_test.csv', index=False, float_format='%.4f')
print("\n✅ Saved to: results/analysis/tables/stats_belief_bias_test.csv")

# ============================================================================
# TEST 6: Correlation Analysis
# ============================================================================
print("\n" + "="*120)
print("TEST 6: CORRELATION ANALYSIS")
print("="*120)

# Load complete table with consistency metrics
complete_df = pd.read_csv('results/analysis/tables/paper_table1_complete.csv')

correlations = []

# Accuracy vs Consistency metrics
for consistency_metric in ['C_all', 'C_N<->X', 'C_O<->OX']:
    pearson_r, pearson_p = pearsonr(complete_df['Syntax_Acc'], complete_df[consistency_metric])
    spearman_r, spearman_p = spearmanr(complete_df['Syntax_Acc'], complete_df[consistency_metric])
    
    correlations.append({
        'Variable_1': 'Syntax_Acc',
        'Variable_2': consistency_metric,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Pearson_Sig': '***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns',
        'Spearman_Sig': '***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'
    })

# Precision vs Recall
pearson_r, pearson_p = pearsonr(complete_df['Syntax_Prec'], complete_df['Syntax_Rec'])
spearman_r, spearman_p = spearmanr(complete_df['Syntax_Prec'], complete_df['Syntax_Rec'])

correlations.append({
    'Variable_1': 'Syntax_Prec',
    'Variable_2': 'Syntax_Rec',
    'Pearson_r': pearson_r,
    'Pearson_p': pearson_p,
    'Spearman_r': spearman_r,
    'Spearman_p': spearman_p,
    'Pearson_Sig': '***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns',
    'Spearman_Sig': '***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'
})

# Load dual eval for Syntax vs NLU correlation
dual_df = pd.read_csv('results/analysis/tables/paper_table2_dual_eval.csv')

pearson_r, pearson_p = pearsonr(dual_df['Syntax_Acc'], dual_df['NLU_Acc'])
spearman_r, spearman_p = spearmanr(dual_df['Syntax_Acc'], dual_df['NLU_Acc'])

correlations.append({
    'Variable_1': 'Syntax_Acc',
    'Variable_2': 'NLU_Acc',
    'Pearson_r': pearson_r,
    'Pearson_p': pearson_p,
    'Spearman_r': spearman_r,
    'Spearman_p': spearman_p,
    'Pearson_Sig': '***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns',
    'Spearman_Sig': '***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'
})

# Accuracy vs Belief Bias - MERGE BY MODEL FIRST TO ENSURE ALIGNMENT
merged_bias = pd.merge(
    complete_df[['Model', 'Syntax_Acc']], 
    bias_df[['Model', 'Bias_Effect']], 
    on='Model'
)
pearson_r, pearson_p = pearsonr(merged_bias['Syntax_Acc'], merged_bias['Bias_Effect'])
spearman_r, spearman_p = spearmanr(merged_bias['Syntax_Acc'], merged_bias['Bias_Effect'])

correlations.append({
    'Variable_1': 'Syntax_Acc',
    'Variable_2': 'Bias_Effect',
    'Pearson_r': pearson_r,
    'Pearson_p': pearson_p,
    'Spearman_r': spearman_r,
    'Spearman_p': spearman_p,
    'Pearson_Sig': '***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns',
    'Spearman_Sig': '***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'
})

corr_df = pd.DataFrame(correlations)
print(corr_df.to_string(index=False, float_format='%.4f'))

corr_df.to_csv('results/analysis/tables/stats_correlations.csv', index=False, float_format='%.4f')
print("\n✅ Saved to: results/analysis/tables/stats_correlations.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*120)
print("STATISTICAL TESTS SUMMARY")
print("="*120)

print("\n📊 Strategy Comparisons (Paired t-tests):")
for _, row in strategy_df.iterrows():
    print(f"   {row['Baseline']} vs {row['Comparison']}: "
          f"Δ={row['Mean_Diff']:.2f}%, t={row['t_statistic']:.2f}, "
          f"p={row['p_value']:.4f} {row['Significant']}, "
          f"p_bonf={row['p_value_bonferroni']:.4f} {row['Significant_bonferroni']}, "
          f"d={row['Cohens_d']:.2f}")

print("\n📊 Strategy Comparisons (Wilcoxon):")
for _, row in wilcoxon_df.iterrows():
    print(f"   {row['Baseline']} vs {row['Comparison']}: "
          f"ΔMedian={row['Median_Diff']:.2f}%, W={row['W_statistic']:.0f}, "
          f"p={row['p_value']:.4f} {row['Significant']}, "
          f"p_bonf={row['p_value_bonferroni']:.4f} {row['Significant_bonferroni']}")

print("\n📊 Strategy Effects (Friedman):")
print(f"   χ²={strategy_friedman_results['Chi2_statistic']:.2f}, df={strategy_friedman_results['df']}, "
      f"p={strategy_friedman_results['p_value']:.4f} {strategy_friedman_results['Significant']}")

print("\n🌡️  Temperature Effects (Friedman):")
print(f"   χ²={temp_results['Chi2_statistic']:.2f}, df={temp_results['df']}, "
      f"p={temp_results['p_value']:.4f} {temp_results['Significant']}")

print("\n🧠 Belief Bias Significance:")
print(f"   Congruent ({bias_test['Mean_Congruent']:.1f}%) vs "
      f"Incongruent ({bias_test['Mean_Incongruent']:.1f}%): "
      f"Δ={bias_test['Mean_Bias_Effect']:.2f}pp, t={bias_test['t_statistic']:.2f}, "
      f"p={bias_test['p_value']:.4f} {bias_test['Significant']}, d={bias_test['Cohens_d']:.2f}")

print("\n🔗 Key Correlations:")
for _, row in corr_df.iterrows():
    print(f"   {row['Variable_1']} × {row['Variable_2']}: "
          f"r={row['Pearson_r']:.3f} {row['Pearson_Sig']}, "
          f"ρ={row['Spearman_r']:.3f} {row['Spearman_Sig']}")

print("\n" + "="*120)
print("✅ ALL STATISTICAL TESTS COMPLETE")
print("="*120)
print(f"\n📁 Results saved to results/analysis/tables/stats_*.csv")
print(f"   • stats_strategy_ttests.csv")
print(f"   • stats_strategy_wilcoxon.csv")
print(f"   • stats_strategy_mcnemar.csv")
print(f"   • stats_strategy_friedman.csv")
print(f"   • stats_temperature_friedman.csv")
print(f"   • stats_belief_bias_test.csv")
print(f"   • stats_correlations.csv")
