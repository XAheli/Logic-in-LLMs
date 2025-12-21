#!/usr/bin/env python3
"""
Proper benchmark correlation analysis
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import re

# Change to project root directory for relative paths
os.chdir(Path(__file__).parent.parent.parent)

# Load our model performance
complete = pd.read_csv('results/analysis/tables/paper_table1_complete.csv')

# Load benchmarks
lmarena = pd.read_csv('LMarena_benchmark.csv')
mmlu = pd.read_csv('data/MMLU_helm.csv')

# Standardize model names for matching
def standardize_name(name):
    """Convert model names to standardized format for matching"""
    name = name.lower()
    name = re.sub(r'[_\-\.\s]+', '_', name)  # Replace separators with underscore
    name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
    name = name.strip('_')
    return name

complete['model_std'] = complete['Model'].apply(standardize_name)
lmarena['model_std'] = lmarena['Model'].apply(standardize_name)
mmlu['model_std'] = mmlu['Model'].apply(standardize_name)

print("="*100)
print("BENCHMARK CORRELATION ANALYSIS - PROPERLY INTERPRETED")
print("="*100)

# ============================================================================
# 1. LMARENA CORRELATION
# ============================================================================
print("\n" + "="*100)
print("1. LMARENA CORRELATION (Lower rank = Better model)")
print("="*100)

lm_merged = pd.merge(
    complete[['Model', 'Syntax_Acc', 'model_std']], 
    lmarena[['Model', 'Overall', 'model_std']], 
    on='model_std', 
    suffixes=('_ours', '_lm')
)

print(f"\nMatched {len(lm_merged)} models:")
print("\nSample of matched data (sorted by LMArena rank):")
print(lm_merged[['Model_ours', 'Syntax_Acc', 'Overall']].sort_values('Overall').head(15).to_string(index=False))

# Calculate correlation
rho_lm, p_lm = spearmanr(lm_merged['Syntax_Acc'], lm_merged['Overall'])

print(f"\n{'='*100}")
print("RESULTS:")
print(f"{'='*100}")
print(f"Spearman ρ = {rho_lm:.4f}")
print(f"p-value = {p_lm:.4f} {'***' if p_lm < 0.001 else '**' if p_lm < 0.01 else '*' if p_lm < 0.05 else 'ns'}")
print(f"\nINTERPRETATION:")
print(f"  • LMArena: Lower rank number (1, 2, 3...) = BETTER performance")
print(f"  • Our metric: Higher accuracy = BETTER performance")
print(f"  • Expected correlation: NEGATIVE (better our accuracy ↔ lower LM rank)")
print(f"  • Observed: ρ = {rho_lm:.4f}")
if rho_lm < 0:
    print(f"  • ✓ CORRECT: Negative correlation means better reasoning correlates with better LMArena rank")
    print(f"  • Magnitude: {abs(rho_lm):.3f} = {'Strong' if abs(rho_lm) > 0.7 else 'Moderate' if abs(rho_lm) > 0.3 else 'Weak'}")
    print(f"  • Conclusion: Instruction-following (LMArena) {'PREDICTS' if p_lm < 0.05 else 'does NOT predict'} logical reasoning")
else:
    print(f"  • ✗ WARNING: Positive correlation is unexpected!")

# ============================================================================
# 2. MMLU CORRELATION
# ============================================================================
print("\n" + "="*100)
print("2. MMLU CORRELATION (Higher score = Better model)")
print("="*100)

# MMLU uses "MMLU All Subjects - EM" column
mmlu_merged = pd.merge(
    complete[['Model', 'Syntax_Acc', 'model_std']], 
    mmlu[['Model', 'MMLU All Subjects - EM', 'model_std']], 
    on='model_std',
    suffixes=('_ours', '_mmlu')
)

print(f"\nMatched {len(mmlu_merged)} models:")
print("\nSample of matched data (sorted by MMLU score):")
print(mmlu_merged[['Model_ours', 'Syntax_Acc', 'MMLU All Subjects - EM']].sort_values('MMLU All Subjects - EM', ascending=False).head(15).to_string(index=False))

# Calculate correlation
rho_mmlu, p_mmlu = spearmanr(mmlu_merged['Syntax_Acc'], mmlu_merged['MMLU All Subjects - EM'])

print(f"\n{'='*100}")
print("RESULTS:")
print(f"{'='*100}")
print(f"Spearman ρ = {rho_mmlu:.4f}")
print(f"p-value = {p_mmlu:.4f} {'***' if p_mmlu < 0.001 else '**' if p_mmlu < 0.01 else '*' if p_mmlu < 0.05 else 'ns'}")
print(f"\nINTERPRETATION:")
print(f"  • MMLU: Higher score = BETTER performance")
print(f"  • Our metric: Higher accuracy = BETTER performance")
print(f"  • Expected correlation: POSITIVE if factual knowledge predicts reasoning")
print(f"  • Observed: ρ = {rho_mmlu:.4f}")
if rho_mmlu > 0:
    print(f"  • Direction: Positive ({'significant' if p_mmlu < 0.05 else 'not significant'})")
    print(f"  • Magnitude: {abs(rho_mmlu):.3f} = {'Strong' if abs(rho_mmlu) > 0.7 else 'Moderate' if abs(rho_mmlu) > 0.3 else 'Weak'}")
    print(f"  • Conclusion: Factual knowledge (MMLU) {'PREDICTS' if p_mmlu < 0.05 else 'does NOT predict'} logical reasoning")
else:
    print(f"  • Direction: Negative (unexpected - would mean more knowledge = worse reasoning)")

# ============================================================================
# SUMMARY FOR PAPER
# ============================================================================
print("\n" + "="*100)
print("SUMMARY FOR PAPER")
print("="*100)

print(f"\nLMArena (N={len(lm_merged)}):")
print(f"  • Spearman ρ = {rho_lm:.3f}, p = {p_lm:.4f}{'***' if p_lm < 0.001 else '**' if p_lm < 0.01 else '*' if p_lm < 0.05 else ' (ns)'}")
print(f"  • Interpretation: {'Strong' if abs(rho_lm) > 0.7 else 'Moderate' if abs(rho_lm) > 0.3 else 'Weak'} negative correlation")
print(f"  • Conclusion: Instruction-following quality (LMArena) {'PREDICTS' if p_lm < 0.05 else 'does NOT predict'} syllogistic reasoning ability")

print(f"\nMMLU (N={len(mmlu_merged)}):")
print(f"  • Spearman ρ = {rho_mmlu:.3f}, p = {p_mmlu:.4f}{'***' if p_mmlu < 0.001 else '**' if p_mmlu < 0.01 else '*' if p_mmlu < 0.05 else ' (ns)'}")
print(f"  • Interpretation: {'Strong' if abs(rho_mmlu) > 0.7 else 'Moderate' if abs(rho_mmlu) > 0.3 else 'Weak'} {'positive' if rho_mmlu > 0 else 'negative'} correlation")
print(f"  • Conclusion: Factual knowledge breadth (MMLU) {'PREDICTS' if p_mmlu < 0.05 else 'does NOT predict'} syllogistic reasoning ability")

print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)
if p_lm < 0.05 and p_mmlu >= 0.05:
    print("✓ Instruction-following (careful task adherence) predicts logical reasoning")
    print("✓ But factual knowledge breadth does NOT predict logical reasoning")
    print("✓ This dissociation suggests reasoning ≠ knowledge accumulation")
elif p_lm < 0.05 and p_mmlu < 0.05:
    print("✓ Both instruction-following AND factual knowledge predict reasoning")
else:
    print("• Results show complex relationship between benchmarks and reasoning")

# Save results
results_df = pd.DataFrame({
    'Benchmark': ['LMArena', 'MMLU'],
    'N': [len(lm_merged), len(mmlu_merged)],
    'Spearman_rho': [rho_lm, rho_mmlu],
    'p_value': [p_lm, p_mmlu],
    'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' for p in [p_lm, p_mmlu]]
})
results_df.to_csv('results/analysis/tables/stats_benchmark_correlations.csv', index=False)
print(f"\n✅ Saved to: results/analysis/tables/stats_benchmark_correlations.csv")
