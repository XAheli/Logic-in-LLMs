#!/usr/bin/env python3
"""
Validation utilities for checking statistical calculations and interpretations

Subcommands:
    correlation  - Debug correlation calculation (model order matching)
    bias         - Verify belief bias interpretation
    all          - Run all validation checks
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def validate_correlation(tables_dir):
    """
    Debug correlation calculation - verify models are matched correctly

    Common issue: Computing correlation without matching models by ID
    leads to incorrect results when tables are sorted differently.
    """
    print("="*80)
    print("VALIDATING CORRELATION CALCULATION")
    print("="*80)

    # Load data
    complete_path = tables_dir / 'paper_table1_complete.csv'
    bias_path = tables_dir / 'paper_table3_belief_bias.csv'

    if not complete_path.exists() or not bias_path.exists():
        print(f"❌ Required tables not found:")
        print(f"   {complete_path}")
        print(f"   {bias_path}")
        return False

    complete_df = pd.read_csv(complete_path)
    bias_df = pd.read_csv(bias_path)

    # Check model order
    print("\n1. Checking model order:")
    print("-" * 80)
    print(f"   Complete table: sorted by {complete_df.columns[1] if len(complete_df.columns) > 1 else 'unknown'}")
    print(f"   Bias table: sorted by Bias_Effect (descending)")

    same_order = complete_df['Model'].tolist() == bias_df['Model'].tolist()
    print(f"   Models in same order? {same_order}")

    if not same_order:
        print("   ⚠️  WARNING: Tables have different model orders!")

    # Correlation WITHOUT matching (incorrect if different order)
    print("\n2. Correlation WITHOUT model matching (INCORRECT if different order):")
    print("-" * 80)

    if 'Syntax_Acc' in complete_df.columns and 'Bias_Effect' in bias_df.columns:
        pearson_r, pearson_p = pearsonr(complete_df['Syntax_Acc'], bias_df['Bias_Effect'])
        spearman_r, spearman_p = spearmanr(complete_df['Syntax_Acc'], bias_df['Bias_Effect'])

        print(f"   Pearson r: {pearson_r:.4f}, p: {pearson_p:.4f}")
        print(f"   Spearman ρ: {spearman_r:.4f}, p: {spearman_p:.4f}")

        if not same_order:
            print("   ⚠️  These values are INCORRECT - models not matched!")

    # Correlation WITH matching (correct)
    print("\n3. Correlation WITH model matching (CORRECT):")
    print("-" * 80)

    merged = pd.merge(
        complete_df[['Model', 'Syntax_Acc']],
        bias_df[['Model', 'Bias_Effect']],
        on='Model'
    )

    spearman_r_correct, spearman_p_correct = spearmanr(
        merged['Syntax_Acc'],
        merged['Bias_Effect']
    )

    print(f"   Spearman ρ: {spearman_r_correct:.4f}, p: {spearman_p_correct:.4f}")
    print(f"   Models matched: {len(merged)}")

    # Show example data
    print("\n4. Sample data (first 5 models from each table):")
    print("-" * 80)
    print(f"{'Complete Table':<40} {'Bias Table':<40}")
    print(f"{'Model':<30} {'Acc':>8}  {'Model':<30} {'Bias':>8}")
    print("-" * 80)

    for i in range(min(5, len(complete_df), len(bias_df))):
        print(f"{complete_df.iloc[i]['Model']:<30} {complete_df.iloc[i]['Syntax_Acc']:>8.2f}  "
              f"{bias_df.iloc[i]['Model']:<30} {bias_df.iloc[i]['Bias_Effect']:>8.2f}")

    print("\n✅ Correlation validation complete")
    print(f"   Correct correlation: ρ = {spearman_r_correct:.4f}, p = {spearman_p_correct:.4f}")

    return True


def validate_bias_interpretation(tables_dir):
    """
    Verify belief bias correlation interpretation

    Check that the interpretation of bias effect correlation is correct:
    - Bias_Effect = Congruent_Acc - Incongruent_Acc
    - Positive values = bias present (better on congruent)
    - Negative values = reverse bias (better on incongruent)
    """
    print("="*80)
    print("VALIDATING BELIEF BIAS INTERPRETATION")
    print("="*80)

    # Load data
    complete_path = tables_dir / 'paper_table1_complete.csv'
    bias_path = tables_dir / 'paper_table3_belief_bias.csv'

    if not complete_path.exists() or not bias_path.exists():
        print(f"❌ Required tables not found")
        return False

    complete = pd.read_csv(complete_path)
    bias = pd.read_csv(bias_path)

    # Merge and sort by accuracy
    merged = pd.merge(
        complete[['Model', 'Syntax_Acc']],
        bias[['Model', 'Bias_Effect']],
        on='Model'
    ).sort_values('Syntax_Acc', ascending=False)

    print("\n1. Models ranked by accuracy (best to worst):")
    print("-" * 80)
    print(f"{'Model':<40} {'Accuracy':>10} {'Bias Effect':>12}")
    print("-" * 80)

    for _, row in merged.head(10).iterrows():
        print(f"{row['Model']:<40} {row['Syntax_Acc']:>10.2f} {row['Bias_Effect']:>12.2f}")

    if len(merged) > 10:
        print("   ...")
        for _, row in merged.tail(3).iterrows():
            print(f"{row['Model']:<40} {row['Syntax_Acc']:>10.2f} {row['Bias_Effect']:>12.2f}")

    # Calculate correlation
    rho, p_value = spearmanr(merged['Syntax_Acc'], merged['Bias_Effect'])

    print("\n2. Correlation analysis:")
    print("-" * 80)
    print(f"   Spearman ρ: {rho:.4f}, p: {p_value:.4f}")

    # Interpretation guide
    print("\n3. Interpretation guide:")
    print("-" * 80)
    print("   Bias_Effect = Congruent_Acc - Incongruent_Acc")
    print("     • Positive value: Model does BETTER on congruent (belief bias present)")
    print("     • Negative value: Model does BETTER on incongruent (reverse bias)")
    print("     • Near-zero: No bias (equal performance)")
    print()

    # Analyze pattern
    best_models = merged.nlargest(3, 'Syntax_Acc')
    worst_models = merged.nsmallest(3, 'Syntax_Acc')

    best_acc_range = f"{best_models['Syntax_Acc'].min():.1f}-{best_models['Syntax_Acc'].max():.1f}%"
    best_bias_range = f"{best_models['Bias_Effect'].min():.2f} to {best_models['Bias_Effect'].max():.2f}pp"

    worst_acc_range = f"{worst_models['Syntax_Acc'].min():.1f}-{worst_models['Syntax_Acc'].max():.1f}%"
    worst_bias_range = f"{worst_models['Bias_Effect'].min():.2f} to {worst_models['Bias_Effect'].max():.2f}pp"

    print(f"   Best models (acc {best_acc_range}): bias = {best_bias_range}")
    print(f"   Worst models (acc {worst_acc_range}): bias = {worst_bias_range}")

    print("\n4. Correct interpretation:")
    print("-" * 80)

    if rho < -0.5:
        print(f"   ✅ NEGATIVE correlation (ρ={rho:.3f})")
        print("      → Better models have SMALLER bias (closer to zero)")
        print("      → Worse models have LARGER bias (farther from zero)")
        print("      → Higher reasoning ability → Less susceptible to belief bias")
    elif rho > 0.5:
        print(f"   ⚠️  POSITIVE correlation (ρ={rho:.3f})")
        print("      → Better models have LARGER bias")
        print("      → This would be unusual - check data!")
    else:
        print(f"   📊 WEAK correlation (ρ={rho:.3f})")
        print("      → No clear relationship between accuracy and bias")

    print("\n✅ Bias interpretation validation complete")

    return True


def validate_all(tables_dir):
    """Run all validation checks"""
    print("\n" + "="*80)
    print("RUNNING ALL VALIDATION CHECKS")
    print("="*80 + "\n")

    success = True

    # 1. Correlation validation
    if not validate_correlation(tables_dir):
        success = False

    print("\n")

    # 2. Bias interpretation validation
    if not validate_bias_interpretation(tables_dir):
        success = False

    print("\n" + "="*80)
    if success:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("="*80)

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Validation utilities for statistical calculations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  correlation  - Validate correlation calculation (check model matching)
  bias         - Validate belief bias interpretation
  all          - Run all validation checks (default)

Examples:
  %(prog)s correlation        # Check correlation calculation
  %(prog)s bias               # Check bias interpretation
  %(prog)s all                # Run all checks
  %(prog)s --tables-dir path  # Use custom tables directory
        """
    )

    parser.add_argument('check', nargs='?', default='all',
                        choices=['correlation', 'bias', 'all'],
                        help='Which validation check to run (default: all)')
    parser.add_argument('--tables-dir', type=Path,
                        default=Path('results/analysis/tables'),
                        help='Directory containing table CSV files')

    args = parser.parse_args()

    # Verify tables directory exists
    if not args.tables_dir.exists():
        print(f"❌ Tables directory not found: {args.tables_dir}")
        print(f"   Make sure to run this from the project root directory")
        return 1

    # Run requested validation
    if args.check == 'correlation':
        success = validate_correlation(args.tables_dir)
    elif args.check == 'bias':
        success = validate_bias_interpretation(args.tables_dir)
    else:  # 'all'
        success = validate_all(args.tables_dir)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
