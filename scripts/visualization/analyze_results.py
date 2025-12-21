#!/usr/bin/env python3
"""
Results Analysis Script for Syllogistic Reasoning Benchmark

This script analyzes experiment results:
1. Calculates accuracy metrics
2. Performs consistency analysis
3. Computes correlation with LM Arena
4. Generates summary reports

Usage:
    python scripts/analyze_results.py [options]
    
    Options:
        --results-dir PATH    Directory containing results (default: from config)
        --output-dir PATH     Directory for analysis outputs (default: results_dir/analysis)
        --temperature T       Analyze specific temperature only
        --strategy S          Analyze specific strategy only
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze syllogistic reasoning benchmark results"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing results (default: from config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for analysis outputs (default: results_dir/analysis)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Analyze specific temperature only'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Analyze specific strategy only'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    
    return parser.parse_args()


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_metrics(results_dir: Path, output_dir: Path, args):
    """Calculate and save accuracy metrics."""
    print("\n📊 Calculating Accuracy Metrics...")
    
    from src.evaluation.calculate_metrics import create_summary_table
    
    df = create_summary_table(results_dir)
    
    if df.empty:
        print("   ⚠️  No results found")
        return None
    
    # Filter if specified
    if args.temperature is not None:
        df = df[df['temperature'] == args.temperature]
    if args.strategy is not None:
        df = df[df['strategy'] == args.strategy]
    
    # Save
    if args.format in ['csv', 'both']:
        csv_path = output_dir / "accuracy_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved: {csv_path}")
    
    if args.format in ['json', 'both']:
        json_path = output_dir / "accuracy_metrics.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"   ✅ Saved: {json_path}")
    
    # Print summary
    print(f"\n   Summary ({len(df)} configurations):")
    print(f"   - Mean accuracy: {df['accuracy'].mean():.2%}")
    print(f"   - Max accuracy: {df['accuracy'].max():.2%}")
    print(f"   - Min accuracy: {df['accuracy'].min():.2%}")
    
    return df


def analyze_consistency(results_dir: Path, output_dir: Path, args):
    """Perform consistency analysis."""
    print("\n🔄 Analyzing Cross-Variant Consistency...")
    
    from src.evaluation.consistency_analysis import create_consistency_summary
    
    summaries = create_consistency_summary(results_dir)
    
    if not summaries:
        print("   ⚠️  No results found for consistency analysis")
        return None
    
    all_data = []
    for config_key, df in summaries.items():
        # Filter if specified
        temp_str = config_key.split('_')[0].replace('temp', '')
        strategy = '_'.join(config_key.split('_')[1:])
        
        if args.temperature is not None and float(temp_str) != args.temperature:
            continue
        if args.strategy is not None and strategy != args.strategy:
            continue
        
        df['config'] = config_key
        all_data.append(df)
    
    if not all_data:
        print("   ⚠️  No matching configurations found")
        return None
    
    import pandas as pd
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save
    if args.format in ['csv', 'both']:
        csv_path = output_dir / "consistency_analysis.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved: {csv_path}")
    
    if args.format in ['json', 'both']:
        json_path = output_dir / "consistency_analysis.json"
        combined_df.to_json(json_path, orient='records', indent=2)
        print(f"   ✅ Saved: {json_path}")
    
    # Print summary
    print(f"\n   Summary ({len(combined_df)} model-config pairs):")
    print(f"   - Mean consistency: {combined_df['overall_consistency'].mean():.2%}")
    print(f"   - Mean name effect: {combined_df['name_effect'].mean():.2%}")
    print(f"   - Mean order effect: {combined_df['order_effect'].mean():.2%}")
    
    return combined_df


def analyze_correlation(results_dir: Path, output_dir: Path, args):
    """Calculate correlation with LM Arena rankings."""
    print("\n📈 Calculating Correlation with LM Arena...")
    
    from src.analysis.correlation import analyze_correlation_across_configs
    
    df = analyze_correlation_across_configs(results_dir)
    
    if df.empty:
        print("   ⚠️  No results found for correlation analysis")
        return None
    
    # Filter if specified
    if args.temperature is not None:
        df = df[df['temperature'] == args.temperature]
    if args.strategy is not None:
        df = df[df['strategy'] == args.strategy]
    
    # Save
    if args.format in ['csv', 'both']:
        csv_path = output_dir / "correlation_analysis.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved: {csv_path}")
    
    if args.format in ['json', 'both']:
        json_path = output_dir / "correlation_analysis.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"   ✅ Saved: {json_path}")
    
    # Print summary
    print(f"\n   Summary ({len(df)} configurations):")
    print(f"   - Mean Spearman ρ: {df['spearman_rho'].mean():.3f}")
    sig_count = df['significant_005'].sum()
    print(f"   - Significant (p<0.05): {sig_count}/{len(df)}")
    
    return df


def analyze_rankings(results_dir: Path, output_dir: Path, args):
    """Create model rankings."""
    print("\n🏆 Creating Model Rankings...")
    
    from src.analysis.ranking import (
        create_ranking_table,
        create_aggregate_ranking_table,
        compare_paid_vs_free
    )
    
    # Aggregate ranking
    agg_df = create_aggregate_ranking_table(results_dir)
    
    if agg_df.empty:
        print("   ⚠️  No results found for ranking")
        return None
    
    # Save aggregate rankings
    if args.format in ['csv', 'both']:
        csv_path = output_dir / "model_rankings.csv"
        agg_df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved: {csv_path}")
    
    if args.format in ['json', 'both']:
        json_path = output_dir / "model_rankings.json"
        agg_df.to_json(json_path, orient='records', indent=2)
        print(f"   ✅ Saved: {json_path}")
    
    # Print top 10
    print(f"\n   Top 10 Models (aggregate accuracy):")
    top10 = agg_df.head(10)
    for _, row in top10.iterrows():
        billing = f"[{row['billing_type']}]" if 'billing_type' in row else ""
        print(f"   #{int(row['rank'])}: {row['model_key']} ({row['accuracy']:.2%}) {billing}")
    
    # Google Studio vs HF Inference comparison
    temp = args.temperature if args.temperature is not None else 0.0
    strat = args.strategy if args.strategy is not None else "zero_shot"
    
    comparison = compare_paid_vs_free(results_dir, temp, strat)
    
    print(f"\n   Google Studio vs HF Inference (T={temp}, {strat}):")
    print(f"   - Google Studio ({comparison['google_studio_paid']['count']} models): {comparison['google_studio_paid']['mean_accuracy']:.2%} mean accuracy")
    print(f"   - HF Inference ({comparison['hf_inf_paid']['count']} models): {comparison['hf_inf_paid']['mean_accuracy']:.2%} mean accuracy")
    
    return agg_df


def generate_summary_report(
    metrics_df,
    consistency_df,
    correlation_df,
    ranking_df,
    output_dir: Path
):
    """Generate a comprehensive summary report."""
    print("\n📄 Generating Summary Report...")
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {}
    }
    
    if metrics_df is not None:
        report["summary"]["accuracy"] = {
            "total_configurations": len(metrics_df),
            "mean_accuracy": float(metrics_df['accuracy'].mean()),
            "max_accuracy": float(metrics_df['accuracy'].max()),
            "min_accuracy": float(metrics_df['accuracy'].min()),
            "std_accuracy": float(metrics_df['accuracy'].std())
        }
    
    if consistency_df is not None:
        report["summary"]["consistency"] = {
            "mean_consistency": float(consistency_df['overall_consistency'].mean()),
            "mean_name_effect": float(consistency_df['name_effect'].mean()),
            "mean_order_effect": float(consistency_df['order_effect'].mean()),
            "mean_combined_effect": float(consistency_df['combined_effect'].mean())
        }
    
    if correlation_df is not None:
        report["summary"]["correlation"] = {
            "mean_spearman_rho": float(correlation_df['spearman_rho'].mean()),
            "significant_configs_005": int(correlation_df['significant_005'].sum()),
            "total_configs": len(correlation_df)
        }
    
    if ranking_df is not None:
        top_model = ranking_df.iloc[0]
        report["summary"]["top_model"] = {
            "model": top_model['model_key'],
            "accuracy": float(top_model['accuracy']),
            "billing_type": str(top_model['billing_type']) if 'billing_type' in top_model else "unknown"
        }
    
    # Save report
    report_path = output_dir / "summary_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✅ Saved: {report_path}")
    
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("SYLLOGISTIC REASONING BENCHMARK - RESULTS ANALYSIS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set directories
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = config.experiment.results_full_path
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results directory: {results_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check results exist
    if not results_dir.exists():
        print(f"\n❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Run analyses
    metrics_df = analyze_metrics(results_dir, output_dir, args)
    consistency_df = analyze_consistency(results_dir, output_dir, args)
    correlation_df = analyze_correlation(results_dir, output_dir, args)
    ranking_df = analyze_rankings(results_dir, output_dir, args)
    
    # Generate summary
    generate_summary_report(
        metrics_df, consistency_df, correlation_df, ranking_df, output_dir
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"📁 All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
