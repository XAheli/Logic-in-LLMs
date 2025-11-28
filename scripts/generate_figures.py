#!/usr/bin/env python3
"""
Figure Generation Script for Syllogistic Reasoning Benchmark

Generates publication-quality figures:
1. Accuracy heatmaps
2. Model ranking charts
3. Correlation scatter plots
4. Consistency visualizations
5. Temperature effect plots

Usage:
    python scripts/generate_figures.py [options]
    
    Options:
        --results-dir PATH    Directory containing analysis results
        --output-dir PATH     Directory for figures (default: results_dir/figures)
        --format FORMAT       Image format: pdf, png, svg (default: pdf)
        --dpi DPI             Resolution for raster formats (default: 300)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import config


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate figures for syllogistic reasoning benchmark"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing analysis results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for figures (default: results_dir/figures)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['pdf', 'png', 'svg'],
        default='pdf',
        help='Image format (default: pdf)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for raster formats (default: 300)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures interactively'
    )
    
    return parser.parse_args()


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_accuracy_figures(
    analysis_dir: Path,
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Generate accuracy-related figures."""
    print("\n📊 Generating Accuracy Figures...")
    
    from src.analysis.visualization import (
        plot_accuracy_heatmap,
        plot_accuracy_by_strategy,
        plot_temperature_effect
    )
    
    # Load metrics
    metrics_file = analysis_dir / "accuracy_metrics.csv"
    if not metrics_file.exists():
        metrics_file = analysis_dir / "accuracy_metrics.json"
    
    if not metrics_file.exists():
        print("   ⚠️  Accuracy metrics not found. Run analyze_results.py first.")
        return
    
    if metrics_file.suffix == '.csv':
        df = pd.read_csv(metrics_file)
    else:
        df = pd.read_json(metrics_file)
    
    # 1. Accuracy heatmap
    print("   - Generating accuracy heatmap...")
    fig = plot_accuracy_heatmap(
        df,
        output_path=output_dir / f"accuracy_heatmap.{fmt}",
        title="Model Accuracy by Configuration"
    )
    print(f"   ✅ Saved: accuracy_heatmap.{fmt}")
    
    # 2. Accuracy by strategy (for T=0)
    print("   - Generating accuracy by strategy...")
    for temp in [0.0, 0.5, 1.0]:
        fig = plot_accuracy_by_strategy(
            df,
            output_path=output_dir / f"accuracy_by_strategy_T{temp}.{fmt}",
            temperature=temp
        )
        print(f"   ✅ Saved: accuracy_by_strategy_T{temp}.{fmt}")
    
    # 3. Temperature effect
    print("   - Generating temperature effect plot...")
    fig = plot_temperature_effect(
        df,
        output_path=output_dir / f"temperature_effect.{fmt}"
    )
    print(f"   ✅ Saved: temperature_effect.{fmt}")


def generate_ranking_figures(
    analysis_dir: Path,
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Generate ranking-related figures."""
    print("\n🏆 Generating Ranking Figures...")
    
    from src.analysis.visualization import plot_model_ranking
    
    # Load rankings
    rankings_file = analysis_dir / "model_rankings.csv"
    if not rankings_file.exists():
        rankings_file = analysis_dir / "model_rankings.json"
    
    if not rankings_file.exists():
        print("   ⚠️  Model rankings not found. Run analyze_results.py first.")
        return
    
    if rankings_file.suffix == '.csv':
        df = pd.read_csv(rankings_file)
    else:
        df = pd.read_json(rankings_file)
    
    # Model ranking chart
    print("   - Generating model ranking chart...")
    fig = plot_model_ranking(
        df,
        output_path=output_dir / f"model_ranking.{fmt}",
        top_n=min(20, len(df))
    )
    print(f"   ✅ Saved: model_ranking.{fmt}")


def generate_correlation_figures(
    analysis_dir: Path,
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Generate correlation-related figures."""
    print("\n📈 Generating Correlation Figures...")
    
    from src.analysis.visualization import plot_accuracy_vs_arena, plot_ranking_comparison
    from src.analysis.correlation import LM_ARENA_RANKINGS, load_model_accuracies
    
    # Load accuracies from results (need original results dir)
    results_dir = analysis_dir.parent
    
    # Try to load for T=0, zero_shot
    accuracies = load_model_accuracies(results_dir, 0.0, "zero_shot")
    
    if not accuracies:
        print("   ⚠️  No accuracy data found for correlation plot.")
        return
    
    # 1. Accuracy vs Arena rank
    print("   - Generating accuracy vs arena plot...")
    fig = plot_accuracy_vs_arena(
        accuracies,
        LM_ARENA_RANKINGS,
        output_path=output_dir / f"accuracy_vs_arena.{fmt}"
    )
    if fig:
        print(f"   ✅ Saved: accuracy_vs_arena.{fmt}")
    else:
        print("   ⚠️  Insufficient data for accuracy vs arena plot")
    
    # 2. Ranking comparison
    print("   - Generating ranking comparison plot...")
    
    # Create our rankings
    sorted_models = sorted(accuracies.items(), key=lambda x: -x[1])
    our_rankings = {m: i + 1 for i, (m, _) in enumerate(sorted_models)}
    
    fig = plot_ranking_comparison(
        our_rankings,
        LM_ARENA_RANKINGS,
        output_path=output_dir / f"ranking_comparison.{fmt}"
    )
    if fig:
        print(f"   ✅ Saved: ranking_comparison.{fmt}")
    else:
        print("   ⚠️  Insufficient data for ranking comparison plot")


def generate_consistency_figures(
    analysis_dir: Path,
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Generate consistency-related figures."""
    print("\n🔄 Generating Consistency Figures...")
    
    from src.analysis.visualization import plot_consistency_by_model, plot_content_effects
    
    # Load consistency data
    consistency_file = analysis_dir / "consistency_analysis.csv"
    if not consistency_file.exists():
        consistency_file = analysis_dir / "consistency_analysis.json"
    
    if not consistency_file.exists():
        print("   ⚠️  Consistency analysis not found. Run analyze_results.py first.")
        return
    
    if consistency_file.suffix == '.csv':
        df = pd.read_csv(consistency_file)
    else:
        df = pd.read_json(consistency_file)
    
    # Filter to T=0, zero_shot if available
    if 'config' in df.columns:
        filtered = df[df['config'] == 'temp0.0_zero_shot']
        if filtered.empty:
            filtered = df.head(26)  # Take first config
    else:
        filtered = df
    
    # 1. Consistency by model
    print("   - Generating consistency by model plot...")
    fig = plot_consistency_by_model(
        filtered,
        output_path=output_dir / f"consistency_by_model.{fmt}"
    )
    print(f"   ✅ Saved: consistency_by_model.{fmt}")
    
    # 2. Content effects
    print("   - Generating content effects plot...")
    if all(col in filtered.columns for col in ['name_effect', 'order_effect', 'combined_effect']):
        fig = plot_content_effects(
            filtered,
            output_path=output_dir / f"content_effects.{fmt}"
        )
        print(f"   ✅ Saved: content_effects.{fmt}")
    else:
        print("   ⚠️  Missing content effect columns")


def generate_summary_figure(
    analysis_dir: Path,
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Generate a summary multi-panel figure."""
    print("\n📋 Generating Summary Figure...")
    
    import matplotlib.pyplot as plt
    from src.analysis.visualization import set_publication_style
    
    set_publication_style()
    
    # This would create a multi-panel figure combining key results
    # For now, just create a placeholder
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fig.suptitle('Syllogistic Reasoning Benchmark - Summary Results', fontsize=14)
    
    # Placeholder text in each panel
    for ax, title in zip(axes.flat, [
        'Accuracy by Model',
        'Accuracy by Strategy',
        'Correlation with LM Arena',
        'Cross-Variant Consistency'
    ]):
        ax.text(0.5, 0.5, f'[{title}]', ha='center', va='center', fontsize=12)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    fig.savefig(output_dir / f"summary_figure.{fmt}", dpi=dpi)
    print(f"   ✅ Saved: summary_figure.{fmt}")
    
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("SYLLOGISTIC REASONING BENCHMARK - FIGURE GENERATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set directories
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = config.experiment.results_full_path
    
    analysis_dir = results_dir / "analysis"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results directory: {results_dir}")
    print(f"📁 Analysis directory: {analysis_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📐 Format: {args.format.upper()}, DPI: {args.dpi}")
    
    # Check analysis exists
    if not analysis_dir.exists():
        print(f"\n❌ Error: Analysis directory not found: {analysis_dir}")
        print("   Run analyze_results.py first to generate analysis files.")
        sys.exit(1)
    
    # Generate figures
    generate_accuracy_figures(analysis_dir, output_dir, args.format, args.dpi)
    generate_ranking_figures(analysis_dir, output_dir, args.format, args.dpi)
    generate_correlation_figures(analysis_dir, output_dir, args.format, args.dpi)
    generate_consistency_figures(analysis_dir, output_dir, args.format, args.dpi)
    generate_summary_figure(analysis_dir, output_dir, args.format, args.dpi)
    
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"📁 All figures saved to: {output_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob(f"*.{args.format}")):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()
