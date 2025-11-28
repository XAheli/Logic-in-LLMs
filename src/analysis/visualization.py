"""
Visualization Module for Syllogistic Reasoning Benchmark

Creates publication-quality figures:
- Accuracy heatmaps
- Ranking comparison charts
- Correlation plots
- Consistency visualizations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.config import config


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Publication-quality settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'pdf'

# Color schemes
PROVIDER_COLORS = {
    'google': '#4285F4',      # Google Blue
    'huggingface': '#FF9D00', # HuggingFace Yellow/Orange
}

STRATEGY_COLORS = {
    'zero_shot': '#2E86AB',
    'one_shot': '#A23B72',
    'few_shot': '#F18F01',
    'zero_shot_cot': '#C73E1D',
}

TEMPERATURE_MARKERS = {
    0.0: 'o',
    0.5: 's',
    1.0: '^',
}


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# =============================================================================
# ACCURACY VISUALIZATIONS
# =============================================================================

def plot_accuracy_heatmap(
    accuracy_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Model Accuracy by Configuration"
) -> plt.Figure:
    """
    Create accuracy heatmap with models vs configurations.
    
    Args:
        accuracy_df: DataFrame with columns [model, temperature, strategy, accuracy]
        output_path: Path to save figure
        title: Figure title
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    # Pivot to create heatmap data
    pivot_df = accuracy_df.pivot_table(
        values='accuracy',
        index='model',
        columns=['strategy', 'temperature'],
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, len(pivot_df) * 0.4 + 2))
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Accuracy'}
    )
    
    ax.set_title(title)
    ax.set_xlabel('Configuration (Strategy, Temperature)')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


def plot_accuracy_by_strategy(
    accuracy_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    temperature: float = 0.0
) -> plt.Figure:
    """
    Create bar chart of accuracy by strategy.
    
    Args:
        accuracy_df: DataFrame with accuracy data
        output_path: Path to save figure
        temperature: Temperature to filter by
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    # Filter and aggregate
    filtered = accuracy_df[accuracy_df['temperature'] == temperature]
    strategy_acc = filtered.groupby('strategy')['accuracy'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [STRATEGY_COLORS.get(s, '#888888') for s in strategy_acc.index]
    bars = ax.bar(strategy_acc.index, strategy_acc.values, color=colors, edgecolor='black')
    
    ax.set_ylabel('Mean Accuracy')
    ax.set_xlabel('Prompting Strategy')
    ax.set_title(f'Accuracy by Prompting Strategy (T={temperature})')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, strategy_acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


# =============================================================================
# RANKING VISUALIZATIONS
# =============================================================================

def plot_model_ranking(
    ranking_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 15
) -> plt.Figure:
    """
    Create horizontal bar chart of model rankings.
    
    Args:
        ranking_df: DataFrame with columns [model, accuracy, provider]
        output_path: Path to save figure
        top_n: Number of top models to show
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    # Sort and take top N
    sorted_df = ranking_df.sort_values('accuracy', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, len(sorted_df) * 0.4 + 2))
    
    colors = [PROVIDER_COLORS.get(p, '#888888') for p in sorted_df['provider']]
    bars = ax.barh(sorted_df['model'], sorted_df['accuracy'], color=colors, edgecolor='black')
    
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Top {top_n} Models by Accuracy')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, sorted_df['accuracy']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2%}', ha='left', va='center', fontsize=9)
    
    # Legend for providers
    legend_patches = [
        mpatches.Patch(color=color, label=provider.capitalize())
        for provider, color in PROVIDER_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


def plot_ranking_comparison(
    our_rankings: Dict[str, int],
    arena_rankings: Dict[str, int],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create scatter plot comparing our rankings with LM Arena.
    
    Args:
        our_rankings: Dict[model, rank] from our benchmark
        arena_rankings: Dict[model, rank] from LM Arena
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    common_models = [m for m in our_rankings if m in arena_rankings and arena_rankings[m] is not None]
    
    if not common_models:
        return None
    
    our_ranks = [our_rankings[m] for m in common_models]
    arena_ranks = [arena_rankings[m] for m in common_models]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(arena_ranks, our_ranks, s=100, alpha=0.7, edgecolors='black')
    
    # Add model labels
    for model, ar, our in zip(common_models, arena_ranks, our_ranks):
        ax.annotate(model.split('-')[0], (ar, our), fontsize=8, alpha=0.8,
                   xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line (perfect correlation)
    max_rank = max(max(our_ranks), max(arena_ranks))
    ax.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5, label='Perfect correlation')
    
    ax.set_xlabel('LM Arena Rank')
    ax.set_ylabel('Our Benchmark Rank')
    ax.set_title('Ranking Comparison: Our Benchmark vs LM Arena')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


# =============================================================================
# CORRELATION VISUALIZATIONS
# =============================================================================

def plot_accuracy_vs_arena(
    accuracies: Dict[str, float],
    arena_rankings: Dict[str, int],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Scatter plot of accuracy vs LM Arena rank.
    
    Args:
        accuracies: Dict[model, accuracy]
        arena_rankings: Dict[model, rank]
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    common_models = [m for m in accuracies if m in arena_rankings and arena_rankings[m] is not None]
    
    if not common_models:
        return None
    
    accs = [accuracies[m] for m in common_models]
    ranks = [arena_rankings[m] for m in common_models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(ranks, accs, s=100, alpha=0.7, edgecolors='black', c='#2E86AB')
    
    # Add trend line
    z = np.polyfit(ranks, accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ranks), max(ranks), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
    
    # Add model labels
    for model, r, a in zip(common_models, ranks, accs):
        short_name = model.split('-')[0][:8]
        ax.annotate(short_name, (r, a), fontsize=8, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('LM Arena Rank (lower = better)')
    ax.set_ylabel('Syllogism Accuracy')
    ax.set_title('Model Accuracy vs LM Arena Ranking')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


# =============================================================================
# CONSISTENCY VISUALIZATIONS
# =============================================================================

def plot_consistency_by_model(
    consistency_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Bar chart of consistency scores by model.
    
    Args:
        consistency_df: DataFrame with [model, overall_consistency]
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    sorted_df = consistency_df.sort_values('overall_consistency', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, len(sorted_df) * 0.3 + 2))
    
    colors = plt.cm.RdYlGn(sorted_df['overall_consistency'])
    bars = ax.barh(sorted_df['model'], sorted_df['overall_consistency'], color=colors, edgecolor='black')
    
    ax.set_xlabel('Consistency Score')
    ax.set_title('Cross-Variant Consistency by Model')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, sorted_df['overall_consistency']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2%}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


def plot_content_effects(
    consistency_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Grouped bar chart of content effects (name, order, combined).
    
    Args:
        consistency_df: DataFrame with [model, name_effect, order_effect, combined_effect]
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    models = consistency_df['model'].tolist()
    name_effects = consistency_df['name_effect'].tolist()
    order_effects = consistency_df['order_effect'].tolist()
    combined_effects = consistency_df['combined_effect'].tolist()
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, name_effects, width, label='Name Effect', color='#2E86AB')
    ax.bar(x, order_effects, width, label='Order Effect', color='#A23B72')
    ax.bar(x + width, combined_effects, width, label='Combined Effect', color='#F18F01')
    
    ax.set_ylabel('Effect Size (proportion of changed predictions)')
    ax.set_xlabel('Model')
    ax.set_title('Content Effects by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


# =============================================================================
# TEMPERATURE EFFECT VISUALIZATION
# =============================================================================

def plot_temperature_effect(
    accuracy_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Line plot showing how accuracy changes with temperature.
    
    Args:
        accuracy_df: DataFrame with [model, temperature, accuracy]
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by temperature
    temp_means = accuracy_df.groupby('temperature')['accuracy'].agg(['mean', 'std'])
    
    ax.errorbar(
        temp_means.index,
        temp_means['mean'],
        yerr=temp_means['std'],
        marker='o',
        markersize=10,
        capsize=5,
        capthick=2,
        linewidth=2,
        color='#2E86AB'
    )
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean Accuracy ± Std')
    ax.set_title('Effect of Temperature on Model Accuracy')
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT, dpi=FIGURE_DPI)
    
    return fig


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION MODULE TEST")
    print("=" * 60)
    
    # Create test data
    test_data = {
        'model': ['gemini-2.0-flash', 'llama-3.1-70b', 'qwen-2.5-72b'] * 6,
        'temperature': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0] * 2,
        'strategy': ['zero_shot'] * 9 + ['few_shot'] * 9,
        'accuracy': [0.85, 0.82, 0.80, 0.78, 0.75, 0.73, 0.70, 0.68, 0.65,
                    0.88, 0.85, 0.83, 0.82, 0.79, 0.77, 0.75, 0.72, 0.70],
        'provider': ['google', 'huggingface', 'huggingface'] * 6
    }
    
    test_df = pd.DataFrame(test_data)
    
    print("\n[Test Data]")
    print(test_df.head(10))
    
    # Test temperature effect plot
    print("\n[Creating temperature effect plot...]")
    fig = plot_temperature_effect(test_df)
    if fig:
        print("  Temperature effect plot created successfully")
    
    # Note: Figures won't be saved in test mode
    print("\nVisualization module ready for use.")
