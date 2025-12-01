"""
Correlation Analysis for Syllogistic Reasoning Benchmark

Calculates correlation between model accuracy and LM Arena rankings.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats
import pandas as pd

from src.config import config
from src.inference.model_registry import MODEL_REGISTRY, get_lm_arena_models


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    spearman_rho: float
    spearman_p_value: float
    kendall_tau: float
    kendall_p_value: float
    pearson_r: float
    pearson_p_value: float
    n_models: int
    model_rankings: Dict[str, Tuple[int, int]]  # model -> (accuracy_rank, arena_rank)
    
    def to_dict(self) -> Dict:
        return {
            "spearman_rho": self.spearman_rho,
            "spearman_p_value": self.spearman_p_value,
            "kendall_tau": self.kendall_tau,
            "kendall_p_value": self.kendall_p_value,
            "pearson_r": self.pearson_r,
            "pearson_p_value": self.pearson_p_value,
            "n_models": self.n_models,
            "model_rankings": self.model_rankings
        }
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if Spearman correlation is statistically significant."""
        return self.spearman_p_value < alpha


# =============================================================================
# LM ARENA RANKINGS
# =============================================================================

# LM Arena rankings from LMarena_benchmark.csv (November 2025)
# Lower rank = better performance
# Only includes models from MODEL_REGISTRY that have LM Arena benchmarks
# Models with lm_arena=False in registry have None here
# Updated for 22-model registry (3 Gemini + 19 HuggingFace)
LM_ARENA_RANKINGS = {
    # Google Gemini models (3 models)
    "gemini-2.5-pro": 9,  
    "gemini-2.5-flash": 44,
    "gemini-2.5-flash-lite": 70, 
    
    # OpenAI OSS (HuggingFace)
    "gpt-oss-20b": 128,
    
    # Meta Llama models (4 models)
    "llama-3.3-70b-instruct": 134,
    "llama-3.2-3b-instruct": 231,
    "llama-3.2-1b-instruct": 260,
    "llama-3.1-8b-instruct": 205,
    
    # Qwen models (5 models)
    "qwen3-next-80b-a3b-instruct": 50,
    "qwen3-next-80b-a3b-thinking": 77,  # Thinking variant
    "qwen3-32b": 97,
    "qwen3-235b-a22b-thinking": 54,
    "qwq-32b": 116,
    
    # Mistral models (1 model)
    "mixtral-8x22b-instruct": 196,
    
    # DeepSeek models (3 models)
    "deepseek-r1": 55,
    "deepseek-v3.1": 33,
    "deepseek-r1-0528": 32,
    
    # Google Gemma models (1 model)
    "gemma-3-27b-it": 79,
    
    # Moonshot Kimi models (2 models) - Not on LM Arena
    "kimi-k2-thinking": 17,
    "kimi-k2-instruct": 30,
    
    # GLM models (2 models)
    "glm-4.5": 40,
    "glm-4.6": 22,
}


def get_arena_ranking(model_key: str) -> Optional[int]:
    """Get LM Arena ranking for a model."""
    return LM_ARENA_RANKINGS.get(model_key)


def get_models_with_arena_rankings() -> List[str]:
    """Get list of model keys that have LM Arena rankings."""
    return [k for k, v in LM_ARENA_RANKINGS.items() if v is not None]


# =============================================================================
# ACCURACY LOADING
# =============================================================================

def load_model_accuracies(
    results_dir: Path,
    temperature: float,
    strategy: str
) -> Dict[str, float]:
    """
    Load accuracy for all models from results.
    
    Returns:
        Dict mapping model_key to accuracy
    """
    from src.evaluation.calculate_metrics import calculate_metrics_from_file
    
    temp_dir = results_dir / "raw_responses" / f"temperature_{temperature}"
    
    if not temp_dir.exists():
        return {}
    
    accuracies = {}
    for filepath in temp_dir.glob(f"*_{strategy}.json"):
        model_key = filepath.stem.replace(f"_{strategy}", "")
        
        try:
            metrics = calculate_metrics_from_file(filepath)
            accuracies[model_key] = metrics.accuracy
        except Exception as e:
            print(f"Error loading accuracy for {model_key}: {e}")
    
    return accuracies


# =============================================================================
# CORRELATION CALCULATION
# =============================================================================

def calculate_correlation(
    accuracies: Dict[str, float],
    arena_rankings: Optional[Dict[str, int]] = None
) -> Optional[CorrelationResult]:
    """
    Calculate correlation between model accuracies and LM Arena rankings.
    
    Args:
        accuracies: Dict mapping model_key to accuracy (0-1)
        arena_rankings: Dict mapping model_key to arena rank (lower = better)
        
    Returns:
        CorrelationResult or None if insufficient data
    """
    if arena_rankings is None:
        arena_rankings = LM_ARENA_RANKINGS
    
    # Filter to models with both accuracy and arena ranking
    common_models = [
        m for m in accuracies.keys()
        if m in arena_rankings and arena_rankings[m] is not None
    ]
    
    if len(common_models) < 3:
        return None
    
    # Extract values
    acc_values = [accuracies[m] for m in common_models]
    arena_values = [arena_rankings[m] for m in common_models]
    
    # Note: Arena rank is lower = better, accuracy is higher = better
    # So we expect negative correlation if models perform consistently
    # Or we can invert arena ranks for positive correlation
    inverted_arena = [-r for r in arena_values]
    
    # Calculate correlations
    spearman_rho, spearman_p = stats.spearmanr(acc_values, inverted_arena)
    kendall_tau, kendall_p = stats.kendalltau(acc_values, inverted_arena)
    pearson_r, pearson_p = stats.pearsonr(acc_values, inverted_arena)
    
    # Calculate accuracy-based rankings (higher accuracy = lower rank number)
    sorted_by_acc = sorted(common_models, key=lambda m: -accuracies[m])
    acc_ranks = {m: i + 1 for i, m in enumerate(sorted_by_acc)}
    
    model_rankings = {
        m: (acc_ranks[m], arena_rankings[m])
        for m in common_models
    }
    
    return CorrelationResult(
        spearman_rho=spearman_rho,
        spearman_p_value=spearman_p,
        kendall_tau=kendall_tau,
        kendall_p_value=kendall_p,
        pearson_r=pearson_r,
        pearson_p_value=pearson_p,
        n_models=len(common_models),
        model_rankings=model_rankings
    )


def analyze_correlation_across_configs(
    results_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analyze correlation across all temperature/strategy configurations.
    
    Returns:
        DataFrame with correlation results for each configuration
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    rows = []
    for temperature in config.experiment.temperatures:
        for strategy in config.experiment.prompting_strategies:
            accuracies = load_model_accuracies(results_dir, temperature, strategy)
            
            if accuracies:
                result = calculate_correlation(accuracies)
                
                if result:
                    rows.append({
                        "temperature": temperature,
                        "strategy": strategy,
                        "spearman_rho": result.spearman_rho,
                        "spearman_p": result.spearman_p_value,
                        "kendall_tau": result.kendall_tau,
                        "kendall_p": result.kendall_p_value,
                        "n_models": result.n_models,
                        "significant_005": result.is_significant(0.05),
                        "significant_001": result.is_significant(0.01)
                    })
    
    return pd.DataFrame(rows)


# =============================================================================
# RANKING COMPARISON
# =============================================================================

def create_ranking_comparison_table(
    accuracies: Dict[str, float],
    arena_rankings: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Create a table comparing accuracy rankings with LM Arena rankings.
    
    Returns:
        DataFrame with model, accuracy, accuracy_rank, arena_rank, rank_diff
    """
    if arena_rankings is None:
        arena_rankings = LM_ARENA_RANKINGS
    
    # Filter to models with arena rankings
    models = [
        m for m in accuracies.keys()
        if m in arena_rankings and arena_rankings[m] is not None
    ]
    
    if not models:
        return pd.DataFrame()
    
    # Sort by accuracy
    sorted_models = sorted(models, key=lambda m: -accuracies[m])
    
    rows = []
    for rank, model in enumerate(sorted_models, 1):
        arena_rank = arena_rankings[model]
        rows.append({
            "model": model,
            "accuracy": accuracies[model],
            "accuracy_rank": rank,
            "arena_rank": arena_rank,
            "rank_diff": abs(rank - arena_rank)
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Try to load real results from the results directory
    results_dir = config.experiment.results_full_path
    
    print(f"\nLooking for results in: {results_dir}")
    print(f"\n[LM Arena Rankings for {len(get_models_with_arena_rankings())} models]")
    for model in sorted(get_models_with_arena_rankings(), key=lambda m: LM_ARENA_RANKINGS[m]):
        print(f"  #{LM_ARENA_RANKINGS[model]:>3}: {model}")
    
    # Try to load accuracies from actual results
    accuracies = load_model_accuracies(results_dir, temperature=0.0, strategy="zero_shot")
    
    if not accuracies:
        print("\n[No results found yet]")
        print("  Run the benchmark first to generate results.")
        print("  Results will be loaded from: results/raw_responses/")
    else:
        print(f"\n[Loaded Accuracies for {len(accuracies)} models]")
        for m, acc in sorted(accuracies.items(), key=lambda x: -x[1]):
            arena = LM_ARENA_RANKINGS.get(m, "N/A")
            print(f"  {m}: {acc:.2%} (Arena rank: {arena})")
        
        result = calculate_correlation(accuracies)
        
        if result:
            print(f"\n[Correlation Results]")
            print(f"  Spearman ρ: {result.spearman_rho:.3f} (p={result.spearman_p_value:.4f})")
            print(f"  Kendall τ: {result.kendall_tau:.3f} (p={result.kendall_p_value:.4f})")
            print(f"  Pearson r: {result.pearson_r:.3f} (p={result.pearson_p_value:.4f})")
            print(f"  N models: {result.n_models}")
            print(f"  Significant (α=0.05): {result.is_significant()}")
            
            print("\n[Ranking Comparison]")
            df = create_ranking_comparison_table(accuracies)
            print(df.to_string(index=False))
        else:
            print("\n[Insufficient data] Need at least 3 models with LM Arena rankings")
