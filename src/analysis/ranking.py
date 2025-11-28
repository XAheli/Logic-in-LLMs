"""
Model Ranking Analysis for Syllogistic Reasoning Benchmark

Creates rankings of models by:
- Overall accuracy
- Per-strategy accuracy
- Per-temperature accuracy
- Consistency scores
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import config
from src.inference.model_registry import MODEL_REGISTRY, ModelConfig


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RankedModel:
    """A model with its ranking information."""
    model_key: str
    rank: int
    accuracy: float
    billing_type: str
    provider: str
    on_lm_arena: bool
    
    def to_dict(self) -> Dict:
        return {
            "model_key": self.model_key,
            "rank": self.rank,
            "accuracy": self.accuracy,
            "billing_type": self.billing_type,
            "provider": self.provider,
            "on_lm_arena": self.on_lm_arena
        }


# =============================================================================
# RANKING FUNCTIONS
# =============================================================================

def rank_models_by_accuracy(
    accuracies: Dict[str, float],
    descending: bool = True
) -> List[RankedModel]:
    """
    Rank models by their accuracy.
    
    Args:
        accuracies: Dict mapping model_key to accuracy
        descending: If True, highest accuracy = rank 1
        
    Returns:
        List of RankedModel sorted by rank
    """
    sorted_models = sorted(
        accuracies.items(),
        key=lambda x: x[1],
        reverse=descending
    )
    
    rankings = []
    for rank, (model_key, accuracy) in enumerate(sorted_models, 1):
        model_info = MODEL_REGISTRY.get(model_key)
        
        rankings.append(RankedModel(
            model_key=model_key,
            rank=rank,
            accuracy=accuracy,
            billing_type=model_info.billing_type if model_info else "unknown",
            provider=model_info.provider if model_info else "unknown",
            on_lm_arena=model_info.on_lm_arena if model_info else False
        ))
    
    return rankings


def load_accuracies_from_results(
    results_dir: Path,
    temperature: float,
    strategy: str
) -> Dict[str, float]:
    """Load model accuracies from result files."""
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
        except Exception:
            pass
    
    return accuracies


def create_ranking_table(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> pd.DataFrame:
    """
    Create a ranking table for models.
    
    Returns:
        DataFrame with columns: rank, model, accuracy, provider, billing_type, on_lm_arena
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    accuracies = load_accuracies_from_results(results_dir, temperature, strategy)
    
    if not accuracies:
        return pd.DataFrame()
    
    rankings = rank_models_by_accuracy(accuracies)
    
    return pd.DataFrame([r.to_dict() for r in rankings])


# =============================================================================
# AGGREGATE RANKINGS
# =============================================================================

def calculate_aggregate_accuracy(
    results_dir: Path,
    model_key: str
) -> Optional[float]:
    """
    Calculate aggregate accuracy across all configurations.
    
    Returns average accuracy across all temperature/strategy combinations.
    """
    from src.evaluation.calculate_metrics import calculate_metrics_from_file
    
    accuracies = []
    
    for temperature in config.experiment.temperatures:
        for strategy in config.experiment.prompting_strategies:
            filepath = (
                results_dir / "raw_responses" /
                f"temperature_{temperature}" /
                f"{model_key}_{strategy}.json"
            )
            
            if filepath.exists():
                try:
                    metrics = calculate_metrics_from_file(filepath)
                    accuracies.append(metrics.accuracy)
                except Exception:
                    pass
    
    return np.mean(accuracies) if accuracies else None


def create_aggregate_ranking_table(
    results_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create ranking based on aggregate accuracy across all configurations.
    
    Returns:
        DataFrame with aggregate rankings
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    aggregate_accuracies = {}
    
    for model_key in MODEL_REGISTRY.keys():
        acc = calculate_aggregate_accuracy(results_dir, model_key)
        if acc is not None:
            aggregate_accuracies[model_key] = acc
    
    if not aggregate_accuracies:
        return pd.DataFrame()
    
    rankings = rank_models_by_accuracy(aggregate_accuracies)
    
    return pd.DataFrame([r.to_dict() for r in rankings])


# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def compare_rankings_across_strategies(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0
) -> pd.DataFrame:
    """
    Compare model rankings across different prompting strategies.
    
    Returns:
        DataFrame with model rankings for each strategy
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    strategy_rankings = {}
    
    for strategy in config.experiment.prompting_strategies:
        accuracies = load_accuracies_from_results(results_dir, temperature, strategy)
        if accuracies:
            rankings = rank_models_by_accuracy(accuracies)
            strategy_rankings[strategy] = {r.model_key: r.rank for r in rankings}
    
    if not strategy_rankings:
        return pd.DataFrame()
    
    # Build comparison table
    all_models = set()
    for ranks in strategy_rankings.values():
        all_models.update(ranks.keys())
    
    rows = []
    for model in sorted(all_models):
        row = {"model": model}
        for strategy, ranks in strategy_rankings.items():
            row[f"rank_{strategy}"] = ranks.get(model, "-")
        rows.append(row)
    
    return pd.DataFrame(rows)


def compare_rankings_across_temperatures(
    results_dir: Optional[Path] = None,
    strategy: str = "zero_shot"
) -> pd.DataFrame:
    """
    Compare model rankings across different temperatures.
    
    Returns:
        DataFrame with model rankings for each temperature
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    temp_rankings = {}
    
    for temperature in config.experiment.temperatures:
        accuracies = load_accuracies_from_results(results_dir, temperature, strategy)
        if accuracies:
            rankings = rank_models_by_accuracy(accuracies)
            temp_rankings[temperature] = {r.model_key: r.rank for r in rankings}
    
    if not temp_rankings:
        return pd.DataFrame()
    
    # Build comparison table
    all_models = set()
    for ranks in temp_rankings.values():
        all_models.update(ranks.keys())
    
    rows = []
    for model in sorted(all_models):
        row = {"model": model}
        for temp, ranks in temp_rankings.items():
            row[f"rank_T{temp}"] = ranks.get(model, "-")
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# PAID VS FREE ANALYSIS
# =============================================================================

def compare_paid_vs_free(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> Dict[str, Dict]:
    """
    Compare performance of Google Studio (Gemini) vs HuggingFace Inference models.
    
    Returns:
        Dict with summary statistics for each billing type
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    accuracies = load_accuracies_from_results(results_dir, temperature, strategy)
    
    google_studio_accs = []
    hf_inf_accs = []
    
    for model_key, acc in accuracies.items():
        model_info = MODEL_REGISTRY.get(model_key)
        if model_info:
            if model_info.billing_type == "google_studio_paid":
                google_studio_accs.append(acc)
            elif model_info.billing_type == "hf_inf_paid":
                hf_inf_accs.append(acc)
    
    return {
        "google_studio_paid": {
            "count": len(google_studio_accs),
            "mean_accuracy": np.mean(google_studio_accs) if google_studio_accs else 0,
            "std_accuracy": np.std(google_studio_accs) if google_studio_accs else 0,
            "max_accuracy": max(google_studio_accs) if google_studio_accs else 0,
            "min_accuracy": min(google_studio_accs) if google_studio_accs else 0,
        },
        "hf_inf_paid": {
            "count": len(hf_inf_accs),
            "mean_accuracy": np.mean(hf_inf_accs) if hf_inf_accs else 0,
            "std_accuracy": np.std(hf_inf_accs) if hf_inf_accs else 0,
            "max_accuracy": max(hf_inf_accs) if hf_inf_accs else 0,
            "min_accuracy": min(hf_inf_accs) if hf_inf_accs else 0,
        }
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RANKING ANALYSIS TEST")
    print("=" * 60)
    
    # Test with simulated data
    test_accuracies = {
        "gemini-2.0-flash": 0.85,
        "llama-3.1-70b-instruct": 0.82,
        "qwen-2.5-72b-instruct": 0.80,
        "gemini-1.5-flash": 0.78,
        "mistral-nemo-instruct": 0.72,
        "llama-3.1-8b-instruct": 0.68,
        "phi-3.5-mini-instruct": 0.65,
        "falcon-7b-instruct": 0.55,
    }
    
    print("\n[Model Rankings]")
    rankings = rank_models_by_accuracy(test_accuracies)
    
    for r in rankings:
        print(f"  #{r.rank}: {r.model_key} ({r.accuracy:.2%}) - {r.provider}")
    
    print("\n[Paid vs Free (simulated)]")
    # Simulate paid/free split
    paid_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
    free_models = [m for m in test_accuracies if m not in paid_models]
    
    paid_mean = np.mean([test_accuracies[m] for m in paid_models])
    free_mean = np.mean([test_accuracies[m] for m in free_models])
    
    print(f"  Paid models mean accuracy: {paid_mean:.2%}")
    print(f"  Free models mean accuracy: {free_mean:.2%}")
