"""
Variant Correlation Analysis for Syllogistic Reasoning Benchmark

Analyzes correlation between different variants (N, X, O, OX) to:
1. Validate that content changes don't affect logical validity judgments
2. Measure consistency of model behavior across variants

Key metric: Pearson correlation between N (sensical) and X/O/OX (modified) variants
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy import stats
import pandas as pd

from src.config import config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VariantCorrelationResult:
    """Result of correlation analysis between two variants."""
    variant_a: str
    variant_b: str
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    n_samples: int
    agreement_rate: float  # % of instances where both variants gave same prediction
    
    def to_dict(self) -> Dict:
        return {
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "pearson_r": self.pearson_r,
            "pearson_p_value": self.pearson_p_value,
            "spearman_rho": self.spearman_rho,
            "spearman_p_value": self.spearman_p_value,
            "n_samples": self.n_samples,
            "agreement_rate": self.agreement_rate
        }
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"{self.variant_a} vs {self.variant_b}: "
            f"r={self.pearson_r:.3f} (p={self.pearson_p_value:.4f}), "
            f"agreement={self.agreement_rate:.1%}"
        )


@dataclass
class ModelVariantAnalysis:
    """Complete variant analysis for a single model."""
    model_key: str
    n_vs_x: VariantCorrelationResult  # Sensical vs nonsensical names
    n_vs_o: VariantCorrelationResult  # Original vs swapped premise order
    n_vs_ox: VariantCorrelationResult  # Original vs both modifications
    x_vs_o: VariantCorrelationResult  # Cross-comparison
    overall_consistency: float
    
    def to_dict(self) -> Dict:
        return {
            "model_key": self.model_key,
            "n_vs_x": self.n_vs_x.to_dict(),
            "n_vs_o": self.n_vs_o.to_dict(),
            "n_vs_ox": self.n_vs_ox.to_dict(),
            "x_vs_o": self.x_vs_o.to_dict(),
            "overall_consistency": self.overall_consistency
        }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def prediction_to_numeric(pred: str) -> int:
    """
    Convert prediction to numeric.
    
    LLM outputs: correct=1, incorrect=0
    Ground truth syntax: valid=1, invalid=0
    Ground truth NLU: believable=1, unbelievable=0
    """
    pred_lower = pred.lower()
    # Handle all positive cases
    if pred_lower in ["correct", "valid", "believable"]:
        return 1
    # Handle all negative cases
    return 0


def calculate_variant_correlation(
    predictions_a: List[str],
    predictions_b: List[str],
    variant_a: str = "A",
    variant_b: str = "B"
) -> VariantCorrelationResult:
    """
    Calculate correlation between predictions from two variants.
    
    Args:
        predictions_a: List of predictions for variant A
        predictions_b: List of predictions for variant B (same order)
        variant_a: Name of variant A
        variant_b: Name of variant B
        
    Returns:
        VariantCorrelationResult
    """
    if len(predictions_a) != len(predictions_b):
        raise ValueError("Prediction lists must have same length")
    
    if len(predictions_a) < 3:
        # Not enough data for meaningful correlation
        return VariantCorrelationResult(
            variant_a=variant_a,
            variant_b=variant_b,
            pearson_r=0.0,
            pearson_p_value=1.0,
            spearman_rho=0.0,
            spearman_p_value=1.0,
            n_samples=len(predictions_a),
            agreement_rate=0.0
        )
    
    # Convert to numeric
    numeric_a = np.array([prediction_to_numeric(p) for p in predictions_a])
    numeric_b = np.array([prediction_to_numeric(p) for p in predictions_b])
    
    # Calculate agreement rate
    agreement_rate = np.mean(numeric_a == numeric_b)
    
    # Calculate correlations
    # Handle constant arrays (all same value)
    if np.std(numeric_a) == 0 or np.std(numeric_b) == 0:
        # If one array is constant, correlation is undefined
        # Use agreement rate as proxy
        pearson_r = 1.0 if agreement_rate == 1.0 else 0.0
        pearson_p = 0.0 if agreement_rate == 1.0 else 1.0
        spearman_rho = pearson_r
        spearman_p = pearson_p
    else:
        pearson_r, pearson_p = stats.pearsonr(numeric_a, numeric_b)
        spearman_rho, spearman_p = stats.spearmanr(numeric_a, numeric_b)
    
    return VariantCorrelationResult(
        variant_a=variant_a,
        variant_b=variant_b,
        pearson_r=float(pearson_r),
        pearson_p_value=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p_value=float(spearman_p),
        n_samples=len(predictions_a),
        agreement_rate=float(agreement_rate)
    )


def load_variant_predictions(
    results_dir: Path,
    model_key: str,
    temperature: float,
    strategy: str
) -> Dict[str, Dict[str, str]]:
    """
    Load predictions organized by syllogism and variant.
    
    Returns:
        Dict[syllogism_id, Dict[variant, prediction]]
    """
    filepath = (
        results_dir / "raw_responses" /
        f"temperature_{temperature}" /
        f"{model_key}_{strategy}.json"
    )
    
    if not filepath.exists():
        return {}
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    organized = defaultdict(dict)
    for result in data.get('results', []):
        syl_id = result.get('syllogism_id', '')
        variant = result.get('variant', '')
        prediction = result.get('predicted', '')
        
        if syl_id and variant and prediction:
            organized[syl_id][variant] = prediction
    
    return dict(organized)


def analyze_model_variants(
    results_dir: Path,
    model_key: str,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> Optional[ModelVariantAnalysis]:
    """
    Perform complete variant analysis for a single model.
    
    Returns:
        ModelVariantAnalysis or None if insufficient data
    """
    predictions_by_syl = load_variant_predictions(
        results_dir, model_key, temperature, strategy
    )
    
    if not predictions_by_syl:
        return None
    
    # Collect predictions for each variant pair
    variants = ['N', 'X', 'O', 'OX']
    variant_preds = {v: [] for v in variants}
    
    # Only include syllogisms that have all 4 variants
    for syl_id, syl_preds in predictions_by_syl.items():
        if all(v in syl_preds for v in variants):
            for v in variants:
                variant_preds[v].append(syl_preds[v])
    
    if len(variant_preds['N']) < 3:
        return None
    
    # Calculate correlations for each pair
    n_vs_x = calculate_variant_correlation(
        variant_preds['N'], variant_preds['X'], 'N', 'X'
    )
    n_vs_o = calculate_variant_correlation(
        variant_preds['N'], variant_preds['O'], 'N', 'O'
    )
    n_vs_ox = calculate_variant_correlation(
        variant_preds['N'], variant_preds['OX'], 'N', 'OX'
    )
    x_vs_o = calculate_variant_correlation(
        variant_preds['X'], variant_preds['O'], 'X', 'O'
    )
    
    # Overall consistency: average agreement rate
    overall_consistency = np.mean([
        n_vs_x.agreement_rate,
        n_vs_o.agreement_rate,
        n_vs_ox.agreement_rate,
        x_vs_o.agreement_rate
    ])
    
    return ModelVariantAnalysis(
        model_key=model_key,
        n_vs_x=n_vs_x,
        n_vs_o=n_vs_o,
        n_vs_ox=n_vs_ox,
        x_vs_o=x_vs_o,
        overall_consistency=overall_consistency
    )


def analyze_all_models_variants(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> pd.DataFrame:
    """
    Analyze variant correlations for all models.
    
    Returns:
        DataFrame with variant correlation metrics for each model
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    from src.inference.model_registry import MODEL_REGISTRY
    
    rows = []
    for model_key in MODEL_REGISTRY.keys():
        analysis = analyze_model_variants(
            results_dir, model_key, temperature, strategy
        )
        
        if analysis:
            rows.append({
                "model": model_key,
                "n_vs_x_pearson": analysis.n_vs_x.pearson_r,
                "n_vs_x_agreement": analysis.n_vs_x.agreement_rate,
                "n_vs_o_pearson": analysis.n_vs_o.pearson_r,
                "n_vs_o_agreement": analysis.n_vs_o.agreement_rate,
                "n_vs_ox_pearson": analysis.n_vs_ox.pearson_r,
                "n_vs_ox_agreement": analysis.n_vs_ox.agreement_rate,
                "overall_consistency": analysis.overall_consistency
            })
    
    return pd.DataFrame(rows)


def aggregate_variant_correlation(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> Dict[str, VariantCorrelationResult]:
    """
    Calculate aggregate variant correlations across ALL models.
    
    This pools all predictions from all models to get overall correlation.
    
    Returns:
        Dict mapping variant pair to VariantCorrelationResult
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    from src.inference.model_registry import MODEL_REGISTRY
    
    # Collect all predictions across models
    variants = ['N', 'X', 'O', 'OX']
    all_preds = {v: [] for v in variants}
    
    for model_key in MODEL_REGISTRY.keys():
        predictions_by_syl = load_variant_predictions(
            results_dir, model_key, temperature, strategy
        )
        
        for syl_id, syl_preds in predictions_by_syl.items():
            if all(v in syl_preds for v in variants):
                for v in variants:
                    all_preds[v].append(syl_preds[v])
    
    if len(all_preds['N']) < 3:
        return {}
    
    return {
        "N_vs_X": calculate_variant_correlation(all_preds['N'], all_preds['X'], 'N', 'X'),
        "N_vs_O": calculate_variant_correlation(all_preds['N'], all_preds['O'], 'N', 'O'),
        "N_vs_OX": calculate_variant_correlation(all_preds['N'], all_preds['OX'], 'N', 'OX'),
        "X_vs_O": calculate_variant_correlation(all_preds['X'], all_preds['O'], 'X', 'O'),
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VARIANT CORRELATION ANALYSIS - DEMO")
    print("=" * 60)
    
    # Simulated data: predictions for 20 syllogisms across 4 variants
    np.random.seed(42)
    
    # Simulate high correlation between N and X (same logical structure)
    n_preds = ["valid" if np.random.random() > 0.4 else "invalid" for _ in range(20)]
    
    # X should be highly correlated with N (r ~ 0.85)
    x_preds = []
    for p in n_preds:
        if np.random.random() > 0.15:  # 85% agreement
            x_preds.append(p)
        else:
            x_preds.append("invalid" if p == "valid" else "valid")
    
    # O also highly correlated
    o_preds = []
    for p in n_preds:
        if np.random.random() > 0.20:  # 80% agreement
            o_preds.append(p)
        else:
            o_preds.append("invalid" if p == "valid" else "valid")
    
    # OX slightly less correlated
    ox_preds = []
    for p in n_preds:
        if np.random.random() > 0.25:  # 75% agreement
            ox_preds.append(p)
        else:
            ox_preds.append("invalid" if p == "valid" else "valid")
    
    print("\n[Simulated Predictions]")
    print(f"N predictions:  {n_preds[:5]}...")
    print(f"X predictions:  {x_preds[:5]}...")
    print(f"O predictions:  {o_preds[:5]}...")
    print(f"OX predictions: {ox_preds[:5]}...")
    
    # Calculate correlations
    print("\n[Variant Correlations]")
    
    n_vs_x = calculate_variant_correlation(n_preds, x_preds, "N", "X")
    print(f"  {n_vs_x.summary()}")
    
    n_vs_o = calculate_variant_correlation(n_preds, o_preds, "N", "O")
    print(f"  {n_vs_o.summary()}")
    
    n_vs_ox = calculate_variant_correlation(n_preds, ox_preds, "N", "OX")
    print(f"  {n_vs_ox.summary()}")
    
    print("\n[Interpretation]")
    print("  High Pearson r (> 0.8) indicates content changes don't affect")
    print("  logical validity judgments - models are reasoning about structure.")
    print("  Low r would suggest models rely on surface content features.")
