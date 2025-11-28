"""
Instance Sufficiency Analysis for Syllogistic Reasoning Benchmark

Validates that our dataset (40 syllogisms × 4 variants = 160 instances)
is sufficient to capture model behavior on logical forms.

Method (from reference paper):
- Calculate Pearson correlation between N and X variants
- High correlation (r > 0.8) indicates instances sufficiently represent logical forms
- Split-half reliability analysis
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
class SufficiencyResult:
    """Result of instance sufficiency analysis."""
    n_syllogisms: int
    n_instances: int  # Total including variants
    n_vs_x_correlation: float
    n_vs_x_p_value: float
    split_half_reliability: float
    cronbach_alpha: float
    is_sufficient: bool  # True if correlation > 0.8
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            "n_syllogisms": self.n_syllogisms,
            "n_instances": self.n_instances,
            "n_vs_x_correlation": self.n_vs_x_correlation,
            "n_vs_x_p_value": self.n_vs_x_p_value,
            "split_half_reliability": self.split_half_reliability,
            "cronbach_alpha": self.cronbach_alpha,
            "is_sufficient": self.is_sufficient,
            "recommendation": self.recommendation
        }


@dataclass
class ModelSufficiencyAnalysis:
    """Sufficiency analysis for a single model."""
    model_key: str
    n_vs_x_correlation: float
    n_vs_o_correlation: float
    n_vs_ox_correlation: float
    average_correlation: float
    n_syllogisms_tested: int
    
    def to_dict(self) -> Dict:
        return {
            "model_key": self.model_key,
            "n_vs_x_correlation": self.n_vs_x_correlation,
            "n_vs_o_correlation": self.n_vs_o_correlation,
            "n_vs_ox_correlation": self.n_vs_ox_correlation,
            "average_correlation": self.average_correlation,
            "n_syllogisms_tested": self.n_syllogisms_tested
        }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def prediction_to_numeric(pred: str) -> int:
    """Convert prediction to numeric (valid=1, invalid=0)."""
    return 1 if pred.lower() == "valid" else 0


def calculate_split_half_reliability(
    predictions: List[str],
    n_splits: int = 100
) -> float:
    """
    Calculate split-half reliability using random splits.
    
    Args:
        predictions: List of predictions
        n_splits: Number of random splits to average
        
    Returns:
        Average Spearman-Brown corrected reliability
    """
    if len(predictions) < 4:
        return 0.0
    
    numeric = np.array([prediction_to_numeric(p) for p in predictions])
    n = len(numeric)
    half = n // 2
    
    reliabilities = []
    for _ in range(n_splits):
        # Random split
        indices = np.random.permutation(n)
        half1 = numeric[indices[:half]]
        half2 = numeric[indices[half:2*half]]
        
        # Calculate correlation between halves
        if np.std(half1) > 0 and np.std(half2) > 0:
            r, _ = stats.pearsonr(half1, half2)
            # Spearman-Brown correction
            reliability = (2 * r) / (1 + r) if r > -1 else 0
            reliabilities.append(reliability)
    
    return np.mean(reliabilities) if reliabilities else 0.0


def calculate_cronbach_alpha(
    predictions_matrix: np.ndarray
) -> float:
    """
    Calculate Cronbach's alpha for internal consistency.
    
    Args:
        predictions_matrix: Shape (n_items, n_variants) - numeric predictions
        
    Returns:
        Cronbach's alpha coefficient
    """
    n_items, n_variants = predictions_matrix.shape
    
    if n_variants < 2 or n_items < 2:
        return 0.0
    
    # Item variances
    item_variances = np.var(predictions_matrix, axis=1, ddof=1)
    
    # Total score variance
    total_scores = np.sum(predictions_matrix, axis=1)
    total_variance = np.var(total_scores, ddof=1)
    
    if total_variance == 0:
        return 1.0 if np.sum(item_variances) == 0 else 0.0
    
    alpha = (n_variants / (n_variants - 1)) * (1 - np.sum(item_variances) / total_variance)
    
    return float(alpha)


def load_all_variant_predictions(
    results_dir: Path,
    model_key: str,
    temperature: float,
    strategy: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load predictions for all variants, aligned by syllogism.
    
    Returns:
        Tuple of (n_preds, x_preds, o_preds, ox_preds) - aligned lists
    """
    filepath = (
        results_dir / "raw_responses" /
        f"temperature_{temperature}" /
        f"{model_key}_{strategy}.json"
    )
    
    if not filepath.exists():
        return [], [], [], []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Organize by syllogism
    by_syllogism = defaultdict(dict)
    for result in data.get('results', []):
        syl_id = result.get('syllogism_id', '')
        variant = result.get('variant', '')
        prediction = result.get('predicted', '')
        
        if syl_id and variant and prediction:
            by_syllogism[syl_id][variant] = prediction
    
    # Extract aligned predictions
    n_preds, x_preds, o_preds, ox_preds = [], [], [], []
    
    for syl_id in sorted(by_syllogism.keys()):
        variants = by_syllogism[syl_id]
        if all(v in variants for v in ['N', 'X', 'O', 'OX']):
            n_preds.append(variants['N'])
            x_preds.append(variants['X'])
            o_preds.append(variants['O'])
            ox_preds.append(variants['OX'])
    
    return n_preds, x_preds, o_preds, ox_preds


def analyze_model_sufficiency(
    results_dir: Path,
    model_key: str,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> Optional[ModelSufficiencyAnalysis]:
    """
    Analyze instance sufficiency for a single model.
    
    Returns:
        ModelSufficiencyAnalysis or None if insufficient data
    """
    n_preds, x_preds, o_preds, ox_preds = load_all_variant_predictions(
        results_dir, model_key, temperature, strategy
    )
    
    if len(n_preds) < 5:
        return None
    
    # Convert to numeric
    n_num = np.array([prediction_to_numeric(p) for p in n_preds])
    x_num = np.array([prediction_to_numeric(p) for p in x_preds])
    o_num = np.array([prediction_to_numeric(p) for p in o_preds])
    ox_num = np.array([prediction_to_numeric(p) for p in ox_preds])
    
    def safe_pearson(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 1.0 if np.array_equal(a, b) else 0.0
        r, _ = stats.pearsonr(a, b)
        return r
    
    n_vs_x = safe_pearson(n_num, x_num)
    n_vs_o = safe_pearson(n_num, o_num)
    n_vs_ox = safe_pearson(n_num, ox_num)
    
    avg_corr = np.mean([n_vs_x, n_vs_o, n_vs_ox])
    
    return ModelSufficiencyAnalysis(
        model_key=model_key,
        n_vs_x_correlation=n_vs_x,
        n_vs_o_correlation=n_vs_o,
        n_vs_ox_correlation=n_vs_ox,
        average_correlation=avg_corr,
        n_syllogisms_tested=len(n_preds)
    )


def analyze_overall_sufficiency(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> SufficiencyResult:
    """
    Analyze overall instance sufficiency across all models.
    
    This is the main analysis that validates whether our 40 syllogisms
    (160 instances) are sufficient.
    
    Returns:
        SufficiencyResult with comprehensive metrics
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    from src.inference.model_registry import MODEL_REGISTRY
    
    # Collect correlations from all models
    all_n_vs_x = []
    all_predictions_matrix = []
    n_syllogisms = 0
    
    for model_key in MODEL_REGISTRY.keys():
        analysis = analyze_model_sufficiency(
            results_dir, model_key, temperature, strategy
        )
        
        if analysis and analysis.n_syllogisms_tested > 0:
            all_n_vs_x.append(analysis.n_vs_x_correlation)
            n_syllogisms = max(n_syllogisms, analysis.n_syllogisms_tested)
            
            # Load predictions for Cronbach's alpha
            n_preds, x_preds, o_preds, ox_preds = load_all_variant_predictions(
                results_dir, model_key, temperature, strategy
            )
            if n_preds:
                row = [
                    prediction_to_numeric(p) for p in n_preds + x_preds + o_preds + ox_preds
                ]
                all_predictions_matrix.append(row)
    
    if not all_n_vs_x:
        return SufficiencyResult(
            n_syllogisms=0,
            n_instances=0,
            n_vs_x_correlation=0.0,
            n_vs_x_p_value=1.0,
            split_half_reliability=0.0,
            cronbach_alpha=0.0,
            is_sufficient=False,
            recommendation="No data available for analysis"
        )
    
    # Average N vs X correlation across models
    avg_n_vs_x = np.mean(all_n_vs_x)
    
    # One-sample t-test: is mean correlation significantly > 0?
    if len(all_n_vs_x) > 1:
        _, p_value = stats.ttest_1samp(all_n_vs_x, 0)
    else:
        p_value = 1.0
    
    # Split-half reliability (aggregate all predictions)
    all_preds_flat = []
    for row in all_predictions_matrix:
        all_preds_flat.extend(['valid' if x == 1 else 'invalid' for x in row])
    
    split_half = calculate_split_half_reliability(all_preds_flat) if all_preds_flat else 0.0
    
    # Cronbach's alpha
    if all_predictions_matrix:
        # Reshape for alpha calculation
        max_len = max(len(row) for row in all_predictions_matrix)
        padded = np.zeros((len(all_predictions_matrix), max_len))
        for i, row in enumerate(all_predictions_matrix):
            padded[i, :len(row)] = row
        cronbach = calculate_cronbach_alpha(padded)
    else:
        cronbach = 0.0
    
    # Determine sufficiency
    is_sufficient = avg_n_vs_x >= 0.80
    
    if avg_n_vs_x >= 0.90:
        recommendation = "Excellent: High correlation indicates instances capture logical forms very well."
    elif avg_n_vs_x >= 0.80:
        recommendation = "Good: Correlation is sufficient. 40 syllogisms adequately represent logical forms."
    elif avg_n_vs_x >= 0.70:
        recommendation = "Moderate: Consider adding more instances for robustness."
    else:
        recommendation = "Low: Instance count may be insufficient. Consider expanding dataset."
    
    return SufficiencyResult(
        n_syllogisms=n_syllogisms,
        n_instances=n_syllogisms * 4,  # 4 variants per syllogism
        n_vs_x_correlation=avg_n_vs_x,
        n_vs_x_p_value=p_value,
        split_half_reliability=split_half,
        cronbach_alpha=cronbach,
        is_sufficient=is_sufficient,
        recommendation=recommendation
    )


def create_sufficiency_report(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> Dict:
    """
    Create comprehensive sufficiency report.
    
    Returns:
        Dict with overall and per-model sufficiency metrics
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    from src.inference.model_registry import MODEL_REGISTRY
    
    # Overall analysis
    overall = analyze_overall_sufficiency(results_dir, temperature, strategy)
    
    # Per-model analysis
    per_model = []
    for model_key in MODEL_REGISTRY.keys():
        analysis = analyze_model_sufficiency(
            results_dir, model_key, temperature, strategy
        )
        if analysis:
            per_model.append(analysis.to_dict())
    
    return {
        "overall": overall.to_dict(),
        "per_model": per_model,
        "summary": {
            "mean_n_vs_x_correlation": overall.n_vs_x_correlation,
            "n_models_analyzed": len(per_model),
            "is_sufficient": overall.is_sufficient,
            "recommendation": overall.recommendation
        }
    }


def analyze_instance_count_sensitivity(
    predictions_by_syl: Dict[str, Dict[str, str]],
    sample_sizes: List[int] = [5, 10, 15, 20, 25, 30, 35, 40],
    n_bootstrap: int = 100
) -> pd.DataFrame:
    """
    Analyze how correlation changes with different instance counts.
    
    This helps determine the minimum number of syllogisms needed.
    
    Args:
        predictions_by_syl: Dict[syl_id, Dict[variant, prediction]]
        sample_sizes: List of sample sizes to test
        n_bootstrap: Number of bootstrap samples per size
        
    Returns:
        DataFrame with sample_size, mean_correlation, std_correlation
    """
    syllogism_ids = list(predictions_by_syl.keys())
    n_total = len(syllogism_ids)
    
    results = []
    
    for sample_size in sample_sizes:
        if sample_size > n_total:
            continue
        
        correlations = []
        
        for _ in range(n_bootstrap):
            # Random sample of syllogisms
            sampled_ids = np.random.choice(syllogism_ids, size=sample_size, replace=False)
            
            n_preds = []
            x_preds = []
            
            for syl_id in sampled_ids:
                variants = predictions_by_syl[syl_id]
                if 'N' in variants and 'X' in variants:
                    n_preds.append(prediction_to_numeric(variants['N']))
                    x_preds.append(prediction_to_numeric(variants['X']))
            
            if len(n_preds) >= 3:
                n_arr = np.array(n_preds)
                x_arr = np.array(x_preds)
                
                if np.std(n_arr) > 0 and np.std(x_arr) > 0:
                    r, _ = stats.pearsonr(n_arr, x_arr)
                    correlations.append(r)
        
        if correlations:
            results.append({
                "sample_size": sample_size,
                "mean_correlation": np.mean(correlations),
                "std_correlation": np.std(correlations),
                "min_correlation": np.min(correlations),
                "max_correlation": np.max(correlations)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INSTANCE SUFFICIENCY ANALYSIS - DEMO")
    print("=" * 60)
    
    # Simulate predictions for 40 syllogisms
    np.random.seed(42)
    n_syllogisms = 40
    
    # Generate simulated predictions
    predictions_by_syl = {}
    
    for i in range(n_syllogisms):
        syl_id = f"SYL_{i+1:03d}"
        
        # N prediction
        n_pred = "valid" if np.random.random() > 0.45 else "invalid"
        
        # X highly correlated with N (simulating ~85% agreement)
        x_pred = n_pred if np.random.random() > 0.15 else ("invalid" if n_pred == "valid" else "valid")
        
        # O also correlated
        o_pred = n_pred if np.random.random() > 0.18 else ("invalid" if n_pred == "valid" else "valid")
        
        # OX slightly less
        ox_pred = n_pred if np.random.random() > 0.22 else ("invalid" if n_pred == "valid" else "valid")
        
        predictions_by_syl[syl_id] = {
            'N': n_pred,
            'X': x_pred,
            'O': o_pred,
            'OX': ox_pred
        }
    
    # Calculate correlations
    n_preds = [predictions_by_syl[s]['N'] for s in predictions_by_syl]
    x_preds = [predictions_by_syl[s]['X'] for s in predictions_by_syl]
    
    n_num = np.array([prediction_to_numeric(p) for p in n_preds])
    x_num = np.array([prediction_to_numeric(p) for p in x_preds])
    
    r, p = stats.pearsonr(n_num, x_num)
    
    print(f"\n[Dataset Info]")
    print(f"  Syllogisms: {n_syllogisms}")
    print(f"  Total instances: {n_syllogisms * 4}")
    
    print(f"\n[N vs X Correlation]")
    print(f"  Pearson r: {r:.3f}")
    print(f"  p-value: {p:.4f}")
    
    # Sufficiency check
    is_sufficient = r >= 0.80
    print(f"\n[Sufficiency Check]")
    print(f"  Threshold: r ≥ 0.80")
    print(f"  Result: {'✓ SUFFICIENT' if is_sufficient else '✗ INSUFFICIENT'}")
    
    # Sample size sensitivity
    print(f"\n[Sample Size Sensitivity Analysis]")
    sensitivity_df = analyze_instance_count_sensitivity(predictions_by_syl)
    print(sensitivity_df.to_string(index=False))
    
    print(f"\n[Interpretation]")
    print(f"  The reference paper used 20 instances and found r=0.85.")
    print(f"  Our 40 syllogisms (160 instances) should provide even better coverage.")
