"""
Consistency Analysis for Syllogistic Reasoning Benchmark

Analyzes cross-variant consistency:
- Do models give same answer for logically equivalent variants?
- N vs X, O vs OX (content should not affect logical validity)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConsistencyResult:
    """Result of consistency analysis for one syllogism across variants."""
    syllogism_id: str
    variant_predictions: Dict[str, str]  # variant -> prediction
    is_consistent: bool  # All variants have same prediction
    consistency_score: float  # Proportion agreeing with majority
    majority_prediction: str
    
    def to_dict(self) -> Dict:
        return {
            "syllogism_id": self.syllogism_id,
            "variant_predictions": self.variant_predictions,
            "is_consistent": self.is_consistent,
            "consistency_score": self.consistency_score,
            "majority_prediction": self.majority_prediction
        }


@dataclass
class ModelConsistencyReport:
    """Consistency report for a single model."""
    model_key: str
    overall_consistency: float
    perfect_consistency_count: int
    total_syllogisms: int
    per_syllogism: List[ConsistencyResult]
    
    # Content effect analysis
    name_effect: float  # N vs X difference
    order_effect: float  # N vs O difference
    combined_effect: float  # N vs OX difference
    
    def to_dict(self) -> Dict:
        return {
            "model_key": self.model_key,
            "overall_consistency": self.overall_consistency,
            "perfect_consistency_count": self.perfect_consistency_count,
            "total_syllogisms": self.total_syllogisms,
            "name_effect": self.name_effect,
            "order_effect": self.order_effect,
            "combined_effect": self.combined_effect,
            "per_syllogism": [r.to_dict() for r in self.per_syllogism]
        }


# =============================================================================
# CONSISTENCY CALCULATION
# =============================================================================

def calculate_syllogism_consistency(
    predictions: Dict[str, str]
) -> ConsistencyResult:
    """
    Calculate consistency for one syllogism across its variants.
    
    Args:
        predictions: Dict mapping variant (N, X, O, OX) to prediction
        
    Returns:
        ConsistencyResult
    """
    syllogism_id = ""  # Will be set by caller
    
    if not predictions:
        return ConsistencyResult(
            syllogism_id=syllogism_id,
            variant_predictions=predictions,
            is_consistent=True,
            consistency_score=1.0,
            majority_prediction=""
        )
    
    # Count predictions
    pred_counts = defaultdict(int)
    for pred in predictions.values():
        pred_counts[pred.lower()] += 1
    
    # Find majority
    majority_pred = max(pred_counts.keys(), key=lambda k: pred_counts[k])
    majority_count = pred_counts[majority_pred]
    
    total = len(predictions)
    consistency_score = majority_count / total if total > 0 else 1.0
    is_consistent = len(set(p.lower() for p in predictions.values())) == 1
    
    return ConsistencyResult(
        syllogism_id=syllogism_id,
        variant_predictions=predictions,
        is_consistent=is_consistent,
        consistency_score=consistency_score,
        majority_prediction=majority_pred
    )


def calculate_content_effects(
    syllogism_results: Dict[str, Dict[str, str]]
) -> Tuple[float, float, float]:
    """
    Calculate content effects.
    
    Content effects measure how much changing names or order affects predictions.
    If model is truly reasoning about logic, content shouldn't matter.
    
    Args:
        syllogism_results: Dict[syllogism_id, Dict[variant, prediction]]
        
    Returns:
        Tuple of (name_effect, order_effect, combined_effect)
        - name_effect: proportion where N != X
        - order_effect: proportion where N != O
        - combined_effect: proportion where N != OX
    """
    name_diffs = []
    order_diffs = []
    combined_diffs = []
    
    for syl_id, variants in syllogism_results.items():
        if "N" in variants and "X" in variants:
            name_diffs.append(1 if variants["N"].lower() != variants["X"].lower() else 0)
        
        if "N" in variants and "O" in variants:
            order_diffs.append(1 if variants["N"].lower() != variants["O"].lower() else 0)
        
        if "N" in variants and "OX" in variants:
            combined_diffs.append(1 if variants["N"].lower() != variants["OX"].lower() else 0)
    
    name_effect = np.mean(name_diffs) if name_diffs else 0.0
    order_effect = np.mean(order_diffs) if order_diffs else 0.0
    combined_effect = np.mean(combined_diffs) if combined_diffs else 0.0
    
    return name_effect, order_effect, combined_effect


# =============================================================================
# RESULT LOADING
# =============================================================================

def load_results_for_model(
    results_dir: Path,
    model_key: str,
    temperature: float,
    strategy: str
) -> Dict[str, Dict[str, str]]:
    """
    Load results for a model and organize by syllogism and variant.
    
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
    
    # Organize by syllogism and variant
    organized = defaultdict(dict)
    for result in data.get('results', []):
        syl_id = result.get('syllogism_id', '')
        variant = result.get('variant', '')
        prediction = result.get('predicted', '')
        
        if syl_id and variant and prediction:
            organized[syl_id][variant] = prediction
    
    return dict(organized)


def analyze_model_consistency(
    results_dir: Path,
    model_key: str,
    temperature: float,
    strategy: str
) -> Optional[ModelConsistencyReport]:
    """
    Analyze consistency for a single model.
    
    Returns:
        ModelConsistencyReport or None if no data
    """
    syllogism_results = load_results_for_model(
        results_dir, model_key, temperature, strategy
    )
    
    if not syllogism_results:
        return None
    
    # Calculate per-syllogism consistency
    per_syllogism = []
    for syl_id, variants in syllogism_results.items():
        result = calculate_syllogism_consistency(variants)
        result.syllogism_id = syl_id
        per_syllogism.append(result)
    
    # Calculate overall metrics
    if per_syllogism:
        overall_consistency = np.mean([r.consistency_score for r in per_syllogism])
        perfect_count = sum(1 for r in per_syllogism if r.is_consistent)
    else:
        overall_consistency = 0.0
        perfect_count = 0
    
    # Calculate content effects
    name_effect, order_effect, combined_effect = calculate_content_effects(syllogism_results)
    
    return ModelConsistencyReport(
        model_key=model_key,
        overall_consistency=overall_consistency,
        perfect_consistency_count=perfect_count,
        total_syllogisms=len(per_syllogism),
        per_syllogism=per_syllogism,
        name_effect=name_effect,
        order_effect=order_effect,
        combined_effect=combined_effect
    )


# =============================================================================
# AGGREGATE ANALYSIS
# =============================================================================

def analyze_all_models_consistency(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> pd.DataFrame:
    """
    Analyze consistency for all models.
    
    Returns:
        DataFrame with consistency metrics for each model
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    from src.inference.model_registry import MODEL_REGISTRY
    
    rows = []
    for model_key in MODEL_REGISTRY.keys():
        report = analyze_model_consistency(
            results_dir, model_key, temperature, strategy
        )
        
        if report:
            rows.append({
                "model": model_key,
                "overall_consistency": report.overall_consistency,
                "perfect_consistency": report.perfect_consistency_count,
                "total_syllogisms": report.total_syllogisms,
                "name_effect": report.name_effect,
                "order_effect": report.order_effect,
                "combined_effect": report.combined_effect
            })
    
    return pd.DataFrame(rows)


def create_consistency_summary(
    results_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create consistency summary across all temperatures and strategies.
    
    Returns:
        Dict mapping "temp_strategy" to consistency DataFrame
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    summaries = {}
    
    for temperature in config.experiment.temperatures:
        for strategy in config.experiment.prompting_strategies:
            key = f"temp{temperature}_{strategy}"
            df = analyze_all_models_consistency(
                results_dir, temperature, strategy
            )
            if not df.empty:
                summaries[key] = df
    
    return summaries


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONSISTENCY ANALYSIS TEST")
    print("=" * 60)
    
    # Test data
    test_predictions = {
        "N": "valid",
        "X": "valid",
        "O": "invalid",  # Different - inconsistent
        "OX": "valid"
    }
    
    print("\n[Test Data]")
    print(f"Predictions: {test_predictions}")
    
    result = calculate_syllogism_consistency(test_predictions)
    print(f"\n[Consistency Analysis]")
    print(f"  Is consistent: {result.is_consistent}")
    print(f"  Consistency score: {result.consistency_score:.2%}")
    print(f"  Majority prediction: {result.majority_prediction}")
    
    # Test content effects
    test_syllogisms = {
        "SYL_001": {"N": "valid", "X": "valid", "O": "valid", "OX": "valid"},
        "SYL_002": {"N": "valid", "X": "invalid", "O": "valid", "OX": "invalid"},  # Name change matters
        "SYL_003": {"N": "invalid", "X": "invalid", "O": "valid", "OX": "valid"},  # Order matters
    }
    
    name_eff, order_eff, comb_eff = calculate_content_effects(test_syllogisms)
    print(f"\n[Content Effects]")
    print(f"  Name effect (N vs X): {name_eff:.2%}")
    print(f"  Order effect (N vs O): {order_eff:.2%}")
    print(f"  Combined effect (N vs OX): {comb_eff:.2%}")
