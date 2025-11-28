"""
Metrics Calculation for Syllogistic Reasoning Benchmark

Calculates:
- Accuracy (overall and per-category)
- Precision, Recall, F1
- Confidence calibration
- Per-model statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetricsResult:
    """Container for calculated metrics."""
    accuracy: float
    precision_valid: float
    recall_valid: float
    f1_valid: float
    precision_invalid: float
    recall_invalid: float
    f1_invalid: float
    total_samples: int
    correct_samples: int
    confusion_matrix: Dict[str, Dict[str, int]]
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "precision_valid": self.precision_valid,
            "recall_valid": self.recall_valid,
            "f1_valid": self.f1_valid,
            "precision_invalid": self.precision_invalid,
            "recall_invalid": self.recall_invalid,
            "f1_invalid": self.f1_invalid,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "confusion_matrix": self.confusion_matrix
        }


# =============================================================================
# CORE METRICS FUNCTIONS
# =============================================================================

def calculate_accuracy(
    predictions: List[str],
    ground_truths: List[str]
) -> float:
    """Calculate accuracy."""
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(
        1 for p, g in zip(predictions, ground_truths)
        if p.lower() == g.lower()
    )
    return correct / len(predictions)


def calculate_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Calculate confusion matrix.
    
    Returns:
        Dict with structure:
        {
            "valid": {"valid": TP, "invalid": FN},
            "invalid": {"valid": FP, "invalid": TN}
        }
    """
    matrix = {
        "valid": {"valid": 0, "invalid": 0},
        "invalid": {"valid": 0, "invalid": 0}
    }
    
    for pred, truth in zip(predictions, ground_truths):
        pred_lower = pred.lower() if pred in ["valid", "invalid"] else "invalid"
        truth_lower = truth.lower()
        
        if truth_lower in matrix and pred_lower in matrix[truth_lower]:
            matrix[truth_lower][pred_lower] += 1
    
    return matrix


def calculate_precision_recall_f1(
    confusion_matrix: Dict[str, Dict[str, int]],
    target_class: str = "valid"
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for a target class.
    
    Args:
        confusion_matrix: Confusion matrix dict
        target_class: "valid" or "invalid"
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if target_class == "valid":
        tp = confusion_matrix["valid"]["valid"]
        fp = confusion_matrix["invalid"]["valid"]
        fn = confusion_matrix["valid"]["invalid"]
    else:
        tp = confusion_matrix["invalid"]["invalid"]
        fp = confusion_matrix["valid"]["invalid"]
        fn = confusion_matrix["invalid"]["valid"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_all_metrics(
    predictions: List[str],
    ground_truths: List[str]
) -> MetricsResult:
    """Calculate all metrics from predictions and ground truths."""
    confusion = calculate_confusion_matrix(predictions, ground_truths)
    
    p_valid, r_valid, f1_valid = calculate_precision_recall_f1(confusion, "valid")
    p_invalid, r_invalid, f1_invalid = calculate_precision_recall_f1(confusion, "invalid")
    
    accuracy = calculate_accuracy(predictions, ground_truths)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p.lower() == g.lower())
    
    return MetricsResult(
        accuracy=accuracy,
        precision_valid=p_valid,
        recall_valid=r_valid,
        f1_valid=f1_valid,
        precision_invalid=p_invalid,
        recall_invalid=r_invalid,
        f1_invalid=f1_invalid,
        total_samples=len(predictions),
        correct_samples=correct,
        confusion_matrix=confusion
    )


# =============================================================================
# RESULT LOADING AND PROCESSING
# =============================================================================

def load_results_file(filepath: Path) -> Dict:
    """Load results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_predictions_from_results(results: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Extract predictions and ground truths from result list.
    
    Returns:
        Tuple of (predictions, ground_truths)
    """
    predictions = []
    ground_truths = []
    
    for r in results:
        if 'predicted' in r and 'ground_truth' in r:
            predictions.append(r['predicted'])
            ground_truths.append(r['ground_truth'])
    
    return predictions, ground_truths


def calculate_metrics_from_file(filepath: Path) -> MetricsResult:
    """Load a results file and calculate metrics."""
    data = load_results_file(filepath)
    predictions, ground_truths = extract_predictions_from_results(data.get('results', []))
    return calculate_all_metrics(predictions, ground_truths)


# =============================================================================
# AGGREGATED METRICS
# =============================================================================

def calculate_metrics_by_category(
    results: List[Dict],
    category_key: str = "syllogism_id"
) -> Dict[str, MetricsResult]:
    """
    Calculate metrics grouped by a category.
    
    Args:
        results: List of result dictionaries
        category_key: Key to group by (e.g., "syllogism_id", "variant")
        
    Returns:
        Dict mapping category values to MetricsResult
    """
    grouped = defaultdict(list)
    
    for r in results:
        if category_key in r:
            grouped[r[category_key]].append(r)
    
    metrics_by_category = {}
    for category, category_results in grouped.items():
        predictions, ground_truths = extract_predictions_from_results(category_results)
        if predictions:
            metrics_by_category[category] = calculate_all_metrics(predictions, ground_truths)
    
    return metrics_by_category


def aggregate_metrics_across_files(
    results_dir: Path,
    temperature: float,
    strategy: str
) -> pd.DataFrame:
    """
    Aggregate metrics across all model files for a given temperature and strategy.
    
    Returns:
        DataFrame with one row per model
    """
    temp_dir = results_dir / "raw_responses" / f"temperature_{temperature}"
    
    if not temp_dir.exists():
        return pd.DataFrame()
    
    rows = []
    for filepath in temp_dir.glob(f"*_{strategy}.json"):
        model_key = filepath.stem.replace(f"_{strategy}", "")
        
        try:
            metrics = calculate_metrics_from_file(filepath)
            row = {
                "model": model_key,
                "temperature": temperature,
                "strategy": strategy,
                **metrics.to_dict()
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return pd.DataFrame(rows)


def create_summary_table(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a summary table of all metrics across all configurations.
    
    Returns:
        DataFrame with metrics for all model/temperature/strategy combinations
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    all_rows = []
    
    for temperature in config.experiment.temperatures:
        for strategy in config.experiment.prompting_strategies:
            df = aggregate_metrics_across_files(results_dir, temperature, strategy)
            if not df.empty:
                all_rows.append(df)
    
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("METRICS CALCULATION TEST")
    print("=" * 60)
    
    # Test data
    predictions = ["valid", "valid", "invalid", "invalid", "valid", "invalid"]
    ground_truths = ["valid", "invalid", "invalid", "valid", "valid", "invalid"]
    
    print("\n[Test Data]")
    print(f"Predictions:   {predictions}")
    print(f"Ground truths: {ground_truths}")
    
    print("\n[Metrics]")
    metrics = calculate_all_metrics(predictions, ground_truths)
    print(f"  Accuracy: {metrics.accuracy:.2%}")
    print(f"  Precision (valid): {metrics.precision_valid:.2%}")
    print(f"  Recall (valid): {metrics.recall_valid:.2%}")
    print(f"  F1 (valid): {metrics.f1_valid:.2%}")
    print(f"  Correct: {metrics.correct_samples}/{metrics.total_samples}")
    
    print("\n[Confusion Matrix]")
    for actual, preds in metrics.confusion_matrix.items():
        print(f"  Actual {actual}: {preds}")
