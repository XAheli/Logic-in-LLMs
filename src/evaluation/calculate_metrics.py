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
    
    Automatically detects the class labels from the ground truths.
    For syntax: valid/invalid
    For NLU: believable/unbelievable
    
    Returns:
        Dict with structure:
        {
            "class1": {"class1": TP, "class2": FN},
            "class2": {"class1": FP, "class2": TN}
        }
    """
    # Detect classes from ground truths
    unique_classes = list(set(g.lower() for g in ground_truths))
    
    # Default to valid/invalid if not detected
    if not unique_classes:
        unique_classes = ["valid", "invalid"]
    elif len(unique_classes) == 1:
        # If only one class, add the opposite
        if unique_classes[0] in ["valid", "invalid"]:
            unique_classes = ["valid", "invalid"]
        else:
            unique_classes = ["believable", "unbelievable"]
    
    # Initialize matrix
    matrix = {c: {c2: 0 for c2 in unique_classes} for c in unique_classes}
    
    for pred, truth in zip(predictions, ground_truths):
        pred_lower = pred.lower()
        truth_lower = truth.lower()
        
        # Handle cases where prediction isn't in our classes
        if pred_lower not in unique_classes:
            # Map to closest match or default
            if "valid" in pred_lower:
                pred_lower = "valid"
            elif "invalid" in pred_lower:
                pred_lower = "invalid"
            elif "believ" in pred_lower:
                pred_lower = "believable"
            elif "unbeliev" in pred_lower:
                pred_lower = "unbelievable"
            else:
                pred_lower = unique_classes[0]  # Default to first class
        
        if truth_lower in matrix and pred_lower in matrix[truth_lower]:
            matrix[truth_lower][pred_lower] += 1
    
    return matrix


def calculate_precision_recall_f1(
    confusion_matrix: Dict[str, Dict[str, int]],
    target_class: str = "valid"
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for a target class.
    
    Confusion matrix terms:
    - TP (True Positive): Correctly predicted as target class
    - FP (False Positive): Other class incorrectly predicted as target class  
    - FN (False Negative): Target class incorrectly predicted as other class
    - TN (True Negative): Other class correctly predicted as other class
    
    Note: TN is not used here because precision, recall, and F1 don't require it:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * Precision * Recall / (Precision + Recall)
    
    Args:
        confusion_matrix: Confusion matrix dict
        target_class: Target class name (e.g., "valid", "invalid", "believable", "unbelievable")
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    # Get all classes
    all_classes = list(confusion_matrix.keys())
    other_classes = [c for c in all_classes if c != target_class]
    
    if target_class not in confusion_matrix:
        return 0.0, 0.0, 0.0
    
    # TP: correctly predicted as target class
    tp = confusion_matrix[target_class].get(target_class, 0)
    
    # FP: incorrectly predicted as target class (from other classes)
    fp = sum(confusion_matrix[c].get(target_class, 0) for c in other_classes)
    
    # FN: target class instances predicted as other classes
    fn = sum(confusion_matrix[target_class].get(c, 0) for c in other_classes)
    
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


def extract_predictions_from_results(
    results: List[Dict], 
    ground_truth_type: str = "syntax"
) -> Tuple[List[str], List[str]]:
    """
    Extract predictions and ground truths from result list.
    
    Maps LLM predictions (correct/incorrect) to ground truth format:
    - syntax: correct→valid, incorrect→invalid
    - NLU: correct→believable, incorrect→unbelievable
    
    Args:
        results: List of result dicts
        ground_truth_type: "syntax" or "NLU"
    
    Returns:
        Tuple of (mapped_predictions, ground_truths)
    """
    predictions = []
    ground_truths = []
    
    for r in results:
        # Get raw prediction
        predicted = r.get('predicted', '')
        
        # Map prediction based on ground truth type
        if ground_truth_type == "syntax":
            # correct→valid, incorrect→invalid
            mapped_pred = "valid" if predicted == "correct" else "invalid"
            gt_key = 'ground_truth_syntax'
        elif ground_truth_type == "NLU":
            # correct→believable, incorrect→unbelievable
            mapped_pred = "believable" if predicted == "correct" else "unbelievable"
            gt_key = 'ground_truth_NLU'
        else:
            raise ValueError(f"Invalid ground_truth_type: {ground_truth_type}. Must be 'syntax' or 'NLU'.")
        
        # Get ground truth
        ground_truth = r.get(gt_key, '')
        
        if mapped_pred and ground_truth:
            predictions.append(mapped_pred)
            ground_truths.append(ground_truth)
    
    return predictions, ground_truths


def calculate_metrics_from_file(
    filepath: Path, 
    ground_truth_type: str = "syntax"
) -> MetricsResult:
    """
    Load a results file and calculate metrics.
    
    Args:
        filepath: Path to results JSON file
        ground_truth_type: "syntax" or "NLU"
    """
    data = load_results_file(filepath)
    predictions, ground_truths = extract_predictions_from_results(
        data.get('results', []), 
        ground_truth_type
    )
    return calculate_all_metrics(predictions, ground_truths)


def calculate_metrics_from_file_both(filepath: Path) -> Dict[str, MetricsResult]:
    """
    Load a results file and calculate metrics against BOTH ground truths.
    
    Returns:
        Dict with "syntax" and "NLU" MetricsResult
    """
    return {
        "syntax": calculate_metrics_from_file(filepath, "syntax"),
        "NLU": calculate_metrics_from_file(filepath, "NLU")
    }


# =============================================================================
# BELIEF BIAS ANALYSIS
# =============================================================================

@dataclass
class BeliefBiasResult:
    """
    Results of belief bias analysis.
    
    Belief bias occurs when the semantic believability of a conclusion
    affects judgments of its logical validity.
    
    The 4 ground truth combinations:
    - valid + believable: Logic and intuition align (easiest)
    - valid + unbelievable: Logically correct but counter-intuitive
    - invalid + believable: Logic wrong but sounds right (BELIEF BIAS TRAP)
    - invalid + unbelievable: Both logic and intuition say wrong
    """
    valid_believable_accuracy: float
    valid_unbelievable_accuracy: float
    invalid_believable_accuracy: float
    invalid_unbelievable_accuracy: float
    
    # Counts per category
    valid_believable_count: int
    valid_unbelievable_count: int
    invalid_believable_count: int
    invalid_unbelievable_count: int
    
    # Derived metrics
    @property
    def congruent_accuracy(self) -> float:
        """Accuracy on congruent cases (logic aligns with intuition)."""
        total = self.valid_believable_count + self.invalid_unbelievable_count
        if total == 0:
            return 0.0
        weighted = (
            self.valid_believable_accuracy * self.valid_believable_count +
            self.invalid_unbelievable_accuracy * self.invalid_unbelievable_count
        )
        return weighted / total
    
    @property
    def incongruent_accuracy(self) -> float:
        """Accuracy on incongruent cases (logic conflicts with intuition)."""
        total = self.valid_unbelievable_count + self.invalid_believable_count
        if total == 0:
            return 0.0
        weighted = (
            self.valid_unbelievable_accuracy * self.valid_unbelievable_count +
            self.invalid_believable_accuracy * self.invalid_believable_count
        )
        return weighted / total
    
    @property
    def belief_bias_effect(self) -> float:
        """
        Belief bias effect size.
        
        Positive = model performs better when logic aligns with intuition
        Negative = model performs better when logic conflicts (unusual)
        Zero = no belief bias
        """
        return self.congruent_accuracy - self.incongruent_accuracy
    
    def to_dict(self) -> Dict:
        return {
            "valid_believable_accuracy": self.valid_believable_accuracy,
            "valid_unbelievable_accuracy": self.valid_unbelievable_accuracy,
            "invalid_believable_accuracy": self.invalid_believable_accuracy,
            "invalid_unbelievable_accuracy": self.invalid_unbelievable_accuracy,
            "valid_believable_count": self.valid_believable_count,
            "valid_unbelievable_count": self.valid_unbelievable_count,
            "invalid_believable_count": self.invalid_believable_count,
            "invalid_unbelievable_count": self.invalid_unbelievable_count,
            "congruent_accuracy": self.congruent_accuracy,
            "incongruent_accuracy": self.incongruent_accuracy,
            "belief_bias_effect": self.belief_bias_effect
        }


def calculate_belief_bias(results: List[Dict]) -> BeliefBiasResult:
    """
    Calculate belief bias metrics from experiment results.
    
    Analyzes accuracy across the 4 combinations of:
    - Syntax ground truth: valid / invalid
    - NLU ground truth: believable / unbelievable
    
    Args:
        results: List of result dictionaries with:
            - predicted: "correct" or "incorrect"
            - ground_truth_syntax: "valid" or "invalid"
            - ground_truth_NLU: "believable" or "unbelievable"
    
    Returns:
        BeliefBiasResult with per-category accuracies
    """
    # Group by ground truth combination
    categories = {
        ('valid', 'believable'): {'correct': 0, 'total': 0},
        ('valid', 'unbelievable'): {'correct': 0, 'total': 0},
        ('invalid', 'believable'): {'correct': 0, 'total': 0},
        ('invalid', 'unbelievable'): {'correct': 0, 'total': 0},
    }
    
    for r in results:
        predicted = r.get('predicted', '').lower()
        syntax_gt = r.get('ground_truth_syntax', '').lower()
        nlu_gt = r.get('ground_truth_NLU', '').lower()
        
        if not predicted or not syntax_gt or not nlu_gt:
            continue
        
        key = (syntax_gt, nlu_gt)
        if key not in categories:
            continue
        
        categories[key]['total'] += 1
        
        # Check if prediction is correct (against syntax)
        # correct→valid, incorrect→invalid
        predicted_syntax = "valid" if predicted == "correct" else "invalid"
        if predicted_syntax == syntax_gt:
            categories[key]['correct'] += 1
    
    # Calculate accuracies
    def safe_accuracy(cat):
        if cat['total'] == 0:
            return 0.0
        return cat['correct'] / cat['total']
    
    return BeliefBiasResult(
        valid_believable_accuracy=safe_accuracy(categories[('valid', 'believable')]),
        valid_unbelievable_accuracy=safe_accuracy(categories[('valid', 'unbelievable')]),
        invalid_believable_accuracy=safe_accuracy(categories[('invalid', 'believable')]),
        invalid_unbelievable_accuracy=safe_accuracy(categories[('invalid', 'unbelievable')]),
        valid_believable_count=categories[('valid', 'believable')]['total'],
        valid_unbelievable_count=categories[('valid', 'unbelievable')]['total'],
        invalid_believable_count=categories[('invalid', 'believable')]['total'],
        invalid_unbelievable_count=categories[('invalid', 'unbelievable')]['total']
    )


def calculate_belief_bias_from_file(filepath: Path) -> BeliefBiasResult:
    """Load a results file and calculate belief bias metrics."""
    data = load_results_file(filepath)
    return calculate_belief_bias(data.get('results', []))


def create_belief_bias_summary(
    results_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create a summary table of belief bias metrics across all configurations.
    
    Returns:
        DataFrame with columns:
            - model, temperature, strategy
            - valid_believable_accuracy, valid_unbelievable_accuracy, etc.
            - belief_bias_effect
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    rows = []
    
    for temperature in config.experiment.temperatures:
        temp_dir = results_dir / "raw_responses" / f"temperature_{temperature}"
        
        if not temp_dir.exists():
            continue
        
        for strategy in config.experiment.prompting_strategies:
            for filepath in temp_dir.glob(f"*_{strategy}.json"):
                model_key = filepath.stem.replace(f"_{strategy}", "")
                
                try:
                    bias_result = calculate_belief_bias_from_file(filepath)
                    row = {
                        "model": model_key,
                        "temperature": temperature,
                        "strategy": strategy,
                        **bias_result.to_dict()
                    }
                    rows.append(row)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    return pd.DataFrame(rows)


def create_belief_bias_heatmap_data(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0,
    strategy: str = "zero_shot"
) -> pd.DataFrame:
    """
    Create DataFrame suitable for belief bias heatmap visualization.
    
    Returns:
        DataFrame with columns [model, syntax_gt, nlu_gt, accuracy, count]
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    temp_dir = results_dir / "raw_responses" / f"temperature_{temperature}"
    
    if not temp_dir.exists():
        return pd.DataFrame()
    
    rows = []
    
    for filepath in temp_dir.glob(f"*_{strategy}.json"):
        model_key = filepath.stem.replace(f"_{strategy}", "")
        
        try:
            bias = calculate_belief_bias_from_file(filepath)
            
            # Create one row per ground truth combination
            for syntax, nlu, acc, count in [
                ('valid', 'believable', bias.valid_believable_accuracy, bias.valid_believable_count),
                ('valid', 'unbelievable', bias.valid_unbelievable_accuracy, bias.valid_unbelievable_count),
                ('invalid', 'believable', bias.invalid_believable_accuracy, bias.invalid_believable_count),
                ('invalid', 'unbelievable', bias.invalid_unbelievable_accuracy, bias.invalid_unbelievable_count),
            ]:
                rows.append({
                    'model': model_key,
                    'syntax_gt': syntax,
                    'nlu_gt': nlu,
                    'accuracy': acc,
                    'count': count
                })
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return pd.DataFrame(rows)


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
