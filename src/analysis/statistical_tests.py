"""
Statistical Testing for Syllogistic Reasoning Benchmark

Implements:
- Paired t-tests for comparing prompting conditions
- Effect size calculations (Cohen's d)
- Multiple comparison corrections (Bonferroni)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats
import pandas as pd

from src.config import config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TTestResult:
    """Result of a paired t-test."""
    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    t_statistic: float
    p_value: float
    cohens_d: float
    n_samples: int
    is_significant_005: bool
    is_significant_001: bool
    
    def to_dict(self) -> Dict:
        return {
            "condition_a": self.condition_a,
            "condition_b": self.condition_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "n_samples": self.n_samples,
            "is_significant_005": self.is_significant_005,
            "is_significant_001": self.is_significant_001
        }
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        sig_str = "***" if self.is_significant_001 else ("**" if self.is_significant_005 else "ns")
        return (
            f"{self.condition_a} vs {self.condition_b}: "
            f"t({self.n_samples-1})={self.t_statistic:.3f}, p={self.p_value:.4f} {sig_str}, "
            f"d={self.cohens_d:.3f}"
        )


@dataclass
class MultipleComparisonResult:
    """Results with multiple comparison correction."""
    tests: List[TTestResult]
    correction_method: str
    alpha: float
    corrected_alpha: float
    significant_after_correction: List[bool]
    
    def to_dict(self) -> Dict:
        return {
            "tests": [t.to_dict() for t in self.tests],
            "correction_method": self.correction_method,
            "alpha": self.alpha,
            "corrected_alpha": self.corrected_alpha,
            "significant_after_correction": self.significant_after_correction
        }


# =============================================================================
# CORE STATISTICAL FUNCTIONS
# =============================================================================

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for paired samples.
    
    Args:
        group1: First group of measurements
        group2: Second group of measurements (paired with group1)
        
    Returns:
        Cohen's d effect size
    """
    diff = group1 - group2
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0


def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
    condition_a: str = "A",
    condition_b: str = "B",
    alpha: float = 0.05
) -> TTestResult:
    """
    Perform a paired t-test between two conditions.
    
    Args:
        scores_a: Accuracy scores for condition A (per model)
        scores_b: Accuracy scores for condition B (per model, same order)
        condition_a: Name of condition A
        condition_b: Name of condition B
        alpha: Significance level
        
    Returns:
        TTestResult with test statistics
    """
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    
    if len(arr_a) != len(arr_b):
        raise ValueError("Paired samples must have same length")
    
    if len(arr_a) < 2:
        raise ValueError("Need at least 2 samples for t-test")
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
    
    # Calculate effect size
    cohens_d = calculate_cohens_d(arr_a, arr_b)
    
    return TTestResult(
        condition_a=condition_a,
        condition_b=condition_b,
        mean_a=float(np.mean(arr_a)),
        mean_b=float(np.mean(arr_b)),
        std_a=float(np.std(arr_a, ddof=1)),
        std_b=float(np.std(arr_b, ddof=1)),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        cohens_d=float(cohens_d),
        n_samples=len(arr_a),
        is_significant_005=p_value < 0.05,
        is_significant_001=p_value < 0.01
    )


def bonferroni_correction(
    tests: List[TTestResult],
    alpha: float = 0.05
) -> MultipleComparisonResult:
    """
    Apply Bonferroni correction to multiple t-tests.
    
    Args:
        tests: List of TTestResult objects
        alpha: Family-wise error rate
        
    Returns:
        MultipleComparisonResult with corrected significance
    """
    n_tests = len(tests)
    corrected_alpha = alpha / n_tests
    
    significant_after = [t.p_value < corrected_alpha for t in tests]
    
    return MultipleComparisonResult(
        tests=tests,
        correction_method="bonferroni",
        alpha=alpha,
        corrected_alpha=corrected_alpha,
        significant_after_correction=significant_after
    )


# =============================================================================
# PROMPTING CONDITION COMPARISONS
# =============================================================================

def load_condition_accuracies(
    results_dir: Path,
    temperature: float = 0.0
) -> Dict[str, Dict[str, float]]:
    """
    Load accuracies for all models across different prompting strategies.
    
    Returns:
        Dict[strategy, Dict[model_key, accuracy]]
    """
    from src.evaluation.calculate_metrics import calculate_metrics_from_file
    
    strategies = config.experiment.prompting_strategies
    condition_accs = {s: {} for s in strategies}
    
    for strategy in strategies:
        temp_dir = results_dir / "raw_responses" / f"temperature_{temperature}"
        
        if not temp_dir.exists():
            continue
        
        for filepath in temp_dir.glob(f"*_{strategy}.json"):
            model_key = filepath.stem.replace(f"_{strategy}", "")
            
            try:
                metrics = calculate_metrics_from_file(filepath)
                condition_accs[strategy][model_key] = metrics.accuracy
            except Exception:
                pass
    
    return condition_accs


def compare_prompting_conditions(
    results_dir: Optional[Path] = None,
    temperature: float = 0.0
) -> Dict[str, TTestResult]:
    """
    Compare all pairs of prompting conditions using paired t-tests.
    
    Comparisons:
    - zero_shot vs few_shot
    - zero_shot vs one_shot  
    - zero_shot vs zero_shot_cot
    - few_shot vs zero_shot_cot
    - one_shot vs few_shot
    - one_shot vs zero_shot_cot
    
    Returns:
        Dict mapping comparison name to TTestResult
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    condition_accs = load_condition_accuracies(results_dir, temperature)
    
    # Define comparisons (matching the reference paper)
    comparisons = [
        ("zero_shot", "few_shot"),
        ("zero_shot", "one_shot"),
        ("zero_shot", "zero_shot_cot"),
        ("one_shot", "few_shot"),
        ("one_shot", "zero_shot_cot"),
        ("few_shot", "zero_shot_cot"),
    ]
    
    results = {}
    
    for cond_a, cond_b in comparisons:
        if cond_a not in condition_accs or cond_b not in condition_accs:
            continue
        
        accs_a = condition_accs[cond_a]
        accs_b = condition_accs[cond_b]
        
        # Get common models
        common_models = sorted(set(accs_a.keys()) & set(accs_b.keys()))
        
        if len(common_models) < 2:
            continue
        
        scores_a = [accs_a[m] for m in common_models]
        scores_b = [accs_b[m] for m in common_models]
        
        comparison_key = f"{cond_a}_vs_{cond_b}"
        results[comparison_key] = paired_ttest(
            scores_a, scores_b,
            condition_a=cond_a,
            condition_b=cond_b
        )
    
    return results


def compare_temperatures(
    results_dir: Optional[Path] = None,
    strategy: str = "zero_shot"
) -> Optional[TTestResult]:
    """
    Compare T=0 vs T=1 (or T=0.5) using paired t-test.
    
    Returns:
        TTestResult or None if insufficient data
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    from src.evaluation.calculate_metrics import calculate_metrics_from_file
    
    temps = config.experiment.temperatures
    if len(temps) < 2:
        return None
    
    temp_low = min(temps)
    temp_high = max(temps)
    
    accs_low = {}
    accs_high = {}
    
    for temp, accs_dict in [(temp_low, accs_low), (temp_high, accs_high)]:
        temp_dir = results_dir / "raw_responses" / f"temperature_{temp}"
        
        if not temp_dir.exists():
            continue
        
        for filepath in temp_dir.glob(f"*_{strategy}.json"):
            model_key = filepath.stem.replace(f"_{strategy}", "")
            
            try:
                metrics = calculate_metrics_from_file(filepath)
                accs_dict[model_key] = metrics.accuracy
            except Exception:
                pass
    
    # Get common models
    common_models = sorted(set(accs_low.keys()) & set(accs_high.keys()))
    
    if len(common_models) < 2:
        return None
    
    scores_low = [accs_low[m] for m in common_models]
    scores_high = [accs_high[m] for m in common_models]
    
    return paired_ttest(
        scores_low, scores_high,
        condition_a=f"T={temp_low}",
        condition_b=f"T={temp_high}"
    )


def run_all_statistical_tests(
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Run all statistical tests and return comprehensive results.
    
    Returns:
        Dict with all test results
    """
    if results_dir is None:
        results_dir = config.experiment.results_full_path
    
    all_results = {
        "prompting_comparisons": {},
        "temperature_comparison": None,
        "multiple_comparison_correction": None
    }
    
    # 1. Prompting condition comparisons at T=0
    prompt_results = compare_prompting_conditions(results_dir, temperature=0.0)
    all_results["prompting_comparisons"] = {
        k: v.to_dict() for k, v in prompt_results.items()
    }
    
    # 2. Temperature comparison
    temp_result = compare_temperatures(results_dir)
    if temp_result:
        all_results["temperature_comparison"] = temp_result.to_dict()
    
    # 3. Multiple comparison correction
    if prompt_results:
        corrected = bonferroni_correction(list(prompt_results.values()))
        all_results["multiple_comparison_correction"] = corrected.to_dict()
    
    return all_results


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STATISTICAL TESTS - DEMO")
    print("=" * 60)
    
    # Simulated data: accuracy scores for 10 models under different conditions
    np.random.seed(42)
    
    # Simulate: few-shot slightly better than zero-shot
    zero_shot_accs = np.random.normal(0.70, 0.10, 10)
    few_shot_accs = zero_shot_accs + np.random.normal(0.02, 0.03, 10)
    cot_accs = zero_shot_accs + np.random.normal(0.08, 0.04, 10)  # CoT significantly better
    
    print("\n[Simulated Data]")
    print(f"Zero-shot mean: {np.mean(zero_shot_accs):.3f} ± {np.std(zero_shot_accs):.3f}")
    print(f"Few-shot mean: {np.mean(few_shot_accs):.3f} ± {np.std(few_shot_accs):.3f}")
    print(f"CoT mean: {np.mean(cot_accs):.3f} ± {np.std(cot_accs):.3f}")
    
    # Test 1: Zero-shot vs Few-shot (expect NOT significant)
    result1 = paired_ttest(
        list(zero_shot_accs), list(few_shot_accs),
        "zero_shot", "few_shot"
    )
    print(f"\n[Test 1] {result1.summary()}")
    
    # Test 2: Zero-shot vs CoT (expect significant)
    result2 = paired_ttest(
        list(zero_shot_accs), list(cot_accs),
        "zero_shot", "zero_shot_cot"
    )
    print(f"[Test 2] {result2.summary()}")
    
    # Multiple comparison correction
    corrected = bonferroni_correction([result1, result2])
    print(f"\n[Bonferroni Correction]")
    print(f"Original α: {corrected.alpha}")
    print(f"Corrected α: {corrected.corrected_alpha:.4f}")
    print(f"Significant after correction: {corrected.significant_after_correction}")
