"""Evaluation module for Syllogistic Reasoning Benchmark.

This module contains:
- parse_responses: Extract predictions from raw LLM responses
- calculate_metrics: Compute accuracy, F1, and other metrics
- consistency_analysis: Analyze cross-variant consistency
- instance_sufficiency: Validate dataset size is sufficient
"""

from src.evaluation.parse_responses import (
    ParsedAnswer,
    ParseResult,
    parse_response,
    parse_cot_response,
    parse_batch_responses,
    extract_answer_string,
    validate_against_ground_truth,
    calculate_parsing_stats
)

from src.evaluation.calculate_metrics import (
    MetricsResult,
    calculate_accuracy,
    calculate_confusion_matrix,
    calculate_precision_recall_f1,
    calculate_all_metrics,
    calculate_metrics_from_file,
    create_summary_table,
    # Belief Bias Analysis
    BeliefBiasResult,
    calculate_belief_bias,
    calculate_belief_bias_from_file,
    create_belief_bias_summary,
    create_belief_bias_heatmap_data
)

from src.evaluation.consistency_analysis import (
    ConsistencyResult,
    ModelConsistencyReport,
    calculate_syllogism_consistency,
    calculate_content_effects,
    analyze_model_consistency,
    analyze_all_models_consistency,
    create_consistency_summary
)

from src.evaluation.instance_sufficiency import (
    SufficiencyResult,
    ModelSufficiencyAnalysis,
    analyze_model_sufficiency,
    analyze_overall_sufficiency,
    create_sufficiency_report,
    analyze_instance_count_sensitivity
)

__all__ = [
    # Response Parsing
    "ParsedAnswer",
    "ParseResult",
    "parse_response",
    "parse_cot_response",
    "parse_batch_responses",
    "extract_answer_string",
    "validate_against_ground_truth",
    "calculate_parsing_stats",
    # Metrics Calculation
    "MetricsResult",
    "calculate_accuracy",
    "calculate_confusion_matrix",
    "calculate_precision_recall_f1",
    "calculate_all_metrics",
    "calculate_metrics_from_file",
    "create_summary_table",
    # Belief Bias Analysis
    "BeliefBiasResult",
    "calculate_belief_bias",
    "calculate_belief_bias_from_file",
    "create_belief_bias_summary",
    "create_belief_bias_heatmap_data",
    # Consistency Analysis
    "ConsistencyResult",
    "ModelConsistencyReport",
    "calculate_syllogism_consistency",
    "calculate_content_effects",
    "analyze_model_consistency",
    "analyze_all_models_consistency",
    "create_consistency_summary",
    # Instance Sufficiency
    "SufficiencyResult",
    "ModelSufficiencyAnalysis",
    "analyze_model_sufficiency",
    "analyze_overall_sufficiency",
    "create_sufficiency_report",
    "analyze_instance_count_sensitivity",
]
