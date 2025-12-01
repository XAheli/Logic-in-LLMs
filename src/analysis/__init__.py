"""Analysis module for Syllogistic Reasoning Benchmark.

This module contains:
- correlation: Correlation analysis with LM Arena rankings
- ranking: Model ranking by accuracy
- visualization: Publication-quality figures
- statistical_tests: Paired t-tests for prompting conditions
- variant_correlation: Correlation between syllogism variants
"""

from src.analysis.correlation import (
    CorrelationResult,
    LM_ARENA_RANKINGS,
    get_arena_ranking,
    get_models_with_arena_rankings,
    calculate_correlation,
    analyze_correlation_across_configs,
    create_ranking_comparison_table
)

from src.analysis.ranking import (
    RankedModel,
    rank_models_by_accuracy,
    create_ranking_table,
    create_aggregate_ranking_table,
    compare_rankings_across_strategies,
    compare_rankings_across_temperatures,
    compare_paid_vs_free
)

from src.analysis.visualization import (
    set_publication_style,
    plot_accuracy_heatmap,
    plot_accuracy_by_strategy,
    plot_model_ranking,
    plot_ranking_comparison,
    plot_accuracy_vs_arena,
    plot_consistency_by_model,
    plot_content_effects,
    plot_temperature_effect,
    # Confusion Matrix
    plot_confusion_matrix_heatmap,
    plot_multi_model_confusion_matrices,
    # Belief Bias
    plot_belief_bias_heatmap,
    plot_belief_bias_comparison,
    # Model Similarity
    plot_model_similarity_heatmap
)

from src.analysis.statistical_tests import (
    TTestResult,
    MultipleComparisonResult,
    paired_ttest,
    bonferroni_correction,
    compare_prompting_conditions,
    compare_temperatures,
    run_all_statistical_tests
)

from src.analysis.variant_correlation import (
    VariantCorrelationResult,
    ModelVariantAnalysis,
    calculate_variant_correlation,
    analyze_model_variants,
    analyze_all_models_variants,
    aggregate_variant_correlation
)

__all__ = [
    # Correlation
    "CorrelationResult",
    "LM_ARENA_RANKINGS",
    "get_arena_ranking",
    "get_models_with_arena_rankings",
    "calculate_correlation",
    "analyze_correlation_across_configs",
    "create_ranking_comparison_table",
    # Ranking
    "RankedModel",
    "rank_models_by_accuracy",
    "create_ranking_table",
    "create_aggregate_ranking_table",
    "compare_rankings_across_strategies",
    "compare_rankings_across_temperatures",
    "compare_paid_vs_free",
    # Visualization
    "set_publication_style",
    "plot_accuracy_heatmap",
    "plot_accuracy_by_strategy",
    "plot_model_ranking",
    "plot_ranking_comparison",
    "plot_accuracy_vs_arena",
    "plot_consistency_by_model",
    "plot_content_effects",
    "plot_temperature_effect",
    # Confusion Matrix
    "plot_confusion_matrix_heatmap",
    "plot_multi_model_confusion_matrices",
    # Belief Bias Visualization
    "plot_belief_bias_heatmap",
    "plot_belief_bias_comparison",
    # Model Similarity
    "plot_model_similarity_heatmap",
    # Statistical Tests
    "TTestResult",
    "MultipleComparisonResult",
    "paired_ttest",
    "bonferroni_correction",
    "compare_prompting_conditions",
    "compare_temperatures",
    "run_all_statistical_tests",
    # Variant Correlation
    "VariantCorrelationResult",
    "ModelVariantAnalysis",
    "calculate_variant_correlation",
    "analyze_model_variants",
    "analyze_all_models_variants",
    "aggregate_variant_correlation",
]
