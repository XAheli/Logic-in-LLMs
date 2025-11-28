"""
Inference module for Syllogistic Reasoning Benchmark.

This module contains:
- model_registry: Model definitions and metadata
- api_clients: API clients for Gemini and HuggingFace
- stopping_strategy: Adaptive stopping for temperature > 0
- batch_processing: Batch experiment runner
"""

from src.inference.model_registry import (
    MODEL_REGISTRY,
    ModelConfig,
    BillingType,
    list_all_models,
    get_google_studio_models,
    get_hf_inference_models,
    get_lm_arena_models
)

from src.inference.api_clients import (
    APIResponse,
    GeminiClient,
    HuggingFaceClient,
    get_client_for_model
)

from src.inference.stopping_strategy import (
    VoteResult,
    AdaptiveStoppingStrategy,
    IterationResult,
    StoppingResult,
    parse_response,
    get_adaptive_vote,
    run_with_stopping
)

from src.inference.batch_processing import (
    ExperimentResult,
    BatchConfig,
    BatchProcessor,
    run_full_benchmark,
    run_quick_test
)

__all__ = [
    # Model Registry
    "MODEL_REGISTRY",
    "ModelConfig",
    "BillingType",
    "list_all_models",
    "get_google_studio_models",
    "get_hf_inference_models",
    "get_lm_arena_models",
    # API Clients
    "APIResponse",
    "GeminiClient",
    "HuggingFaceClient",
    "get_client_for_model",
    # Stopping Strategy
    "VoteResult",
    "AdaptiveStoppingStrategy",
    "IterationResult",
    "StoppingResult",
    "parse_response",
    "get_adaptive_vote",
    "run_with_stopping",
    # Batch Processing
    "ExperimentResult",
    "BatchConfig",
    "BatchProcessor",
    "run_full_benchmark",
    "run_quick_test",
]
