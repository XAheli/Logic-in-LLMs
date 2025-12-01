"""
Model Registry for Syllogistic Reasoning Benchmark

Defines 22 models across 2 providers:
- Google AI Studio (Gemini): 3 models (google_studio_paid)
- HuggingFace Inference API: 19 models (hf_inf_paid) with :cheapest routing

Gemini models:
- gemini-2.5-pro: Most capable, best for complex reasoning
- gemini-2.5-flash: Fast and efficient
- gemini-2.5-flash-lite: Lightweight, fastest

HuggingFace models use the :cheapest suffix for automatic provider routing.
This routes requests to the cheapest available inference provider.

Each model includes:
- provider: API provider identifier (GOOGLE or HUGGINGFACE)
- model_id: Official model identifier for API calls
- display_name: Human-readable name for reports
- billing_type: "google_studio_paid" for Gemini, "hf_inf_paid" for HuggingFace
- lm_arena: Whether model is on LM Arena leaderboard for correlation analysis

Iteration Limits (GLOBAL for all models):
- max_iterations: 10
- threshold: 5

All models are now paid via their respective providers:
- Gemini: Direct Google AI Studio API
- Others: HuggingFace Inference API with :cheapest routing
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class Provider(Enum):
    """API Provider identifiers."""
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class BillingType(Enum):
    """Billing type identifiers."""
    GOOGLE_STUDIO_PAID = "google_studio_paid"
    HF_INF_PAID = "hf_inf_paid"


# =============================================================================
# ITERATION LIMITS (GLOBAL - same for all models)
# =============================================================================

MAX_ITERATIONS = 10
THRESHOLD = 5


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: Provider
    model_id: str
    display_name: str
    billing_type: BillingType
    lm_arena: bool
    
    @property
    def max_iterations(self) -> int:
        """Global iteration limit: 10 for all models."""
        return MAX_ITERATIONS
    
    @property
    def threshold(self) -> int:
        """Global threshold: 5 for all models."""
        return THRESHOLD
    
    @property
    def is_google_studio(self) -> bool:
        """Check if model uses Google AI Studio directly."""
        return self.billing_type == BillingType.GOOGLE_STUDIO_PAID
    
    @property
    def is_hf_inference(self) -> bool:
        """Check if model uses HuggingFace Inference API."""
        return self.billing_type == BillingType.HF_INF_PAID
    
    @property
    def api_model_id(self) -> str:
        """
        Model ID to send to the API client.
        
        For HuggingFace models, appends :cheapest for automatic provider routing.
        For Google models, returns the raw model_id.
        """
        if self.is_hf_inference:
            return f"{self.model_id}:cheapest"
        return self.model_id


# =============================================================================
# MODEL REGISTRY - 23 MODELS TOTAL
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    
    # =========================================================================
    # GOOGLE AI STUDIO - Gemini Models (3 models)
    # Direct API access via Google AI Studio
    # Billing: google_studio_paid
    # =========================================================================
    
    "gemini-2.5-pro": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        billing_type=BillingType.GOOGLE_STUDIO_PAID,
        lm_arena=True,
    ),
    
    "gemini-2.5-flash": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        billing_type=BillingType.GOOGLE_STUDIO_PAID,
        lm_arena=True,
    ),
    
    "gemini-2.5-flash-lite": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
        billing_type=BillingType.GOOGLE_STUDIO_PAID,
        lm_arena=True,
    ),
    
    # =========================================================================
    # HUGGINGFACE INFERENCE API (20 models)
    # Uses :cheapest routing for automatic provider selection
    # Billing: hf_inf_paid
    # =========================================================================
    
    # --- OpenAI OSS Model (1 model) ---
    "gpt-oss-20b": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="openai/gpt-oss-20b",
        display_name="GPT-OSS 20B",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    # --- Meta Llama Models (5 models) ---
    "llama-3.3-70b-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama 3.3 70B Instruct",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "llama-3.2-3b-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B Instruct",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "llama-3.2-1b-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        display_name="Llama 3.2 1B Instruct",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "llama-3.1-8b-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B Instruct",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    # --- Qwen Models (5 models) ---
    "qwen3-next-80b-a3b-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        display_name="Qwen3 Next 80B A3B Instruct",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "qwen3-next-80b-a3b-thinking": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
        display_name="Qwen3 Next 80B A3B Thinking",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "qwen3-32b": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="Qwen/Qwen3-32B",
        display_name="Qwen3 32B",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "qwen3-235b-a22b-thinking": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        display_name="Qwen3 235B A22B Thinking",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "qwq-32b": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="Qwen/QwQ-32B",
        display_name="QwQ 32B",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    # --- Mistral Models (1 model) ---
    "mixtral-8x22b-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        display_name="Mixtral 8x22B Instruct v0.1",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    # --- DeepSeek Models (3 models) ---
    "deepseek-r1": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="deepseek-ai/DeepSeek-R1",
        display_name="DeepSeek R1",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "deepseek-v3.1": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="deepseek-ai/DeepSeek-V3.1",
        display_name="DeepSeek V3.1",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "deepseek-r1-0528": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="deepseek-ai/DeepSeek-R1-0528",
        display_name="DeepSeek R1 0528",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    # --- Google Gemma Models (1 model) ---
    "gemma-3-27b-it": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="google/gemma-3-27b-it",
        display_name="Gemma 3 27B IT",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    # --- Moonshot AI Kimi Models (2 models) ---
    "kimi-k2-thinking": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="moonshotai/Kimi-K2-Thinking",
        display_name="Kimi K2 Thinking",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=False,
    ),
    
    "kimi-k2-instruct": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="moonshotai/Kimi-K2-Instruct",
        display_name="Kimi K2 Instruct",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=False,
    ),
    
    # --- GLM Models (2 models) ---
    "glm-4.5": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="zai-org/GLM-4.5",
        display_name="GLM 4.5",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
    
    "glm-4.6": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="zai-org/GLM-4.6",
        display_name="GLM 4.6",
        billing_type=BillingType.HF_INF_PAID,
        lm_arena=True,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model(model_key: str) -> ModelConfig:
    """Get a model configuration by its key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]


def get_models_by_provider(provider: Provider) -> Dict[str, ModelConfig]:
    """Get all models for a specific provider."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.provider == provider}


def get_models_by_billing_type(billing_type: BillingType) -> Dict[str, ModelConfig]:
    """Get all models with a specific billing type."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.billing_type == billing_type}


def get_google_studio_models() -> Dict[str, ModelConfig]:
    """Get all models using Google AI Studio (Gemini)."""
    return get_models_by_billing_type(BillingType.GOOGLE_STUDIO_PAID)


def get_hf_inference_models() -> Dict[str, ModelConfig]:
    """Get all models using HuggingFace Inference API."""
    return get_models_by_billing_type(BillingType.HF_INF_PAID)


def get_lm_arena_models() -> Dict[str, ModelConfig]:
    """Get all models that are on the LM Arena leaderboard."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.lm_arena}


def list_all_models() -> List[str]:
    """List all available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_model_summary() -> Dict[str, int]:
    """Get a summary of models by provider and billing type."""
    google_models = get_models_by_provider(Provider.GOOGLE)
    hf_models = get_models_by_provider(Provider.HUGGINGFACE)
    
    summary = {
        "total": len(MODEL_REGISTRY),
        "google_studio": len(google_models),
        "hf_inference": len(hf_models),
        "google_studio_paid": len(get_google_studio_models()),
        "hf_inf_paid": len(get_hf_inference_models()),
        "lm_arena": len(get_lm_arena_models()),
    }
    return summary


def print_model_table():
    """Print a formatted table of all models."""
    print("\n" + "=" * 110)
    print(f"{'Key':<35} {'Model ID (API)':<50} {'LM Arena':<10}")
    print("=" * 110)
    
    for key, model in MODEL_REGISTRY.items():
        api_id = model.api_model_id[:48] + ".." if len(model.api_model_id) > 50 else model.api_model_id
        print(f"{key:<35} {api_id:<50} {'Yes' if model.lm_arena else 'No':<10}")
    
    print("=" * 110)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_registry():
    """Validate the model registry for consistency."""
    errors = []
    
    # Expected counts: 22 models total
    # 3 Gemini (google_studio_paid) + 19 HuggingFace (hf_inf_paid) = 22
    expected_total = 22
    expected_google = 3
    expected_huggingface = 19
    
    actual_total = len(MODEL_REGISTRY)
    if actual_total != expected_total:
        errors.append(f"Expected {expected_total} models, found {actual_total}")
    
    # Check Google models count and billing type
    google_models = get_models_by_provider(Provider.GOOGLE)
    if len(google_models) != expected_google:
        errors.append(f"Expected {expected_google} Google models, found {len(google_models)}")
    for key, model in google_models.items():
        if model.billing_type != BillingType.GOOGLE_STUDIO_PAID:
            errors.append(f"{key}: Google model should have billing_type=GOOGLE_STUDIO_PAID")
    
    # Check HuggingFace models count and billing type
    hf_models = get_models_by_provider(Provider.HUGGINGFACE)
    if len(hf_models) != expected_huggingface:
        errors.append(f"Expected {expected_huggingface} HuggingFace models, found {len(hf_models)}")
    for key, model in hf_models.items():
        if model.billing_type != BillingType.HF_INF_PAID:
            errors.append(f"{key}: HuggingFace model should have billing_type=HF_INF_PAID")
        # Check that api_model_id has :cheapest suffix
        if not model.api_model_id.endswith(":cheapest"):
            errors.append(f"{key}: HuggingFace model api_model_id should end with :cheapest")
    
    # Check iteration limits are global (same for all)
    for key, model in MODEL_REGISTRY.items():
        if model.max_iterations != MAX_ITERATIONS:
            errors.append(f"{key}: should have max_iterations={MAX_ITERATIONS}")
        if model.threshold != THRESHOLD:
            errors.append(f"{key}: should have threshold={THRESHOLD}")
    
    if errors:
        raise ValueError("Registry validation failed:\n" + "\n".join(errors))
    
    return True


if __name__ == "__main__":
    # Run validation and print summary
    validate_registry()
    print("✓ Model registry validated successfully!")
    print()
    
    summary = get_model_summary()
    print("Model Summary:")
    print(f"  Total models: {summary['total']}")
    print(f"  Google AI Studio (Gemini): {summary['google_studio']} (google_studio_paid)")
    print(f"  HuggingFace Inference: {summary['hf_inference']} (hf_inf_paid with :cheapest routing)")
    print(f"  On LM Arena: {summary['lm_arena']}")
    print()
    print(f"Global Iteration Limits:")
    print(f"  Max Iterations: {MAX_ITERATIONS}")
    print(f"  Threshold: {THRESHOLD}")
    
    # Print detailed table
    print_model_table()
