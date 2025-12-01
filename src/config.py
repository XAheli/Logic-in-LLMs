"""
Configuration Module for Syllogistic Reasoning Benchmark

Loads configuration from config.toml and environment variables.
Environment variables take precedence over config.toml values.

Usage:
    from src.config import config
    
    # Access API keys
    google_key = config.api_keys.google_api_key
    
    # Access experiment settings
    temps = config.experiment.temperatures
    
    # Get iteration limits (global for all models)
    max_iter, threshold = config.get_iteration_limits()
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_FILE = PROJECT_ROOT / "config.toml"


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class APIKeys:
    """API key configuration."""
    google_api_key: str = ""
    hf_token: str = ""
    
    def __post_init__(self):
        """Load from environment variables if available (takes precedence)."""
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.hf_token = os.getenv("HF_TOKEN", self.hf_token)
        
        # IMPORTANT: Set HF_TOKEN as environment variable for downstream libraries
        # This ensures huggingface_hub and other HF libraries have access to the token
        # and prevents rate limiting issues
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGINGFACE_TOKEN"] = self.hf_token  # Some libs use this variant
    
    def validate(self) -> dict:
        """Check which API keys are configured."""
        return {
            "google": bool(self.google_api_key),
            "huggingface": bool(self.hf_token),
        }
    
    def get_configured_providers(self) -> List[str]:
        """Return list of providers with configured API keys."""
        validation = self.validate()
        return [provider for provider, configured in validation.items() if configured]


@dataclass
class ExperimentSettings:
    """Experiment configuration."""
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    prompting_strategies: List[str] = field(default_factory=lambda: [
        "zero_shot", "one_shot", "few_shot", "zero_shot_cot"
    ])
    dataset_path: str = "data/syllogisms_master_dataset.json"
    results_dir: str = "results"
    
    @property
    def dataset_full_path(self) -> Path:
        """Get full path to dataset file."""
        return PROJECT_ROOT / self.dataset_path
    
    @property
    def results_full_path(self) -> Path:
        """Get full path to results directory."""
        return PROJECT_ROOT / self.results_dir


@dataclass
class IterationLimits:
    """
    Iteration limits for adaptive stopping strategy.
    
    Now global for all models since all use paid inference:
    - Google AI Studio (Gemini): google_studio_paid
    - HuggingFace via Fireworks: hf_inf_paid
    """
    max_iterations: int = 10
    threshold: int = 5
    
    def get_limits(self) -> tuple:
        """Get (max_iterations, threshold) - same for all models."""
        return self.max_iterations, self.threshold


@dataclass
class APISettings:
    """API call settings."""
    max_tokens: int = 12000  # High for Gemini 2.5 Pro's internal "thinking" tokens
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit_rpm: int = 0
    timeout: int = 60
    # Sampling parameters for reproducibility
    top_p: float = 1.0  # Nucleus sampling disabled (using temperature only)
    presence_penalty: float = 0.0  # No penalty for topic repetition
    frequency_penalty: float = 0.0  # No penalty for token repetition
    # HuggingFace specific rate limiting
    hf_min_request_interval: float = 1.5  # Seconds between HF requests
    hf_max_retries_429: int = 10  # Max retries for rate limit errors
    hf_base_backoff: float = 5.0  # Base wait time for backoff


@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/benchmark.log"
    
    @property
    def log_file_path(self) -> Path:
        """Get full path to log file."""
        return PROJECT_ROOT / self.log_file


@dataclass
class OutputSettings:
    """Output configuration."""
    save_raw_responses: bool = True
    save_parsed_results: bool = True
    results_format: str = "json"


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """Main configuration container."""
    api_keys: APIKeys = field(default_factory=APIKeys)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    iteration_limits: IterationLimits = field(default_factory=IterationLimits)
    api_settings: APISettings = field(default_factory=APISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    
    @classmethod
    def from_toml(cls, config_path: Path = CONFIG_FILE) -> "Config":
        """Load configuration from TOML file."""
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return cls()
        
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        
        return cls(
            api_keys=APIKeys(**data.get("api_keys", {})),
            experiment=ExperimentSettings(**data.get("experiment", {})),
            iteration_limits=IterationLimits(**data.get("iteration_limits", {})),
            api_settings=APISettings(**data.get("api_settings", {})),
            logging=LoggingSettings(**data.get("logging", {})),
            output=OutputSettings(**data.get("output", {})),
        )
    
    def get_iteration_limits(self) -> tuple:
        """Get (max_iterations, threshold) - same for all models."""
        return self.iteration_limits.get_limits()
    
    def get_temperature_output_dir(self, temperature: float) -> Path:
        """Get the output directory for a specific temperature."""
        temp_dir = self.experiment.results_full_path / "raw_responses" / f"temperature_{temperature}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def ensure_directories(self):
        """Create all necessary directories."""
        # Results directories
        results_path = self.experiment.results_full_path
        (results_path / "raw_responses").mkdir(parents=True, exist_ok=True)
        (results_path / "parsed_results").mkdir(parents=True, exist_ok=True)
        
        # Temperature-specific directories
        for temp in self.experiment.temperatures:
            (results_path / "raw_responses" / f"temperature_{temp}").mkdir(exist_ok=True)
        
        # Logs directory
        self.logging.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # API Keys
        print("\n[API Keys]")
        validation = self.api_keys.validate()
        for provider, configured in validation.items():
            status = "✓ Configured" if configured else "✗ Not configured"
            print(f"  {provider.capitalize()}: {status}")
        
        # Experiment Settings
        print("\n[Experiment Settings]")
        print(f"  Temperatures: {self.experiment.temperatures}")
        print(f"  Prompting strategies: {self.experiment.prompting_strategies}")
        print(f"  Dataset: {self.experiment.dataset_path}")
        print(f"  Results dir: {self.experiment.results_dir}")
        
        # Iteration Limits
        print("\n[Iteration Limits]")
        print(f"  Global limits (all models): max={self.iteration_limits.max_iterations}, "
              f"threshold={self.iteration_limits.threshold}")
        
        # API Settings
        print("\n[API Settings]")
        print(f"  Max tokens: {self.api_settings.max_tokens}")
        print(f"  Max retries: {self.api_settings.max_retries}")
        print(f"  Retry delay: {self.api_settings.retry_delay}s")
        print(f"  Timeout: {self.api_settings.timeout}s")
        
        print("\n" + "=" * 60)


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

# Load configuration on module import
config = Config.from_toml()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def reload_config() -> Config:
    """Reload configuration from file."""
    global config
    config = Config.from_toml()
    return config


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider."""
    provider = provider.lower()
    if provider == "google":
        return config.api_keys.google_api_key or None
    elif provider in ("huggingface", "hf"):
        return config.api_keys.hf_token or None
    return None


def is_provider_configured(provider: str) -> bool:
    """Check if a provider's API key is configured."""
    return bool(get_api_key(provider))


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Print configuration summary
    config.print_summary()
    
    # Test iteration limits
    print("\nIteration Limits Test:")
    max_iter, threshold = config.get_iteration_limits()
    print(f"  Global limits: max={max_iter}, threshold={threshold}")
    
    # Test paths
    print("\nPaths:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Config file: {CONFIG_FILE}")
    print(f"  Dataset path: {config.experiment.dataset_full_path}")
    print(f"  Results path: {config.experiment.results_full_path}")
    
    # Test configured providers
    print(f"\nConfigured providers: {config.api_keys.get_configured_providers()}")
