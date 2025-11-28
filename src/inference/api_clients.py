"""
API Clients for Syllogistic Reasoning Benchmark

Provides unified interface for querying LLMs across 2 providers:
- Google AI Studio (Gemini models) - google_studio_paid
- HuggingFace Inference API via Fireworks (21 models) - hf_inf_paid

Each client handles:
- API authentication
- Request formatting
- Error handling with retries
- Rate limiting (if configured)
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Third-party imports (will be imported conditionally)
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    InferenceClient = None

# Local imports
from src.config import config, get_api_key
from src.inference.model_registry import (
    MODEL_REGISTRY, 
    ModelConfig, 
    Provider,
    get_model
)


# =============================================================================
# BASE CLIENT
# =============================================================================

@dataclass
class APIResponse:
    """Standardized response from any API."""
    text: str
    model: str
    provider: str
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class BaseAPIClient(ABC):
    """Base class for all API clients."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.max_retries = config.api_settings.max_retries
        self.retry_delay = config.api_settings.retry_delay
        self.timeout = config.api_settings.timeout
        self.max_tokens = config.api_settings.max_tokens
        self.top_p = config.api_settings.top_p
        self.presence_penalty = config.api_settings.presence_penalty
        self.frequency_penalty = config.api_settings.frequency_penalty
    
    @abstractmethod
    def query(
        self, 
        prompt: str, 
        temperature: float,
        model_id: str
    ) -> str:
        """
        Query the model and return response text.
        
        Args:
            prompt: The prompt to send
            temperature: Temperature setting
            model_id: Model identifier for this provider
            
        Returns:
            Response text from the model
        """
        pass
    
    def query_with_retry(
        self,
        prompt: str,
        temperature: float,
        model_id: str
    ) -> str:
        """Query with automatic retries on failure."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.query(prompt, temperature, model_id)
            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"    [Retry {attempt + 1}/{self.max_retries}] Error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        raise last_error


# =============================================================================
# GOOGLE GEMINI CLIENT (using google-genai SDK)
# =============================================================================

class GeminiClient(BaseAPIClient):
    """Client for Google AI Studio (Gemini models) using google-genai SDK."""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose)
        
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
        
        self.api_key = api_key or get_api_key("google")
        if not self.api_key:
            raise ValueError("Google API key not configured")
        
        # Create the client with API key
        self.client = genai.Client(api_key=self.api_key)
        
        if self.verbose:
            print("[GeminiClient] Initialized with google-genai SDK")
    
    def query(
        self,
        prompt: str,
        temperature: float,
        model_id: str
    ) -> str:
        """Query Gemini model."""
        start_time = time.time()
        
        try:
            # Use the new google-genai SDK's generate_content method
            # Config is passed as a types.GenerateContentConfig or dict
            response = self.client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.top_p,
                    # Note: Gemini doesn't support presence/frequency penalties
                )
            )
            
            latency = (time.time() - start_time) * 1000
            
            if self.verbose:
                print(f"    [Gemini] {model_id} responded in {latency:.0f}ms")
            
            return response.text.strip()
            
        except Exception as e:
            if self.verbose:
                print(f"    [Gemini ERROR] {model_id}: {e}")
            raise


# =============================================================================
# HUGGINGFACE CLIENT (with rate limiting and 429 handling)
# =============================================================================

import random

class HuggingFaceClient(BaseAPIClient):
    """
    Client for HuggingFace Inference API with robust rate limiting.
    
    Features:
    - Proactive rate limiting (configurable delay between requests)
    - Exponential backoff with jitter on 429 errors
    - Automatic retry with increasing wait times
    """
    
    # Rate limiting settings for free tier
    MIN_REQUEST_INTERVAL = 1.0  # Minimum seconds between requests
    MAX_RETRIES_429 = 10  # Max retries specifically for rate limit errors
    BASE_BACKOFF = 5.0  # Base wait time for 429 errors (seconds)
    MAX_BACKOFF = 120.0  # Maximum wait time (2 minutes)
    
    # Class-level rate limiting (shared across all instances)
    _last_request_time = 0.0
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose)
        
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub package not installed. "
                "Run: pip install huggingface_hub"
            )
        
        self.api_key = api_key or get_api_key("huggingface")
        if not self.api_key:
            raise ValueError("HuggingFace token not configured")
        
        # Create inference client with Fireworks as the provider
        # This routes all requests through Fireworks using the Fireworks API key
        # configured in HuggingFace Settings -> Inference Providers
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=self.api_key
        )
        
        if self.verbose:
            print("[HuggingFaceClient] Initialized with Fireworks provider routing")
    
    def _wait_for_rate_limit(self):
        """Proactively wait to respect rate limits."""
        now = time.time()
        elapsed = now - HuggingFaceClient._last_request_time
        
        if elapsed < self.MIN_REQUEST_INTERVAL:
            wait_time = self.MIN_REQUEST_INTERVAL - elapsed
            if self.verbose:
                print(f"    [Rate Limit] Waiting {wait_time:.1f}s before next request...")
            time.sleep(wait_time)
        
        HuggingFaceClient._last_request_time = time.time()
    
    def _calculate_backoff(self, attempt: int, reset_time: Optional[float] = None) -> float:
        """Calculate wait time with exponential backoff and jitter."""
        if reset_time and reset_time > 0:
            # Use the reset time from headers if available
            wait = reset_time + random.uniform(1, 3)
        else:
            # Exponential backoff: 5, 10, 20, 40, 80, 120 (capped)
            wait = min(self.BASE_BACKOFF * (2 ** attempt), self.MAX_BACKOFF)
            # Add jitter (±20%)
            wait = wait * (1 + random.uniform(-0.2, 0.2))
        
        return wait
    
    def _extract_reset_time(self, error: Exception) -> Optional[float]:
        """Try to extract reset time from error response."""
        try:
            # HuggingFace errors sometimes contain headers or metadata
            error_str = str(error)
            if 'reset' in error_str.lower():
                # Try to parse reset time from error message
                import re
                match = re.search(r'reset[^\d]*(\d+)', error_str, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except:
            pass
        return None
    
    def query(
        self,
        prompt: str,
        temperature: float,
        model_id: str
    ) -> str:
        """Query HuggingFace model with rate limiting and 429 handling."""
        
        for attempt in range(self.MAX_RETRIES_429):
            # Proactive rate limiting
            self._wait_for_rate_limit()
            
            start_time = time.time()
            
            try:
                # Use chat completion for instruction-tuned models
                response = self.client.chat_completion(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature if temperature > 0 else 0.01,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                )
                
                latency = (time.time() - start_time) * 1000
                
                if self.verbose:
                    print(f"    [HuggingFace] {model_id} responded in {latency:.0f}ms")
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error (429)
                if '429' in str(e) or 'rate limit' in error_str or 'too many requests' in error_str:
                    reset_time = self._extract_reset_time(e)
                    wait_time = self._calculate_backoff(attempt, reset_time)
                    
                    if self.verbose:
                        print(f"    [429 Rate Limit] Attempt {attempt + 1}/{self.MAX_RETRIES_429}, waiting {wait_time:.1f}s...")
                    
                    time.sleep(wait_time)
                    continue  # Retry
                
                # Check if it's a model loading error (503)
                if '503' in str(e) or 'loading' in error_str:
                    wait_time = 20 + random.uniform(5, 15)
                    if self.verbose:
                        print(f"    [503 Model Loading] Waiting {wait_time:.1f}s for model to load...")
                    time.sleep(wait_time)
                    continue  # Retry
                
                # For other errors, try text_generation fallback
                if self.verbose:
                    print(f"    [HuggingFace] Chat failed ({e}), trying text_generation...")
                
                try:
                    self._wait_for_rate_limit()
                    
                    response = self.client.text_generation(
                        prompt,
                        model=model_id,
                        temperature=temperature if temperature > 0 else 0.01,
                        max_new_tokens=self.max_tokens,
                        top_p=self.top_p,
                        repetition_penalty=1.0,  # Equivalent to no frequency/presence penalty
                        return_full_text=False,
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    
                    if self.verbose:
                        print(f"    [HuggingFace] {model_id} responded in {latency:.0f}ms")
                    
                    return response.strip()
                    
                except Exception as e2:
                    error_str2 = str(e2).lower()
                    
                    # Rate limit on fallback too
                    if '429' in str(e2) or 'rate limit' in error_str2:
                        reset_time = self._extract_reset_time(e2)
                        wait_time = self._calculate_backoff(attempt, reset_time)
                        
                        if self.verbose:
                            print(f"    [429 Rate Limit] Attempt {attempt + 1}/{self.MAX_RETRIES_429}, waiting {wait_time:.1f}s...")
                        
                        time.sleep(wait_time)
                        continue  # Retry from the top
                    
                    if self.verbose:
                        print(f"    [HuggingFace ERROR] {model_id}: {e2}")
                    raise e2
        
        # If we exhausted all retries
        raise RuntimeError(f"Max retries ({self.MAX_RETRIES_429}) exceeded for HuggingFace API - persistent rate limiting")


# =============================================================================
# UNIFIED CLIENT
# =============================================================================

class UniversalClient:
    """
    Unified client that routes queries to the appropriate provider.
    
    Usage:
        client = UniversalClient(verbose=True)
        response = client.query("gemini-2.5-flash", prompt, temperature=0.5)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._clients: Dict[Provider, BaseAPIClient] = {}
        
        if self.verbose:
            print("[UniversalClient] Initializing...")
    
    def _get_client(self, provider: Provider) -> BaseAPIClient:
        """Get or create client for a provider."""
        if provider not in self._clients:
            if provider == Provider.GOOGLE:
                self._clients[provider] = GeminiClient(verbose=self.verbose)
            elif provider == Provider.HUGGINGFACE:
                self._clients[provider] = HuggingFaceClient(verbose=self.verbose)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        return self._clients[provider]
    
    def query(
        self,
        model_key: str,
        prompt: str,
        temperature: float = 0.0
    ) -> str:
        """
        Query a model by its registry key.
        
        Args:
            model_key: Key from MODEL_REGISTRY (e.g., "gemini-2.5-flash")
            prompt: The prompt to send
            temperature: Temperature setting
            
        Returns:
            Response text from the model
        """
        # Get model config from registry
        model_config = get_model(model_key)
        
        # Get appropriate client
        client = self._get_client(model_config.provider)
        
        if self.verbose:
            print(f"  [Query] {model_key} (T={temperature})")
        
        # Query with the model's API identifier
        # Uses api_model_id which adds :fireworks-ai suffix for HuggingFace models
        return client.query_with_retry(
            prompt=prompt,
            temperature=temperature,
            model_id=model_config.api_model_id
        )
    
    def create_query_function(
        self,
        model_key: str,
        prompt: str
    ):
        """
        Create a query function for use with stopping strategy.
        
        Args:
            model_key: Key from MODEL_REGISTRY
            prompt: The prompt to send
            
        Returns:
            Function that takes temperature and returns response
        """
        def query_fn(temperature: float) -> str:
            return self.query(model_key, prompt, temperature)
        
        return query_fn
    
    def test_connection(self, model_key: str) -> bool:
        """Test if a model is accessible."""
        try:
            response = self.query(
                model_key,
                "Say 'OK' if you can read this.",
                temperature=0.0
            )
            return len(response) > 0
        except Exception as e:
            if self.verbose:
                print(f"  [Connection Test FAILED] {model_key}: {e}")
            return False
    
    def test_all_providers(self) -> Dict[str, bool]:
        """Test connection to one model from each provider."""
        results = {}
        
        # Test models (one per provider)
        test_models = {
            "google": "gemini-2.5-flash",
            "huggingface": "llama-3.2-1b-instruct",  # Small model for fast test
        }
        
        for provider, model_key in test_models.items():
            if self.verbose:
                print(f"\n[Testing {provider}] {model_key}")
            results[provider] = self.test_connection(model_key)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def query_model(
    model_key: str,
    prompt: str,
    temperature: float = 0.0,
    verbose: bool = False
) -> str:
    """
    Convenience function to query a model.
    
    Args:
        model_key: Key from MODEL_REGISTRY
        prompt: The prompt to send
        temperature: Temperature setting
        verbose: Print progress
        
    Returns:
        Response text
    """
    client = UniversalClient(verbose=verbose)
    return client.query(model_key, prompt, temperature)


def get_client_for_model(model_key: str, verbose: bool = False) -> BaseAPIClient:
    """
    Get the appropriate API client for a model.
    
    Args:
        model_key: Key from MODEL_REGISTRY (e.g., "gemini-2.5-flash")
        verbose: Whether to print progress messages
        
    Returns:
        The appropriate API client instance (GeminiClient or HuggingFaceClient)
        
    Raises:
        ValueError: If model_key is not found in MODEL_REGISTRY
    """
    model_config = get_model(model_key)
    provider = model_config.provider
    
    if provider == Provider.GOOGLE:
        return GeminiClient(verbose=verbose)
    elif provider == Provider.HUGGINGFACE:
        return HuggingFaceClient(verbose=verbose)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def test_api_connections(verbose: bool = True) -> Dict[str, bool]:
    """Test connections to all API providers."""
    client = UniversalClient(verbose=verbose)
    return client.test_all_providers()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("API CLIENTS TEST")
    print("=" * 70)
    
    # Check which packages are available
    print("\n[Package Availability]")
    print(f"  google-genai: {'✓' if GENAI_AVAILABLE else '✗'}")
    print(f"  huggingface_hub: {'✓' if HF_AVAILABLE else '✗'}")
    
    # Check API key configuration
    print("\n[API Key Configuration]")
    print(f"  Google: {'✓' if get_api_key('google') else '✗'}")
    print(f"  HuggingFace: {'✓' if get_api_key('huggingface') else '✗'}")
    
    # Test connections
    print("\n[Connection Tests]")
    try:
        results = test_api_connections(verbose=True)
        print("\n[Results]")
        for provider, success in results.items():
            status = "✓ Connected" if success else "✗ Failed"
            print(f"  {provider}: {status}")
    except Exception as e:
        print(f"  Error during tests: {e}")
    
    # Test a simple query
    print("\n[Simple Query Test]")
    test_prompt = """Determine whether the following syllogism is valid or invalid.

Premise 1: All men are mortal
Premise 2: Socrates is a man
Conclusion: Socrates is mortal

Is this syllogism valid or invalid? Respond with exactly one word: "valid" or "invalid"."""
    
    try:
        # Try Gemini first (it's configured)
        print("\nQuerying gemini-2.5-flash...")
        response = query_model("gemini-2.5-flash", test_prompt, temperature=0.0, verbose=True)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Gemini query failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
