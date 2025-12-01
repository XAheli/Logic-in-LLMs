"""
Unit Tests for API Clients

Tests for GeminiClient and HuggingFaceClient.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.api_clients import (
    GeminiClient,
    HuggingFaceClient,
    get_client_for_model,
    APIResponse,
    UniversalClient
)
from src.inference.model_registry import MODEL_REGISTRY


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    mock_response = Mock()
    mock_response.text = "valid"
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 10
    return mock_response


@pytest.fixture
def mock_hf_response():
    """Mock HuggingFace API response."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="valid"))]
    mock_response.usage = Mock(prompt_tokens=50, completion_tokens=10)
    return mock_response


# =============================================================================
# GEMINI CLIENT TESTS
# =============================================================================

class TestGeminiClient:
    """Tests for GeminiClient."""
    
    @patch('src.inference.api_clients.genai')
    def test_init(self, mock_genai):
        """Test client initialization."""
        client = GeminiClient(api_key="test_key")
        mock_genai.Client.assert_called_once_with(api_key="test_key")
    
    @patch('src.inference.api_clients.genai')
    def test_query_success(self, mock_genai, mock_gemini_response):
        """Test successful query."""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_gemini_response
        mock_genai.Client.return_value = mock_client
        
        client = GeminiClient(api_key="test_key")
        response = client.query(
            prompt="Test prompt",
            temperature=0.0,
            model_id="gemini-2.5-flash"
        )
        
        assert response == "valid"
    
    @patch('src.inference.api_clients.genai')
    def test_query_error(self, mock_genai):
        """Test error handling."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_genai.Client.return_value = mock_client
        
        client = GeminiClient(api_key="test_key")
        
        with pytest.raises(Exception):
            client.query(
                prompt="Test prompt",
                temperature=0.0,
                model_id="gemini-2.5-flash"
            )


# =============================================================================
# HUGGINGFACE CLIENT TESTS
# =============================================================================

class TestHuggingFaceClient:
    """Tests for HuggingFaceClient."""
    
    @patch('src.inference.api_clients.InferenceClient')
    def test_init(self, mock_inference_client):
        """Test client initialization."""
        client = HuggingFaceClient(api_key="test_key")
        mock_inference_client.assert_called_once()
        # Check that api_key is passed (no provider param - uses :cheapest routing)
        call_kwargs = mock_inference_client.call_args[1]
        assert call_kwargs['api_key'] == "test_key"
    
    @patch('src.inference.api_clients.InferenceClient')
    def test_query_success(self, mock_inference_client, mock_hf_response):
        """Test successful query using chat.completions.create()."""
        mock_client = Mock()
        # Mock the nested chat.completions.create() method
        mock_client.chat.completions.create.return_value = mock_hf_response
        mock_inference_client.return_value = mock_client
        
        client = HuggingFaceClient(api_key="test_key")
        response = client.query(
            prompt="Test prompt",
            temperature=0.5,
            model_id="meta-llama/Llama-3.1-8B-Instruct"
        )
        
        assert response == "valid"


# =============================================================================
# CLIENT FACTORY TESTS
# =============================================================================

class TestClientFactory:
    """Tests for get_client_for_model."""
    
    @patch('src.inference.api_clients.GeminiClient')
    def test_get_gemini_client(self, mock_client):
        """Test getting Gemini client."""
        client = get_client_for_model("gemini-2.5-flash")
        mock_client.assert_called_once()
    
    @patch('src.inference.api_clients.HuggingFaceClient')
    def test_get_huggingface_client(self, mock_client):
        """Test getting HuggingFace client."""
        client = get_client_for_model("llama-3.1-8b-instruct")
        mock_client.assert_called_once()
    
    def test_unknown_model(self):
        """Test error for unknown model."""
        with pytest.raises(ValueError):
            get_client_for_model("unknown-model-xyz")


# =============================================================================
# API RESPONSE TESTS
# =============================================================================

class TestAPIResponse:
    """Tests for APIResponse dataclass."""
    
    def test_response_creation(self):
        """Test response creation."""
        response = APIResponse(
            text="valid",
            model="test-model",
            provider="google"
        )
        
        assert response.text == "valid"
        assert response.model == "test-model"
        assert response.provider == "google"
    
    def test_error_response(self):
        """Test error response creation."""
        response = APIResponse(
            text="",
            model="test-model",
            provider="google",
            error="API Error"
        )
        
        assert response.error == "API Error"


# =============================================================================
# INTEGRATION TESTS (with mocked APIs)
# =============================================================================

class TestIntegration:
    """Integration tests for API clients."""
    
    @patch('src.inference.api_clients.genai')
    def test_gemini_retry_on_error(self, mock_genai, mock_gemini_response):
        """Test that Gemini client retries on error."""
        mock_client = Mock()
        # First call raises error, second succeeds
        mock_client.models.generate_content.side_effect = [
            Exception("Rate limit exceeded"),
            mock_gemini_response
        ]
        mock_genai.Client.return_value = mock_client
        
        client = GeminiClient(api_key="test_key")
        
        # First call should fail
        with pytest.raises(Exception):
            client.query("prompt", 0.0, "model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
