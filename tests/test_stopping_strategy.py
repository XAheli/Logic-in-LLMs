"""
Unit Tests for Stopping Strategy

Tests for adaptive stopping mechanism used in temperature > 0 experiments.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.stopping_strategy import (
    VoteResult,
    IterationResult,
    StoppingResult,
    AdaptiveStoppingStrategy,
    parse_response,
    get_adaptive_vote,
    run_with_stopping
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def global_strategy():
    """Create a stopping strategy with global limits."""
    return AdaptiveStoppingStrategy(
        max_iterations=10,
        threshold=5
    )


@pytest.fixture
def custom_strategy():
    """Create a stopping strategy with custom limits."""
    return AdaptiveStoppingStrategy(
        max_iterations=15,
        threshold=8
    )


# =============================================================================
# PARSE RESPONSE TESTS
# =============================================================================

class TestParseResponse:
    """Tests for parse_response function."""
    
    def test_parse_valid(self):
        """Test parsing 'valid' responses."""
        assert parse_response("valid") == VoteResult.VALID
        assert parse_response("Valid") == VoteResult.VALID
        assert parse_response("VALID") == VoteResult.VALID
        assert parse_response("The answer is valid.") == VoteResult.VALID
    
    def test_parse_invalid(self):
        """Test parsing 'invalid' responses."""
        assert parse_response("invalid") == VoteResult.INVALID
        assert parse_response("Invalid") == VoteResult.INVALID
        assert parse_response("The argument is invalid.") == VoteResult.INVALID
    
    def test_parse_invalid_takes_precedence(self):
        """Test that 'invalid' is detected even when 'valid' is substring."""
        # "invalid" contains "valid" as substring
        assert parse_response("invalid") == VoteResult.INVALID
    
    def test_parse_empty(self):
        """Test parsing empty responses."""
        assert parse_response("") == VoteResult.ERROR
        assert parse_response("   ") == VoteResult.ERROR
    
    def test_parse_unclear(self):
        """Test parsing unclear responses."""
        assert parse_response("I'm not sure") == VoteResult.UNCLEAR
        assert parse_response("Maybe") == VoteResult.UNCLEAR


# =============================================================================
# ADAPTIVE STOPPING STRATEGY TESTS
# =============================================================================

class TestAdaptiveStoppingStrategy:
    """Tests for AdaptiveStoppingStrategy."""
    
    def test_init_global(self, global_strategy):
        """Test initialization with global limits."""
        assert global_strategy.max_iterations == 10
        assert global_strategy.threshold == 5
    
    def test_init_custom(self, custom_strategy):
        """Test initialization with custom limits."""
        assert custom_strategy.max_iterations == 15
        assert custom_strategy.threshold == 8
    
    def test_with_global_limits(self):
        """Test factory method for global limits."""
        strategy = AdaptiveStoppingStrategy.with_global_limits()
        assert strategy.max_iterations == 10
        assert strategy.threshold == 5
    
    def test_from_model_config(self):
        """Test factory method from_model_config uses global limits."""
        strategy = AdaptiveStoppingStrategy.from_model_config()
        assert strategy.max_iterations == 10
        assert strategy.threshold == 5


# =============================================================================
# DETERMINISTIC (T=0) TESTS
# =============================================================================

class TestDeterministicBehavior:
    """Tests for deterministic (temperature=0) behavior."""
    
    def test_single_query_at_temp_zero(self, global_strategy):
        """Test that only one query is made at T=0."""
        call_count = 0
        
        def mock_query(temp):
            nonlocal call_count
            call_count += 1
            return "valid"
        
        result = global_strategy.run(mock_query, temperature=0.0)
        
        assert call_count == 1
        assert result.final_answer == "valid"
        assert result.total_iterations == 1
        assert result.confidence == 1.0
    
    def test_deterministic_invalid(self, global_strategy):
        """Test deterministic response for invalid."""
        def mock_query(temp):
            return "The argument is invalid."
        
        result = global_strategy.run(mock_query, temperature=0.0)
        
        assert result.final_answer == "invalid"
        assert result.total_iterations == 1


# =============================================================================
# ADAPTIVE STOPPING TESTS (T > 0)
# =============================================================================

class TestAdaptiveStopping:
    """Tests for adaptive stopping at temperature > 0."""
    
    def test_early_stop_on_threshold_valid(self, global_strategy):
        """Test early stopping when first 5 iterations are ALL valid."""
        call_count = 0
        
        def mock_query(temp):
            nonlocal call_count
            call_count += 1
            return "valid"  # Always return valid
        
        result = global_strategy.run(mock_query, temperature=0.5)
        
        # Should stop after 5 queries (first 5 all valid)
        assert call_count == 5
        assert result.final_answer == "valid"
        assert result.stopped_early is True
        assert result.valid_count == 5
    
    def test_early_stop_on_threshold_invalid(self, global_strategy):
        """Test early stopping when first 5 iterations are ALL invalid."""
        call_count = 0
        
        def mock_query(temp):
            nonlocal call_count
            call_count += 1
            return "invalid"
        
        result = global_strategy.run(mock_query, temperature=0.5)
        
        # Should stop after 5 queries (first 5 all invalid)
        assert call_count == 5
        assert result.final_answer == "invalid"
        assert result.stopped_early is True
        assert result.invalid_count == 5
    
    def test_max_iterations_reached(self, global_strategy):
        """Test that mixed first 5 iterations continues to max iterations."""
        call_count = 0
        
        def mock_query(temp):
            nonlocal call_count
            call_count += 1
            # First 5: V,I,V,I,V (mixed) -> continues
            # Rest: V,V,V,V,V
            if call_count <= 5:
                return "valid" if call_count % 2 == 1 else "invalid"
            else:
                return "valid"
        
        result = global_strategy.run(mock_query, temperature=0.5)
        
        # Should run all 10 iterations (first 5 mixed)
        assert call_count == 10
        assert result.total_iterations == 10
        assert result.stopped_early is False
        # 3 valid in first 5 + 5 valid after = 8 valid, 2 invalid
        assert result.valid_count == 8
        assert result.invalid_count == 2
    
    def test_majority_vote(self, global_strategy):
        """Test majority vote when max iterations reached."""
        call_count = 0
        
        def mock_query(temp):
            nonlocal call_count
            call_count += 1
            # First 5: V,I,V,I,V (mixed) -> continues
            # After 5: I,V,I,V,I -> 6 valid total, 4 invalid
            # Wait, let's do: first 5 mixed, then more valid
            # V,I,V,I,V (3V,2I), then V,V,V,V,V = 8V, 2I
            if call_count <= 5:
                return "valid" if call_count % 2 == 1 else "invalid"
            else:
                return "valid"
        
        result = global_strategy.run(mock_query, temperature=0.5)
        
        # Should run all 10 iterations
        assert result.total_iterations == 10
        # Majority is valid (8 > 2)
        assert result.final_answer == "valid"
        assert result.valid_count == 8
        assert result.invalid_count == 2
    
    def test_tie_defaults_to_invalid(self, global_strategy):
        """Test that tie (equal valid/invalid) defaults to invalid (conservative)."""
        call_count = 0
        
        def mock_query(temp):
            nonlocal call_count
            call_count += 1
            # First 5: V,I,V,I,V (mixed) -> continues
            # After 5: I,I,I,I,I -> 3V, 7I? No...
            # Let's do: V,I,V,I,V,I,V,I,V,I = 5V, 5I (tie)
            return "valid" if call_count % 2 == 1 else "invalid"
        
        result = global_strategy.run(mock_query, temperature=0.5)
        
        # Should run all 10 iterations
        assert result.total_iterations == 10
        # Tie (5 == 5) -> defaults to invalid
        assert result.final_answer == "invalid"
        assert result.valid_count == 5
        assert result.invalid_count == 5
        assert result.confidence == 0.5


# =============================================================================
# GLOBAL LIMITS TESTS
# =============================================================================

class TestGlobalLimits:
    """Tests verifying global limits behavior (all models use same limits)."""
    
    def test_same_limits_for_all(self):
        """Test that all strategies created use the same global limits."""
        strategy1 = AdaptiveStoppingStrategy.with_global_limits()
        strategy2 = AdaptiveStoppingStrategy.from_model_config()
        
        assert strategy1.max_iterations == strategy2.max_iterations
        assert strategy1.threshold == strategy2.threshold
    
    def test_global_limits_values(self):
        """Test that global limits are max=10, threshold=5."""
        strategy = AdaptiveStoppingStrategy.with_global_limits()
        
        assert strategy.max_iterations == 10
        assert strategy.threshold == 5


# =============================================================================
# STOPPING RESULT TESTS
# =============================================================================

class TestStoppingResult:
    """Tests for StoppingResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = StoppingResult(
            final_answer="valid",
            confidence=0.8,
            total_iterations=5,
            valid_count=4,
            invalid_count=1,
            error_count=0,
            stopped_early=True,
            all_responses=[]
        )
        
        d = result.to_dict()
        assert d['final_answer'] == "valid"
        assert d['confidence'] == 0.8
        assert d['total_iterations'] == 5
        assert d['stopped_early'] is True
    
    def test_unclear_count(self):
        """Test unclear_count property."""
        result = StoppingResult(
            final_answer="valid",
            confidence=0.5,
            total_iterations=10,
            valid_count=4,
            invalid_count=3,
            error_count=1,
            stopped_early=False
        )
        
        # unclear = total - valid - invalid - error = 10 - 4 - 3 - 1 = 2
        assert result.unclear_count == 2


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_adaptive_vote(self):
        """Test get_adaptive_vote function."""
        def mock_query(temp):
            return "valid"
        
        answer, confidence = get_adaptive_vote(mock_query, temperature=0.0)
        
        assert answer == "valid"
        assert confidence == 1.0
    
    def test_run_with_stopping(self):
        """Test run_with_stopping function."""
        def mock_query(temp):
            return "invalid"
        
        result = run_with_stopping(mock_query, temperature=0.0)
        
        assert result.final_answer == "invalid"
        assert result.total_iterations == 1


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_error_at_temp_zero(self, global_strategy):
        """Test when query returns error at T=0."""
        def mock_query(temp):
            raise Exception("API Error")
        
        result = global_strategy.run(mock_query, temperature=0.0)
        
        assert result.final_answer == "error"
        assert result.error_count == 1
    
    def test_unclear_at_temp_zero(self, global_strategy):
        """Test when query returns unclear at T=0."""
        def mock_query(temp):
            return "I don't know"
        
        result = global_strategy.run(mock_query, temperature=0.0)
        
        # Single query at T=0, unclear response
        assert result.final_answer == "unclear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
