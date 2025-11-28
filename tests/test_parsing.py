"""
Unit Tests for Response Parsing

Tests for extracting valid/invalid predictions from raw LLM responses.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


# =============================================================================
# BASIC PARSING TESTS
# =============================================================================

class TestParseResponse:
    """Tests for parse_response function."""
    
    def test_simple_valid(self):
        """Test parsing 'valid' from simple response."""
        result = parse_response("valid")
        assert result.answer == ParsedAnswer.VALID
        
        result = parse_response("Valid")
        assert result.answer == ParsedAnswer.VALID
        
        result = parse_response("VALID")
        assert result.answer == ParsedAnswer.VALID
    
    def test_simple_invalid(self):
        """Test parsing 'invalid' from simple response."""
        result = parse_response("invalid")
        assert result.answer == ParsedAnswer.INVALID
        
        result = parse_response("Invalid")
        assert result.answer == ParsedAnswer.INVALID
        
        result = parse_response("INVALID")
        assert result.answer == ParsedAnswer.INVALID
    
    def test_answer_with_context(self):
        """Test extracting answer from longer response."""
        result = parse_response("After analyzing the syllogism, I conclude that it is valid.")
        assert result.answer == ParsedAnswer.VALID
        
        result = parse_response("The argument is invalid because the conclusion doesn't follow.")
        assert result.answer == ParsedAnswer.INVALID
    
    def test_invalid_takes_precedence(self):
        """Test that 'invalid' is detected even when 'valid' is substring."""
        result = parse_response("invalid")
        assert result.answer == ParsedAnswer.INVALID
        
        result = parse_response("This argument is invalid")
        assert result.answer == ParsedAnswer.INVALID
    
    def test_empty_response(self):
        """Test parsing empty response."""
        result = parse_response("")
        assert result.answer == ParsedAnswer.ERROR
        assert result.confidence == 0.0
    
    def test_none_response(self):
        """Test parsing None response."""
        result = parse_response(None)
        assert result.answer == ParsedAnswer.ERROR
    
    def test_unclear_response(self):
        """Test parsing unclear response."""
        result = parse_response("I'm not sure about this")
        assert result.answer == ParsedAnswer.UNCLEAR
        
        result = parse_response("Maybe")
        assert result.answer == ParsedAnswer.UNCLEAR


# =============================================================================
# CHAIN-OF-THOUGHT PARSING TESTS
# =============================================================================

class TestParseCotResponse:
    """Tests for parse_cot_response function."""
    
    def test_cot_valid(self):
        """Test parsing CoT response ending in valid."""
        cot_response = """
        Let's think step by step.
        
        Premise 1: All A are B.
        Premise 2: All B are C.
        
        From these premises, we can conclude that all A are C.
        The conclusion follows logically from the premises.
        
        Therefore, the argument is valid.
        """
        result = parse_cot_response(cot_response)
        assert result.answer == ParsedAnswer.VALID
        assert result.reasoning is not None
    
    def test_cot_invalid(self):
        """Test parsing CoT response ending in invalid."""
        cot_response = """
        Let's analyze this carefully.
        
        The first premise states that some A are B.
        The second premise states that some B are C.
        
        However, this doesn't guarantee that any A are C.
        
        The argument is invalid.
        """
        result = parse_cot_response(cot_response)
        assert result.answer == ParsedAnswer.INVALID
        assert result.reasoning is not None
    
    def test_cot_with_final_answer(self):
        """Test CoT with explicit final answer."""
        cot_response = """
        Let me work through this...
        
        [analysis here]
        
        Final answer: valid
        """
        result = parse_cot_response(cot_response)
        assert result.answer == ParsedAnswer.VALID


# =============================================================================
# EXTRACT ANSWER STRING TESTS
# =============================================================================

class TestExtractAnswerString:
    """Tests for extract_answer_string function."""
    
    def test_extract_valid(self):
        """Test extracting 'valid' string."""
        assert extract_answer_string("valid") == "valid"
        assert extract_answer_string("The answer is valid") == "valid"
    
    def test_extract_invalid(self):
        """Test extracting 'invalid' string."""
        assert extract_answer_string("invalid") == "invalid"
        assert extract_answer_string("The answer is invalid") == "invalid"
    
    def test_extract_unclear(self):
        """Test extracting 'unclear' for ambiguous responses."""
        assert extract_answer_string("I don't know") == "unclear"


# =============================================================================
# PARSE RESULT TESTS
# =============================================================================

class TestParseResult:
    """Tests for ParseResult dataclass."""
    
    def test_is_valid_property(self):
        """Test is_valid property."""
        result = ParseResult(
            answer=ParsedAnswer.VALID,
            raw_response="valid"
        )
        assert result.is_valid is True
        assert result.is_invalid is False
    
    def test_is_invalid_property(self):
        """Test is_invalid property."""
        result = ParseResult(
            answer=ParsedAnswer.INVALID,
            raw_response="invalid"
        )
        assert result.is_valid is False
        assert result.is_invalid is True
    
    def test_is_clear_property(self):
        """Test is_clear property."""
        valid_result = ParseResult(answer=ParsedAnswer.VALID, raw_response="valid")
        invalid_result = ParseResult(answer=ParsedAnswer.INVALID, raw_response="invalid")
        unclear_result = ParseResult(answer=ParsedAnswer.UNCLEAR, raw_response="maybe")
        
        assert valid_result.is_clear is True
        assert invalid_result.is_clear is True
        assert unclear_result.is_clear is False


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidateAgainstGroundTruth:
    """Tests for validate_against_ground_truth function."""
    
    def test_correct_valid(self):
        """Test validation when correctly predicting valid."""
        parsed = parse_response("valid")
        result = validate_against_ground_truth(parsed, "valid")
        
        assert result["is_correct"] is True
        assert result["predicted"] == "valid"
        assert result["ground_truth"] == "valid"
    
    def test_correct_invalid(self):
        """Test validation when correctly predicting invalid."""
        parsed = parse_response("invalid")
        result = validate_against_ground_truth(parsed, "invalid")
        
        assert result["is_correct"] is True
    
    def test_incorrect_prediction(self):
        """Test validation when prediction is wrong."""
        parsed = parse_response("valid")
        result = validate_against_ground_truth(parsed, "invalid")
        
        assert result["is_correct"] is False


# =============================================================================
# BATCH PARSING TESTS
# =============================================================================

class TestParseBatchResponses:
    """Tests for parse_batch_responses function."""
    
    def test_batch_parsing(self):
        """Test batch parsing."""
        responses = [
            "valid",
            "invalid",
            "The answer is valid",
            "I don't know"
        ]
        
        results = parse_batch_responses(responses)
        
        assert len(results) == 4
        assert results[0].answer == ParsedAnswer.VALID
        assert results[1].answer == ParsedAnswer.INVALID
        assert results[2].answer == ParsedAnswer.VALID
        assert results[3].answer == ParsedAnswer.UNCLEAR
    
    def test_empty_batch(self):
        """Test empty batch."""
        results = parse_batch_responses([])
        assert len(results) == 0


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestCalculateParsingStats:
    """Tests for calculate_parsing_stats function."""
    
    def test_stats_calculation(self):
        """Test statistics calculation."""
        results = [
            ParseResult(answer=ParsedAnswer.VALID, raw_response="valid"),
            ParseResult(answer=ParsedAnswer.VALID, raw_response="valid"),
            ParseResult(answer=ParsedAnswer.INVALID, raw_response="invalid"),
            ParseResult(answer=ParsedAnswer.UNCLEAR, raw_response="maybe"),
        ]
        
        stats = calculate_parsing_stats(results)
        
        assert stats["total"] == 4
        assert stats["valid"] == 2
        assert stats["invalid"] == 1
        assert stats["unclear"] == 1
        assert stats["clear_rate"] == 0.75  # 3/4
    
    def test_empty_stats(self):
        """Test stats for empty results."""
        stats = calculate_parsing_stats([])
        assert stats["total"] == 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_whitespace_response(self):
        """Test response with only whitespace."""
        result = parse_response("   ")
        assert result.answer == ParsedAnswer.ERROR
    
    def test_multiline_response(self):
        """Test multiline response."""
        text = """
        First, let me analyze the premises.
        
        After careful consideration...
        
        The syllogism is valid.
        """
        result = parse_response(text)
        assert result.answer == ParsedAnswer.VALID
    
    def test_mixed_case(self):
        """Test mixed case responses."""
        assert parse_response("VaLiD").answer == ParsedAnswer.VALID
        assert parse_response("InVaLiD").answer == ParsedAnswer.INVALID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
