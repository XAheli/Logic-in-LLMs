"""
Unit Tests for Response Parsing

Tests for extracting correct/incorrect predictions from raw LLM responses.
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
    
    def test_simple_correct(self):
        """Test parsing 'correct' from simple response."""
        result = parse_response("correct")
        assert result.answer == ParsedAnswer.CORRECT
        
        result = parse_response("Correct")
        assert result.answer == ParsedAnswer.CORRECT
        
        result = parse_response("CORRECT")
        assert result.answer == ParsedAnswer.CORRECT
    
    def test_simple_incorrect(self):
        """Test parsing 'incorrect' from simple response."""
        result = parse_response("incorrect")
        assert result.answer == ParsedAnswer.INCORRECT
        
        result = parse_response("Incorrect")
        assert result.answer == ParsedAnswer.INCORRECT
        
        result = parse_response("INCORRECT")
        assert result.answer == ParsedAnswer.INCORRECT
    
    def test_answer_with_context(self):
        """Test extracting answer from longer response."""
        result = parse_response("After analyzing the syllogism, I conclude that it is correct.")
        assert result.answer == ParsedAnswer.CORRECT
        
        result = parse_response("The argument is incorrect because the conclusion doesn't follow.")
        assert result.answer == ParsedAnswer.INCORRECT
    
    def test_incorrect_takes_precedence(self):
        """Test that 'incorrect' is detected even when 'correct' is substring."""
        result = parse_response("incorrect")
        assert result.answer == ParsedAnswer.INCORRECT
        
        result = parse_response("This argument is incorrect")
        assert result.answer == ParsedAnswer.INCORRECT
    
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
    
    def test_cot_correct(self):
        """Test parsing CoT response ending in correct."""
        cot_response = """
        Let's think step by step.
        
        Premise 1: All A are B.
        Premise 2: All B are C.
        
        From these premises, we can conclude that all A are C.
        The conclusion follows logically from the premises.
        
        Therefore, the argument is correct.
        """
        result = parse_cot_response(cot_response)
        assert result.answer == ParsedAnswer.CORRECT
        assert result.reasoning is not None
    
    def test_cot_incorrect(self):
        """Test parsing CoT response ending in incorrect."""
        cot_response = """
        Let's analyze this carefully.
        
        The first premise states that some A are B.
        The second premise states that some B are C.
        
        However, this doesn't guarantee that any A are C.
        
        The argument is incorrect.
        """
        result = parse_cot_response(cot_response)
        assert result.answer == ParsedAnswer.INCORRECT
        assert result.reasoning is not None
    
    def test_cot_with_final_answer(self):
        """Test CoT with explicit final answer."""
        cot_response = """
        Let me work through this...
        
        [analysis here]
        
        Final answer: correct
        """
        result = parse_cot_response(cot_response)
        assert result.answer == ParsedAnswer.CORRECT


# =============================================================================
# EXTRACT ANSWER STRING TESTS
# =============================================================================

class TestExtractAnswerString:
    """Tests for extract_answer_string function."""
    
    def test_extract_correct(self):
        """Test extracting 'correct' string."""
        assert extract_answer_string("correct") == "correct"
        assert extract_answer_string("The answer is correct") == "correct"
    
    def test_extract_incorrect(self):
        """Test extracting 'incorrect' string."""
        assert extract_answer_string("incorrect") == "incorrect"
        assert extract_answer_string("The answer is incorrect") == "incorrect"
    
    def test_extract_unclear(self):
        """Test extracting 'unclear' for ambiguous responses."""
        assert extract_answer_string("I don't know") == "unclear"


# =============================================================================
# PARSE RESULT TESTS
# =============================================================================

class TestParseResult:
    """Tests for ParseResult dataclass."""
    
    def test_is_correct_property(self):
        """Test is_correct property."""
        result = ParseResult(
            answer=ParsedAnswer.CORRECT,
            raw_response="correct"
        )
        assert result.is_correct is True
        assert result.is_incorrect is False
    
    def test_is_incorrect_property(self):
        """Test is_incorrect property."""
        result = ParseResult(
            answer=ParsedAnswer.INCORRECT,
            raw_response="incorrect"
        )
        assert result.is_correct is False
        assert result.is_incorrect is True
    
    def test_is_clear_property(self):
        """Test is_clear property."""
        correct_result = ParseResult(answer=ParsedAnswer.CORRECT, raw_response="correct")
        incorrect_result = ParseResult(answer=ParsedAnswer.INCORRECT, raw_response="incorrect")
        unclear_result = ParseResult(answer=ParsedAnswer.UNCLEAR, raw_response="maybe")
        
        assert correct_result.is_clear is True
        assert incorrect_result.is_clear is True
        assert unclear_result.is_clear is False


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidateAgainstGroundTruth:
    """Tests for validate_against_ground_truth function."""
    
    def test_correct_maps_to_valid(self):
        """Test validation: LLM says 'correct' and ground truth is 'valid' (syntax)."""
        parsed = parse_response("correct")
        result = validate_against_ground_truth(parsed, "valid", "syntax")
        
        assert result["is_correct"] is True
        assert result["predicted"] == "correct"
        assert result["mapped_prediction"] == "valid"
        assert result["ground_truth"] == "valid"
    
    def test_incorrect_maps_to_invalid(self):
        """Test validation: LLM says 'incorrect' and ground truth is 'invalid' (syntax)."""
        parsed = parse_response("incorrect")
        result = validate_against_ground_truth(parsed, "invalid", "syntax")
        
        assert result["is_correct"] is True
        assert result["mapped_prediction"] == "invalid"
    
    def test_correct_maps_to_believable(self):
        """Test validation: LLM says 'correct' and ground truth is 'believable' (NLU)."""
        parsed = parse_response("correct")
        result = validate_against_ground_truth(parsed, "believable", "NLU")
        
        assert result["is_correct"] is True
        assert result["mapped_prediction"] == "believable"
    
    def test_incorrect_maps_to_unbelievable(self):
        """Test validation: LLM says 'incorrect' and ground truth is 'unbelievable' (NLU)."""
        parsed = parse_response("incorrect")
        result = validate_against_ground_truth(parsed, "unbelievable", "NLU")
        
        assert result["is_correct"] is True
        assert result["mapped_prediction"] == "unbelievable"
    
    def test_wrong_prediction(self):
        """Test validation when prediction is wrong."""
        parsed = parse_response("correct")
        # LLM says "correct" (maps to "valid"), but ground truth is "invalid"
        result = validate_against_ground_truth(parsed, "invalid", "syntax")
        
        assert result["is_correct"] is False


# =============================================================================
# BATCH PARSING TESTS
# =============================================================================

class TestParseBatchResponses:
    """Tests for parse_batch_responses function."""
    
    def test_batch_parsing(self):
        """Test batch parsing."""
        responses = [
            "correct",
            "incorrect",
            "The answer is correct",
            "I don't know"
        ]
        
        results = parse_batch_responses(responses)
        
        assert len(results) == 4
        assert results[0].answer == ParsedAnswer.CORRECT
        assert results[1].answer == ParsedAnswer.INCORRECT
        assert results[2].answer == ParsedAnswer.CORRECT
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
            ParseResult(answer=ParsedAnswer.CORRECT, raw_response="correct"),
            ParseResult(answer=ParsedAnswer.CORRECT, raw_response="correct"),
            ParseResult(answer=ParsedAnswer.INCORRECT, raw_response="incorrect"),
            ParseResult(answer=ParsedAnswer.UNCLEAR, raw_response="maybe"),
        ]
        
        stats = calculate_parsing_stats(results)
        
        assert stats["total"] == 4
        assert stats["correct"] == 2
        assert stats["incorrect"] == 1
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
        
        The syllogism is correct.
        """
        result = parse_response(text)
        assert result.answer == ParsedAnswer.CORRECT
    
    def test_mixed_case(self):
        """Test mixed case responses."""
        assert parse_response("CoRrEcT").answer == ParsedAnswer.CORRECT
        assert parse_response("InCoRrEcT").answer == ParsedAnswer.INCORRECT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
