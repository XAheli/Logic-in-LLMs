"""
Response Parsing for Syllogistic Reasoning Benchmark

Parses raw model responses to extract:
- Valid/Invalid classification
- Confidence scores
- Reasoning (for CoT prompts)
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


# =============================================================================
# PARSING CLASSES
# =============================================================================

class ParsedAnswer(Enum):
    """Possible parsed answers."""
    VALID = "valid"
    INVALID = "invalid"
    UNCLEAR = "unclear"
    ERROR = "error"


@dataclass
class ParseResult:
    """Result of parsing a model response."""
    answer: ParsedAnswer
    raw_response: str
    confidence: float = 1.0
    reasoning: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        return self.answer == ParsedAnswer.VALID
    
    @property
    def is_invalid(self) -> bool:
        return self.answer == ParsedAnswer.INVALID
    
    @property
    def is_clear(self) -> bool:
        return self.answer in (ParsedAnswer.VALID, ParsedAnswer.INVALID)


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_response(response: str) -> ParseResult:
    """
    Parse a model response to extract the answer.
    
    Parsing logic (order matters):
    1. If response is empty/None → ERROR
    2. If "invalid" appears → INVALID
    3. If "valid" appears → VALID
    4. Otherwise → UNCLEAR
    
    Args:
        response: Raw model response string
        
    Returns:
        ParseResult with answer and metadata
    """
    if not response or not response.strip():
        return ParseResult(
            answer=ParsedAnswer.ERROR,
            raw_response=response or "",
            confidence=0.0
        )
    
    clean = response.lower().strip()
    
    # Check for invalid first (since "invalid" contains "valid")
    if "invalid" in clean:
        return ParseResult(
            answer=ParsedAnswer.INVALID,
            raw_response=response,
            confidence=1.0
        )
    elif "valid" in clean:
        return ParseResult(
            answer=ParsedAnswer.VALID,
            raw_response=response,
            confidence=1.0
        )
    else:
        return ParseResult(
            answer=ParsedAnswer.UNCLEAR,
            raw_response=response,
            confidence=0.0
        )


def parse_cot_response(response: str) -> ParseResult:
    """
    Parse a chain-of-thought response.
    
    Extracts:
    - The final answer (valid/invalid)
    - The reasoning leading to the answer
    
    Args:
        response: Raw CoT response
        
    Returns:
        ParseResult with answer and reasoning
    """
    if not response or not response.strip():
        return ParseResult(
            answer=ParsedAnswer.ERROR,
            raw_response=response or "",
            confidence=0.0
        )
    
    # Try to find explicit final answer patterns
    final_patterns = [
        r"(?:final\s+)?answer[:\s]+(\w+)",
        r"(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:syllogism\s+)?(?:is\s+)?(\w+)",
        r"(?:conclusion|verdict)[:\s]+(\w+)",
        r"(\w+)\s*$",  # Last word
    ]
    
    clean = response.lower().strip()
    reasoning = response  # Full response is the reasoning for CoT
    
    # Try each pattern
    for pattern in final_patterns:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            word = match.group(1).lower()
            if "invalid" in word:
                return ParseResult(
                    answer=ParsedAnswer.INVALID,
                    raw_response=response,
                    confidence=1.0,
                    reasoning=reasoning
                )
            elif "valid" in word:
                return ParseResult(
                    answer=ParsedAnswer.VALID,
                    raw_response=response,
                    confidence=1.0,
                    reasoning=reasoning
                )
    
    # Fallback to simple parsing
    result = parse_response(response)
    result.reasoning = reasoning
    return result


def parse_batch_responses(responses: List[str]) -> List[ParseResult]:
    """Parse a batch of responses."""
    return [parse_response(r) for r in responses]


def extract_answer_string(response: str) -> str:
    """
    Extract just the answer string from a response.
    
    Returns "valid", "invalid", or "unclear".
    """
    result = parse_response(response)
    return result.answer.value


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_against_ground_truth(
    parsed: ParseResult,
    ground_truth: str
) -> Dict:
    """
    Validate a parsed response against ground truth.
    
    Args:
        parsed: Parsed response
        ground_truth: Expected answer ("valid" or "invalid")
        
    Returns:
        Dict with validation results
    """
    is_correct = parsed.answer.value == ground_truth.lower()
    
    return {
        "predicted": parsed.answer.value,
        "ground_truth": ground_truth.lower(),
        "is_correct": is_correct,
        "is_clear": parsed.is_clear,
        "confidence": parsed.confidence
    }


def calculate_parsing_stats(results: List[ParseResult]) -> Dict:
    """
    Calculate statistics about parsing results.
    
    Args:
        results: List of parse results
        
    Returns:
        Dict with parsing statistics
    """
    total = len(results)
    if total == 0:
        return {"total": 0}
    
    valid_count = sum(1 for r in results if r.answer == ParsedAnswer.VALID)
    invalid_count = sum(1 for r in results if r.answer == ParsedAnswer.INVALID)
    unclear_count = sum(1 for r in results if r.answer == ParsedAnswer.UNCLEAR)
    error_count = sum(1 for r in results if r.answer == ParsedAnswer.ERROR)
    
    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "unclear": unclear_count,
        "error": error_count,
        "clear_rate": (valid_count + invalid_count) / total,
        "valid_rate": valid_count / total if total > 0 else 0,
        "invalid_rate": invalid_count / total if total > 0 else 0,
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESPONSE PARSING TEST")
    print("=" * 60)
    
    # Test cases
    test_responses = [
        "valid",
        "invalid",
        "The syllogism is valid.",
        "This is invalid reasoning.",
        "I think the answer is VALID",
        "The conclusion is INVALID because...",
        "I'm not sure about this one.",
        "",
        None,
        "Let's think step by step. First... Therefore, the syllogism is valid.",
    ]
    
    print("\n[Parsing Tests]")
    for response in test_responses:
        result = parse_response(response)
        print(f"  '{response}' -> {result.answer.value}")
    
    print("\n[CoT Parsing Test]")
    cot_response = """Let's analyze this step by step.
    
    Premise 1 says all men are mortal.
    Premise 2 says Socrates is a man.
    
    If all men are mortal, and Socrates is a man, then Socrates must be mortal.
    
    Therefore, the syllogism is valid."""
    
    result = parse_cot_response(cot_response)
    print(f"  Answer: {result.answer.value}")
    print(f"  Has reasoning: {result.reasoning is not None}")
    
    print("\n[Validation Test]")
    parsed = parse_response("valid")
    validation = validate_against_ground_truth(parsed, "valid")
    print(f"  Predicted: {validation['predicted']}")
    print(f"  Ground truth: {validation['ground_truth']}")
    print(f"  Correct: {validation['is_correct']}")
