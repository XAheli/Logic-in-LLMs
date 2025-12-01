"""
Response Parsing for Syllogistic Reasoning Benchmark

Parses raw model responses to extract:
- Correct/Incorrect classification
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
    CORRECT = "correct"
    INCORRECT = "incorrect"
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
    def is_correct(self) -> bool:
        return self.answer == ParsedAnswer.CORRECT
    
    @property
    def is_incorrect(self) -> bool:
        return self.answer == ParsedAnswer.INCORRECT
    
    @property
    def is_clear(self) -> bool:
        return self.answer in (ParsedAnswer.CORRECT, ParsedAnswer.INCORRECT)


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_response(response: str) -> ParseResult:
    """
    Parse a model response to extract the answer.
    
    Parsing logic (order matters):
    1. If response is empty/None → ERROR
    2. If "incorrect" appears → INCORRECT
    3. If "correct" appears → CORRECT
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
    
    # Check for incorrect first (since "incorrect" contains "correct")
    if "incorrect" in clean:
        return ParseResult(
            answer=ParsedAnswer.INCORRECT,
            raw_response=response,
            confidence=1.0
        )
    elif "correct" in clean:
        return ParseResult(
            answer=ParsedAnswer.CORRECT,
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
    - The final answer (correct/incorrect)
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
            if "incorrect" in word:
                return ParseResult(
                    answer=ParsedAnswer.INCORRECT,
                    raw_response=response,
                    confidence=1.0,
                    reasoning=reasoning
                )
            elif "correct" in word:
                return ParseResult(
                    answer=ParsedAnswer.CORRECT,
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
    
    Returns "correct", "incorrect", or "unclear".
    """
    result = parse_response(response)
    return result.answer.value


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_against_ground_truth(
    parsed: ParseResult,
    ground_truth: str,
    ground_truth_type: str = "syntax"
) -> Dict:
    """
    Validate a parsed response against ground truth.
    
    LLM predicts: "correct" or "incorrect"
    
    For syntax comparison (ground_truth_type="syntax"):
        - correct ↔ valid
        - incorrect ↔ invalid
        
    For NLU comparison (ground_truth_type="NLU"):
        - correct ↔ believable
        - incorrect ↔ unbelievable
    
    Args:
        parsed: Parsed response
        ground_truth: Expected ground truth ("valid"/"invalid" or "believable"/"unbelievable")
        ground_truth_type: "syntax" or "NLU"
        
    Returns:
        Dict with validation results
    """
    prediction = parsed.answer.value  # "correct" or "incorrect"
    
    if ground_truth_type == "syntax":
        # Map: correct↔valid, incorrect↔invalid
        mapped_prediction = "valid" if prediction == "correct" else "invalid"
        is_correct = mapped_prediction == ground_truth.lower()
    elif ground_truth_type == "NLU":
        # Map: correct↔believable, incorrect↔unbelievable
        mapped_prediction = "believable" if prediction == "correct" else "unbelievable"
        is_correct = mapped_prediction == ground_truth.lower()
    else:
        raise ValueError(f"Invalid ground_truth_type: {ground_truth_type}. Must be 'syntax' or 'NLU'.")
    
    return {
        "predicted": prediction,
        "mapped_prediction": mapped_prediction,
        "ground_truth": ground_truth.lower(),
        "ground_truth_type": ground_truth_type,
        "is_correct": is_correct,
        "is_clear": parsed.is_clear,
        "confidence": parsed.confidence
    }


def validate_against_both_ground_truths(
    parsed: ParseResult,
    ground_truth_syntax: str,
    ground_truth_NLU: str
) -> Dict:
    """
    Validate a parsed response against both ground truths.
    
    Args:
        parsed: Parsed response
        ground_truth_syntax: Expected syntax validity ("valid"/"invalid")
        ground_truth_NLU: Expected NLU believability ("believable"/"unbelievable")
        
    Returns:
        Dict with validation results for both ground truths
    """
    syntax_result = validate_against_ground_truth(parsed, ground_truth_syntax, "syntax")
    nlu_result = validate_against_ground_truth(parsed, ground_truth_NLU, "NLU")
    
    return {
        "predicted": parsed.answer.value,
        "ground_truth_syntax": ground_truth_syntax.lower(),
        "ground_truth_NLU": ground_truth_NLU.lower(),
        "is_correct_syntax": syntax_result["is_correct"],
        "is_correct_NLU": nlu_result["is_correct"],
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
    
    correct_count = sum(1 for r in results if r.answer == ParsedAnswer.CORRECT)
    incorrect_count = sum(1 for r in results if r.answer == ParsedAnswer.INCORRECT)
    unclear_count = sum(1 for r in results if r.answer == ParsedAnswer.UNCLEAR)
    error_count = sum(1 for r in results if r.answer == ParsedAnswer.ERROR)
    
    return {
        "total": total,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "unclear": unclear_count,
        "error": error_count,
        "clear_rate": (correct_count + incorrect_count) / total,
        "correct_rate": correct_count / total if total > 0 else 0,
        "incorrect_rate": incorrect_count / total if total > 0 else 0,
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
        "correct",
        "incorrect",
        "The syllogism is correct.",
        "This is incorrect reasoning.",
        "I think the answer is CORRECT",
        "The conclusion is INCORRECT because...",
        "I'm not sure about this one.",
        "",
        None,
        "Let's think step by step. First... Therefore, the syllogism is correct.",
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
    
    Therefore, the syllogism is correct."""
    
    result = parse_cot_response(cot_response)
    print(f"  Answer: {result.answer.value}")
    print(f"  Has reasoning: {result.reasoning is not None}")
    
    print("\n[Validation Test]")
    parsed = parse_response("correct")
    validation = validate_against_ground_truth(parsed, "correct")
    print(f"  Predicted: {validation['predicted']}")
    print(f"  Ground truth: {validation['ground_truth']}")
    print(f"  Correct: {validation['is_correct']}")
