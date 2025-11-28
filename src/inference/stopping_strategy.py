"""
Adaptive Stopping Strategy for Syllogistic Reasoning Benchmark

Implements the stopping strategy inspired by the EMNLP 2024 paper
"Conditional and Modal Reasoning in Large Language Models".

Strategy:
- Temperature = 0.0: Single query (deterministic, greedy decoding)
- Temperature > 0.0: Adaptive stopping with early termination

For Temperature > 0:
    - Continue querying until threshold reached OR max iterations
    - Early stop if valid_count >= threshold OR invalid_count >= threshold
    - Return majority vote with confidence score

Iteration Limits:
    - PAID models (Gemini): max_iterations=10, threshold=5
    - FREE models: max_iterations=20, threshold=10
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional, Dict, Any
from enum import Enum
import time


# =============================================================================
# RESULT CLASSES
# =============================================================================

class VoteResult(Enum):
    """Possible vote outcomes."""
    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"
    UNCLEAR = "unclear"


@dataclass
class IterationResult:
    """Result of a single iteration/query."""
    iteration: int
    raw_response: str
    parsed_vote: VoteResult
    timestamp: float


@dataclass
class StoppingResult:
    """Final result after stopping strategy completes."""
    final_answer: str  # "valid" or "invalid"
    confidence: float  # Proportion of votes for the answer (0.0 to 1.0)
    total_iterations: int
    valid_count: int
    invalid_count: int
    error_count: int
    stopped_early: bool  # True if threshold reached before max_iterations
    all_responses: List[IterationResult] = field(default_factory=list)
    
    @property
    def unclear_count(self) -> int:
        """Count of unclear/unparseable responses."""
        return self.total_iterations - self.valid_count - self.invalid_count - self.error_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "total_iterations": self.total_iterations,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "error_count": self.error_count,
            "unclear_count": self.unclear_count,
            "stopped_early": self.stopped_early,
            "responses": [
                {
                    "iteration": r.iteration,
                    "raw_response": r.raw_response,
                    "parsed_vote": r.parsed_vote.value,
                }
                for r in self.all_responses
            ]
        }


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_response(response: str) -> VoteResult:
    """
    Parse a model response to determine the vote.
    
    Parsing logic:
    - If "invalid" appears in response → INVALID
    - Else if "valid" appears → VALID
    - Else → UNCLEAR
    
    Args:
        response: Raw model response string
        
    Returns:
        VoteResult enum value
    """
    if not response or response.strip() == "":
        return VoteResult.ERROR
    
    clean_response = response.lower().strip()
    
    # Check for "invalid" first (since "invalid" contains "valid")
    if "invalid" in clean_response:
        return VoteResult.INVALID
    elif "valid" in clean_response:
        return VoteResult.VALID
    else:
        return VoteResult.UNCLEAR


# =============================================================================
# STOPPING STRATEGY
# =============================================================================

class AdaptiveStoppingStrategy:
    """
    Implements adaptive stopping for LLM queries.
    
    For temperature = 0.0:
        - Single deterministic query
        
    For temperature > 0.0:
        - Query repeatedly until threshold or max_iterations
        - Early stop when valid_count >= threshold OR invalid_count >= threshold
        - Return majority vote with confidence
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
        threshold: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the stopping strategy.
        
        Args:
            max_iterations: Maximum number of queries (20 for free, 10 for paid)
            threshold: Stop early if this many votes for one answer (10 for free, 5 for paid)
            verbose: Print progress information
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.verbose = verbose
    
    @classmethod
    def for_paid_model(cls, verbose: bool = False) -> "AdaptiveStoppingStrategy":
        """Create strategy for PAID models (Gemini): max=10, threshold=5.
        
        DEPRECATED: Use with_global_limits() instead - all models now use same limits.
        """
        return cls(max_iterations=10, threshold=5, verbose=verbose)
    
    @classmethod
    def for_free_model(cls, verbose: bool = False) -> "AdaptiveStoppingStrategy":
        """Create strategy for FREE models: max=10, threshold=5.
        
        DEPRECATED: Use with_global_limits() instead - all models now use same limits.
        """
        return cls(max_iterations=10, threshold=5, verbose=verbose)
    
    @classmethod
    def with_global_limits(cls, verbose: bool = False) -> "AdaptiveStoppingStrategy":
        """Create strategy with global limits: max=10, threshold=5."""
        return cls(max_iterations=10, threshold=5, verbose=verbose)
    
    @classmethod
    def from_model_config(cls, verbose: bool = False) -> "AdaptiveStoppingStrategy":
        """Create strategy with global limits (same for all models).
        
        Previously accepted is_paid parameter, now uses global limits for all models.
        """
        return cls.with_global_limits(verbose=verbose)
    
    def run(
        self,
        query_fn: Callable[[float], str],
        temperature: float
    ) -> StoppingResult:
        """
        Run the adaptive stopping strategy.
        
        Args:
            query_fn: Function that takes temperature and returns model response
            temperature: Temperature setting for the model
            
        Returns:
            StoppingResult with final answer and statistics
        """
        # Temperature 0 = deterministic, single query
        if temperature == 0.0:
            return self._run_deterministic(query_fn)
        else:
            return self._run_adaptive(query_fn, temperature)
    
    def _run_deterministic(self, query_fn: Callable[[float], str]) -> StoppingResult:
        """Run single deterministic query for temperature=0."""
        if self.verbose:
            print(f"  [T=0.0] Running single deterministic query...")
        
        try:
            response = query_fn(0.0)
            vote = parse_response(response)
        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] Query failed: {e}")
            response = f"ERROR: {str(e)}"
            vote = VoteResult.ERROR
        
        iteration_result = IterationResult(
            iteration=1,
            raw_response=response,
            parsed_vote=vote,
            timestamp=time.time()
        )
        
        # Determine final answer
        if vote == VoteResult.VALID:
            final_answer = "valid"
            valid_count, invalid_count, error_count = 1, 0, 0
        elif vote == VoteResult.INVALID:
            final_answer = "invalid"
            valid_count, invalid_count, error_count = 0, 1, 0
        elif vote == VoteResult.ERROR:
            final_answer = "error"
            valid_count, invalid_count, error_count = 0, 0, 1
        else:  # UNCLEAR
            final_answer = "unclear"
            valid_count, invalid_count, error_count = 0, 0, 0
        
        if self.verbose:
            print(f"  [T=0.0] Result: {final_answer}")
        
        return StoppingResult(
            final_answer=final_answer,
            confidence=1.0,  # Single query = 100% confidence in what we got
            total_iterations=1,
            valid_count=valid_count,
            invalid_count=invalid_count,
            error_count=error_count,
            stopped_early=False,  # N/A for deterministic
            all_responses=[iteration_result]
        )
    
    def _run_adaptive(
        self,
        query_fn: Callable[[float], str],
        temperature: float
    ) -> StoppingResult:
        """Run adaptive stopping strategy for temperature > 0.
        
        Early Stopping Logic:
        - If the FIRST 5 iterations are ALL the same (all valid OR all invalid),
          stop early at iteration 5.
        - Otherwise, continue to max_iterations (10).
        
        Final Answer Logic (when max iterations reached):
        - If valid_count > invalid_count → "valid"
        - If invalid_count > valid_count → "invalid"  
        - If valid_count == invalid_count → "invalid" (conservative default)
        """
        if self.verbose:
            print(f"  [T={temperature}] Running adaptive strategy "
                  f"(max={self.max_iterations}, threshold={self.threshold})...")
        
        valid_count = 0
        invalid_count = 0
        error_count = 0
        all_responses: List[IterationResult] = []
        stopped_early = False
        
        for i in range(self.max_iterations):
            # Query the model
            try:
                response = query_fn(temperature)
                vote = parse_response(response)
            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {i+1}: ERROR - {e}")
                response = f"ERROR: {str(e)}"
                vote = VoteResult.ERROR
            
            # Record result
            iteration_result = IterationResult(
                iteration=i + 1,
                raw_response=response,
                parsed_vote=vote,
                timestamp=time.time()
            )
            all_responses.append(iteration_result)
            
            # Update counts
            if vote == VoteResult.VALID:
                valid_count += 1
            elif vote == VoteResult.INVALID:
                invalid_count += 1
            elif vote == VoteResult.ERROR:
                error_count += 1
            # UNCLEAR responses don't count toward either
            
            if self.verbose:
                print(f"    Iteration {i+1}: {vote.value} "
                      f"(valid={valid_count}, invalid={invalid_count})")
            
            # Check early stopping condition: FIRST 5 iterations must ALL be the same
            # Only check at exactly iteration 5 (index 4)
            if i + 1 == self.threshold:  # After 5th iteration
                # Check if first 5 are all valid or all invalid
                first_5_votes = [r.parsed_vote for r in all_responses[:self.threshold]]
                all_valid = all(v == VoteResult.VALID for v in first_5_votes)
                all_invalid = all(v == VoteResult.INVALID for v in first_5_votes)
                
                if all_valid:
                    stopped_early = True
                    if self.verbose:
                        print(f"  [EARLY STOP] First {self.threshold} iterations ALL valid")
                    break
                elif all_invalid:
                    stopped_early = True
                    if self.verbose:
                        print(f"  [EARLY STOP] First {self.threshold} iterations ALL invalid")
                    break
                else:
                    if self.verbose:
                        print(f"  [CONTINUE] First {self.threshold} iterations mixed, continuing to {self.max_iterations}...")
        
        # Calculate final result
        total_iterations = len(all_responses)
        total_valid_invalid = valid_count + invalid_count
        
        if total_valid_invalid == 0:
            # No valid votes at all
            final_answer = "error" if error_count > 0 else "unclear"
            confidence = 0.0
        elif valid_count > invalid_count:
            final_answer = "valid"
            confidence = valid_count / total_valid_invalid
        elif invalid_count > valid_count:
            final_answer = "invalid"
            confidence = invalid_count / total_valid_invalid
        else:
            # Tie - default to invalid (more conservative)
            final_answer = "invalid"
            confidence = 0.5
        
        if self.verbose:
            print(f"  [RESULT] {final_answer} (confidence={confidence:.2%}, "
                  f"iterations={total_iterations})")
        
        return StoppingResult(
            final_answer=final_answer,
            confidence=confidence,
            total_iterations=total_iterations,
            valid_count=valid_count,
            invalid_count=invalid_count,
            error_count=error_count,
            stopped_early=stopped_early,
            all_responses=all_responses
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_adaptive_vote(
    query_fn: Callable[[float], str],
    temperature: float,
    verbose: bool = False
) -> Tuple[str, float]:
    """
    Convenience function to get vote using adaptive stopping.
    
    Args:
        query_fn: Function that takes temperature and returns model response
        temperature: Temperature setting
        verbose: Print progress
        
    Returns:
        Tuple of (answer, confidence)
    """
    strategy = AdaptiveStoppingStrategy.with_global_limits(verbose)
    result = strategy.run(query_fn, temperature)
    return result.final_answer, result.confidence


def run_with_stopping(
    query_fn: Callable[[float], str],
    temperature: float,
    max_iterations: int = 20,
    threshold: int = 10,
    verbose: bool = False
) -> StoppingResult:
    """
    Run stopping strategy with custom limits.
    
    Args:
        query_fn: Function that takes temperature and returns model response
        temperature: Temperature setting
        max_iterations: Maximum queries
        threshold: Early stopping threshold
        verbose: Print progress
        
    Returns:
        Full StoppingResult
    """
    strategy = AdaptiveStoppingStrategy(
        max_iterations=max_iterations,
        threshold=threshold,
        verbose=verbose
    )
    return strategy.run(query_fn, temperature)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("=" * 70)
    print("ADAPTIVE STOPPING STRATEGY TEST")
    print("=" * 70)
    
    # Mock query function that returns "valid" or "invalid" randomly
    def mock_query_mostly_valid(temperature: float) -> str:
        """Mock that returns valid 70% of the time."""
        if temperature == 0.0:
            return "valid"  # Deterministic
        return "valid" if random.random() < 0.7 else "invalid"
    
    def mock_query_mostly_invalid(temperature: float) -> str:
        """Mock that returns invalid 80% of the time."""
        if temperature == 0.0:
            return "invalid"
        return "invalid" if random.random() < 0.8 else "valid"
    
    def mock_query_with_noise(temperature: float) -> str:
        """Mock with some unclear responses."""
        if temperature == 0.0:
            return "valid"
        r = random.random()
        if r < 0.6:
            return "The syllogism is valid."
        elif r < 0.9:
            return "This is invalid reasoning."
        else:
            return "I'm not sure about this one."
    
    # Test 1: Deterministic (T=0)
    print("\n[TEST 1] Temperature = 0.0 (deterministic)")
    print("-" * 50)
    strategy = AdaptiveStoppingStrategy.with_global_limits(verbose=True)
    result = strategy.run(mock_query_mostly_valid, temperature=0.0)
    print(f"Final: {result.final_answer}, Confidence: {result.confidence:.2%}")
    
    # Test 2: Global limits with T=0.5
    print("\n[TEST 2] Global limits, Temperature = 0.5")
    print("-" * 50)
    strategy = AdaptiveStoppingStrategy.with_global_limits(verbose=True)
    result = strategy.run(mock_query_mostly_valid, temperature=0.5)
    print(f"Final: {result.final_answer}, Confidence: {result.confidence:.2%}")
    print(f"Stopped early: {result.stopped_early}, Iterations: {result.total_iterations}")
    
    # Test 3: Global limits with T=1.0
    print("\n[TEST 3] Global limits, Temperature = 1.0")
    print("-" * 50)
    strategy = AdaptiveStoppingStrategy.with_global_limits(verbose=True)
    result = strategy.run(mock_query_mostly_invalid, temperature=1.0)
    print(f"Final: {result.final_answer}, Confidence: {result.confidence:.2%}")
    print(f"Stopped early: {result.stopped_early}, Iterations: {result.total_iterations}")
    
    # Test 4: With noisy responses
    print("\n[TEST 4] Noisy responses")
    print("-" * 50)
    strategy = AdaptiveStoppingStrategy(max_iterations=15, threshold=8, verbose=True)
    result = strategy.run(mock_query_with_noise, temperature=0.5)
    print(f"Final: {result.final_answer}, Confidence: {result.confidence:.2%}")
    print(f"Valid: {result.valid_count}, Invalid: {result.invalid_count}, "
          f"Unclear: {result.unclear_count}, Errors: {result.error_count}")
    
    # Test 5: Convenience function
    print("\n[TEST 5] Convenience function")
    print("-" * 50)
    answer, conf = get_adaptive_vote(
        mock_query_mostly_valid, 
        temperature=0.5, 
        verbose=True
    )
    print(f"Answer: {answer}, Confidence: {conf:.2%}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
