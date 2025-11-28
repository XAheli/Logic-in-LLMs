"""
Prompting Strategies for Syllogistic Reasoning Benchmark

This module provides 4 prompting strategies:
1. Zero-shot: No examples, direct question
2. One-shot: One example provided
3. Few-shot: Multiple examples (2 valid, 2 invalid)
4. Zero-shot CoT: Chain-of-thought reasoning without examples

Usage:
    from src.prompts import get_prompt_module, get_prompt, AVAILABLE_STRATEGIES
    
    # Get a specific strategy module
    strategy = get_prompt_module("zero_shot")
    prompt = strategy.format_prompt(syllogism)
    
    # Or use the unified interface
    prompt = get_prompt(syllogism, strategy="few_shot")
"""

from typing import Dict, List, Optional
import importlib

from . import zero_shot
from . import one_shot
from . import few_shot
from . import zero_shot_COT


# =============================================================================
# AVAILABLE STRATEGIES
# =============================================================================

AVAILABLE_STRATEGIES = ["zero_shot", "one_shot", "few_shot", "zero_shot_cot"]

STRATEGY_MODULES = {
    "zero_shot": zero_shot,
    "one_shot": one_shot,
    "few_shot": few_shot,
    "zero_shot_cot": zero_shot_COT,
}


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def get_prompt_module(strategy: str):
    """
    Get the prompt module for a specific strategy.
    
    Args:
        strategy: One of 'zero_shot', 'one_shot', 'few_shot', 'zero_shot_cot'
        
    Returns:
        The corresponding prompt module
        
    Raises:
        ValueError: If strategy is not recognized
    """
    strategy = strategy.lower()
    if strategy not in STRATEGY_MODULES:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available strategies: {AVAILABLE_STRATEGIES}"
        )
    return STRATEGY_MODULES[strategy]


def get_prompt(syllogism: Dict, strategy: str = "zero_shot") -> str:
    """
    Get a formatted prompt for a syllogism using the specified strategy.
    
    Args:
        syllogism: Dictionary with 'statement_1', 'statement_2', 'conclusion'
        strategy: Prompting strategy to use
        
    Returns:
        Formatted prompt string
    """
    module = get_prompt_module(strategy)
    return module.format_prompt(syllogism)


def get_messages(syllogism: Dict, strategy: str = "zero_shot") -> List[Dict]:
    """
    Get chat messages for a syllogism using the specified strategy.
    
    Args:
        syllogism: Dictionary with 'statement_1', 'statement_2', 'conclusion'
        strategy: Prompting strategy to use
        
    Returns:
        List of message dictionaries for chat API
    """
    module = get_prompt_module(strategy)
    return module.get_messages(syllogism)


def get_prompt_only(syllogism: Dict, strategy: str = "zero_shot") -> str:
    """
    Get a single combined prompt (system + user) for completion APIs.
    
    Args:
        syllogism: Dictionary with 'statement_1', 'statement_2', 'conclusion'
        strategy: Prompting strategy to use
        
    Returns:
        Combined prompt string
    """
    module = get_prompt_module(strategy)
    return module.get_prompt_only(syllogism)


def get_system_prompt(strategy: str = "zero_shot") -> str:
    """
    Get the system prompt for a specific strategy.
    
    Args:
        strategy: Prompting strategy to use
        
    Returns:
        System prompt string
    """
    module = get_prompt_module(strategy)
    return module.SYSTEM_PROMPT


def get_strategy_info(strategy: str) -> Dict:
    """
    Get metadata about a prompting strategy.
    
    Args:
        strategy: Prompting strategy name
        
    Returns:
        Dictionary with strategy name and description
    """
    module = get_prompt_module(strategy)
    return {
        "name": module.STRATEGY_NAME,
        "description": module.STRATEGY_DESCRIPTION,
    }


def list_strategies() -> List[Dict]:
    """
    List all available strategies with their descriptions.
    
    Returns:
        List of dictionaries with strategy info
    """
    return [get_strategy_info(s) for s in AVAILABLE_STRATEGIES]


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test all strategies
    test_syllogism = {
        "statement_1": "All men are mortal",
        "statement_2": "Socrates is a man",
        "conclusion": "Socrates is mortal"
    }
    
    print("=" * 70)
    print("PROMPTING STRATEGIES TEST")
    print("=" * 70)
    
    print("\nAvailable Strategies:")
    for info in list_strategies():
        print(f"  - {info['name']}: {info['description']}")
    
    print("\n" + "=" * 70)
    
    for strategy in AVAILABLE_STRATEGIES:
        print(f"\n[{strategy.upper()}]")
        print("-" * 50)
        prompt = get_prompt(test_syllogism, strategy)
        # Print first 200 chars to keep output manageable
        preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
        print(preview)
        print("-" * 50)
