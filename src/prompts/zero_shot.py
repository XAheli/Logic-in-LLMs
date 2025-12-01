"""
Zero-Shot Prompting Strategy for Syllogistic Reasoning

The model is given only the syllogism and asked to determine correctness
without any examples or demonstrations.
"""

from typing import Dict


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert in syllogistic reasoning. 
Your task is to determine whether the conclusion of a given syllogism follows from the premises.

A syllogism is CORRECT if the conclusion follows from the premises.
A syllogism is INCORRECT if the conclusion does not follow from the premises.

You must respond with exactly one word: either "correct" or "incorrect"."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """Determine whether the following syllogism is correct or incorrect.

Premise 1: {statement_1}
Premise 2: {statement_2}
Conclusion: {conclusion}

Is this syllogism correct or incorrect? Respond with exactly one word: "correct" or "incorrect"."""


# =============================================================================
# PROMPT GENERATION FUNCTIONS
# =============================================================================

def format_prompt(syllogism: Dict) -> str:
    """
    Format a syllogism into a zero-shot prompt.
    
    Args:
        syllogism: Dictionary containing 'statement_1', 'statement_2', 'conclusion'
        
    Returns:
        Formatted prompt string
    """
    return USER_PROMPT_TEMPLATE.format(
        statement_1=syllogism["statement_1"],
        statement_2=syllogism["statement_2"],
        conclusion=syllogism["conclusion"]
    )


def get_messages(syllogism: Dict) -> list:
    """
    Get the full message list for chat-based APIs.
    
    Args:
        syllogism: Dictionary containing 'statement_1', 'statement_2', 'conclusion'
        
    Returns:
        List of message dictionaries for chat API
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(syllogism)}
    ]


def get_prompt_only(syllogism: Dict) -> str:
    """
    Get a single combined prompt for completion-based APIs.
    
    Args:
        syllogism: Dictionary containing 'statement_1', 'statement_2', 'conclusion'
        
    Returns:
        Combined system + user prompt as single string
    """
    return f"{SYSTEM_PROMPT}\n\n{format_prompt(syllogism)}"


# =============================================================================
# STRATEGY METADATA
# =============================================================================

STRATEGY_NAME = "zero_shot"
STRATEGY_DESCRIPTION = "Zero-shot prompting - no examples provided, asks for correct/incorrect"


if __name__ == "__main__":
    # Test the prompt generation with example from master dataset
    test_syllogism = {
        "statement_1": "All things that are smoked are bad for your health.",
        "statement_2": "Cigarettes are smoked.",
        "conclusion": "Therefore, cigarettes are bad for your health."
    }
    
    print("=" * 60)
    print("ZERO-SHOT PROMPT TEST")
    print("=" * 60)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Prompt]")
    print(format_prompt(test_syllogism))
    print("\n[Combined Prompt]")
    print(get_prompt_only(test_syllogism))
