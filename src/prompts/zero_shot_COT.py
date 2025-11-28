"""
Zero-Shot Chain-of-Thought (CoT) Prompting Strategy for Syllogistic Reasoning

The model is asked to think step-by-step before giving its final answer,
without any examples provided.
"""

from typing import Dict


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a logic expert specializing in syllogistic reasoning. 
Your task is to determine whether a given syllogism is logically valid or invalid.

A syllogism is VALID if and only if the conclusion follows necessarily from the premises,
regardless of whether the premises or conclusion are true in the real world.

A syllogism is INVALID if the conclusion does not follow necessarily from the premises.

Think through the problem step by step, then provide your final answer."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """Determine whether the following syllogism is valid or invalid.

Premise 1: {statement_1}
Premise 2: {statement_2}
Conclusion: {conclusion}

Is this syllogism valid or invalid? Let's think step by step."""


# =============================================================================
# PROMPT GENERATION FUNCTIONS
# =============================================================================

def format_prompt(syllogism: Dict) -> str:
    """
    Format a syllogism into a zero-shot chain-of-thought prompt.
    
    Args:
        syllogism: Dictionary containing 'statement_1', 'statement_2', 'conclusion'
        
    Returns:
        Formatted prompt string with CoT instructions
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

STRATEGY_NAME = "zero_shot_cot"
STRATEGY_DESCRIPTION = "Zero-shot Chain-of-Thought prompting - step-by-step reasoning"


if __name__ == "__main__":
    # Test the prompt generation
    test_syllogism = {
        "statement_1": "All men are mortal",
        "statement_2": "Socrates is a man",
        "conclusion": "Socrates is mortal"
    }
    
    print("=" * 60)
    print("ZERO-SHOT CHAIN-OF-THOUGHT PROMPT TEST")
    print("=" * 60)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Prompt]")
    print(format_prompt(test_syllogism))
