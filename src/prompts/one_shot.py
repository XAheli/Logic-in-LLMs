"""
One-Shot Prompting Strategy for Syllogistic Reasoning

The model is given ONE example of a valid syllogism before being asked
to determine the validity of the target syllogism.
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

You must respond with exactly one word: either "valid" or "invalid"."""


# =============================================================================
# ONE-SHOT EXAMPLE
# =============================================================================

ONE_SHOT_EXAMPLE = """Here is an example:

Premise 1: All mammals are warm-blooded
Premise 2: All whales are mammals
Conclusion: All whales are warm-blooded

This syllogism is: valid

The conclusion follows necessarily from the premises because if all mammals are warm-blooded, and all whales are mammals, then all whales must be warm-blooded."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """{example}

Now determine whether the following syllogism is valid or invalid.

Premise 1: {statement_1}
Premise 2: {statement_2}
Conclusion: {conclusion}

Is this syllogism valid or invalid? Respond with exactly one word: "valid" or "invalid"."""


# =============================================================================
# PROMPT GENERATION FUNCTIONS
# =============================================================================

def format_prompt(syllogism: Dict) -> str:
    """
    Format a syllogism into a one-shot prompt with example.
    
    Args:
        syllogism: Dictionary containing 'statement_1', 'statement_2', 'conclusion'
        
    Returns:
        Formatted prompt string with one example
    """
    return USER_PROMPT_TEMPLATE.format(
        example=ONE_SHOT_EXAMPLE,
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

STRATEGY_NAME = "one_shot"
STRATEGY_DESCRIPTION = "One-shot prompting - one valid example provided"


if __name__ == "__main__":
    # Test the prompt generation
    test_syllogism = {
        "statement_1": "All men are mortal",
        "statement_2": "Socrates is a man",
        "conclusion": "Socrates is mortal"
    }
    
    print("=" * 60)
    print("ONE-SHOT PROMPT TEST")
    print("=" * 60)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Prompt]")
    print(format_prompt(test_syllogism))
