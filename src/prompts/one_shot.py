"""
One-Shot Prompting Strategy for Syllogistic Reasoning

The model is given ONE example of a correct syllogism before being asked
to determine the correctness of the target syllogism.
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
# ONE-SHOT EXAMPLE (shows a correct syllogism with "correct" answer format)
# =============================================================================

ONE_SHOT_EXAMPLE = """Here is an example:

Premise 1: All things that are smoked are bad for your health.
Premise 2: Cigarettes are smoked.
Conclusion: Therefore, cigarettes are bad for your health.

Answer: correct

The conclusion follows from the premises because if all things that are smoked are bad for your health, and cigarettes are smoked, then cigarettes must be bad for your health."""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """{example}

Now determine whether the following syllogism is correct or incorrect.

Premise 1: {statement_1}
Premise 2: {statement_2}
Conclusion: {conclusion}

Is this syllogism correct or incorrect? Respond with exactly one word: "correct" or "incorrect"."""


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
STRATEGY_DESCRIPTION = "One-shot prompting - one correct example provided"


if __name__ == "__main__":
    # Test the prompt generation with example from master dataset
    test_syllogism = {
        "statement_1": "All calculators are machines.",
        "statement_2": "All computers are calculators.",
        "conclusion": "Therefore, some machines are not computers."
    }
    
    print("=" * 60)
    print("ONE-SHOT PROMPT TEST")
    print("=" * 60)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Prompt]")
    print(format_prompt(test_syllogism))
