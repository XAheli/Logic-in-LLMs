"""
Few-Shot Prompting Strategy for Syllogistic Reasoning

The model is given MULTIPLE examples (both correct and incorrect) before being asked
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
# FEW-SHOT EXAMPLES (2 correct, 2 incorrect - using "correct"/"incorrect" format)
# =============================================================================

FEW_SHOT_EXAMPLES = """Here are some examples:

Example 1:
Premise 1: All things that are smoked are bad for your health.
Premise 2: Cigarettes are smoked.
Conclusion: Therefore, cigarettes are bad for your health.
Answer: correct

Example 2:
Premise 1: No pieces of furniture are attractive things.
Premise 2: Some tables are attractive things.
Conclusion: Therefore, some tables are not pieces of furniture.
Answer: correct

Example 3:
Premise 1: All calculators are machines.
Premise 2: All computers are calculators.
Conclusion: Therefore, some machines are not computers.
Answer: incorrect

Example 4:
Premise 1: No screwdrivers are heavy.
Premise 2: Some tools are heavy.
Conclusion: Therefore, some screwdrivers are not tools.
Answer: incorrect"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """{examples}

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
    Format a syllogism into a few-shot prompt with multiple examples.
    
    Args:
        syllogism: Dictionary containing 'statement_1', 'statement_2', 'conclusion'
        
    Returns:
        Formatted prompt string with multiple examples
    """
    return USER_PROMPT_TEMPLATE.format(
        examples=FEW_SHOT_EXAMPLES,
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

STRATEGY_NAME = "few_shot"
STRATEGY_DESCRIPTION = "Few-shot prompting - multiple examples (2 correct, 2 incorrect) provided"


if __name__ == "__main__":
    # Test the prompt generation with example from master dataset
    test_syllogism = {
        "statement_1": "No fruits are fungi.",
        "statement_2": "All mushrooms are fungi.",
        "conclusion": "Therefore, some mushrooms are fruits."
    }
    
    print("=" * 60)
    print("FEW-SHOT PROMPT TEST")
    print("=" * 60)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Prompt]")
    print(format_prompt(test_syllogism))
