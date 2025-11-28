"""
Few-Shot Prompting Strategy for Syllogistic Reasoning

The model is given MULTIPLE examples (both valid and invalid) before being asked
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
# FEW-SHOT EXAMPLES (2 valid, 2 invalid)
# =============================================================================

FEW_SHOT_EXAMPLES = """Here are some examples:

Example 1:
Premise 1: All mammals are warm-blooded
Premise 2: All whales are mammals
Conclusion: All whales are warm-blooded
Answer: valid

Example 2:
Premise 1: All birds have feathers
Premise 2: Some animals are birds
Conclusion: Some animals have feathers
Answer: valid

Example 3:
Premise 1: All cats are animals
Premise 2: All dogs are animals
Conclusion: All cats are dogs
Answer: invalid

Example 4:
Premise 1: Some flowers are red
Premise 2: Some red things are roses
Conclusion: Some flowers are roses
Answer: invalid"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """{examples}

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
STRATEGY_DESCRIPTION = "Few-shot prompting - multiple examples (2 valid, 2 invalid) provided"


if __name__ == "__main__":
    # Test the prompt generation
    test_syllogism = {
        "statement_1": "All men are mortal",
        "statement_2": "Socrates is a man",
        "conclusion": "Socrates is mortal"
    }
    
    print("=" * 60)
    print("FEW-SHOT PROMPT TEST")
    print("=" * 60)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Prompt]")
    print(format_prompt(test_syllogism))
