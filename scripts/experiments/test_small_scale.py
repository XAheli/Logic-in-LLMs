#!/usr/bin/env python3
"""
Small-Scale Test Script for Syllogistic Reasoning Benchmark

This script tests a small subset of the experiment to verify:
1. Each API call is independent (no session/context carryover)
2. Prompts are correctly formatted for each strategy
3. Model responses are parseable
4. Token counting works correctly

Usage:
    python scripts/test_small_scale.py --model MODEL_KEY [--dry-run]
    
Examples:
    # Test with Gemini (prints prompts and outputs)
    python scripts/test_small_scale.py --model gemini-2.5-flash
    
    # Dry run - just show prompts, no API calls
    python scripts/test_small_scale.py --model gemini-2.5-flash --dry-run
    
    # Test with HuggingFace model
    python scripts/test_small_scale.py --model llama-3.2-3b-instruct
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken not installed. Install with: pip install tiktoken")

from src.config import config
from src.prompts import get_prompt, AVAILABLE_STRATEGIES
from src.inference.model_registry import MODEL_REGISTRY, get_model
from src.inference.api_clients import UniversalClient
from src.inference.stopping_strategy import parse_response


# =============================================================================
# TOKEN COUNTING
# =============================================================================

class TokenCounter:
    """Token counter using tiktoken (GPT tokenizer)."""
    
    def __init__(self, model: str = "gpt-4"):
        if not TIKTOKEN_AVAILABLE:
            self.encoder = None
            return
        
        try:
            # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not self.encoder:
            # Rough estimate: ~4 chars per token
            return len(text) // 4
        return len(self.encoder.encode(text))
    
    def count_messages(self, messages: List[Dict]) -> int:
        """Count tokens in chat messages."""
        total = 0
        for msg in messages:
            # Add overhead for message structure
            total += 4  # role, content structure
            total += self.count(msg.get("role", ""))
            total += self.count(msg.get("content", ""))
        total += 2  # priming tokens
        return total


# =============================================================================
# TEST DATA
# =============================================================================

def get_test_instances(dataset_path: Path, n_syllogisms: int = 2) -> List[Dict]:
    """Get a small subset of syllogism instances for testing."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    instances = []
    for syllogism in dataset['syllogisms'][:n_syllogisms]:
        base_id = syllogism['id']
        
        # Get all variants for this syllogism
        for variant_key, variant_data in syllogism['variants'].items():
            instances.append({
                'syllogism_id': base_id,
                'variant': variant_key,
                'variant_id': variant_data['variant_id'],
                'statement_1': variant_data['statement_1'],
                'statement_2': variant_data['statement_2'],
                'conclusion': variant_data['conclusion'],
                'ground_truth_syntax': variant_data['ground_truth_syntax'],
                'ground_truth_NLU': variant_data['ground_truth_NLU']
            })
    
    return instances


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(
    model_key: str,
    n_syllogisms: int = 2,
    temperature: float = 0.0,
    dry_run: bool = False,
    verbose: bool = True
):
    """
    Run small-scale test with detailed output.
    
    Args:
        model_key: Model to test
        n_syllogisms: Number of syllogisms to test (default: 2)
        temperature: Temperature to use (default: 0.0 for deterministic)
        dry_run: If True, only show prompts without calling API
        verbose: Print detailed output
    """
    
    # Validate model
    if model_key not in MODEL_REGISTRY:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        return
    
    model_config = get_model(model_key)
    
    print("\n" + "=" * 80)
    print("SMALL-SCALE TEST")
    print("=" * 80)
    print(f"Model: {model_key}")
    print(f"Provider: {model_config.provider.value}")
    print(f"Billing: {model_config.billing_type}")
    print(f"Temperature: {temperature}")
    print(f"Dry Run: {dry_run}")
    print("=" * 80)
    
    # Initialize token counter
    token_counter = TokenCounter()
    
    # Load test instances
    dataset_path = config.experiment.dataset_full_path
    instances = get_test_instances(dataset_path, n_syllogisms)
    
    print(f"\n📊 Testing with {len(instances)} instances ({n_syllogisms} syllogisms × 4 variants)")
    print(f"📝 Strategies: {AVAILABLE_STRATEGIES}")
    
    # Calculate expected API calls
    total_calls = len(instances) * len(AVAILABLE_STRATEGIES)
    print(f"📞 Expected API calls: {total_calls}")
    
    # Initialize client (only if not dry run)
    client = None
    if not dry_run:
        try:
            client = UniversalClient(verbose=False)
            print("✅ Client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize client: {e}")
            return
    
    # Track statistics
    total_input_tokens = 0
    total_output_tokens = 0
    results = []
    
    # Run tests
    print("\n" + "-" * 80)
    print("RUNNING TESTS")
    print("-" * 80)
    
    for i, instance in enumerate(instances, 1):
        syl_id = instance['syllogism_id']
        variant = instance['variant']
        ground_truth_syntax = instance['ground_truth_syntax']
        ground_truth_NLU = instance['ground_truth_NLU']
        
        print(f"\n{'─' * 40}")
        print(f"[{i}/{len(instances)}] {syl_id} - Variant {variant}")
        print(f"{'─' * 40}")
        print(f"Statement 1: {instance['statement_1']}")
        print(f"Statement 2: {instance['statement_2']}")
        print(f"Conclusion:  {instance['conclusion']}")
        print(f"Ground Truth Syntax: {ground_truth_syntax.upper()}")
        print(f"Ground Truth NLU:    {ground_truth_NLU.upper()}")
        
        for strategy in AVAILABLE_STRATEGIES:
            print(f"\n  📝 Strategy: {strategy}")
            
            # Generate prompt
            prompt = get_prompt(instance, strategy)
            
            # Count input tokens
            input_tokens = token_counter.count(prompt)
            total_input_tokens += input_tokens
            
            print(f"  📏 Prompt length: {len(prompt)} chars, ~{input_tokens} tokens")
            
            if verbose:
                # Show truncated prompt
                prompt_preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
                print(f"  📋 Prompt preview:")
                for line in prompt_preview.split('\n'):
                    print(f"     {line}")
            
            if dry_run:
                print(f"  ⏭️  [DRY RUN] Skipping API call")
                continue
            
            # Make API call
            try:
                response = client.query(model_key, prompt, temperature)
                
                # Count output tokens
                output_tokens = token_counter.count(response)
                total_output_tokens += output_tokens
                
                # Parse response (returns VoteResult enum)
                parsed = parse_response(response)
                
                # Map prediction: correct→valid, incorrect→invalid for syntax comparison
                predicted_syntax = "valid" if parsed.value == "correct" else "invalid"
                is_correct_syntax = predicted_syntax == ground_truth_syntax
                
                # Map prediction: correct→believable, incorrect→unbelievable for NLU comparison
                predicted_NLU = "believable" if parsed.value == "correct" else "unbelievable"
                is_correct_NLU = predicted_NLU == ground_truth_NLU
                
                # Store result
                results.append({
                    'syllogism_id': syl_id,
                    'variant': variant,
                    'strategy': strategy,
                    'ground_truth_syntax': ground_truth_syntax,
                    'ground_truth_NLU': ground_truth_NLU,
                    'predicted': parsed.value,
                    'is_correct_syntax': is_correct_syntax,
                    'is_correct_NLU': is_correct_NLU,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'response_preview': response[:200] if len(response) > 200 else response
                })
                
                # Display result
                syntax_status = "✅" if is_correct_syntax else "❌"
                nlu_status = "✅" if is_correct_NLU else "❌"
                print(f"  {syntax_status} Syntax | {nlu_status} NLU | Response ({output_tokens} tokens):")
                response_preview = response[:200] + "..." if len(response) > 200 else response
                for line in response_preview.split('\n')[:5]:
                    print(f"     {line}")
                print(f"  → Parsed: {parsed.value.upper()}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'syllogism_id': syl_id,
                    'variant': variant,
                    'strategy': strategy,
                    'error': str(e)
                })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Token Usage:")
    print(f"   Total Input Tokens:  {total_input_tokens:,}")
    print(f"   Total Output Tokens: {total_output_tokens:,}")
    print(f"   Total Tokens:        {total_input_tokens + total_output_tokens:,}")
    
    if not dry_run and results:
        correct_syntax = sum(1 for r in results if r.get('is_correct_syntax', False))
        correct_NLU = sum(1 for r in results if r.get('is_correct_NLU', False))
        errors = sum(1 for r in results if 'error' in r)
        total = len(results)
        successful = total - errors
        
        print(f"\n📈 Results:")
        if successful > 0:
            print(f"   Correct (Syntax): {correct_syntax}/{successful} ({100*correct_syntax/successful:.1f}%)")
            print(f"   Correct (NLU):    {correct_NLU}/{successful} ({100*correct_NLU/successful:.1f}%)")
        else:
            print(f"   Correct:  0/0 (no successful API calls)")
        print(f"   Errors:   {errors}/{total}")
        
        # Breakdown by strategy
        print(f"\n📊 By Strategy (Syntax / NLU):")
        for strategy in AVAILABLE_STRATEGIES:
            strat_results = [r for r in results if r.get('strategy') == strategy and 'error' not in r]
            if strat_results:
                strat_correct_syntax = sum(1 for r in strat_results if r.get('is_correct_syntax', False))
                strat_correct_NLU = sum(1 for r in strat_results if r.get('is_correct_NLU', False))
                print(f"   {strategy}: {strat_correct_syntax}/{len(strat_results)} syntax, {strat_correct_NLU}/{len(strat_results)} NLU")
        
        # Breakdown by variant
        print(f"\n📊 By Variant (Syntax / NLU):")
        for variant in ['N', 'O', 'X', 'OX']:
            var_results = [r for r in results if r.get('variant') == variant and 'error' not in r]
            if var_results:
                var_correct_syntax = sum(1 for r in var_results if r.get('is_correct_syntax', False))
                var_correct_NLU = sum(1 for r in var_results if r.get('is_correct_NLU', False))
                print(f"   {variant}: {var_correct_syntax}/{len(var_results)} syntax, {var_correct_NLU}/{len(var_results)} NLU")
    
    # Cost estimate
    if not dry_run:
        print(f"\n💰 Cost Estimate (for this test only):")
        
        # Get model pricing (rough estimates)
        if 'gemini' in model_key.lower():
            input_price = 0.30 / 1_000_000
            output_price = 2.50 / 1_000_000
            # Add thinking tokens estimate for Gemini
            thinking_tokens = total_output_tokens * 5  # Estimate thinking is 5x output
            adjusted_output = total_output_tokens + thinking_tokens
            print(f"   ⚠️  Gemini thinking tokens estimated: ~{thinking_tokens:,}")
            output_cost = adjusted_output * output_price
        else:
            # Use generic Fireworks pricing
            input_price = 0.20 / 1_000_000
            output_price = 0.20 / 1_000_000
            output_cost = total_output_tokens * output_price
        
        input_cost = total_input_tokens * input_price
        total_cost = input_cost + output_cost
        
        print(f"   Input Cost:  ${input_cost:.6f}")
        print(f"   Output Cost: ${output_cost:.6f}")
        print(f"   Total Cost:  ${total_cost:.6f}")
        
        # Extrapolate to full experiment
        scale_factor = 160 * 3 * 10 / total_calls  # full dataset × 3 temps × avg 10 iters
        print(f"\n📈 Extrapolated Full Experiment Cost:")
        print(f"   Scale factor: {scale_factor:.1f}x")
        print(f"   Estimated: ${total_cost * scale_factor:.2f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Small-scale test for syllogistic reasoning benchmark"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model key to test (e.g., gemini-2.5-flash, llama-3.2-3b-instruct)'
    )
    
    parser.add_argument(
        '--n-syllogisms',
        type=int,
        default=2,
        help='Number of syllogisms to test (default: 2, means 8 instances)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show prompts without making API calls'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Less verbose output'
    )
    
    args = parser.parse_args()
    
    run_test(
        model_key=args.model,
        n_syllogisms=args.n_syllogisms,
        temperature=args.temperature,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
