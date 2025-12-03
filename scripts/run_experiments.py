#!/usr/bin/env python3
"""
Main Experiment Runner for Syllogistic Reasoning Benchmark

This script orchestrates the entire experiment:
1. Loads configuration and dataset
2. Runs all models across all configurations
3. Saves raw responses
4. Provides progress tracking and resumption

Usage:
    python scripts/run_experiments.py [options]
    
    Options:
        --models MODEL1,MODEL2     Comma-separated list of models (default: all)
        --strategies STRAT1,STRAT2 Comma-separated strategies (default: all)
        --temperatures T1,T2       Comma-separated temperatures (default: all)
        --dry-run                  Print what would be run without executing
        --resume                   Resume from last checkpoint
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.inference.model_registry import MODEL_REGISTRY, list_all_models
from src.inference.batch_processing import BatchProcessor, BatchConfig


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run syllogistic reasoning benchmark experiments"
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='Comma-separated list of model keys to run (default: all)'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        default=None,
        help='Comma-separated list of strategies (default: all)'
    )
    
    parser.add_argument(
        '--temperatures',
        type=str,
        default=None,
        help='Comma-separated list of temperatures (default: all)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print experiment plan without running'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: from config)'
    )
    
    parser.add_argument(
        '--paid-only',
        action='store_true',
        help='Only run paid models'
    )
    
    parser.add_argument(
        '--free-only',
        action='store_true',
        help='Only run free models'
    )
    
    return parser.parse_args()


def get_models_from_args(args) -> List[str]:
    """Get list of models based on arguments."""
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
        # Validate
        for m in models:
            if m not in MODEL_REGISTRY:
                print(f"WARNING: Unknown model '{m}', skipping")
                models.remove(m)
        return models
    
    all_models = list_all_models()
    
    # Note: --paid-only and --free-only flags are deprecated
    # All models now have billing_type instead of is_paid
    # Keeping for backward compatibility but filtering by provider
    if args.paid_only:
        from src.inference.model_registry import get_google_studio_models
        return get_google_studio_models()
    elif args.free_only:
        from src.inference.model_registry import get_hf_inference_models
        return get_hf_inference_models()
    
    return all_models


def get_strategies_from_args(args) -> List[str]:
    """Get list of strategies based on arguments."""
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]
        valid_strategies = config.experiment.prompting_strategies
        return [s for s in strategies if s in valid_strategies]
    return config.experiment.prompting_strategies


def get_temperatures_from_args(args) -> List[float]:
    """Get list of temperatures based on arguments."""
    if args.temperatures:
        return [float(t.strip()) for t in args.temperatures.split(',')]
    return config.experiment.temperatures


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def print_experiment_plan(
    models: List[str],
    strategies: List[str],
    temperatures: List[float],
    dataset_size: int
):
    """Print the experiment plan."""
    print("\n" + "=" * 70)
    print("EXPERIMENT PLAN")
    print("=" * 70)
    
    print(f"\n📊 Dataset: {dataset_size} syllogism instances")
    
    print(f"\n🤖 Models ({len(models)}):")
    for m in models:
        info = MODEL_REGISTRY[m]
        billing = f"[{info.billing_type}]"
        print(f"   - {m} ({info.provider}) {billing}")
    
    print(f"\n📝 Prompting Strategies ({len(strategies)}):")
    for s in strategies:
        print(f"   - {s}")
    
    print(f"\n🌡️  Temperatures ({len(temperatures)}):")
    for t in temperatures:
        print(f"   - {t}")
    
    total_configs = len(models) * len(strategies) * len(temperatures)
    total_queries = total_configs * dataset_size
    
    print(f"\n📈 Total configurations: {total_configs}")
    print(f"📈 Total queries (T=0): {total_queries}")
    print(f"📈 Note: T>0 may require additional queries for adaptive stopping")
    
    # Estimate time (rough)
    avg_time_per_query = 2.0  # seconds
    estimated_time = total_queries * avg_time_per_query / 60
    print(f"\n⏱️  Estimated time: ~{estimated_time:.0f} minutes")


def load_checkpoint(results_dir: Path) -> dict:
    """Load checkpoint if exists."""
    checkpoint_file = results_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(results_dir: Path, checkpoint: dict):
    """Save checkpoint."""
    checkpoint_file = results_dir / "checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_experiments(
    models: List[str],
    strategies: List[str],
    temperatures: List[float],
    results_dir: Path,
    resume: bool = False
):
    """Run all experiments."""
    
    # Load dataset
    dataset_path = config.experiment.dataset_full_path
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    num_syllogisms = len(dataset['syllogisms'])
    num_instances = sum(len(syl.get('variants', {})) for syl in dataset['syllogisms'])
    print(f"\n✅ Loaded dataset: {num_syllogisms} syllogisms × 4 variants = {num_instances} instances")
    
    # Setup results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw_responses").mkdir(exist_ok=True)
    
    # Load checkpoint if resuming
    completed = set()
    if resume:
        checkpoint = load_checkpoint(results_dir)
        completed = set(checkpoint.get('completed', []))
        print(f"📁 Resuming: {len(completed)} configurations already completed")
    
    # Create batch config
    batch_config = BatchConfig(
        models=models,
        temperatures=temperatures,
        strategies=strategies,
        save_raw_responses=True,
        verbose=False
    )
    
    # Initialize processor
    processor = BatchProcessor(
        dataset_path=dataset_path,
        results_dir=results_dir,
        verbose=False
    )
    
    # Run experiments
    start_time = time.time()
    
    total_configs = len(models) * len(strategies) * len(temperatures)
    completed_count = len(completed)
    
    for model_key in models:
        for strategy in strategies:
            for temperature in temperatures:
                config_key = f"{model_key}_{strategy}_{temperature}"
                
                if config_key in completed:
                    print(f"⏭️  Skipping (already done): {config_key}")
                    continue
                
                completed_count += 1
                print(f"\n[{completed_count}/{total_configs}] Running: {config_key}")
                
                # Prepare output path early for potential partial saves
                temp_dir = results_dir / "raw_responses" / f"temperature_{temperature}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                output_file = temp_dir / f"{model_key}_{strategy}.json"
                partial_file = temp_dir / f"{model_key}_{strategy}.partial.json"
                
                # Check for existing partial results to resume from
                results = []
                completed_instances = set()
                start_index = 0
                
                if partial_file.exists():
                    try:
                        with open(partial_file, 'r') as f:
                            partial_data = json.load(f)
                        results = partial_data.get('results', [])
                        completed_instances = {
                            (r['syllogism_id'], r['variant']) 
                            for r in results
                        }
                        start_index = len(results)
                        print(f"   📂 Resuming from partial: {start_index}/160 instances already done")
                    except Exception as e:
                        print(f"   ⚠️  Could not load partial file: {e}")
                        results = []
                        completed_instances = set()
                        start_index = 0
                
                try:
                    # Get instances and run each
                    instances = processor._get_syllogism_instances()
                    total_instances = len(instances)
                    
                    import sys
                    for i, instance in enumerate(instances, 1):
                        # Skip already completed instances (for partial resume)
                        instance_key = (instance['syllogism_id'], instance['variant'])
                        if instance_key in completed_instances:
                            continue
                        
                        result = processor.run_single_experiment(
                            instance=instance,
                            model_key=model_key,
                            temperature=temperature,
                            strategy=strategy
                        )
                        results.append(result.to_dict())
                        
                        # Progress indicator (every instance) - check is_correct_syntax
                        correct_syntax = sum(1 for r in results if r.get('is_correct_syntax', False))
                        current_count = len(results)
                        pct = (current_count / total_instances) * 100
                        bar_len = 20
                        filled = int(bar_len * current_count / total_instances)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"\r   [{bar}] {pct:5.1f}% ({current_count}/{total_instances}) | ✓ {correct_syntax}/{current_count} syntax", end='', flush=True)
                    
                    # Final stats on new line
                    final_correct_syntax = sum(1 for r in results if r.get('is_correct_syntax', False))
                    final_correct_nlu = sum(1 for r in results if r.get('is_correct_NLU', False))
                    print(f"\n   ✅ Done: {final_correct_syntax}/{total_instances} syntax ({100*final_correct_syntax/total_instances:.1f}%), {final_correct_nlu}/{total_instances} NLU ({100*final_correct_nlu/total_instances:.1f}%)")
                    
                    # Save final results
                    output_data = {
                        'metadata': {
                            'model': model_key,
                            'strategy': strategy,
                            'temperature': temperature,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'complete'
                        },
                        'results': results
                    }
                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    
                    print(f"   ✅ Saved: {output_file}")
                    
                    # Remove partial file if exists (completed successfully)
                    if partial_file.exists():
                        partial_file.unlink()
                        print(f"   🗑️  Removed partial file")
                    
                    # Update checkpoint
                    completed.add(config_key)
                    save_checkpoint(results_dir, {'completed': list(completed)})
                    
                except KeyboardInterrupt:
                    # Save partial results on interrupt
                    print(f"\n\n⚠️  Interrupted! Saving partial progress ({len(results)}/{total_instances} instances)...")
                    
                    partial_data = {
                        'metadata': {
                            'model': model_key,
                            'strategy': strategy,
                            'temperature': temperature,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'partial',
                            'completed_instances': len(results),
                            'total_instances': total_instances
                        },
                        'results': results
                    }
                    with open(partial_file, 'w') as f:
                        json.dump(partial_data, f, indent=2)
                    
                    print(f"   💾 Partial results saved: {partial_file}")
                    print(f"   ℹ️  Run with --resume to continue from {len(results)}/{total_instances}")
                    
                    save_checkpoint(results_dir, {'completed': list(completed)})
                    sys.exit(0)
                    
                except Exception as e:
                    print(f"\n   ❌ Error: {e}")
                    
                    # Save partial results on error too
                    if results:
                        print(f"   💾 Saving partial progress ({len(results)} instances)...")
                        partial_data = {
                            'metadata': {
                                'model': model_key,
                                'strategy': strategy,
                                'temperature': temperature,
                                'timestamp': datetime.now().isoformat(),
                                'status': 'error',
                                'error': str(e),
                                'completed_instances': len(results),
                                'total_instances': total_instances
                            },
                            'results': results
                        }
                        with open(partial_file, 'w') as f:
                            json.dump(partial_data, f, indent=2)
                        print(f"   💾 Partial results saved: {partial_file}")
                    
                    import traceback
                    traceback.print_exc()
                    # Continue with next configuration
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📁 Results saved to: {results_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("SYLLOGISTIC REASONING BENCHMARK")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get configuration
    models = get_models_from_args(args)
    strategies = get_strategies_from_args(args)
    temperatures = get_temperatures_from_args(args)
    
    # Load dataset to get size
    dataset_path = config.experiment.dataset_full_path
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    dataset_size = sum(
        len(syl.get('variants', {}))
        for syl in dataset.get('syllogisms', [])
    )
    
    # Set output directory
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = config.experiment.results_full_path
    
    # Print plan
    print_experiment_plan(models, strategies, temperatures, dataset_size)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No experiments executed")
        return
    
    # Confirm
    print("\n" + "-" * 70)
    response = input("Proceed with experiments? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Run
    run_experiments(models, strategies, temperatures, results_dir, args.resume)


if __name__ == "__main__":
    main()
