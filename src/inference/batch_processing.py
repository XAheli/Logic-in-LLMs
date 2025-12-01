"""Batch Processing for Syllogistic Reasoning Benchmark

Handles running experiments across:
- Multiple models (24 total)
- Multiple temperatures (0.0, 0.5, 1.0)
- Multiple prompting strategies (4 total)
- All syllogism variants (160 instances)

Implements:
- Progress tracking with tqdm
- Result saving (raw responses + parsed results)
- Resume capability for interrupted runs
- Parallel processing (optional)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from src.config import config
from src.inference.model_registry import (
    MODEL_REGISTRY,
    get_model,
    list_all_models,
    Provider
)
from src.inference.api_clients import UniversalClient
from src.inference.stopping_strategy import (
    AdaptiveStoppingStrategy,
    StoppingResult,
    parse_response
)
from src.prompts import get_prompt, AVAILABLE_STRATEGIES


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentResult:
    """Result of a single experiment (one syllogism + one model + one config)."""
    syllogism_id: str
    variant: str
    model_key: str
    temperature: float
    prompting_strategy: str
    # Dual ground truths
    ground_truth_syntax: str  # valid/invalid
    ground_truth_NLU: str  # believable/unbelievable
    # Model prediction (correct/incorrect)
    predicted: str
    confidence: float
    # Correctness against each ground truth
    # correct→valid, incorrect→invalid for syntax comparison
    # correct→believable, incorrect→unbelievable for NLU comparison  
    is_correct_syntax: bool
    is_correct_NLU: bool
    total_iterations: int
    correct_count: int
    incorrect_count: int
    stopped_early: bool
    timestamp: str
    raw_responses: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class BatchConfig:
    """Configuration for a batch experiment run."""
    models: List[str] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    strategies: List[str] = field(default_factory=lambda: AVAILABLE_STRATEGIES)
    save_raw_responses: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        if not self.models:
            self.models = list_all_models()


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """
    Processes experiments in batches with progress tracking and result saving.
    """
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        verbose: bool = False
    ):
        self.dataset_path = dataset_path or config.experiment.dataset_full_path
        self.results_dir = results_dir or config.experiment.results_full_path
        self.verbose = verbose
        
        # Initialize client
        self.client = UniversalClient(verbose=verbose)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Ensure results directories exist
        config.ensure_directories()
    
    def _load_dataset(self) -> Dict:
        """Load the syllogisms dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def _get_syllogism_instances(self) -> List[Dict]:
        """
        Flatten dataset into list of individual instances.
        Each instance has: syllogism_id, variant, statement_1, statement_2, conclusion,
        ground_truth_syntax (valid/invalid), ground_truth_NLU (believable/unbelievable)
        """
        instances = []
        for syllogism in self.dataset['syllogisms']:
            base_id = syllogism['id']
            
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
    
    def _get_output_path(
        self,
        model_key: str,
        temperature: float,
        strategy: str
    ) -> Path:
        """Get the output file path for a specific configuration."""
        temp_dir = self.results_dir / "raw_responses" / f"temperature_{temperature}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"{model_key}_{strategy}.json"
    
    def _load_existing_results(self, output_path: Path) -> Dict[str, ExperimentResult]:
        """Load existing results for resume capability."""
        if not output_path.exists():
            return {}
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        results = {}
        for item in data.get('results', []):
            key = f"{item['syllogism_id']}_{item['variant']}"
            results[key] = item
        
        return results
    
    def _save_results(
        self,
        output_path: Path,
        results: List[ExperimentResult],
        metadata: Dict
    ):
        """Save results to JSON file."""
        output_data = {
            'metadata': metadata,
            'results': [r.to_dict() if isinstance(r, ExperimentResult) else r for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def run_single_experiment(
        self,
        instance: Dict,
        model_key: str,
        temperature: float,
        strategy: str
    ) -> ExperimentResult:
        """
        Run a single experiment: one syllogism instance with one model configuration.
        
        The LLM predicts "correct" or "incorrect".
        We compare this against:
        - ground_truth_syntax: correct↔valid, incorrect↔invalid
        - ground_truth_NLU: correct↔believable, incorrect↔unbelievable
        """
        # Get model config (for validation/logging purposes)
        model_config = get_model(model_key)
        
        # Generate prompt
        prompt = get_prompt(instance, strategy)
        
        # Create query function for stopping strategy
        def query_fn(temp: float) -> str:
            return self.client.query(model_key, prompt, temp)
        
        # Create stopping strategy with global limits (same for all models)
        stopping = AdaptiveStoppingStrategy.with_global_limits(
            verbose=self.verbose
        )
        
        # Run with stopping strategy
        result = stopping.run(query_fn, temperature)
        
        # Map LLM prediction to ground truth comparisons
        # LLM says "correct" or "incorrect"
        # Syntax: correct↔valid, incorrect↔invalid
        # NLU: correct↔believable, incorrect↔unbelievable
        predicted = result.final_answer  # "correct" or "incorrect"
        
        # Map prediction to syntax comparison
        predicted_syntax = "valid" if predicted == "correct" else "invalid"
        is_correct_syntax = predicted_syntax == instance['ground_truth_syntax']
        
        # Map prediction to NLU comparison
        predicted_NLU = "believable" if predicted == "correct" else "unbelievable"
        is_correct_NLU = predicted_NLU == instance['ground_truth_NLU']
        
        return ExperimentResult(
            syllogism_id=instance['syllogism_id'],
            variant=instance['variant'],
            model_key=model_key,
            temperature=temperature,
            prompting_strategy=strategy,
            ground_truth_syntax=instance['ground_truth_syntax'],
            ground_truth_NLU=instance['ground_truth_NLU'],
            predicted=predicted,
            confidence=result.confidence,
            is_correct_syntax=is_correct_syntax,
            is_correct_NLU=is_correct_NLU,
            total_iterations=result.total_iterations,
            correct_count=result.correct_count,
            incorrect_count=result.incorrect_count,
            stopped_early=result.stopped_early,
            timestamp=datetime.now().isoformat(),
            raw_responses=[
                {
                    'iteration': r.iteration,
                    'response': r.raw_response,
                    'vote': r.parsed_vote.value
                }
                for r in result.all_responses
            ] if config.output.save_raw_responses else []
        )
    
    def run_batch(
        self,
        batch_config: Optional[BatchConfig] = None,
        resume: bool = True
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run a batch of experiments.
        
        Args:
            batch_config: Configuration for the batch
            resume: Whether to skip already completed experiments
            
        Returns:
            Dictionary mapping output paths to results
        """
        if batch_config is None:
            batch_config = BatchConfig(verbose=self.verbose)
        
        instances = self._get_syllogism_instances()
        all_results = {}
        
        # Calculate total experiments
        total = (
            len(batch_config.models) * 
            len(batch_config.temperatures) * 
            len(batch_config.strategies) * 
            len(instances)
        )
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING")
        print(f"{'='*60}")
        print(f"Models: {len(batch_config.models)}")
        print(f"Temperatures: {batch_config.temperatures}")
        print(f"Strategies: {batch_config.strategies}")
        print(f"Instances: {len(instances)}")
        print(f"Total experiments: {total}")
        print(f"{'='*60}\n")
        
        # Main progress bar
        pbar = tqdm(total=total, desc="Overall Progress")
        
        for model_key in batch_config.models:
            for temperature in batch_config.temperatures:
                for strategy in batch_config.strategies:
                    output_path = self._get_output_path(model_key, temperature, strategy)
                    
                    # Load existing results for resume
                    existing = self._load_existing_results(output_path) if resume else {}
                    results = list(existing.values())
                    
                    # Metadata for this run
                    metadata = {
                        'model_key': model_key,
                        'model_id': get_model(model_key).model_id,
                        'temperature': temperature,
                        'strategy': strategy,
                        'started_at': datetime.now().isoformat(),
                        'total_instances': len(instances)
                    }
                    
                    for instance in instances:
                        key = f"{instance['syllogism_id']}_{instance['variant']}"
                        
                        # Skip if already completed
                        if key in existing:
                            pbar.update(1)
                            continue
                        
                        try:
                            result = self.run_single_experiment(
                                instance, model_key, temperature, strategy
                            )
                            results.append(result)
                            
                            # Save after each result (for resume capability)
                            self._save_results(output_path, results, metadata)
                            
                        except Exception as e:
                            if self.verbose:
                                print(f"\n  [ERROR] {model_key}/{instance['instance_id']}: {e}")
                            # Record error
                            results.append({
                                'syllogism_id': instance['syllogism_id'],
                                'variant': instance['variant'],
                                'model_key': model_key,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        pbar.update(1)
                    
                    # Final save with completion timestamp
                    metadata['completed_at'] = datetime.now().isoformat()
                    self._save_results(output_path, results, metadata)
                    all_results[str(output_path)] = results
        
        pbar.close()
        return all_results
    
    def run_single_model(
        self,
        model_key: str,
        temperatures: Optional[List[float]] = None,
        strategies: Optional[List[str]] = None,
        resume: bool = True
    ) -> Dict[str, List[ExperimentResult]]:
        """Run experiments for a single model."""
        batch_config = BatchConfig(
            models=[model_key],
            temperatures=temperatures or config.experiment.temperatures,
            strategies=strategies or AVAILABLE_STRATEGIES,
            verbose=self.verbose
        )
        return self.run_batch(batch_config, resume=resume)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_full_benchmark(
    models: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """Run the full benchmark on all or specified models."""
    processor = BatchProcessor(verbose=verbose)
    batch_config = BatchConfig(
        models=models or list_all_models(),
        verbose=verbose
    )
    return processor.run_batch(batch_config)


def run_quick_test(
    model_key: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    strategy: str = "zero_shot",
    num_instances: int = 5,
    verbose: bool = True
) -> List[ExperimentResult]:
    """Run a quick test on a few instances."""
    processor = BatchProcessor(verbose=verbose)
    instances = processor._get_syllogism_instances()[:num_instances]
    
    results = []
    for instance in tqdm(instances, desc=f"Testing {model_key}"):
        result = processor.run_single_experiment(
            instance, model_key, temperature, strategy
        )
        results.append(result)
        if verbose:
            print(f"  {instance['instance_id']}: "
                  f"predicted={result.predicted}, "
                  f"actual={result.ground_truth}, "
                  f"correct={result.is_correct}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Batch Processing Module")
    print("=" * 50)
    
    # Quick test
    print("\nRunning quick test with gemini-2.5-flash...")
    try:
        results = run_quick_test(
            model_key="gemini-2.5-flash",
            temperature=0.0,
            strategy="zero_shot",
            num_instances=3,
            verbose=True
        )
        
        correct = sum(1 for r in results if r.is_correct)
        print(f"\nResults: {correct}/{len(results)} correct")
        
    except Exception as e:
        print(f"Test failed: {e}")
