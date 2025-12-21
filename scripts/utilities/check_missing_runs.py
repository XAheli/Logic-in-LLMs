#!/usr/bin/env python3
"""
Check for missing raw response files across all models, temperatures, and strategies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.model_registry import list_all_models

# Get all available models
all_models = sorted(list_all_models())

# Expected strategies
strategies = ['zero_shot', 'one_shot', 'few_shot', 'zero_shot_cot']

# Expected temperatures
temperatures = [0.0, 0.5, 1.0]

print('='*80)
print('COMPREHENSIVE RAW RESPONSE FILES CHECK')
print('='*80)

# Track missing files
missing_by_temp = {}
total_missing = 0
total_existing = 0
total_expected = len(all_models) * len(strategies) * len(temperatures)

for temp in temperatures:
    raw_dir = project_root / 'results' / 'raw_responses' / f'temperature_{temp}'
    
    if not raw_dir.exists():
        print(f'\n❌ Directory does not exist: {raw_dir}')
        missing_by_temp[temp] = {model: strategies[:] for model in all_models}
        total_missing += len(all_models) * len(strategies)
        continue
    
    existing_files = set(f.stem for f in raw_dir.glob('*.json'))
    total_existing += len(existing_files)
    
    missing_models = {}
    for model in all_models:
        missing_strats = []
        for strat in strategies:
            expected_file = f'{model}_{strat}'
            if expected_file not in existing_files:
                missing_strats.append(strat)
        
        if missing_strats:
            missing_models[model] = missing_strats
    
    if missing_models:
        missing_by_temp[temp] = missing_models
        total_missing += sum(len(s) for s in missing_models.values())

# Display results by temperature
for temp in temperatures:
    print(f'\n{"="*80}')
    print(f'TEMPERATURE {temp}')
    print(f'{"="*80}')
    
    if temp not in missing_by_temp:
        print('✅ All models have complete data!')
    else:
        missing = missing_by_temp[temp]
        complete = len(all_models) - len(missing)
        print(f'\n❌ Models with missing strategies: {len(missing)}/{len(all_models)}')
        
        print(f'\nMISSING FILES:')
        for model, strats in sorted(missing.items()):
            strat_str = ', '.join(strats)
            print(f'  • {model}: {strat_str}')

# Overall summary
print(f'\n{"="*80}')
print(f'OVERALL SUMMARY')
print(f'{"="*80}')
print(f'Total models: {len(all_models)}')
print(f'Total strategies per model: {len(strategies)}')
print(f'Total temperatures: {len(temperatures)}')
print(f'Total expected files: {total_expected}')
print(f'Total existing files: {total_existing}')
print(f'Total missing files: {total_missing}')
print(f'Completion: {100 * total_existing / total_expected:.1f}%')

# Generate run command
if total_missing > 0:
    print(f'\n{"="*80}')
    print('MODELS NEEDING RUNS')
    print(f'{"="*80}')
    
    all_missing_models = set()
    for temp_missing in missing_by_temp.values():
        all_missing_models.update(temp_missing.keys())
    
    print(f'\nModels with any missing data: {len(all_missing_models)}')
    for model in sorted(all_missing_models):
        print(f'  • {model}')
    
    print(f'\n{"="*80}')
    print('SUGGESTED COMMANDS TO RUN')
    print(f'{"="*80}')
    
    # Group by models
    models_str = ','.join(sorted(all_missing_models))
    print(f'\n# Run all missing combinations:')
    print(f'python scripts/run_experiments.py --models {models_str}')
    
    # Or by temperature
    print(f'\n# Or run by temperature:')
    for temp in temperatures:
        if temp in missing_by_temp:
            temp_models = sorted(missing_by_temp[temp].keys())
            models_csv = ','.join(temp_models)
            print(f'python scripts/run_experiments.py --models {models_csv} --temperatures {temp}')
else:
    print(f'\n{"="*80}')
    print('🎉 ALL DATA COMPLETE!')
    print(f'{"="*80}')
    print('\n✅ All models have been run for all strategies and temperatures.')
