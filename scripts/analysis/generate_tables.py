#!/usr/bin/env python3
"""
Consolidated table generation script from raw response data
Creates paper tables with flexible model selection and table options

Usage:
    python generate_tables.py                    # Generate all tables for 14 models (default)
    python generate_tables.py --models 15        # Include all 15 models
    python generate_tables.py --tables basic     # Generate only basic performance tables
    python generate_tables.py --tables complete  # Generate all tables with consistency
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Model configurations
MODELS_14 = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gpt-oss-20b',
    'glm-4.6',
    'kimi-k2-instruct',
    'deepseek-v3.1',
    'gemini-2.5-flash-lite',
    'qwen3-next-80b-a3b-instruct',
    'qwen3-next-80b-a3b-thinking',
    'llama-3.3-70b-instruct',
    'gemma-3-27b-it',
    'llama-3.1-8b-instruct',
    'llama-3.2-3b-instruct',
    'llama-3.2-1b-instruct'
]

MODELS_15 = MODELS_14 + ['mixtral-8x22b-instruct']

STRATEGIES = ['zero_shot', 'one_shot', 'few_shot', 'zero_shot_cot']
TEMPERATURES = ['0.0', '0.5', '1.0']
VARIANTS = ['N', 'O', 'X', 'OX']


def load_raw_data(models, results_dir):
    """Load all raw response data for specified models"""
    print(f"\n📥 Loading raw response data for {len(models)} models...")

    all_data = defaultdict(list)
    missing_files = []

    for model in models:
        for temp in TEMPERATURES:
            for strategy in STRATEGIES:
                filepath = results_dir / f"temperature_{temp}" / f"{model}_{strategy}.json"

                if not filepath.exists():
                    missing_files.append(filepath.name)
                    continue

                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    results = data.get('results', [])
                    for result in results:
                        result['model'] = model
                        result['temperature'] = temp
                        result['strategy'] = strategy
                        all_data[model].append(result)
                except Exception as e:
                    print(f"⚠️  Error loading {filepath.name}: {e}")

    if missing_files:
        print(f"⚠️  Missing {len(missing_files)} files (will be skipped)")

    loaded_models = len([m for m in models if all_data[m]])
    print(f"✅ Loaded data for {loaded_models}/{len(models)} models")

    return all_data


def calculate_syntax_metrics(model_results):
    """Calculate syntax accuracy, precision, recall, F1"""
    syntax_predictions = []
    syntax_ground_truth = []

    for r in model_results:
        pred = r.get('predicted', '').lower()
        gt_syntax = r.get('ground_truth_syntax', '').lower()

        if pred in ['correct', 'incorrect'] and gt_syntax in ['valid', 'invalid']:
            # Convert to binary: valid/correct=1, invalid/incorrect=0
            pred_bin = 1 if pred == 'correct' else 0
            gt_bin = 1 if gt_syntax == 'valid' else 0

            syntax_predictions.append(pred_bin)
            syntax_ground_truth.append(gt_bin)

    if not syntax_predictions:
        return 0, 0, 0, 0

    syntax_predictions = np.array(syntax_predictions)
    syntax_ground_truth = np.array(syntax_ground_truth)

    # Accuracy
    syntax_correct = (syntax_predictions == syntax_ground_truth).sum()
    syntax_acc = (syntax_correct / len(syntax_predictions) * 100)

    # Precision & recall (for "valid" class = positive class)
    tp = ((syntax_predictions == 1) & (syntax_ground_truth == 1)).sum()
    fp = ((syntax_predictions == 1) & (syntax_ground_truth == 0)).sum()
    fn = ((syntax_predictions == 0) & (syntax_ground_truth == 1)).sum()

    syntax_prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    syntax_rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    syntax_f1 = (2 * syntax_prec * syntax_rec / (syntax_prec + syntax_rec)) if (syntax_prec + syntax_rec) > 0 else 0

    return syntax_acc, syntax_prec, syntax_rec, syntax_f1


def calculate_nlu_metrics(model_results):
    """Calculate NLU accuracy, precision, recall based on believability"""
    nlu_predictions = []
    nlu_ground_truth = []

    for r in model_results:
        pred = r.get('predicted', '').lower()
        gt_nlu = r.get('ground_truth_NLU', '').lower()

        if pred in ['correct', 'incorrect'] and gt_nlu in ['believable', 'unbelievable']:
            # Convert: believable/correct=1, unbelievable/incorrect=0
            pred_bin = 1 if pred == 'correct' else 0
            gt_bin = 1 if gt_nlu == 'believable' else 0

            nlu_predictions.append(pred_bin)
            nlu_ground_truth.append(gt_bin)

    if not nlu_predictions:
        return 0, 0, 0

    nlu_predictions = np.array(nlu_predictions)
    nlu_ground_truth = np.array(nlu_ground_truth)

    # Accuracy
    nlu_correct = (nlu_predictions == nlu_ground_truth).sum()
    nlu_acc = (nlu_correct / len(nlu_predictions) * 100)

    # Precision & recall
    tp_nlu = ((nlu_predictions == 1) & (nlu_ground_truth == 1)).sum()
    fp_nlu = ((nlu_predictions == 1) & (nlu_ground_truth == 0)).sum()
    fn_nlu = ((nlu_predictions == 0) & (nlu_ground_truth == 1)).sum()

    nlu_prec = (tp_nlu / (tp_nlu + fp_nlu) * 100) if (tp_nlu + fp_nlu) > 0 else 0
    nlu_rec = (tp_nlu / (tp_nlu + fn_nlu) * 100) if (tp_nlu + fn_nlu) > 0 else 0

    return nlu_acc, nlu_prec, nlu_rec


def calculate_belief_bias(model_results):
    """Calculate congruent vs incongruent accuracy"""
    congruent_correct = 0
    congruent_total = 0
    incongruent_correct = 0
    incongruent_total = 0

    for r in model_results:
        pred = r.get('predicted', '').lower()
        gt_syntax = r.get('ground_truth_syntax', '').lower()
        gt_nlu = r.get('ground_truth_NLU', '').lower()

        if gt_syntax in ['valid', 'invalid'] and gt_nlu in ['believable', 'unbelievable']:
            is_congruent = (gt_syntax == 'valid' and gt_nlu == 'believable') or \
                          (gt_syntax == 'invalid' and gt_nlu == 'unbelievable')

            is_correct = (pred == 'correct' and gt_syntax == 'valid') or \
                        (pred == 'incorrect' and gt_syntax == 'invalid')

            if is_congruent:
                congruent_total += 1
                if is_correct:
                    congruent_correct += 1
            else:
                incongruent_total += 1
                if is_correct:
                    incongruent_correct += 1

    congruent_acc = (congruent_correct / congruent_total * 100) if congruent_total > 0 else 0
    incongruent_acc = (incongruent_correct / incongruent_total * 100) if incongruent_total > 0 else 0
    bias_effect = congruent_acc - incongruent_acc

    return congruent_acc, incongruent_acc, bias_effect


def calculate_consistency_metrics(model_results):
    """Calculate consistency across variants (N, O, X, OX)"""
    # Group by syllogism, strategy, temperature
    consistency_data = defaultdict(lambda: defaultdict(list))

    for r in model_results:
        syl_id = r.get('syllogism_id', '')
        variant = r.get('variant', '')
        strategy = r.get('strategy', '')
        temp = r.get('temperature', '')
        pred = r.get('predicted', '').lower()

        key = (syl_id, strategy, temp)
        consistency_data[key][variant].append(pred)

    # Calculate consistency percentages
    c_all_scores = []
    c_nx_scores = []
    c_oox_scores = []

    for (syl_id, strategy, temp), variants_dict in consistency_data.items():
        # All 4 variants
        if len(variants_dict) == 4 and all(len(v) > 0 for v in variants_dict.values()):
            preds = [variants_dict[v][0] for v in ['N', 'O', 'X', 'OX']]
            c_all_scores.append(1 if len(set(preds)) == 1 else 0)

        # N vs X (name effect)
        if 'N' in variants_dict and 'X' in variants_dict:
            if len(variants_dict['N']) > 0 and len(variants_dict['X']) > 0:
                c_nx_scores.append(1 if variants_dict['N'][0] == variants_dict['X'][0] else 0)

        # O vs OX (order effect)
        if 'O' in variants_dict and 'OX' in variants_dict:
            if len(variants_dict['O']) > 0 and len(variants_dict['OX']) > 0:
                c_oox_scores.append(1 if variants_dict['O'][0] == variants_dict['OX'][0] else 0)

    c_all = (np.mean(c_all_scores) * 100) if c_all_scores else 0
    c_nx = (np.mean(c_nx_scores) * 100) if c_nx_scores else 0
    c_oox = (np.mean(c_oox_scores) * 100) if c_oox_scores else 0

    return c_all, c_nx, c_oox


def calculate_all_metrics(models, all_data, include_consistency=True):
    """Calculate all metrics for specified models"""
    print(f"\n📊 Calculating metrics (consistency: {include_consistency})...")

    model_metrics = []

    for model in models:
        model_results = all_data[model]

        if not model_results:
            print(f"⚠️  No results for {model}")
            continue

        # Calculate metrics
        syntax_acc, syntax_prec, syntax_rec, syntax_f1 = calculate_syntax_metrics(model_results)
        nlu_acc, nlu_prec, nlu_rec = calculate_nlu_metrics(model_results)
        gap = syntax_acc - nlu_acc
        congruent_acc, incongruent_acc, bias_effect = calculate_belief_bias(model_results)

        metrics = {
            'Model': model,
            'Syntax_Acc': round(syntax_acc, 2),
            'Syntax_Prec': round(syntax_prec, 2),
            'Syntax_Rec': round(syntax_rec, 2),
            'Syntax_F1': round(syntax_f1, 2),
            'NLU_Acc': round(nlu_acc, 2),
            'NLU_Prec': round(nlu_prec, 2),
            'NLU_Rec': round(nlu_rec, 2),
            'Gap': round(gap, 2),
            'Congruent_Acc': round(congruent_acc, 2),
            'Incongruent_Acc': round(incongruent_acc, 2),
            'Bias_Effect': round(bias_effect, 2)
        }

        # Add consistency metrics if requested
        if include_consistency:
            c_all, c_nx, c_oox = calculate_consistency_metrics(model_results)
            metrics.update({
                'C_all': round(c_all, 2),
                'C_N<->X': round(c_nx, 2),
                'C_O<->OX': round(c_oox, 2)
            })

        model_metrics.append(metrics)

    return model_metrics


def save_tables(model_metrics, output_dir, table_set='complete'):
    """Save tables to CSV files"""
    print(f"\n💾 Saving tables (set: {table_set})...")

    df_all = pd.DataFrame(model_metrics)
    df_all = df_all.sort_values('Syntax_Acc', ascending=False).reset_index(drop=True)

    tables_saved = []

    if table_set in ['basic', 'complete']:
        # Table 1: Performance metrics
        if 'C_all' in df_all.columns:
            # Complete version with consistency
            table1_complete = df_all[['Model', 'Syntax_Acc', 'Syntax_Prec', 'Syntax_Rec', 'Syntax_F1',
                                       'C_all', 'C_N<->X', 'C_O<->OX']]
            table1_complete.to_csv(output_dir / 'paper_table1_complete.csv', index=False)
            tables_saved.append('paper_table1_complete.csv')
            print(f"✅ Saved: paper_table1_complete.csv ({len(table1_complete)} models)")

        # Basic performance
        table1_performance = df_all[['Model', 'Syntax_Acc', 'Syntax_Prec', 'Syntax_Rec', 'Syntax_F1']]
        table1_performance.to_csv(output_dir / 'paper_table1_performance.csv', index=False)
        tables_saved.append('paper_table1_performance.csv')
        print(f"✅ Saved: paper_table1_performance.csv ({len(table1_performance)} models)")

    if table_set == 'complete':
        # Table 2: Dual Evaluation (Syntax vs NLU)
        table2_dual = df_all[['Model', 'Syntax_Acc', 'Syntax_Prec', 'Syntax_Rec',
                               'NLU_Acc', 'NLU_Prec', 'NLU_Rec', 'Gap']]
        table2_dual.to_csv(output_dir / 'paper_table2_dual_eval.csv', index=False)
        tables_saved.append('paper_table2_dual_eval.csv')
        print(f"✅ Saved: paper_table2_dual_eval.csv ({len(table2_dual)} models)")

        # Table 3: Belief Bias (sorted by bias magnitude)
        table3_bias = df_all[['Model', 'Congruent_Acc', 'Incongruent_Acc', 'Bias_Effect']]
        table3_bias = table3_bias.sort_values('Bias_Effect', ascending=False).reset_index(drop=True)
        table3_bias.to_csv(output_dir / 'paper_table3_belief_bias.csv', index=False)
        tables_saved.append('paper_table3_belief_bias.csv')
        print(f"✅ Saved: paper_table3_belief_bias.csv ({len(table3_bias)} models)")

    return df_all, tables_saved


def print_summary(df_all):
    """Print summary statistics"""
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    print(f"\n📊 Top 5 models by Syntax Accuracy:")
    top5 = df_all[['Model', 'Syntax_Acc']].head()
    print(top5.to_string(index=False))

    if 'Bias_Effect' in df_all.columns:
        print(f"\n🧠 Belief Bias Range:")
        print(f"   Max bias: {df_all['Bias_Effect'].max():.2f}pp ({df_all.loc[df_all['Bias_Effect'].idxmax(), 'Model']})")
        print(f"   Min bias: {df_all['Bias_Effect'].min():.2f}pp ({df_all.loc[df_all['Bias_Effect'].idxmin(), 'Model']})")
        print(f"   Mean bias: {df_all['Bias_Effect'].mean():.2f}pp")

    if 'NLU_Acc' in df_all.columns:
        print(f"\n🔍 Syntax vs NLU:")
        print(f"   Mean Syntax Acc: {df_all['Syntax_Acc'].mean():.2f}%")
        print(f"   Mean NLU Acc: {df_all['NLU_Acc'].mean():.2f}%")
        print(f"   Mean Gap: {df_all['Gap'].mean():.2f}pp")

    if 'C_all' in df_all.columns:
        print(f"\n📏 Consistency Metrics:")
        print(f"   Mean C_all: {df_all['C_all'].mean():.2f}%")
        print(f"   Mean C_N<->X: {df_all['C_N<->X'].mean():.2f}%")
        print(f"   Mean C_O<->OX: {df_all['C_O<->OX'].mean():.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper tables from raw response data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Default: 14 models, complete tables
  %(prog)s --models 15          # Include all 15 models (including Mixtral)
  %(prog)s --tables basic       # Generate only basic performance tables
  %(prog)s --no-consistency     # Skip consistency metrics calculation
        """
    )

    parser.add_argument('--models', type=int, choices=[14, 15], default=14,
                        help='Number of models to include (default: 14, excludes Mixtral)')
    parser.add_argument('--tables', choices=['basic', 'complete'], default='complete',
                        help='Which tables to generate (default: complete)')
    parser.add_argument('--no-consistency', action='store_true',
                        help='Skip consistency metrics (faster computation)')
    parser.add_argument('--results-dir', type=Path, default=Path('results/raw_responses'),
                        help='Directory containing raw response files')
    parser.add_argument('--output-dir', type=Path, default=Path('results/analysis/tables'),
                        help='Output directory for generated tables')

    args = parser.parse_args()

    # Setup
    models = MODELS_15 if args.models == 15 else MODELS_14
    include_consistency = not args.no_consistency and args.tables == 'complete'

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print(f"GENERATING TABLES FROM RAW RESPONSES")
    print("="*100)
    print(f"Models: {args.models} ({'including Mixtral' if args.models == 15 else 'excluding Mixtral'})")
    print(f"Table set: {args.tables}")
    print(f"Consistency metrics: {'Yes' if include_consistency else 'No'}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")

    # Load data
    all_data = load_raw_data(models, args.results_dir)

    # Calculate metrics
    model_metrics = calculate_all_metrics(models, all_data, include_consistency)

    if not model_metrics:
        print("\n❌ No metrics calculated. Check that raw response files exist.")
        return 1

    # Save tables
    df_all, tables_saved = save_tables(model_metrics, args.output_dir, args.tables)

    # Print summary
    print_summary(df_all)

    print("\n" + "="*100)
    print(f"✅ SUCCESSFULLY GENERATED {len(tables_saved)} TABLES")
    print("="*100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
