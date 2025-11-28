# Logic in LLMs: Syllogistic Reasoning Benchmark

A comprehensive benchmark for evaluating Large Language Models on syllogistic reasoning tasks.

## Overview

This project evaluates LLMs' logical reasoning capabilities using **categorical syllogisms** - a fundamental form of deductive reasoning. We test models across multiple prompting strategies and temperature settings to analyze both accuracy and consistency.

## Features

- **24 Models**: 3 Google Gemini + 21 HuggingFace models (via Fireworks.ai inferencing)
- **160 Syllogism Instances**: 40 base syllogisms × 4 content variants
- **4 Prompting Strategies**: `zero_shot`, `one_shot`, `few_shot`, `zero_shot_cot`
- **3 Temperature Settings**: 0.0, 0.5, 1.0
- **Adaptive Stopping**: Efficient sampling for non-deterministic runs

## Project Structure

```
Logic_in_LLMs/
├── src/
│   ├── config.py              # Configuration management
│   ├── inference/
│   │   ├── model_registry.py  # 24 model definitions
│   │   ├── api_clients.py     # Gemini & HuggingFace clients
│   │   ├── stopping_strategy.py # Adaptive voting logic
│   │   └── batch_processing.py  # Experiment runner
│   ├── prompts/               # Prompting strategies
│   ├── evaluation/            # Response parsing & metrics
│   └── analysis/              # Results analysis
├── scripts/
│   └── run_experiments.py     # CLI for experiments
├── tests/                     # Unit tests
├── data/
│   └── syllogisms_master_dataset.json
├── config.toml                # API keys & settings
└── requirements.txt
```

## Installation

```bash
# Clone repository
git clone https://github.com/XAheli/Logic_in_LLMs.git
cd Logic_in_LLMs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp config.toml.example config.toml
# Edit config.toml with your API keys
```

## Configuration

Edit `config.toml`:

```toml
[api_keys]
google_api_key = "your-gemini-api-key"
huggingface_api_key = "your-hf-api-key"  # Fireworks routing

[api_settings]
max_tokens = 600
top_p = 1.0
```

## Usage

### Run Experiments

```bash
# Single model, all strategies, all temperatures
python scripts/run_experiments.py --models "gemini-2.5-pro"

# Multiple models with specific settings
python scripts/run_experiments.py \
    --models "gemini-2.5-pro,llama-3.3-70b-instruct" \
    --strategies "zero_shot,zero_shot_cot" \
    --temperatures "0.0,0.5,1.0"

# Dry run (preview without execution)
python scripts/run_experiments.py --models "gemini-2.5-pro" --dry-run

# Run all 24 models
python scripts/run_experiments.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Adaptive Stopping Strategy

For temperature > 0 (non-deterministic), we use an adaptive voting mechanism:

| Condition | Action |
|-----------|--------|
| First 5 iterations ALL same | **Stop early**, use that answer |
| First 5 iterations mixed | **Continue to 10** iterations |
| After 10: valid > invalid | Final answer = "valid" |
| After 10: invalid > valid | Final answer = "invalid" |
| After 10: **tie** | Final answer = "invalid" (conservative) |

## Models

### Google Gemini (3)
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-preview-09-25`

### HuggingFace via Fireworks (21)
- Llama: `llama-3.3-70b-instruct`, `llama-3.1-70b-instruct`, `llama-3.1-8b-instruct`, etc.
- Qwen: `qwq-32b`, `qwen3-235b-a22b-thinking`, `qwen3-next-80b-a3b-instruct`
- DeepSeek: `deepseek-r1`, `deepseek-v3.1`
- Mistral: `mixtral-8x7b-instruct`, `mistral-7b-instruct-v0.3`
- Others: `gemma-3-27b-it`, `yi-34b-chat`, `codellama-34b-instruct`, etc.

## Dataset

- **40 syllogisms**: 19 valid (47.5%) + 21 invalid (52.5%)
- **4 variants per syllogism**:
  - **N (Normal)**: Original sensical predicates
  - **X (Nonsense)**: Abstract/meaningless predicates
  - **O (Order-switched)**: Premises in reversed order
  - **OX (Nonsense + Order)**: Combined modifications
- **160 total instances**: 40 × 4 variants
- **Purpose**: Evaluate reasoning independent of content (X), premise ordering (O), and both (OX)

## Output

Results are saved to `results/raw_responses/temperature_{T}/`:
```
{model_key}_{strategy}.json
```

Each file contains:
- Metadata (model, temperature, strategy)
- Per-instance results (prediction, ground truth, confidence, raw responses)

## References

1. [llm-logic](https://github.com/wesholliday/llm-logic)
2. [SR-FoT](https://github.com/RodeWayne/SR-FoT)
3. [AAAI 2026 LM Reasoning Workshop](https://sites.google.com/view/aaai-2026-lmreasoning)