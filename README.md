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
- **Dual Ground Truth**: Syntax (valid/invalid) and NLU (believable/unbelievable)
- **Belief Bias Analysis**: Detect when semantic plausibility overrides logical reasoning

---

## Experiments

### Models

We evaluate **24 models** across 2 providers:

#### Google AI Studio (3 models)
| Model Key | Display Name | LM Arena |
|-----------|--------------|----------|
| `gemini-2.5-pro` | Gemini 2.5 Pro | ✓ |
| `gemini-2.5-flash` | Gemini 2.5 Flash | ✓ |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | ✓ |

#### HuggingFace Inference API with :cheapest routing (19 models)

| Family | Models |
|--------|--------|
| **OpenAI OSS** | `gpt-oss-20b` |
| **Meta Llama** | `llama-3.3-70b-instruct`, `llama-3.2-3b-instruct`, `llama-3.2-1b-instruct`, `llama-3.1-8b-instruct` |
| **Qwen** | `qwen3-next-80b-a3b-instruct`, `qwen3-next-80b-a3b-thinking`, `qwen3-32b`, `qwen3-235b-a22b-thinking`, `qwq-32b` |
| **Mistral** | `mixtral-8x22b-instruct` |
| **DeepSeek** | `deepseek-r1`, `deepseek-v3.1`, `deepseek-r1-0528` |
| **Google Gemma** | `gemma-3-27b-it` |
| **Moonshot Kimi** | `kimi-k2-thinking`, `kimi-k2-instruct` |
| **GLM** | `glm-4.5`, `glm-4.6` |

---

### Data & Methodology

#### Dataset Structure

| Component | Description |
|-----------|-------------|
| **40 Base Syllogisms** | 19 valid (47.5%) + 21 invalid (52.5%) |
| **4 Variants per Syllogism** | N (normal), X (nonsense), O (order-switched), OX (combined) |
| **160 Total Instances** | 40 × 4 variants |
| **Dual Ground Truth** | Syntax (valid/invalid) + NLU (believable/unbelievable) |

#### Variant Types

| Variant | Description | Purpose |
|---------|-------------|---------|
| **N (Normal)** | Original sensical predicates | Baseline performance |
| **X (Nonsense)** | Abstract/meaningless predicates (e.g., "blargs", "zimons") | Test pure logical reasoning without semantic interference |
| **O (Order-switched)** | Premises in reversed order | Test sensitivity to premise ordering |
| **OX (Combined)** | Nonsense + Order-switched | Combined robustness test |

#### Ground Truth Mapping

The LLM responds with **"correct"** or **"incorrect"**. We evaluate against two ground truths:

| LLM Response | Syntax Mapping | NLU Mapping |
|--------------|----------------|-------------|
| `"correct"` | → `valid` | → `believable` |
| `"incorrect"` | → `invalid` | → `unbelievable` |

#### Belief Bias Detection

The 4 ground truth combinations reveal **belief bias** (when intuition overrides logic):

| Syntax | NLU | Interpretation |
|--------|-----|----------------|
| Valid | Believable | **Congruent** - Logic & intuition align ✓ |
| Valid | Unbelievable | **Incongruent** - Logically correct but counter-intuitive |
| Invalid | Believable | **Belief Bias Trap** - Sounds right but logically wrong ⚠️ |
| Invalid | Unbelievable | **Congruent** - Both say wrong ✓ |

---

### Prompting Strategies

#### Strategy Overview

| Strategy | Examples | CoT | Description |
|----------|----------|-----|-------------|
| `zero_shot` | 0 | ✗ | Direct question, no examples |
| `one_shot` | 1 | ✗ | One correct example provided |
| `few_shot` | 4 | ✗ | 2 correct + 2 incorrect examples |
| `zero_shot_cot` | 0 | ✓ | "Let's think step by step" |

#### Prompt Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM PROMPT                            │
├─────────────────────────────────────────────────────────────────┤
│ You are an expert in syllogistic reasoning.                     │
│ Your task is to determine whether the conclusion of a given     │
│ syllogism follows from the premises.                            │
│                                                                 │
│ A syllogism is CORRECT if the conclusion follows.               │
│ A syllogism is INCORRECT if the conclusion does not follow.     │
│                                                                 │
│ [Strategy-specific instruction]                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        USER PROMPT                              │
├─────────────────────────────────────────────────────────────────┤
│ [Examples if one_shot/few_shot]                                 │
│                                                                 │
│ Determine whether the following syllogism is correct/incorrect  │
│                                                                 │
│ Premise 1: {statement_1}                                        │
│ Premise 2: {statement_2}                                        │
│ Conclusion: {conclusion}                                        │
│                                                                 │
│ [CoT: "Let's think step by step" / Direct: "Respond with one    │
│  word: correct or incorrect"]                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      LLM RESPONSE                               │
├─────────────────────────────────────────────────────────────────┤
│ [CoT: Step-by-step reasoning...]                                │
│                                                                 │
│ Final Answer: "correct" or "incorrect"                          │
└─────────────────────────────────────────────────────────────────┘
```

#### Few-Shot Examples Used

| # | Syllogism | Answer |
|---|-----------|--------|
| 1 | All smoked things → bad for health; Cigarettes → smoked; ∴ Cigarettes → bad | `correct` |
| 2 | No furniture → attractive; Some tables → attractive; ∴ Some tables ≠ furniture | `correct` |
| 3 | All calculators → machines; All computers → calculators; ∴ Some machines ≠ computers | `incorrect` |
| 4 | No screwdrivers → heavy; Some tools → heavy; ∴ Some screwdrivers ≠ tools | `incorrect` |

---

### Adaptive Stopping Strategy

For non-deterministic sampling (temperature > 0):

| Condition | Action |
|-----------|--------|
| `temperature == 0.0` | **Single query** (deterministic, greedy decoding) |
| First 5 iterations ALL same | **Stop early**, use that answer |
| First 5 iterations mixed | **Continue to 10** iterations |
| After 10: correct > incorrect | Final answer = `"correct"` |
| After 10: incorrect > correct | Final answer = `"incorrect"` |
| After 10: **tie** | Final answer = `"incorrect"` (conservative) |

**Global Limits**: `max_iterations=10`, `threshold=5` (same for all models)

---

### Analysis Outputs

| Analysis Type | Description | Module |
|---------------|-------------|--------|
| **Accuracy Metrics** | Precision, Recall, F1 per class | `src/evaluation/calculate_metrics.py` |
| **Confusion Matrix** | TP/FP/FN/TN visualization | `src/analysis/visualization.py` |
| **Belief Bias** | Accuracy by (syntax × NLU) combination | `src/evaluation/calculate_metrics.py` |
| **Consistency** | Cross-variant agreement (N vs X vs O vs OX) | `src/evaluation/consistency_analysis.py` |
| **LM Arena Correlation** | Ranking correlation with benchmarks | `src/analysis/correlation.py` |
| **Model Similarity** | Prediction agreement between models | `src/analysis/visualization.py` |

---

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
│   ├── prompts/               # 4 prompting strategies
│   ├── evaluation/            # Response parsing, metrics, belief bias
│   └── analysis/              # Visualization, correlation, rankings
├── scripts/
│   ├── run_experiments.py     # CLI for experiments
│   ├── analyze_results.py     # Results analysis
│   └── generate_figures.py    # Publication figures
├── tests/                     # 63 unit tests
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
max_tokens = 12000
top_p = 1.0
```

## Usage

### Run Experiments

```bash
# Single model, all strategies, all temperatures
python scripts/run_experiments.py --models "gemini-2.5-flash"

# Multiple models with specific settings
python scripts/run_experiments.py \
    --models "gemini-2.5-flash,llama-3.3-70b-instruct" \
    --strategies "zero_shot,zero_shot_cot" \
    --temperatures "0.0,0.5,1.0"

# Dry run (preview without execution)
python scripts/run_experiments.py --models "gemini-2.5-flash" --dry-run

# Run all 24 models
python scripts/run_experiments.py
```

### Analyze Results

```bash
# Generate metrics, consistency analysis, rankings
python scripts/analyze_results.py

# Generate publication figures
python scripts/generate_figures.py
```

### Run Tests

```bash
pytest tests/ -v  # 63 tests
```

## Output

Results are saved to `results/raw_responses/temperature_{T}/`:
```
{model_key}_{strategy}.json
```

Each file contains:
- Metadata (model, temperature, strategy)
- Per-instance results (prediction, ground truth, confidence, raw responses)

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Syntax Accuracy** | Match against valid/invalid ground truth |
| **NLU Accuracy** | Match against believable/unbelievable ground truth |
| **Belief Bias Effect** | Congruent accuracy − Incongruent accuracy |
| **Cross-Variant Consistency** | Agreement rate across N/X/O/OX variants |
| **LM Arena Correlation** | Spearman ρ with benchmark rankings |

## References

1. [llm-logic](https://github.com/wesholliday/llm-logic) - Syllogistic reasoning benchmarks
2. [SR-FoT](https://github.com/RodeWayne/SR-FoT) - Syllogistic reasoning with faithful CoT
3. [AAAI 2026 LM Reasoning Workshop](https://sites.google.com/view/aaai-2026-lmreasoning)
4. Lambell, N. J., Evans, J. S. B. T., & Handley, S. J. (1999). Belief bias, logical reasoning and presentation order on the syllogistic evaluation task. In M. Hahn, & S. C. Stoness (Eds.), Proceedings of the 21st Annual Conference of the Cognitive Science Society (pp. 282-287). Lawrence Erlbaum. https://mindmodeling.org/cogscihistorical/
