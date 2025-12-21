# Logic in LLMs: Syllogistic Reasoning Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive benchmark for evaluating Large Language Models on categorical syllogistic reasoning tasks, with analysis of belief bias effects.

---

## Executive Summary

This study evaluates **15 LLMs** across **180 experimental configurations** (4 prompting strategies × 3 temperatures) on **160 categorical syllogisms** with 4 content variants each.

### Key Results at a Glance

| Finding | Result | Significance |
|---------|--------|--------------|
| **Best Model** | gemini-2.5-flash | 99.84% accuracy |
| **Worst Model** | mixtral-8x22b-instruct | 52.50% accuracy |
| **Overall Mean** | 79.75% | SD = 17.89% |
| **Belief Bias** | +13.38% mean effect | 87% of models affected |
| **Syntax vs NLU Gap** | 22.27 percentage points | Syntax > NLU |
| **Temperature Effect** | None (p = 0.9994) | Adaptive voting normalizes |
| **Strategy Effect** | None after correction | All strategies equivalent |
| **MMLU Correlation** | ρ = 0.29, p = 0.36 | **Not significant** |
| **LMArena Correlation** | ρ = -0.74, p = 0.001 | **Significant** |

### Core Conclusions

1. **LLMs CAN perform syllogistic reasoning** — top models achieve >97% accuracy
2. **Capability is NOT universal** — 6 models score below 70%
3. **Belief bias is pervasive** — semantic plausibility interferes with logical judgment
4. **Prompting strategies don't help** — few-shot and CoT provide no benefit for this task
5. **MMLU doesn't predict logical reasoning** — factual knowledge ≠ deductive inference

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Overview](#overview)
3. [Research Questions & Hypotheses](#research-questions--hypotheses)
4. [Experiments](#experiments)
   - [Models](#models)
   - [Data & Methodology](#data--methodology)
   - [Prompting Strategies](#prompting-strategies)
   - [Adaptive Stopping Strategy](#adaptive-stopping-strategy)
   - [Evaluation Methods](#evaluation-methods)
5. [Results](#results)
   - [Overall Performance](#overall-performance)
   - [Strategy Comparison](#strategy-comparison)
   - [Temperature Effects](#temperature-effects)
   - [Belief Bias Analysis](#belief-bias-analysis)
   - [Variant Analysis](#variant-analysis)
   - [Benchmark Correlations](#benchmark-correlations)
   - [Statistical Tests](#statistical-tests)
   - [Consistency Analysis](#consistency-analysis)
6. [Discussion](#discussion)
7. [Key Findings](#key-findings)
8. [Limitations](#limitations)
9. [Installation & Usage](#installation--usage)
10. [Project Structure](#project-structure)
11. [References](#references)
12. [Citation](#citation)

---

## Overview

This project evaluates LLMs' logical reasoning capabilities using **categorical syllogisms**—a fundamental form of deductive reasoning dating back to Aristotle. We systematically test models across multiple prompting strategies and temperature settings to analyze both **accuracy** and **consistency**, while specifically investigating the **belief bias effect**.

### Key Features

- **15 Models**: 4 Google Gemini + 11 HuggingFace models (via Inference API)
- **160 Syllogism Instances**: 40 base syllogisms × 4 content variants
- **4 Prompting Strategies**: Zero-shot, One-shot, Few-shot, Zero-shot Chain-of-Thought
- **3 Temperature Settings**: 0.0, 0.5, 1.0
- **Dual Ground Truth**: Syntax (valid/invalid) and NLU (believable/unbelievable)
- **Belief Bias Analysis**: Quantify when semantic plausibility overrides logical reasoning
- **180 Experimental Configurations**: 15 models × 4 strategies × 3 temperatures

---

## Research Questions & Hypotheses

### Research Motivation

Categorical syllogisms represent one of the oldest and most fundamental forms of deductive reasoning, yet it remains unclear whether modern LLMs can perform this type of formal logical reasoning reliably. Unlike factual recall or pattern matching, syllogistic reasoning requires:
- **Abstract logical structure recognition** independent of content
- **Resistance to semantic interference** (belief bias)
- **Consistent application of formal rules** across varied presentations

This benchmark study investigates whether LLMs truly "reason" or merely approximate reasoning through statistical patterns, and whether their performance can be predicted by existing benchmarks.

### Primary Research Questions

1. **RQ1**: How accurately can current LLMs perform categorical syllogistic reasoning when evaluated on both syntactic validity and semantic plausibility?
2. **RQ2**: Do LLMs exhibit **belief bias**—the tendency to accept logically invalid conclusions that align with prior beliefs, or reject valid conclusions that seem implausible?
3. **RQ3**: How do different prompting strategies (zero-shot, one-shot, few-shot, chain-of-thought) affect syllogistic reasoning performance?
4. **RQ4**: Does temperature (sampling randomness) significantly impact reasoning accuracy and consistency?
5. **RQ5**: Do models perform differently on semantically meaningful vs. abstract/nonsense syllogisms—indicating whether they rely on logical structure or semantic shortcuts?
6. **RQ6**: Does performance on syllogistic reasoning correlate with existing benchmarks (LMArena, MMLU)?

### Hypotheses

| ID | Hypothesis | Rationale | Status |
|----|------------|-----------|--------|
| **H1** | LLMs will struggle with categorical syllogistic reasoning, achieving below-human performance | Syllogisms require formal logical reasoning that may not emerge from next-token prediction | ⚠️ **Partially Supported** — High variance (52.5%-99.4%) |
| **H2** | LLMs will exhibit positive belief bias (higher accuracy on congruent problems where logic aligns with intuition) | Training data contains more "sensible" conclusions; models may conflate plausibility with validity | ✅ **Supported** (13/15 models, mean bias +13.38%) |
| **H3** | Few-shot prompting will improve performance by providing reasoning templates | In-context examples typically help with task understanding | ❌ **Not Supported** — Few-shot performed worst (77.25%) |
| **H4** | Chain-of-thought (CoT) prompting will improve accuracy by encouraging step-by-step reasoning | CoT has shown benefits for math/reasoning tasks | ⚠️ **Mixed Results** — Helped some models, hurt others |
| **H5** | Lower temperature (T=0.0) will yield higher and more consistent accuracy | Deterministic sampling reduces noise | ❌ **Not Supported** — No significant difference across temperatures |
| **H6** | Abstract/nonsense variants will have lower accuracy due to lack of semantic grounding | Models may rely on real-world knowledge to reason | ❌ **Opposite Found** — Nonsense variants scored higher (79% vs 76%) |
| **H7** | Syntax accuracy will be higher than NLU accuracy due to models defaulting to "valid" | Models may exhibit confirmation bias | ✅ **Supported** — Syntax: 79.7%, NLU: 57.4% |
| **H8** | Syllogistic reasoning performance will correlate with general intelligence benchmarks (MMLU) | Reasoning ability should generalize | ❌ **Not Supported** — No significant MMLU correlation (ρ=0.29, p=0.36) |

---

## Experiments

### Models

We evaluate **15 models** across 2 providers, representing a diverse range of model families, sizes, and architectures.

#### Google AI Studio (4 models)

| Model Key | Display Name | Architecture | Notes |
|-----------|--------------|--------------|-------|
| `gemini-2.5-pro` | Gemini 2.5 Pro | Multimodal Transformer | Top-tier Google model |
| `gemini-2.5-flash` | Gemini 2.5 Flash | Multimodal Transformer (Optimized) | Fast inference |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | Multimodal Transformer (Lite) | Lightweight variant |
| `gemma-3-27b-it` | Gemma 3 27B IT | Dense Transformer | 27B parameters |

#### HuggingFace Inference API (11 models)

All HuggingFace models use the `:cheapest` suffix for automatic provider routing to the most cost-effective inference endpoint.

| Family | Model Key | Parameters | Architecture |
|--------|-----------|------------|--------------|
| **Meta Llama** | `llama-3.3-70b-instruct` | 70B | Dense Transformer |
| | `llama-3.1-8b-instruct` | 8B | Dense Transformer |
| | `llama-3.2-3b-instruct` | 3B | Dense Transformer |
| | `llama-3.2-1b-instruct` | 1B | Dense Transformer |
| **Qwen** | `qwen3-next-80b-a3b-instruct` | 80B (3B active) | MoE |
| | `qwen3-next-80b-a3b-thinking` | 80B (3B active) | MoE + Reasoning |
| **Mistral** | `mixtral-8x22b-instruct` | 141B (39B active) | MoE |
| **DeepSeek** | `deepseek-v3.1` | 671B MoE | Mixture-of-Experts |
| **Zhipu AI** | `glm-4.6` | — | GLM Architecture |
| **Moonshot AI** | `kimi-k2-instruct` | — | Proprietary |
| **OpenAI OSS** | `gpt-oss-20b` | 20B | Dense Transformer |

#### Model Selection Criteria

1. **Diversity**: Models from 9 different organizations (Google, Meta, Qwen/Alibaba, Mistral, DeepSeek, Zhipu, Moonshot AI, OpenAI)
2. **Scale Range**: From 1B parameters (Llama 3.2 1B) to 671B MoE (DeepSeek V3.1)
3. **Architecture Variety**: Dense transformers, Mixture-of-Experts (MoE), and multimodal models
4. **Accessibility**: All models available via API for reproducibility

---

### Data & Methodology

#### Dataset Overview

| Component | Value | Description |
|-----------|-------|-------------|
| **Source** | Cognitive Science Literature | Adapted from human syllogistic reasoning studies |
| **Base Syllogisms** | 40 | Unique logical argument structures |
| **Variants per Syllogism** | 4 | N (normal), X (nonsense), O (order-switched), OX (combined) |
| **Total Instances** | 160 | 40 syllogisms × 4 variants |
| **Valid Syllogisms** | 76 (47.5%) | Conclusions logically follow from premises |
| **Invalid Syllogisms** | 84 (52.5%) | Conclusions do NOT logically follow |
| **Believable Conclusions** | 38 (23.8%) | Semantically plausible conclusions |
| **Unbelievable Conclusions** | 122 (76.2%) | Counter-intuitive or abstract conclusions |

#### Syllogism Structure

Each syllogism follows the classical form:

```
Premise 1: [Major premise - general statement]
Premise 2: [Minor premise - specific statement]
Conclusion: [Derived statement]
```

**Example (Valid Syllogism):**
```
Premise 1: All things that are smoked are bad for your health.
Premise 2: Cigarettes are smoked.
Conclusion: Therefore, cigarettes are bad for your health.

Ground Truth (Syntax): valid
Ground Truth (NLU): believable
```

**Example (Invalid Syllogism - Belief Bias Trap):**
```
Premise 1: All things with an engine need oil.
Premise 2: Cars need oil.
Conclusion: Therefore, cars have engines.

Ground Truth (Syntax): invalid (affirming the consequent fallacy)
Ground Truth (NLU): believable (conclusion sounds correct despite being logically invalid)
```

#### Variant Types

| Variant | Code | Description | Purpose |
|---------|------|-------------|---------|
| **Normal** | N | Original semantically meaningful predicates | Baseline performance with real-world content |
| **Nonsense** | X | Abstract/meaningless predicates (e.g., "blargs", "zimons") | Test pure logical reasoning without semantic interference |
| **Order-switched** | O | Premise order reversed (P2 before P1) | Test sensitivity to premise presentation order |
| **Combined** | OX | Nonsense predicates + reversed order | Combined robustness test |

**Nonsense Variant Example:**
```
Original (N):
  Premise 1: All calculators are machines.
  Premise 2: All computers are calculators.
  Conclusion: Therefore, some machines are not computers.

Nonsense (X):
  Premise 1: All blargs are zimons.
  Premise 2: All glorps are blargs.
  Conclusion: Therefore, some zimons are not glorps.
```

#### Dual Ground Truth System

Each syllogism has **two independent ground truths**:

| Ground Truth | Values | Description |
|--------------|--------|-------------|
| **Syntax** | `valid` / `invalid` | Logical validity based on formal syllogistic rules |
| **NLU** | `believable` / `unbelievable` | Semantic plausibility of the conclusion |

The LLM responds with `"correct"` or `"incorrect"`, mapped as:

| LLM Response | Syntax Mapping | NLU Mapping |
|--------------|----------------|-------------|
| `"correct"` | → `valid` | → `believable` |
| `"incorrect"` | → `invalid` | → `unbelievable` |

#### Ground Truth Distribution (Belief Bias Categories)

| Syntax | NLU | Count | % | Category | Difficulty |
|--------|-----|-------|---|----------|------------|
| Valid | Believable | 18 | 11.2% | **Congruent** | Easy |
| Valid | Unbelievable | 58 | 36.2% | **Incongruent** | Hard |
| Invalid | Believable | 20 | 12.5% | **Incongruent** (Belief Bias Trap) | **Hardest** |
| Invalid | Unbelievable | 64 | 40.0% | **Congruent** | Easy |

**Summary:**
- **Congruent (82 instances, 51.2%)**: Logic and intuition align—easier for models
- **Incongruent (78 instances, 48.8%)**: Logic and intuition conflict—tests true reasoning

---

### Prompting Strategies

We implement 4 prompting strategies to evaluate models under different conditions.

#### Strategy Comparison Table

| Strategy | Examples | Chain-of-Thought | Response Format |
|----------|----------|------------------|-----------------|
| `zero_shot` | 0 | ✗ | One word: "correct"/"incorrect" |
| `one_shot` | 1 | ✗ | One word: "correct"/"incorrect" |
| `few_shot` | 4 (2✓ + 2✗) | ✗ | One word: "correct"/"incorrect" |
| `zero_shot_cot` | 0 | ✓ | Step-by-step reasoning + answer |

#### Prompt Schema (Algorithmic Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ALGORITHM: LLM_Syllogism_Eval                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Syllogism S = {premise_1, premise_2, conclusion}                    │
│         Strategy ∈ {zero_shot, one_shot, few_shot, zero_shot_cot}           │
│         Temperature T ∈ {0.0, 0.5, 1.0}                                     │
│                                                                             │
│  STEP 1: Construct System Prompt                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ "You are an expert in syllogistic reasoning.                        │    │
│  │  Your task is to determine whether the conclusion of a given        │    │
│  │  syllogism follows from the premises.                               │    │
│  │                                                                     │    │
│  │  A syllogism is CORRECT if the conclusion follows from premises.    │    │
│  │  A syllogism is INCORRECT if the conclusion does not follow.        │    │
│  │                                                                     │    │
│  │  [IF zero_shot_cot]: Think through step by step.                    │    │
│  │  [ELSE]: Respond with exactly one word: 'correct' or 'incorrect'."  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  STEP 2: Construct User Prompt                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ [IF one_shot OR few_shot]: Include example(s)                       │    │
│  │                                                                     │    │
│  │ "Determine whether the following syllogism is correct or incorrect. │    │
│  │                                                                     │    │
│  │  Premise 1: {S.premise_1}                                           │    │
│  │  Premise 2: {S.premise_2}                                           │    │
│  │  Conclusion: {S.conclusion}                                         │    │
│  │                                                                     │    │
│  │  [IF zero_shot_cot]: Let's think step by step.                      │    │
│  │  [ELSE]: Respond with one word: 'correct' or 'incorrect'."          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  STEP 3: Query Model with Adaptive Stopping (see next section)              │
│                                                                             │
│  STEP 4: Parse Response                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ IF "incorrect" in response.lower() → prediction = "incorrect"       │    │
│  │ ELSE IF "correct" in response.lower() → prediction = "correct"      │    │
│  │ ELSE → prediction = "unclear"                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  OUTPUT: prediction ∈ {"correct", "incorrect", "unclear"}                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Few-Shot Examples Used

| # | Syllogism | Validity | Answer |
|---|-----------|----------|--------|
| 1 | All smoked things → bad for health; Cigarettes → smoked; ∴ Cigarettes → bad | Valid | `correct` |
| 2 | No furniture → attractive; Some tables → attractive; ∴ Some tables ≠ furniture | Valid | `correct` |
| 3 | All calculators → machines; All computers → calculators; ∴ Some machines ≠ computers | Invalid | `incorrect` |
| 4 | No screwdrivers → heavy; Some tools → heavy; ∴ Some screwdrivers ≠ tools | Invalid | `incorrect` |

---

### Adaptive Stopping Strategy

For non-deterministic sampling (temperature > 0), we implement an adaptive stopping strategy inspired by Hu et al. (EMNLP 2024).

#### Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ALGORITHM: Adaptive_Stopping                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PARAMETERS:                                                                │
│    max_iterations = 10                                                      │
│    threshold = 5                                                            │
│                                                                             │
│  IF temperature == 0.0:                                                     │
│      # Deterministic: single query                                          │
│      response = query_model(temperature=0.0)                                │
│      RETURN parse(response)                                                 │
│                                                                             │
│  ELSE:                                                                      │
│      # Stochastic: adaptive voting                                          │
│      correct_count = 0                                                      │
│      incorrect_count = 0                                                    │
│      responses = []                                                         │
│                                                                             │
│      FOR i = 1 TO max_iterations:                                           │
│          response = query_model(temperature)                                │
│          vote = parse(response)                                             │
│          responses.append(vote)                                             │
│                                                                             │
│          IF vote == "correct": correct_count += 1                           │
│          IF vote == "incorrect": incorrect_count += 1                       │
│                                                                             │
│          # Early stopping: check at iteration 5                             │
│          IF i == threshold:                                                 │
│              IF all(v == "correct" for v in responses[:5]):                 │
│                  RETURN "correct" (early stop)                              │
│              IF all(v == "incorrect" for v in responses[:5]):               │
│                  RETURN "incorrect" (early stop)                            │
│              # Otherwise continue to max_iterations                         │
│                                                                             │
│      # Final decision: majority vote                                        │
│      IF correct_count > incorrect_count: RETURN "correct"                   │
│      IF incorrect_count > correct_count: RETURN "incorrect"                 │
│      ELSE: RETURN "incorrect"  # Tie → conservative default                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Stopping Strategy Summary

| Temperature | Behavior | Max Queries |
|-------------|----------|-------------|
| `0.0` | Single deterministic query (greedy decoding) | 1 |
| `0.5`, `1.0` | Adaptive stopping with early termination | 5-10 |

---

### Evaluation Methods

#### Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / N | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Correctness of positive predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positives |
| **F1 Score** | 2 × P × R / (P + R) | Harmonic mean of P and R |

#### Dual Evaluation

Each prediction is evaluated against **both** ground truths:

| Evaluation | LLM "correct" maps to | LLM "incorrect" maps to | Ground Truth |
|------------|----------------------|------------------------|--------------|
| **Syntax Accuracy** | `valid` | `invalid` | `ground_truth_syntax` |
| **NLU Accuracy** | `believable` | `unbelievable` | `ground_truth_NLU` |

#### Belief Bias Metric

```
Belief Bias Effect = Accuracy_congruent - Accuracy_incongruent
```

Where:
- `Accuracy_congruent` = accuracy on (valid, believable) + (invalid, unbelievable)
- `Accuracy_incongruent` = accuracy on (valid, unbelievable) + (invalid, believable)

**Interpretation:**
- **Positive bias effect** → Model performs better when logic aligns with intuition
- **Negative bias effect** → Model performs better on counter-intuitive problems (unusual)
- **Zero bias effect** → Model uses pure logical reasoning

---

## Results

### Overall Performance

#### Syntax Accuracy by Model (T=0.0, Zero-Shot)

**Baseline performance at deterministic sampling with minimal prompting:**

| Rank | Model | Syntax Acc (%) | Classification |
|------|-------|---------------|----------------|
| 1 | gemini-2.5-flash | **100.00** | Top Tier |
| 2 | glm-4.6 | **100.00** | Top Tier |
| 3 | gpt-oss-20b | **100.00** | Top Tier |
| 4 | gemini-2.5-pro | **98.75** | Top Tier |
| 5 | deepseek-v3.1 | **97.50** | Top Tier |
| 6 | kimi-k2-instruct | **95.62** | Top Tier |
| 7 | gemini-2.5-flash-lite | **90.62** | High Tier |
| 8 | qwen3-next-80b-a3b-instruct | **84.38** | Mid Tier |
| 9 | qwen3-next-80b-a3b-thinking | **83.12** | Mid Tier |
| 10 | llama-3.3-70b-instruct | **69.38** | Lower Tier |
| 11 | gemma-3-27b-it | **68.12** | Lower Tier |
| 12 | llama-3.1-8b-instruct | **61.25** | Lower Tier |
| 13 | llama-3.2-3b-instruct | **57.50** | Lower Tier |
| 14 | llama-3.2-1b-instruct | **53.12** | Lower Tier |
| 15 | mixtral-8x22b-instruct | **52.50** | Lower Tier |

#### Average Syntax Accuracy by Model (Across All Configurations)

**Performance averaged across all temperatures (0.0, 0.5, 1.0) and strategies (zero_shot, one_shot, few_shot, zero_shot_cot):**

| Rank | Model | Mean Acc (%) | Std (%) | Min (%) | Max (%) |
|------|-------|-------------|---------|---------|---------|
| 1 | gemini-2.5-flash | **99.84** | 0.36 | 99.38 | 100.00 |
| 2 | gemini-2.5-pro | **99.38** | 0.52 | 98.12 | 100.00 |
| 3 | glm-4.6 | **98.96** | 1.58 | 95.62 | 100.00 |
| 4 | gpt-oss-20b | **97.86** | 1.77 | 93.75 | 100.00 |
| 5 | deepseek-v3.1 | **95.68** | 2.42 | 91.25 | 98.12 |
| 6 | kimi-k2-instruct | **95.31** | 1.89 | 91.25 | 98.12 |
| 7 | gemini-2.5-flash-lite | **87.66** | 4.12 | 78.12 | 93.12 |
| 8 | qwen3-next-80b-a3b-instruct | **77.92** | 6.13 | 66.88 | 86.88 |
| 9 | qwen3-next-80b-a3b-thinking | **73.54** | 7.84 | 58.75 | 86.25 |
| 10 | llama-3.3-70b-instruct | **69.22** | 3.96 | 61.88 | 76.88 |
| 11 | gemma-3-27b-it | **68.85** | 4.52 | 60.00 | 76.25 |
| 12 | llama-3.1-8b-instruct | **64.32** | 5.28 | 54.38 | 74.38 |
| 13 | llama-3.2-3b-instruct | **60.57** | 4.89 | 51.88 | 68.75 |
| 14 | llama-3.2-1b-instruct | **54.64** | 6.42 | 42.50 | 66.25 |
| 15 | mixtral-8x22b-instruct | **52.50** | 0.00 | 52.50 | 52.50 |

**Overall Statistics:**
- **Grand Mean Accuracy: 79.75%**
- **Between-Model Std: 17.89%**
- **Range: 52.50% - 99.84%**

#### Accuracy by Provider

| Provider | Mean Acc (%) | Std (%) | Models |
|----------|-------------|---------|--------|
| Google AI Studio | **88.93** | 14.18 | 4 |
| HuggingFace | **75.60** | 18.22 | 11 |

### Strategy Comparison

#### Syntax Accuracy by Prompting Strategy (T=0.0)

| Strategy | Mean (%) | Std (%) | Min (%) | Max (%) |
|----------|----------|---------|---------|---------|
| **one_shot** | **81.12** | 16.13 | 52.50 | 100.00 |
| zero_shot | 80.75 | 16.57 | 52.50 | 100.00 |
| zero_shot_cot | 79.67 | 19.47 | 52.50 | 98.75 |
| few_shot | 77.25 | 19.35 | 46.88 | 100.00 |

#### Syntax Accuracy by Strategy (Averaged Across All Temperatures)

| Strategy | Mean (%) | Std (%) | 95% CI Lower | 95% CI Upper |
|----------|----------|---------|--------------|--------------|
| **one_shot** | **80.42** | 17.21 | 75.87 | 84.97 |
| zero_shot | 79.86 | 17.62 | 75.22 | 84.50 |
| zero_shot_cot | 79.58 | 19.14 | 74.54 | 84.62 |
| few_shot | 79.14 | 18.65 | 74.23 | 84.05 |

**Key Finding:** One-shot prompting slightly outperforms other strategies, but the differences are not statistically significant after correction for multiple comparisons (see Statistical Tests section).

#### Strategy Effect Size Analysis

| Comparison | Cohen's d | Effect Size |
|------------|-----------|-------------|
| one_shot vs few_shot | 0.07 | Negligible |
| one_shot vs zero_shot | 0.03 | Negligible |
| one_shot vs zero_shot_cot | 0.05 | Negligible |
| zero_shot vs few_shot | 0.04 | Negligible |
| zero_shot vs zero_shot_cot | 0.02 | Negligible |
| few_shot vs zero_shot_cot | 0.02 | Negligible |

### Temperature Effects

#### Syntax Accuracy by Temperature (Averaged across all models and strategies)

| Temperature | Mean (%) | Std (%) | N Configurations |
|-------------|----------|---------|------------------|
| T=0.0 | 79.70 | 18.01 | 60 |
| T=0.5 | 79.76 | 18.81 | 60 |
| T=1.0 | 79.71 | 18.86 | 60 |

**Key Finding:** Temperature has **negligible effect** on accuracy (Δ < 0.1%). The adaptive stopping strategy effectively normalizes stochastic sampling.

#### Temperature Effect by Model Category

| Model Tier | T=0.0 (%) | T=0.5 (%) | T=1.0 (%) | Variance |
|------------|-----------|-----------|-----------|----------|
| Top Tier (>95%) | 97.71 | 97.68 | 97.65 | 0.00 |
| Mid Tier (70-95%) | 79.53 | 79.61 | 79.58 | 0.00 |
| Lower Tier (<70%) | 59.68 | 59.82 | 59.75 | 0.01 |

**Conclusion:** Temperature does not differentially affect models across performance tiers.

### Belief Bias Analysis

Belief bias occurs when semantic plausibility of a conclusion influences judgment of its logical validity.

#### Belief Bias Categories Explained

| Category | Logic | Intuition | Example | Expected Difficulty |
|----------|-------|-----------|---------|---------------------|
| **Congruent-Valid** | Valid | Believable | "Cigarettes are bad for health" | Easy ✓ |
| **Congruent-Invalid** | Invalid | Unbelievable | "Some fish are not birds" (obvious) | Easy ✓ |
| **Incongruent-Valid** | Valid | Unbelievable | "Some dogs are not pets" (valid but sounds wrong) | Hard ✗ |
| **Incongruent-Invalid** | Invalid | Believable | "Cars have engines" (true but not logical from premises) | **Hardest** ✗✗ |

#### Belief Bias Effect by Model (T=0.0, Zero-Shot)

| Model | Congruent (%) | Incongruent (%) | Bias Effect | Category |
|-------|---------------|-----------------|-------------|----------|
| mixtral-8x22b-instruct | 78.05 | 25.64 | **+52.41%** | Severe Bias |
| llama-3.2-3b-instruct | 82.01 | 39.10 | **+42.91%** | Severe Bias |
| llama-3.3-70b-instruct | 84.76 | 52.24 | **+32.51%** | High Bias |
| qwen3-next-80b-a3b-thinking | 85.37 | 58.65 | **+26.71%** | High Bias |
| llama-3.2-1b-instruct | 63.72 | 44.23 | **+19.49%** | Moderate Bias |
| gemini-2.5-flash-lite | 94.21 | 81.41 | **+12.80%** | Moderate Bias |
| llama-3.1-8b-instruct | 69.51 | 58.01 | **+11.50%** | Moderate Bias |
| kimi-k2-instruct | 99.39 | 90.71 | **+8.69%** | Low Bias |
| deepseek-v3.1 | 100.00 | 91.35 | **+8.65%** | Low Bias |
| glm-4.6 | 100.00 | 97.44 | **+2.56%** | Minimal Bias |
| gemini-2.5-pro | 100.00 | 98.40 | **+1.60%** | Minimal Bias |
| gpt-oss-20b | 98.78 | 97.44 | **+1.34%** | Minimal Bias |
| gemini-2.5-flash | 100.00 | 98.72 | **+1.28%** | Minimal Bias |
| qwen3-next-80b-a3b-instruct | 75.00 | 83.33 | **-8.33%** | Reverse Bias* |
| gemma-3-27b-it | 62.50 | 75.96 | **-13.46%** | Reverse Bias* |

*Reverse bias indicates counter-intuitive pattern where models perform better on incongruent problems

#### Belief Bias Summary Statistics

| Metric | Value |
|--------|-------|
| **Overall Mean Bias Effect** | +13.38% |
| **Median Bias Effect** | +8.65% |
| **Standard Deviation** | 17.23% |
| **Models with Positive Bias** | 13/15 (87%) |
| **Models with Negative Bias** | 2/15 (13%) |
| **Models with Minimal Bias (<5%)** | 5/15 (33%) |

#### Statistical Significance of Belief Bias

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Paired t-test (Cong. vs Incong.) | t = 4.82 | <0.0001 | Significant |
| Wilcoxon signed-rank | W = 89 | 0.0012 | Significant |
| Effect Size (Cohen's d) | d = 0.72 | — | Medium-Large |

**Key Finding:** Belief bias is statistically significant across models. The effect size (d = 0.72) indicates a practically meaningful performance gap between problems where logic aligns vs. conflicts with intuition.

### Variant Analysis

#### Accuracy by Syllogism Variant (Averaged across all configurations)

| Variant | Mean Acc (%) | Std (%) | Description |
|---------|-------------|---------|-------------|
| **X** (Nonsense) | **79.75** | 17.84 | Abstract predicates |
| **OX** (Combined) | **79.72** | 17.86 | Nonsense + reversed |
| **N** (Normal) | 79.77 | 17.92 | Meaningful content |
| **O** (Order-switched) | 79.76 | 17.91 | Reversed premise order |

**Observation:** Variant type has minimal impact on overall accuracy when averaged across all models and configurations.

#### Variant Accuracy by Model (T=0.0, Zero-Shot)

| Model | N (%) | O (%) | X (%) | OX (%) | Best Variant |
|-------|-------|-------|-------|--------|--------------|
| gemini-2.5-flash | 100.0 | 100.0 | 100.0 | 100.0 | All Equal |
| gemini-2.5-pro | 100.0 | 97.5 | 100.0 | 97.5 | N, X |
| glm-4.6 | 100.0 | 100.0 | 100.0 | 100.0 | All Equal |
| gpt-oss-20b | 100.0 | 100.0 | 100.0 | 100.0 | All Equal |
| deepseek-v3.1 | 100.0 | 97.5 | 95.0 | 97.5 | N |
| kimi-k2-instruct | 97.5 | 92.5 | 97.5 | 95.0 | N, X |
| gemini-2.5-flash-lite | 92.5 | 87.5 | 92.5 | 90.0 | N, X |
| qwen3-next-80b-a3b-instruct | 82.5 | 85.0 | 87.5 | 82.5 | X |
| qwen3-next-80b-a3b-thinking | 85.0 | 77.5 | 90.0 | 80.0 | X |
| llama-3.3-70b-instruct | 70.0 | 72.5 | 70.0 | 65.0 | O |
| gemma-3-27b-it | 65.0 | 65.0 | 70.0 | 72.5 | OX |
| llama-3.1-8b-instruct | 57.5 | 60.0 | 65.0 | 62.5 | X |
| llama-3.2-3b-instruct | 57.5 | 57.5 | 57.5 | 57.5 | All Equal |
| llama-3.2-1b-instruct | 42.5 | 50.0 | 60.0 | 60.0 | X, OX |
| mixtral-8x22b-instruct | 52.5 | 52.5 | 52.5 | 52.5 | All Equal |

#### Variant Correlation Analysis

| Variant Pair | Pearson r | p-value | Interpretation |
|--------------|-----------|---------|----------------|
| N vs X | **0.9201** | <0.0001 | Very strong positive |
| N vs O | **0.9478** | <0.0001 | Very strong positive |
| N vs OX | **0.9312** | <0.0001 | Very strong positive |
| O vs X | **0.9156** | <0.0001 | Very strong positive |
| O vs OX | **0.9401** | <0.0001 | Very strong positive |
| X vs OX | **0.9523** | <0.0001 | Very strong positive |

**Key Finding:** All variant pairs show very strong positive correlations (r > 0.91), indicating that models perform consistently across variant types. A model that performs well on normal syllogisms tends to perform well on all variants.

#### Content Type Effect (Semantic vs. Nonsense)

| Content Type | Variants | Mean Acc (%) | Std (%) |
|--------------|----------|-------------|---------|
| **Semantic** | N, O | 79.76 | 17.91 |
| **Nonsense** | X, OX | 79.74 | 17.85 |
| **Difference** | — | **0.02%** | — |

**Statistical Test (Paired t-test):**
- t-statistic: 0.12
- p-value: 0.905
- **Conclusion:** No significant difference between semantic and nonsense content (contradicts H6)

### Benchmark Correlations

#### LMArena Correlation

LMArena ranks models based on human preference in conversations.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Spearman ρ** | **-0.744** | Strong negative correlation |
| **p-value** | **0.0015** | Highly significant |
| **N** | 14 | Models with LMArena rank |
| **Pearson r** | -0.712 | Strong linear relationship |

**Interpretation:** Lower LMArena rank (= better model) correlates with higher syllogistic reasoning accuracy. Models that are better at following human instructions and providing quality responses also tend to reason more logically.

#### MMLU Correlation

MMLU tests factual knowledge across 57 subjects.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Spearman ρ** | 0.288 | Weak positive correlation |
| **p-value** | **0.364** | **NOT significant** |
| **N** | 12 | Models with MMLU scores |
| **Pearson r** | 0.314 | Weak linear relationship |

**Critical Inference:** The lack of correlation between MMLU and syllogistic reasoning is a key finding:

1. **MMLU measures factual knowledge recall** — knowing facts from science, history, law, etc.
2. **Syllogistic reasoning measures formal logical deduction** — applying inference rules independent of content
3. **These are distinct cognitive capabilities** that do not transfer automatically
4. **High MMLU ≠ Good logical reasoning**: A model can know many facts but fail to apply valid inference rules
5. **Implication for benchmarking**: Current LLM evaluations overemphasize factual knowledge; logical reasoning capability should be tested separately

#### Correlation Summary Table

| Benchmark | Correlation | p-value | Significant | Measures |
|-----------|-------------|---------|-------------|----------|
| LMArena | ρ = -0.744 | 0.0015 | ✅ Yes | Instruction-following quality |
| MMLU | ρ = 0.288 | 0.3637 | ❌ No | Factual knowledge breadth |

**Conclusion:** Syllogistic reasoning correlates with conversational/instruction quality (LMArena) but NOT with factual knowledge (MMLU). This suggests that logical reasoning is more related to careful instruction processing than encyclopedic knowledge.

### Statistical Tests

#### Understanding the Test Hierarchy

We use two complementary tests for strategy comparison that answer **different questions**:

| Test | Unit of Analysis | N | Question Answered |
|------|------------------|---|-------------------|
| **McNemar** | Individual instances | 2400 | "Do strategies solve *different* syllogisms?" |
| **Wilcoxon** | Model averages | 15 | "Does one strategy *consistently* outperform across models?" |

**Why results differ:** McNemar detects instance-level changes (high power, N=2400), while Wilcoxon tests for consistent directional effects across models (low power, N=15). A strategy can change *which* problems are solved without *consistently* improving performance.

#### Normality Tests (Shapiro-Wilk)

Before choosing parametric vs. non-parametric tests, we assessed normality of accuracy distributions:

| Distribution | W Statistic | p-value | Normal? |
|--------------|-------------|---------|---------|
| Overall Accuracy | 0.923 | 0.0012 | ❌ No |
| By Strategy (zero_shot) | 0.917 | 0.0031 | ❌ No |
| By Strategy (one_shot) | 0.921 | 0.0024 | ❌ No |
| By Strategy (few_shot) | 0.912 | 0.0048 | ❌ No |
| By Strategy (zero_shot_cot) | 0.908 | 0.0061 | ❌ No |

**Conclusion:** Accuracy data is not normally distributed. Non-parametric tests are appropriate.

#### McNemar's Test (Pairwise Strategy Comparison)

Tests whether strategy pairs produce significantly different error patterns **at the instance level**. Each instance (syllogism × model) is classified as correct/incorrect under each strategy, and McNemar tests if the off-diagonal cells (disagreements) are balanced.

| Comparison | S1 Better | S2 Better | χ² | p-value | Bonferroni p | Significant |
|------------|-----------|-----------|-----|---------|--------------|-------------|
| zero_shot vs few_shot | 274 | 190 | 14.85 | 0.0001 | 0.0006 | ✅ Yes |
| one_shot vs few_shot | 229 | 136 | 23.19 | <0.0001 | <0.0006 | ✅ Yes |
| few_shot vs zero_shot_cot | 249 | 307 | 5.84 | 0.0156 | 0.0942 | ❌ No* |
| zero_shot vs one_shot | 90 | 99 | 0.34 | 0.5606 | 1.0000 | ❌ No |
| zero_shot vs zero_shot_cot | 169 | 143 | 2.00 | 0.1570 | 0.9420 | ❌ No |
| one_shot vs zero_shot_cot | 188 | 153 | 3.39 | 0.0656 | 0.3936 | ❌ No |

*After Bonferroni correction (α = 0.05/6 = 0.0083)

**Interpretation:** Few-shot prompting causes models to fail on 84 more instances than zero-shot (274 vs 190). This is a significant redistribution of errors, not just noise.

#### Wilcoxon Signed-Rank Test (Strategy Pairs, Model-Level)

Tests whether one strategy **consistently** outperforms another **across models**. Each model contributes one data point (its average accuracy under each strategy).

| Comparison | W Statistic | p-value | p (Holm) | Significant |
|------------|-------------|---------|----------|-------------|
| few_shot vs one_shot | 17.5 | 0.0914 | 0.5486 | ❌ No |
| few_shot vs zero_shot | 21.0 | 0.1575 | 0.7873 | ❌ No |
| few_shot vs zero_shot_cot | 49.0 | 0.8260 | 0.8260 | ❌ No |
| one_shot vs zero_shot | 33.0 | 0.3807 | 1.0000 | ❌ No |
| one_shot vs zero_shot_cot | 45.0 | 0.6377 | 1.0000 | ❌ No |
| zero_shot vs zero_shot_cot | 37.0 | 0.5520 | 1.0000 | ❌ No |

**Interpretation:** Although few-shot changes error patterns (McNemar), it doesn't *consistently* help or hurt across models. Some models improve with few-shot, others worsen—the net effect is zero.

#### Reconciling McNemar vs. Wilcoxon Results

| Comparison | McNemar | Wilcoxon | Interpretation |
|------------|---------|----------|----------------|
| zero_shot vs few_shot | ✅ Sig. | ❌ Not Sig. | Few-shot changes *which* problems are solved, but not *how many* consistently |
| one_shot vs few_shot | ✅ Sig. | ❌ Not Sig. | Same pattern—error redistribution without consistent benefit |
| Others | ❌ | ❌ | Strategies produce similar error patterns and similar overall accuracy |

**Practical Conclusion:** Prompting strategies significantly alter error patterns at the instance level (McNemar: p < 0.001), but these changes do not translate to consistent performance improvements across models (Wilcoxon: p > 0.05 after Holm correction). This suggests that strategy effects are model-specific rather than universal.

#### Friedman Test (Strategy Effect Across Models)

Non-parametric repeated measures test for strategy effect.

| Test | Statistic | df | p-value | Significant |
|------|-----------|----|---------| ------------|
| Friedman χ² | 3.24 | 3 | 0.3562 | ❌ No |

**Conclusion:** No significant overall strategy effect across models.

#### Kruskal-Wallis Test (Temperature Effect)

Non-parametric one-way ANOVA across temperature settings.

| Test | H Statistic | df | p-value | Significant |
|------|-------------|----|---------| ------------|
| Temperature Effect | 0.0012 | 2 | 0.9994 | ❌ No |

**Conclusion:** Temperature has no significant effect on accuracy.

#### Mann-Whitney U Test (Syntax vs NLU Accuracy)

Comparing the two ground truth evaluations.

| Comparison | U Statistic | p-value | Effect Size (r) |
|------------|-------------|---------|-----------------|
| Syntax vs NLU | 10847.5 | <0.0001 | 0.53 (Large) |

**Conclusion:** Syntax accuracy (79.75%) is significantly higher than NLU accuracy (57.48%) with large effect size.

#### Paired t-test (Congruent vs. Incongruent)

Testing belief bias effect.

| Test | t Statistic | df | p-value | Cohen's d |
|------|-------------|----|---------|-----------|
| Congruent - Incongruent | 4.82 | 179 | <0.0001 | 0.72 (Medium-Large) |

**Conclusion:** Models perform significantly better on congruent problems (mean +13.38%), confirming belief bias.

### Consistency Analysis

Consistency measures how reliably a model gives the same answer across different content variants of logically equivalent syllogisms.

#### Consistency by Model (T=0.0, Zero-Shot)

| Model | Overall (%) | N vs X (%) | O vs OX (%) |
|-------|-------------|------------|-------------|
| gemini-2.5-flash | **100.0** | 100.0 | 100.0 |
| glm-4.6 | **100.0** | 100.0 | 100.0 |
| gpt-oss-20b | **100.0** | 100.0 | 100.0 |
| mixtral-8x22b-instruct | **100.0** | 100.0 | 100.0 |
| llama-3.2-3b-instruct | **100.0** | 100.0 | 100.0 |
| gemini-2.5-pro | 98.75 | 97.5 | 100.0 |
| deepseek-v3.1 | 96.88 | 95.0 | 98.75 |
| kimi-k2-instruct | 95.62 | 97.5 | 93.75 |
| gemini-2.5-flash-lite | 95.00 | 95.0 | 95.0 |
| qwen3-next-80b-a3b-instruct | 93.12 | 90.0 | 96.25 |
| qwen3-next-80b-a3b-thinking | 91.25 | 87.5 | 95.0 |
| llama-3.3-70b-instruct | 90.62 | 92.5 | 88.75 |
| llama-3.1-8b-instruct | 86.88 | 87.5 | 86.25 |
| gemma-3-27b-it | 86.25 | 87.5 | 85.0 |
| llama-3.2-1b-instruct | 78.12 | 77.5 | 78.75 |

**Key Observation:** High consistency does NOT imply high accuracy. Mixtral-8x22b-instruct shows 100% consistency but only 52.5% accuracy—it consistently gives the same (wrong) answer.

#### Average Consistency by Model (All Configurations)

| Model | Mean Consistency (%) | Std (%) |
|-------|---------------------|---------|
| gemini-2.5-flash | **99.90** | 0.31 |
| mixtral-8x22b-instruct | **99.79** | 0.65 |
| glm-4.6 | **98.85** | 1.42 |
| gpt-oss-20b | **98.44** | 1.67 |
| gemini-2.5-pro | **98.12** | 1.23 |
| deepseek-v3.1 | **95.94** | 2.35 |
| kimi-k2-instruct | **94.48** | 2.89 |
| gemini-2.5-flash-lite | **92.71** | 3.56 |
| llama-3.2-3b-instruct | **91.56** | 4.12 |
| qwen3-next-80b-a3b-instruct | **89.27** | 5.23 |
| llama-3.3-70b-instruct | **88.02** | 4.78 |
| qwen3-next-80b-a3b-thinking | **85.42** | 6.34 |
| gemma-3-27b-it | **84.90** | 5.67 |
| llama-3.1-8b-instruct | **83.44** | 5.89 |
| llama-3.2-1b-instruct | **76.35** | 7.42 |

#### Accuracy-Consistency Relationship

| Metric | Value |
|--------|-------|
| Pearson r | 0.71 |
| p-value | 0.0032 |

**Finding:** Moderate positive correlation between accuracy and consistency. High-performing models tend to be more consistent, but consistency alone is not sufficient for good performance.

#### Bootstrap Confidence Intervals (95% CI)

Model accuracy confidence intervals from 10,000 bootstrap resamples:

| Model | Mean (%) | 95% CI Lower | 95% CI Upper | CI Width |
|-------|----------|--------------|--------------|----------|
| gemini-2.5-flash | 99.84 | 99.38 | 100.00 | 0.62 |
| gemini-2.5-pro | 99.38 | 98.54 | 100.00 | 1.46 |
| glm-4.6 | 98.96 | 97.08 | 100.00 | 2.92 |
| gpt-oss-20b | 97.86 | 95.73 | 99.58 | 3.85 |
| deepseek-v3.1 | 95.68 | 93.02 | 97.92 | 4.90 |
| kimi-k2-instruct | 95.31 | 92.81 | 97.50 | 4.69 |
| gemini-2.5-flash-lite | 87.66 | 83.33 | 91.56 | 8.23 |
| qwen3-next-80b-a3b-instruct | 77.92 | 72.19 | 83.23 | 11.04 |
| qwen3-next-80b-a3b-thinking | 73.54 | 66.88 | 79.79 | 12.91 |
| llama-3.3-70b-instruct | 69.22 | 64.90 | 73.33 | 8.43 |
| gemma-3-27b-it | 68.85 | 64.27 | 73.23 | 8.96 |
| llama-3.1-8b-instruct | 64.32 | 59.48 | 69.06 | 9.58 |
| llama-3.2-3b-instruct | 60.57 | 55.94 | 65.10 | 9.16 |
| llama-3.2-1b-instruct | 54.64 | 48.23 | 61.04 | 12.81 |
| mixtral-8x22b-instruct | 52.50 | 52.50 | 52.50 | 0.00 |

**Observation:** Higher-performing models have narrower confidence intervals, indicating more consistent performance.

---

## Discussion

### Addressing the Research Questions

#### RQ1: How accurately can LLMs perform categorical syllogistic reasoning?

**Answer:** Performance varies dramatically across models (52.5% - 99.8%). Top-tier models (Gemini 2.5 Flash/Pro, GLM-4.6, GPT-OSS-20B) achieve near-perfect syntax accuracy (>97%), while lower-tier models (Mixtral, smaller Llamas) perform near chance level. The mean accuracy of 79.75% masks a bimodal distribution—models either "get" syllogistic reasoning or struggle significantly with it.

#### RQ2: Do LLMs exhibit belief bias?

**Answer:** Yes, emphatically. 87% of models (13/15) show positive belief bias, with an average effect of +13.38 percentage points. This means models are systematically worse at judging syllogisms where logical validity conflicts with intuitive plausibility. Critically, top-performing models show minimal bias (<3%), suggesting that robust logical reasoning and bias resistance co-occur.

#### RQ3: How do prompting strategies affect performance?

**Answer:** Minimally and inconsistently. One-shot prompting marginally outperforms others (80.42% vs 79.14-79.86%), but no differences survive multiple comparison correction. Surprisingly, few-shot prompting—typically beneficial for many tasks—performed worst. This suggests syllogistic reasoning may not benefit from in-context examples in the same way that arithmetic or factual recall does.

#### RQ4: Does temperature significantly impact accuracy?

**Answer:** No. With adaptive stopping, temperature has zero measurable effect on accuracy (79.70-79.76% across T=0.0, 0.5, 1.0). The voting mechanism effectively filters out stochastic noise, making the final classification deterministic regardless of sampling temperature.

#### RQ5: Do models perform differently on semantic vs. nonsense syllogisms?

**Answer:** No significant difference. Variant correlations are extremely high (r > 0.91 across all pairs), indicating that models perform consistently regardless of whether predicates are meaningful ("dogs," "engines") or abstract ("blargs," "zimons"). This undermines the hypothesis that semantic grounding helps reasoning—and also undermines the hypothesis that it hurts. Models simply apply the same (correct or incorrect) reasoning process regardless of content.

#### RQ6: Does syllogistic reasoning correlate with existing benchmarks?

**Answer:** Partially. Strong correlation with LMArena (ρ = -0.74, p = 0.001) suggests that instruction-following quality predicts logical reasoning ability. However, **no correlation with MMLU** (ρ = 0.29, p = 0.36) indicates that factual knowledge breadth does NOT predict syllogistic reasoning. This is a crucial finding for LLM evaluation—current benchmarks may miss fundamental reasoning capabilities.

### Theoretical Implications

#### 1. LLMs Can Perform Formal Logical Reasoning (Sometimes)

The near-perfect performance of top models demonstrates that transformer architectures are capable of learning valid syllogistic inference rules. However, this capability is not universal—training methodology, architecture, and scale all influence whether this capability emerges.

#### 2. Belief Bias Reveals Semantic Interference

The robust belief bias effect suggests that even in "reasoning" tasks, LLMs are influenced by their language model priors. When a conclusion "sounds right" based on training data distributions, models are more likely to accept it as logically valid. This mirrors human cognitive biases documented since Evans et al. (1983).

#### 3. Knowledge ≠ Reasoning

The dissociation between MMLU (no correlation) and LMArena (strong correlation) suggests a crucial distinction:
- **MMLU tests factual recall**: "What is the capital of France?"
- **Syllogistic reasoning tests inference**: "Given A→B and B→C, does A→C?"

LLMs can accumulate encyclopedic knowledge without developing robust inference capabilities. Conversely, good instruction-following (LMArena) may require the same careful attention to logical structure that syllogistic reasoning demands.

#### 4. Prompting Limitations for Formal Reasoning

The failure of few-shot prompting to improve (and possibly worsen) performance is notable. Possible explanations:
- Few-shot examples may introduce irrelevant patterns
- Syllogistic reasoning requires rule application, not pattern matching
- In-context examples may encourage "similar surface" matching rather than structural analysis

---

## Key Findings

### 1. Model Performance Hierarchy

- **Top Tier (>95%)**: gemini-2.5-flash (99.84%), gemini-2.5-pro (99.38%), glm-4.6 (98.96%), gpt-oss-20b (97.86%), deepseek-v3.1 (95.68%), kimi-k2-instruct (95.31%)
- **Mid Tier (70-95%)**: gemini-2.5-flash-lite (87.66%), qwen3-next-80b-a3b-instruct (77.92%), qwen3-next-80b-a3b-thinking (73.54%)
- **Lower Tier (<70%)**: llama-3.3-70b-instruct (69.22%), gemma-3-27b-it (68.85%), llama-3.1-8b-instruct (64.32%), llama-3.2-3b-instruct (60.57%), llama-3.2-1b-instruct (54.64%), mixtral-8x22b-instruct (52.50%)

**Key Insight:** Performance gap between top and bottom tier is massive (47+ percentage points), suggesting syllogistic reasoning capability varies dramatically across model families.

### 2. Belief Bias is Pervasive and Substantial

- **87% of models** (13/15) exhibit positive belief bias
- **Average bias effect: +13.38 percentage points**
- **Strongest bias**: mixtral-8x22b-instruct (+52.41%) — near-random performance on incongruent problems
- **Most resistant**: gemini-2.5-flash (+1.28%) — nearly bias-free logical reasoning
- **Interpretation**: Most LLMs conflate semantic plausibility with logical validity, especially when faced with counter-intuitive but valid conclusions

### 3. Syntax Accuracy >> NLU Accuracy

- **Syntax Accuracy: 79.75%** (predicting valid/invalid)
- **NLU Accuracy: 57.48%** (predicting believable/unbelievable)
- **Gap: 22.27 percentage points** (statistically significant, p < 0.0001)
- **Interpretation**: LLMs are better at assessing logical structure than semantic plausibility—they may default to "valid" judgments

### 4. Prompting Strategy Has Minimal Impact

- **One-shot marginally best** (80.42% mean)
- **Few-shot underperforms** (79.14% mean) — contrary to common assumptions
- **Chain-of-thought provides no consistent benefit** (79.58% mean)
- **No significant differences after multiple comparison correction**
- **Interpretation**: Unlike arithmetic or factual tasks, syllogistic reasoning may not benefit from in-context examples or explicit reasoning traces

### 5. Temperature Has Zero Effect

- Adaptive stopping completely normalizes stochastic sampling
- **T=0.0 vs T=0.5 vs T=1.0: identical mean accuracy (79.70-79.76%)**
- Kruskal-Wallis H = 0.0012, p = 0.9994
- **Interpretation**: For classification tasks with voting, temperature is irrelevant to final accuracy

### 6. Semantic Content May Impair Rather Than Help

- Nonsense variants (X, OX) achieve **slightly higher accuracy** than normal variants (N, O)
- However, difference is not statistically significant (p = 0.905)
- All variant pairs show very strong correlation (r > 0.91)
- **Interpretation**: Models that can reason correctly do so regardless of content; those that cannot are not helped by meaningful content

### 7. Model Architecture and Training Trump Size

| Model | Parameters | Accuracy |
|-------|------------|----------|
| mixtral-8x22b-instruct | 141B MoE | 52.50% |
| gpt-oss-20b | 20B | 97.86% |
| llama-3.3-70b-instruct | 70B | 69.22% |
| llama-3.2-1b-instruct | 1B | 54.64% |

**Key Finding:** Parameter count poorly predicts syllogistic reasoning ability. Training methodology and architecture matter more.

### 8. Syllogistic Reasoning ≠ General Knowledge/Intelligence

#### LMArena Correlation (Chat/Instruction Quality)
- **Spearman ρ = -0.744**, p = 0.0015
- Strong correlation: better chat models → better syllogistic reasoning
- **Interpretation**: Quality of instruction-following correlates with logical reasoning

#### MMLU Correlation (Factual Knowledge)
- **Spearman ρ = 0.288**, p = 0.364 (NOT SIGNIFICANT)
- No correlation: MMLU score does NOT predict syllogistic reasoning
- **Interpretation**: Categorical logical reasoning requires specialized capabilities distinct from broad factual knowledge. MMLU benchmarks world knowledge recall, not formal deductive reasoning.

### 9. Consistency ≠ Accuracy

- Mixtral-8x22b shows 100% consistency but only 52.5% accuracy
- **Interpretation**: Some models are "consistently wrong"—they systematically apply incorrect reasoning rules
- High consistency is necessary but not sufficient for good performance

### 10. Google AI Studio Models Dominate

| Provider | Mean Accuracy | Best Model | Worst Model |
|----------|--------------|------------|-------------|
| Google AI Studio | **88.93%** | gemini-2.5-flash (99.84%) | gemma-3-27b-it (68.85%) |
| HuggingFace | **75.60%** | glm-4.6 (98.96%) | mixtral-8x22b-instruct (52.50%) |

**Note:** glm-4.6 and gpt-oss-20b (both HuggingFace) match Google's top performers, indicating provider is not the determining factor.

---

## Limitations

### 1. Dataset Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Size**: 160 instances | May limit statistical power for fine-grained analyses | Multiple testing corrections applied; bootstrap CIs computed |
| **Scope**: Only categorical syllogisms | Does not generalize to propositional, predicate, or modal logic | Focused study design; future work planned |
| **Complexity**: Basic syllogistic figures only | Does not test sorites, complex chained reasoning | Baseline establishment; extension possible |
| **Source**: Cognitive science adaptations | May not represent full diversity of real-world reasoning | Standard stimuli enable comparison with human studies |
| **Language**: English only | No multilingual evaluation | Prioritized depth over breadth |

### 2. Model Access Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **API-only evaluation** | No access to model internals (attention, hidden states) | Black-box evaluation matches deployment reality |
| **Provider routing** | HuggingFace `:cheapest` routing may introduce variability | Multiple runs with adaptive voting |
| **Version changes** | API models may be updated between experiments | Experiments run in short time window |
| **Rate limits** | Some models may have different API behaviors | Retry logic implemented |
| **Missing models** | Could not test GPT-4, Claude, Llama 3.2 90B | Focused on accessible, reproducible models |

### 3. Methodological Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Binary classification** | Forces nuanced reasoning into correct/incorrect | Standard approach for syllogism evaluation |
| **Response parsing** | Simple keyword matching may miss edge cases | CoT responses parsed for final answer |
| **No error categorization** | Did not classify fallacy types (undistributed middle, affirming consequent, etc.) | Future work planned |
| **Fixed prompts** | No prompt optimization or engineering | Standardized prompts enable fair comparison |
| **No chain-of-thought analysis** | Did not examine reasoning quality in CoT responses | Focused on accuracy outcomes |

### 4. Statistical Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Multiple comparisons** | Risk of Type I errors despite corrections | Conservative corrections applied (Bonferroni, Holm) |
| **Non-independence** | Same models tested across conditions | Repeated measures designs used |
| **Small N for correlations** | Only 12-14 models with benchmark scores | Non-parametric tests used |
| **Missing data** | Not all models have LMArena/MMLU scores | Analyses restricted to available data |
| **Single evaluation** | No repeated measurements for reliability | Adaptive voting provides implicit replication |

### 5. Generalization Limitations

| Limitation | Impact | Future Work |
|------------|--------|-------------|
| **English only** | Unknown cross-linguistic performance | Multilingual extension planned |
| **Base/IT models only** | No fine-tuned logic specialists | Could test logic-tuned models |
| **No human baseline** | Cannot directly compare to human performance | Human study could be conducted |
| **Static dataset** | No adaptive testing | Could develop computerized adaptive testing |

### 6. Scope Limitations

This study specifically tests **categorical syllogistic reasoning**, which is a narrow but fundamental form of logical reasoning. Our findings may not generalize to:

- **Propositional logic** (if-then reasoning)
- **Predicate logic** (quantified statements)
- **Modal logic** (necessity/possibility)
- **Defeasible reasoning** (non-monotonic logic)
- **Probabilistic reasoning** (Bayesian inference)
- **Causal reasoning** (counterfactuals)

However, categorical syllogisms provide a rigorous baseline for assessing whether LLMs can perform formal deductive inference independent of semantic content.

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- Google AI Studio API key (for Gemini models)
- HuggingFace API token with Inference API access

### Installation

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

### Configuration

Edit `config.toml`:

```toml
[api_keys]
google_api_key = "your-gemini-api-key"
huggingface_api_key = "your-hf-api-key"

[api_settings]
max_tokens = 12000
top_p = 1.0
```

### Running Experiments

```bash
# Run all models
python scripts/run_experiments.py

# Run specific models
python scripts/run_experiments.py --models "gemini-2.5-pro,llama-3.3-70b-instruct"

# Run with specific strategies and temperatures
python scripts/run_experiments.py \
    --models "gemini-2.5-flash" \
    --strategies "zero_shot,few_shot" \
    --temperatures "0.0,0.5"

# Resume interrupted experiments
python scripts/run_experiments.py --resume

# Dry run (preview without execution)
python scripts/run_experiments.py --dry-run
```

### Analyzing Results

```bash
# Generate metrics and analysis
python scripts/analyze_results.py

# Generate publication figures
python scripts/generate_revamped_figures.py
```

---

## Project Structure

```
Logic_in_LLMs/
├── data/                                 # Input datasets
│   ├── syllogisms_master_dataset.json    # 160 syllogism instances
│   ├── syllogisms_main_summary.json/.csv # Summary statistics
│   ├── LMarena_benchmark.csv             # LMArena rankings
│   └── MMLU_helm.csv                     # MMLU benchmark scores
│
├── results/                              # Experimental outputs
│   ├── raw_responses/                    # Raw API responses (168 files: 14 models × 4 strategies × 3 temps)
│   │   ├── temperature_0.0/
│   │   ├── temperature_0.5/
│   │   └── temperature_1.0/
│   └── analysis/
│       ├── tables/                       # Generated analysis tables (CSV)
│       │   ├── paper_table1_complete.csv
│       │   ├── paper_table2_dual_eval.csv
│       │   ├── paper_table3_belief_bias.csv
│       │   └── stats_*.csv               # Statistical test results
│       └── figures/                      # Generated visualizations
│           ├── paper_figures_14_models/  # Publication-ready figures (14 models)
│           ├── static/                   # PNG/PDF exports
│           └── plotly/                   # Interactive HTML figures
│
├── scripts/                              # Executable scripts (organized by purpose)
│   ├── experiments/
│   │   ├── run_experiments.py            # Main experiment runner
│   │   └── test_small_scale.py           # Small-scale testing utility
│   ├── analysis/
│   │   ├── generate_tables.py            # Generate paper tables from raw data
│   │   ├── run_statistical_tests.py      # Comprehensive statistical analysis
│   │   └── calculate_benchmark_correlations.py  # Benchmark correlation analysis
│   ├── visualization/
│   │   ├── generate_figures.py           # Main figure generation (12 figures)
│   │   ├── generate_figures_14_models.py # Paper figures (4 specific figures)
│   │   └── analyze_results.py            # Results analysis orchestrator
│   └── utilities/
│       ├── check_missing_runs.py         # Validate experimental completeness
│       └── validate_results.py           # Statistical validation utilities
│
├── src/                                  # Core source code
│   ├── config.py                         # Configuration management
│   ├── inference/                        # Model interaction & API clients
│   │   ├── model_registry.py             # Model definitions
│   │   ├── api_clients.py                # API wrappers (Google, HuggingFace)
│   │   ├── batch_processing.py           # Batch experiment orchestration
│   │   └── stopping_strategy.py          # Response parsing & extraction
│   ├── evaluation/                       # Metrics calculation
│   │   ├── calculate_metrics.py          # Accuracy, precision, recall, F1
│   │   ├── consistency_analysis.py       # Cross-variant consistency
│   │   ├── parse_responses.py            # Response parsing utilities
│   │   └── instance_sufficiency.py       # Dataset sufficiency analysis
│   ├── analysis/                         # Statistical analysis & visualization
│   │   ├── statistical_tests.py          # Statistical testing functions
│   │   ├── correlation.py                # Correlation analysis
│   │   ├── ranking.py                    # Model ranking
│   │   ├── variant_correlation.py        # Variant-specific correlations
│   │   └── visualization.py              # Figure generation
│   └── prompts/                          # Prompting strategies
│       ├── zero_shot.py
│       ├── one_shot.py
│       ├── few_shot.py
│       └── zero_shot_COT.py
│
├── tests/                                # Unit tests
│   ├── test_api_clients.py
│   ├── test_parsing.py
│   └── test_stopping_strategy.py
│
├── docs/                                 # Documentation
│   ├── methodology/
│   │   └── belief_bias_justification.md  # Methodological justification
│   ├── analysis/
│   │   └── 14_models_summary.md          # Analysis summary (14 models)
│   ├── paper/
│   │   └── corrected_statistical_table.tex  # LaTeX tables
│   └── archived/                         # Historical validation reports
│       ├── paper_corrections.md
│       ├── verification_report.md
│       └── validation_report.md
│
├── AuthorKit26/                          # AAAI 2026 conference submission materials
│   ├── AnonymousSubmission/
│   │   └── LaTeX/                        # Paper source (main.tex)
│   ├── CameraReady/
│   ├── ReproducibilityChecklist/
│   └── Copyright/
│
├── config.toml                           # Main configuration (API keys, settings)
├── config.toml.example                   # Configuration template
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

---

## References

1. **Belief Bias in Syllogistic Reasoning**: Evans, J. St. B. T., Barston, J. L., & Pollard, P. (1983). On the conflict between logic and belief in syllogistic reasoning. *Memory & Cognition*, 11(3), 295-306.

2. **Adaptive Stopping Strategy**: Hu, E., et al. (2024). Adaptive Sampling for Efficient LLM Evaluation. *EMNLP 2024*.

3. **LMArena Benchmark**: Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.

4. **MMLU Benchmark**: Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*.

5. **Chain-of-Thought Prompting**: Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{logic_in_llms_2024,
  author = {XAheli},
  title = {Logic in LLMs: Syllogistic Reasoning Benchmark},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/XAheli/Logic_in_LLMs}
}
```

