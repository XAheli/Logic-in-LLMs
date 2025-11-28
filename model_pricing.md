# Model Pricing Guide

> Comprehensive pricing breakdown for the Syllogistic Reasoning Benchmark


## Table of Contents

1. [Overview](#overview)
2. [Experiment Parameters](#experiment-parameters)
3. [Google Gemini Models (Google AI Studio)](#google-gemini-models-google-ai-studio)
4. [HuggingFace Models (via Fireworks.ai)](#huggingface-models-via-fireworksai)
5. [Cost Summary](#cost-summary)
6. [Rate Limits](#rate-limits)


## Overview

This benchmark uses **23 models** across **2 providers**:

| Provider | Models | Billing Type | Authentication |
|----------|--------|--------------|----------------|
| Google AI Studio | 2 | `google_studio_paid` | Google API Key |
| HuggingFace via Fireworks | 21 | `hf_inf_paid` | HF Token (routed to Fireworks) |

### Why These Providers?

- **Google AI Studio**: Direct access to Gemini models with pay-per-token pricing
- **HuggingFace + Fireworks**: Access to 21 open-source models through HuggingFace's inference API, routed to Fireworks.ai for execution

### Important Note on Gemini 2.5 Pro

⚠️ **Gemini 2.5 Pro was removed from this benchmark** due to its high cost (~$400 for the full experiment). This is because:

1. It's a **thinking model** that generates >2,600 internal reasoning tokens before producing output
2. Output pricing is $10/1M tokens (including thinking tokens)
3. Even for simple "valid/invalid" responses, you pay for thousands of thinking tokens

## Experiment Parameters

### Dataset
- **40 syllogisms** × **4 variants** = **160 instances**
- Variants: N (Normal), X (Nonsense), O (Switched Order), OX (Both)

### Strategies
- `zero_shot` - Direct question, no examples
- `one_shot` - One example provided
- `few_shot` - Four examples (2 valid, 2 invalid)
- `zero_shot_cot` - Chain-of-thought reasoning

### Temperatures
| Temperature | Behavior | Iterations |
|-------------|----------|------------|
| 0.0 | Deterministic | 1 |
| 0.5 | Moderate randomness | Up to 10 |
| 1.0 | High randomness | Up to 10 |

### Stopping Strategy (for T > 0)
- If first 5 iterations produce **same answer** → Stop early
- Otherwise → Continue to 10 iterations, majority vote

### API Calls Calculation (Per Model)

| Temperature | Formula | Calls |
|-------------|---------|-------|
| 0.0 | 160 × 4 × 1 | 640 |
| 0.5 | 160 × 4 × 10 (max) | 6,400 |
| 1.0 | 160 × 4 × 10 (max) | 6,400 |
| **Total** | | **13,440** |

**With early stopping (realistic)**: ~8,000-10,000 calls per model


## Google Gemini Models (Google AI Studio)

> **Source**: [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)

### Available Models (2)

| Model Key | Model ID | Display Name |
|-----------|----------|--------------|
| `gemini-2.5-flash` | gemini-2.5-flash | Gemini 2.5 Flash |
| `gemini-2.5-flash-preview-09-25` | gemini-2.5-flash-preview-09-25 | Gemini 2.5 Flash Preview |

### Gemini 2.5 Flash Pricing (Paid Tier)

| Component | Price per 1M Tokens |
|-----------|---------------------|
| **Input** (text/image/video) | $0.30 |
| **Output** (including thinking tokens) | $2.50 |
| Context Caching (input) | $0.03 |
| Context Caching (storage) | $1.00/hr per 1M tokens |
| Grounding (Google Search) | 1,500 RPD free, then $35/1K |

### ⚠️ Important: Thinking Tokens

Gemini 2.5 Flash is a **hybrid reasoning model** that uses internal "thinking" tokens before producing output. From our testing:

| Query Type | Thinking Tokens | Output Tokens | Total Output |
|------------|-----------------|---------------|--------------|
| Simple (valid/invalid) | ~500-1,500 | ~10 | ~510-1,510 |
| CoT reasoning | ~1,000-2,500 | ~100-300 | ~1,100-2,800 |

**You pay for thinking tokens as part of output!**

### Token Estimates (Gemini Flash)

| Component | Tokens/Query | Notes |
|-----------|--------------|-------|
| Input (prompt) | ~400 | System + user prompt |
| Output (thinking + response) | ~1,000-1,500 | Flash uses moderate thinking |

Using **1,200 tokens average** for output (thinking + response):

### Cost Calculation (Per Gemini Flash Model)

```
API Calls: 13,440 (worst case) / ~9,000 (with early stopping)

WORST CASE (13,440 calls):
─────────────────────────────────────────────────────────────
Input:  13,440 × 400 tokens = 5,376,000 = 5.4M tokens
Output: 13,440 × 1,200 tokens = 16,128,000 = 16.1M tokens

Input Cost:  5.4M × $0.30/1M  = $1.62
Output Cost: 16.1M × $2.50/1M = $40.32
─────────────────────────────────────────────────────────────
Total per model (worst case): ~$42

REALISTIC (9,000 calls with early stopping):
─────────────────────────────────────────────────────────────
Input:  9,000 × 400 tokens = 3,600,000 = 3.6M tokens
Output: 9,000 × 1,200 tokens = 10,800,000 = 10.8M tokens

Input Cost:  3.6M × $0.30/1M  = $1.08
Output Cost: 10.8M × $2.50/1M = $27.00
─────────────────────────────────────────────────────────────
Total per model (realistic): ~$28
```

### Total Gemini Cost

| Model | Worst Case | Realistic |
|-------|------------|-----------|
| gemini-2.5-flash | $42 | $28 |
| gemini-2.5-flash-preview-09-25 | $42 | $28 |
| **Subtotal** | **$84** | **$56** |

### Gemini Free Tier (Not Recommended)

| Model | RPM | RPD | TPM |
|-------|-----|-----|-----|
| Gemini 2.5 Flash | 10 | 250 | 250,000 |
| Gemini 2.5 Flash Preview | 10 | 250 | 250,000 |

⚠️ **Why Free Tier Won't Work**:
- You need 13,440 requests per model
- Free tier only allows 250 requests per day
- Would take **54 days per model**!

### Gemini API Configuration

```toml
# config.toml
[api_settings]
max_tokens = 12000  # High for thinking tokens
top_p = 1.0
timeout = 60
```


## HuggingFace Models (via Fireworks.ai)

> **Source**: [fireworks.ai/pricing](https://fireworks.ai/pricing) and [docs.fireworks.ai/ecosystem/integrations/hugging-face](https://docs.fireworks.ai/ecosystem/integrations/hugging-face)

### How It Works

```
Your Code → HuggingFace Inference API → Fireworks.ai → Model → Response
```

### Authentication Options

| Method | How | Billing |
|--------|-----|---------|
| **Routed** | Use HF token only | Billed to HuggingFace account |
| **Direct (Recommended)** | Add Fireworks API key in [HF Settings](https://huggingface.co/settings/inference-providers) | Billed to Fireworks account |

### Fireworks.ai Pricing Tiers (per 1M tokens)

| Model Size | Input | Output |
|------------|-------|--------|
| < 4B parameters | $0.10 | $0.10 |
| 4B - 16B parameters | $0.20 | $0.20 |
| > 16B parameters | $0.90 | $0.90 |
| MoE 0B - 56B (e.g., Mixtral 8x7B) | $0.50 | $0.50 |
| MoE 56B - 176B | $1.20 | $1.20 |

### Special Model Pricing (per 1M tokens)

| Model Family | Input | Output |
|--------------|-------|--------|
| DeepSeek V3 family | $0.56 | $1.68 |
| DeepSeek R1 0528 | $1.35 | $5.40 |
| GLM-4.5, GLM-4.6 | $0.55 | $2.19 |
| Qwen3 235B Family | $0.22 | $0.88 |
| Kimi K2 Instruct/Thinking | $0.60 | $2.50 |
| OpenAI gpt-oss-20b | $0.07 | $0.30 |

---

### Complete HuggingFace Model Pricing (21 Models)

#### Token Estimates (Standard Models)
- **Input**: ~400 tokens/query
- **Output**: ~100 tokens/query (no thinking for most)

#### Per-Model Cost Calculation

| # | Model Key | Params | Input $/1M | Output $/1M | Input Cost | Output Cost | **Total** |
|---|-----------|--------|------------|-------------|------------|-------------|-----------|
| 1 | `gpt-oss-20b` | 20B | $0.07 | $0.30 | $0.38 | $0.40 | **$0.78** |
| 2 | `llama-3.3-70b-instruct` | 70B | $0.90 | $0.90 | $4.86 | $1.21 | **$6.07** |
| 3 | `llama-3.2-3b-instruct` | 3B | $0.10 | $0.10 | $0.54 | $0.13 | **$0.67** |
| 4 | `llama-3.2-1b-instruct` | 1B | $0.10 | $0.10 | $0.54 | $0.13 | **$0.67** |
| 5 | `llama-3.1-70b-instruct` | 70B | $0.90 | $0.90 | $4.86 | $1.21 | **$6.07** |
| 6 | `llama-3.1-8b-instruct` | 8B | $0.20 | $0.20 | $1.08 | $0.27 | **$1.35** |
| 7 | `codellama-34b-instruct` | 34B | $0.90 | $0.90 | $4.86 | $1.21 | **$6.07** |
| 8 | `qwen3-next-80b-a3b-instruct` | 80B MoE | $0.22 | $0.88 | $1.19 | $1.18 | **$2.37** |
| 9 | `qwen3-235b-a22b-thinking` | 235B MoE | $0.22 | $0.88 | $1.19 | $1.18 | **$2.37** |
| 10 | `qwq-32b` | 32B | $0.90 | $0.90 | $4.86 | $1.21 | **$6.07** |
| 11 | `mistral-7b-v0.3` | 7B | $0.20 | $0.20 | $1.08 | $0.27 | **$1.35** |
| 12 | `mistral-7b-instruct-v0.3` | 7B | $0.20 | $0.20 | $1.08 | $0.27 | **$1.35** |
| 13 | `mixtral-8x7b-instruct` | 8x7B MoE | $0.50 | $0.50 | $2.70 | $0.67 | **$3.37** |
| 14 | `deepseek-r1` | R1 | $1.35 | $5.40 | $7.29 | $7.26 | **$14.55** |
| 15 | `deepseek-v3.1` | V3 | $0.56 | $1.68 | $3.02 | $2.26 | **$5.28** |
| 16 | `gemma-3-27b-it` | 27B | $0.90 | $0.90 | $4.86 | $1.21 | **$6.07** |
| 17 | `kimi-k2-instruct` | K2 | $0.60 | $2.50 | $3.24 | $3.36 | **$6.60** |
| 18 | `kimi-k2-thinking` | K2 | $0.60 | $2.50 | $3.24 | $3.36 | **$6.60** |
| 19 | `glm-4.5` | GLM | $0.55 | $2.19 | $2.97 | $2.94 | **$5.91** |
| 20 | `glm-4.6` | GLM | $0.55 | $2.19 | $2.97 | $2.94 | **$5.91** |
| 21 | `yi-34b-chat` | 34B | $0.90 | $0.90 | $4.86 | $1.21 | **$6.07** |

### HuggingFace Models Summary

| Category | Count | Cost Range | Total |
|----------|-------|------------|-------|
| Budget (<$2) | 6 | $0.67 - $1.35 | ~$6 |
| Mid-range ($2-$6) | 6 | $2.37 - $5.91 | ~$24 |
| Expensive (>$6) | 9 | $6.07 - $14.55 | ~$65 |
| **Total (21 models)** | | | **~$95** |

### Models by Cost (Sorted)

#### Cheapest (Under $2)
1. `llama-3.2-1b-instruct` - **$0.67**
2. `llama-3.2-3b-instruct` - **$0.67**
3. `gpt-oss-20b` - **$0.78**
4. `llama-3.1-8b-instruct` - **$1.35**
5. `mistral-7b-v0.3` - **$1.35**
6. `mistral-7b-instruct-v0.3` - **$1.35**

#### Mid-Range ($2-$6)
7. `qwen3-next-80b-a3b-instruct` - **$2.37**
8. `qwen3-235b-a22b-thinking` - **$2.37**
9. `mixtral-8x7b-instruct` - **$3.37**
10. `deepseek-v3.1` - **$5.28**
11. `glm-4.5` - **$5.91**
12. `glm-4.6` - **$5.91**

#### Most Expensive (>$6)
13. `codellama-34b-instruct` - **$6.07**
14. `llama-3.1-70b-instruct` - **$6.07**
15. `llama-3.3-70b-instruct` - **$6.07**
16. `qwq-32b` - **$6.07**
17. `gemma-3-27b-it` - **$6.07**
18. `yi-34b-chat` - **$6.07**
19. `kimi-k2-instruct` - **$6.60**
20. `kimi-k2-thinking` - **$6.60**
21. `deepseek-r1` - **$14.55** ⚠️ Most expensive (thinking model)


## Cost Summary

### Total Cost Breakdown

| Provider | Models | Worst Case | Realistic (Early Stop) |
|----------|--------|------------|------------------------|
| Google Gemini | 2 | ~$84 | ~$56 |
| HuggingFace/Fireworks | 21 | ~$95 | ~$60 |
| **GRAND TOTAL** | **23** | **~$179** | **~$116** |



## Rate Limits

### Google AI Studio (Paid Tier 1)

| Model | RPM | TPM | RPD |
|-------|-----|-----|-----|
| Gemini 2.5 Flash | 1,000 | 1,000,000 | 10,000 |
| Gemini 2.5 Flash Preview | 1,000 | 1,000,000 | 10,000 |

### Fireworks.ai (with Payment Method)

| Limit | Value |
|-------|-------|
| Requests per Minute (RPM) | 6,000 |
| Concurrent Streaming Connections | 10 |

**Note**: Without a payment method, Fireworks limits you to 10 RPM.

### Rate Limit Handling in Code

The codebase includes automatic rate limit handling:

```python
# src/inference/api_clients.py
class HuggingFaceClient:
    MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
    MAX_RETRIES_429 = 10        # retries for rate limits
    BASE_BACKOFF = 5.0          # exponential backoff base
```

