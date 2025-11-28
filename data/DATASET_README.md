# Syllogistic Reasoning Benchmark Dataset

## 📊 Dataset Overview

**Version:** 1.0  
**Created:** 2025-11-27  
**Total Syllogisms:** 40  
**Total Instances:** 160 (40 × 4 variants)  

## 📁 Files Generated

- `syllogisms_master_dataset.json` - Complete dataset with all variants

## 🎯 Dataset Structure

Each syllogism contains:
- **Base ID:** SYL_001 to SYL_040
- **Ground Truth:** valid or invalid
- **4 Variants:**
  - **N (Normal):** Original sensical predicates
  - **X (Nonsense):** Abstract/meaningless predicates
  - **O (Order-switched):** Premises in reversed order
  - **OX (Nonsense + Order):** Combined modifications

## 📈 Statistics

- **Valid syllogisms:** 19 (47.5%)
- **Invalid syllogisms:** 21 (52.5%)
- **Balanced distribution** for unbiased evaluation

## 🔍 Example Structure

```json
{
  "id": "SYL_001",
  "ground_truth": "invalid",
  "variants": {
    "N": {
      "variant_id": "SYL_001_N",
      "variant_type": "normal",
      "statement_1": "All calculators are machines.",
      "statement_2": "All computers are calculators.",
      "conclusion": "Some machines are not computers."
    },
    "X": { ... },
    "O": { ... },
    "OX": { ... }
  }
}
```

## 🎓 Purpose

This dataset is designed to evaluate LLMs' ability to:
1. Recognize valid vs. invalid syllogistic arguments
2. Apply logical reasoning independent of content (X variants)
3. Maintain consistency across premise ordering (O variants)
4. Combine both challenges (OX variants)

## 📝 Usage for LLM Testing

### Prompt Template

```
Statement 1: {statement_1}
Statement 2: {statement_2}
Conclusion: {conclusion}

Is this syllogism valid or invalid?
```

Expected response: "valid" or "invalid"
