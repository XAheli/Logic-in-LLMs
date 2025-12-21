# REQUIRED CORRECTIONS TO PAPER RESULTS SECTION

## Executive Summary

Your paper contains **3 CRITICAL ERRORS** and **1 MAJOR OMISSION** that must be corrected before submission:

### Critical Errors
1. ❌ **Bias correlation: ρ = 1.000 (WRONG)** → Should be ρ = -0.632
2. ❌ **McNemar test: 274 vs 190, χ²=14.85 (WRONG)** → Should be 786 vs 546, χ²=42.88
3. ❌ **LMArena correlation: ρ = -0.744, p=0.0015 (WRONG)** → Should be ρ = -0.825, p=0.0010

### Major Omission
4. ⚠️ **MMLU correlation: ρ = 0.288 (INVALID)** → No model matches; must be removed entirely

---

## CORRECTION 1: Bias Correlation (CRITICAL)

### What the Paper Says (WRONG):
> "The magnitude of belief bias correlates inversely with overall performance, showing a **perfect rank correlation (Spearman ρ = 1.000, p < 0.0001)**."

> "This **perfect monotonic relationship** provides compelling evidence..."

> Table 3: "Syntax Acc. × Bias Effect | Spearman ρ | **1.000** | <0.0001 | **Perfect** | Significant"

### Correct Values:
- **Spearman ρ = -0.632** (p = 0.0115*)
- **Pearson r = -0.632** (p = 0.0114*)
- **Effect: Strong negative correlation** (NOT perfect)

### Why the Paper is Wrong:
The original analysis compared two CSV files that were sorted differently:
- `table1_complete.csv` sorted by accuracy (descending)
- `table3_belief_bias.csv` sorted by bias (descending)

Comparing row-by-row without matching by Model name created a **spurious perfect correlation**.

### Required Changes to Paper:

#### Change 1: Section 4.4 (Belief Bias Analysis)
**REMOVE:**
```latex
The magnitude of belief bias correlates inversely with overall performance, 
showing a perfect rank correlation (Spearman ρ = 1.000, p < 0.0001).
```

**REPLACE WITH:**
```latex
The magnitude of belief bias correlates strongly and inversely with overall 
performance (Spearman ρ = -0.632, p = 0.0115). Better-performing models 
exhibit significantly reduced belief bias effects.
```

#### Change 2: Section 4.6 (Statistical Summary)
**REMOVE:**
```latex
This perfect monotonic relationship provides compelling evidence that higher 
reasoning ability directly reduces reliance on content-based heuristics.
```

**REPLACE WITH:**
```latex
This strong negative correlation provides compelling evidence that higher 
reasoning ability reduces reliance on content-based heuristics. The correlation 
indicates that as accuracy increases, bias magnitude decreases toward zero, 
with top-tier models showing near-zero bias effects.
```

#### Change 3: Table 3 (Statistical Summary)
**CHANGE:**
```latex
Syntax Acc. × Bias Effect & Spearman ρ & 1.000 & --- & <0.0001 & Perfect & Perfect rank correlation
```

**TO:**
```latex
Syntax Acc. × Bias Effect & Spearman ρ & -0.632 & --- & 0.0115 & Strong & Negative correlation
```

#### Add Interpretation Note:
After the table, add:
```latex
\multicolumn{7}{l}{\footnotesize Note: Bias\_Effect = Congruent\_Acc - Incongruent\_Acc. 
Negative correlation means higher accuracy correlates with smaller bias magnitude (closer to zero).}
```

### Critical Understanding:
- **Bias_Effect** is measured as: Congruent_Acc - Incongruent_Acc
- Positive Bias_Effect = presence of bias (better on congruent)
- **Negative correlation** means: Higher accuracy → Smaller bias (closer to zero)
- Example:
  - Gemini 2.5 Flash: 99.6% acc, **+0.9 pp bias** (minimal)
  - Mixtral 8×22B: 52.5% acc, **+52.4 pp bias** (severe)

---

## CORRECTION 2: McNemar Test (CRITICAL)

### What the Paper Says (WRONG):
> "McNemar's test at the instance level (N = 2400 model-instance pairs) detects whether 
> strategies produce different error patterns. We find significant error redistribution: 
> **ZS solves 274 instances that FS fails, while FS solves only 190 that ZS fails** 
> **(χ² = 14.85, Bonferroni-corrected p = 0.0006)**."

> Table 3: "Zero-shot vs Few-shot | McNemar χ² | **14.85** | 1 | **0.0001** | --- | Error redistribution"

### Correct Values:
- **N = 7,200 instances** (not 2,400)
  - Calculation: 15 models × 3 temps × 160 syllogisms = 7,200
- **ZS correct, FS wrong: 786 instances** (not 274)
- **FS correct, ZS wrong: 546 instances** (not 190)
- **χ² = 42.88** (not 14.85)
- **p < 0.0001*** (this is correct)
- **Difference: 240 more instances** (786 - 546 = 240)

### Why the Paper is Wrong:
The original script used **configuration-level** (45 configs) instead of **instance-level** (7,200 instances).

### Required Changes to Paper:

#### Change 1: Section 4.2 (Prompting Strategy Effects)
**CHANGE:**
```latex
McNemar's test at the instance level (N = 2400 model-instance pairs) detects whether 
strategies produce different error patterns. We find significant error redistribution: 
ZS solves 274 instances that FS fails, while FS solves only 190 that ZS fails 
(χ² = 14.85, Bonferroni-corrected p = 0.0006).
```

**TO:**
```latex
McNemar's test at the instance level (N = 7200 syllogism evaluations: 15 models × 
3 temperatures × 160 syllogisms) detects whether strategies produce different error 
patterns. We find highly significant error redistribution: ZS solves 786 instances 
that FS fails, while FS solves only 546 that ZS fails (χ² = 42.88, p < 0.0001), 
a net advantage of 240 instances for zero-shot.
```

#### Change 2: Table 3 (Statistical Summary)
**CHANGE:**
```latex
McNemar Tests (Instance-level, N = 2400)
Zero-shot vs Few-shot & McNemar χ² & 14.85 & 1 & 0.0001 & --- & Error redistribution
Zero-shot vs Few-shot (Bonferroni) & McNemar χ² & 14.85 & 1 & 0.0006† & --- & Significant
```

**TO:**
```latex
McNemar Tests (Instance-level, N = 7200)
Zero-shot vs Few-shot & McNemar χ² & 42.88 & 1 & <0.0001 & --- & 786 vs 546 (ZS advantage)
```

#### Change 3: Remove Bonferroni Line
The Bonferroni-corrected McNemar line should be removed since p < 0.0001 is so strong that correction is unnecessary to mention.

---

## CORRECTION 3: LMArena Correlation (CRITICAL)

### What the Paper Says (WRONG):
> Figure caption: "LMArena correlation **(ρ = -0.744, p = 0.0015)**"

> Text: "Syllogistic reasoning shows a **strong negative correlation with LMArena rank 
> (Spearman ρ = -0.744, p = 0.0015, N = 14**)"

> Table 3: "LMArena rank (lower = better) | Spearman ρ | **-0.744** | --- | **0.0015** | Strong | Significant"

### Correct Values:
- **Spearman ρ = -0.825** (p = 0.0010***)
- **N = 12** (not 14)
- **Effect: Strong negative** (even stronger than reported)

### Why the Paper is Wrong:
The original calculation was based on preliminary data. The corrected analysis with proper model name matching found:
- 12 matched models (not 14)
- Stronger correlation than originally reported

### The Negative Correlation is CORRECT:
- **LMArena:** Lower rank number = Better model (1st place = rank 1)
- **Our accuracy:** Higher percentage = Better model
- **Expected correlation:** NEGATIVE (as one goes up, the other goes down)
- **Result:** ρ = -0.825 ✅ CORRECT direction!

### Required Changes to Paper:

#### Change 1: Section 4.5 (Benchmark Correlations)
**CHANGE:**
```latex
Syllogistic reasoning shows a strong negative correlation with LMArena rank 
(Spearman ρ = -0.744, p = 0.0015, N = 14; lower rank indicates better performance).
```

**TO:**
```latex
Syllogistic reasoning shows a strong negative correlation with LMArena rank 
(Spearman ρ = -0.825, p = 0.0010, N = 12; lower rank indicates better performance). 
The negative correlation is the expected direction: models with higher reasoning 
accuracy achieve numerically lower (better) LMArena rankings.
```

#### Change 2: Figure Caption
**CHANGE:**
```latex
\caption{... LMArena correlation (ρ = -0.744, p = 0.0015). ...}
```

**TO:**
```latex
\caption{... LMArena correlation (ρ = -0.825, p = 0.0010). ...}
```

#### Change 3: Table 3 (Statistical Summary)
**CHANGE:**
```latex
LMArena rank (lower = better) & Spearman ρ & -0.744 & --- & 0.0015 & Strong & Significant
```

**TO:**
```latex
LMArena rank (lower = better) & Spearman ρ & -0.825 & --- & 0.0010 & Strong & Predicts reasoning
```

---

## CORRECTION 4: MMLU Correlation (MUST REMOVE)

### What the Paper Says (WRONG):
> "However, we find **no significant correlation with MMLU scores 
> (Spearman ρ = 0.288, p = 0.364, N = 12)**."

> Figure: Shows MMLU correlation plot

> Table 3: "MMLU score | Spearman ρ | +0.288 | --- | 0.364 | Weak | Not significant"

### Why This is INVALID:
**There are ZERO model matches between our dataset and the MMLU dataset.**

- Our models: `gemini-2.5-flash`, `llama-3.3-70b-instruct`, `qwen3-next-80b-a3b-instruct`
- MMLU models: `Gemini 1.5 Pro (002)`, `Llama 3.1 Instruct Turbo (405B)`, `Qwen 2.5 72B`

These are **different model versions** entirely. The reported ρ = 0.288 cannot be calculated with N=0 matches.

### Required Changes to Paper:

#### Change 1: Section 4.5 (Benchmark Correlations)
**REMOVE ENTIRELY:**
```latex
However, we find no significant correlation with MMLU scores (Spearman ρ = 0.288, 
p = 0.364, N = 12). This dissociation has important implications. MMLU primarily 
tests factual knowledge retrieval---knowing that Paris is the capital of France or 
that mitochondria are cellular organelles. Syllogistic reasoning, by contrast, tests 
formal deductive inference---applying validity rules independent of whether premises 
or conclusions correspond to reality. A model can accumulate encyclopedic knowledge 
without developing robust inference capabilities. This finding suggests that current 
LLM evaluations, which emphasize breadth of factual knowledge, may systematically 
miss fundamental reasoning competencies. High MMLU scores should not be taken as 
evidence of logical reasoning proficiency.
```

**REPLACE WITH:**
```latex
We attempted to evaluate correlation with MMLU scores (factual knowledge across 
57 domains) but could not obtain sufficient model overlap due to different model 
versions in the benchmark datasets. The lack of standardized model identifiers 
across benchmarks remains a limitation for cross-benchmark correlation analysis.
```

#### Change 2: Figure - Remove MMLU subplot
**CHANGE:**
```latex
\begin{figure*}[t]
\centering
\begin{subfigure}[t]{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{images/fig05_lmarena_correlation.pdf}
    \caption{LMArena correlation (ρ = -0.744, p = 0.0015).}
    \label{fig:lmarena}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{images/fig06_mmlu_correlation.pdf}
    \caption{MMLU correlation (ρ = 0.288, p = 0.364).}
    \label{fig:mmlu}
\end{subfigure}
\caption{Correlation between syllogistic reasoning accuracy and established benchmarks...}
\label{fig:benchmark_corr}
\end{figure*}
```

**TO (single figure):**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\columnwidth]{images/fig05_lmarena_correlation.pdf}
\caption{Correlation between syllogistic reasoning accuracy and LMArena rankings 
(Spearman ρ = -0.825, p = 0.0010, N = 12). Lower rank indicates better performance. 
The strong negative correlation suggests that instruction-following quality predicts 
formal reasoning capability.}
\label{fig:lmarena}
\end{figure}
```

#### Change 3: Table 3 (Statistical Summary)
**REMOVE:**
```latex
MMLU score & Spearman ρ & +0.288 & --- & 0.364 & Weak & Not significant
```

#### Change 4: Section 4.6 (Statistical Summary)
**REMOVE:**
```latex
However, zero correlation with MMLU (ρ = 0.288, p = 0.364) indicates that factual 
knowledge breadth does not predict formal inference capability. High MMLU scores 
should not be interpreted as evidence of logical reasoning proficiency.
```

**REPLACE WITH:**
```latex
The strong LMArena correlation suggests that careful instruction-following overlaps 
with precise rule application required for formal reasoning.
```

---

## CORRECTION 5: Holm vs Bonferroni Terminology (MINOR)

### Issue:
Paper inconsistently uses "Holm correction" and "Bonferroni correction" terminology.

### What is Implemented:
The script NOW correctly implements **Holm-Bonferroni sequential correction**, not simple Bonferroni.

### Formula:
- **Bonferroni (simple):** p_adj = p × n
- **Holm (sequential):** p_adj = p × (n - rank + 1), where tests are sorted by p-value

### Required Changes:

#### In Table 3 footnote:
**CHANGE:**
```latex
$^*$Bonferroni correction for 3 comparisons: α = 0.05 / 3 = 0.0167
```

**TO:**
```latex
$^*$Holm-Bonferroni correction: Sequential method sorting by p-value, 
p_adj = p × (n - rank + 1)
```

#### In text (Section 4.2):
Keep "Holm correction" references - they are now correct.

---

## CORRECTION 6: Sample Size Clarifications (MINOR)

### Issues:
- McNemar: Paper says N=2400, should be N=7200
- Wilcoxon: Paper says N=15, should be N=45

### Required Changes:

#### Change 1: Section 4.2
**CHANGE:**
```latex
Wilcoxon signed-rank tests at the model level (N = 15) reveal no consistent 
directional effect after Holm correction (all p_adj > 0.05).
```

**TO:**
```latex
Wilcoxon signed-rank tests at the configuration level (N = 45: 15 models × 3 temperatures) 
reveal no consistent directional effect after Holm correction (all p_adj > 0.05).
```

---

## SUMMARY TABLE: ALL CORRECTIONS

| Location | Current (WRONG) | Corrected | Priority |
|----------|----------------|-----------|----------|
| **Bias correlation** | ρ = 1.000, p < 0.0001 | ρ = -0.632, p = 0.0115* | **CRITICAL** |
| **McNemar instances** | 274 vs 190, χ²=14.85 | 786 vs 546, χ²=42.88 | **CRITICAL** |
| **McNemar N** | N = 2400 | N = 7200 | **CRITICAL** |
| **LMArena ρ** | ρ = -0.744, p = 0.0015 | ρ = -0.825, p = 0.0010*** | **CRITICAL** |
| **LMArena N** | N = 14 | N = 12 | Medium |
| **MMLU correlation** | ρ = 0.288 (reported) | **REMOVE ENTIRELY** | **CRITICAL** |
| **MMLU figure** | Two-panel figure | Single LMArena figure | **CRITICAL** |
| **Wilcoxon N** | N = 15 (model level) | N = 45 (configuration level) | Minor |
| **Correction method** | "Bonferroni" (mixed) | "Holm-Bonferroni" (consistent) | Minor |

---

## VERIFICATION CHECKLIST

Before submitting, verify:

- [ ] **Table 3:** ρ = -0.632 for bias correlation
- [ ] **Table 3:** χ² = 42.88, 786 vs 546 for McNemar
- [ ] **Table 3:** ρ = -0.825, p = 0.0010 for LMArena
- [ ] **Table 3:** MMLU row completely removed
- [ ] **Section 4.2:** "786 instances" and "546 instances" in text
- [ ] **Section 4.4:** "ρ = -0.632" replacing "ρ = 1.000"
- [ ] **Section 4.5:** "ρ = -0.825" replacing "ρ = -0.744"
- [ ] **Section 4.5:** MMLU paragraph completely removed
- [ ] **Figure:** MMLU subplot removed (single LMArena figure)
- [ ] **Figure caption:** "ρ = -0.825" in LMArena caption
- [ ] **Text:** No mention of "perfect rank correlation" for bias

---

## POSITIVE FINDINGS (UNCHANGED)

These results are **CORRECT** and do not need changes:

✅ Few-shot underperforms zero-shot (Δ=-3.33%, p=0.0166*, survives Holm correction)  
✅ Temperature has no effect (χ²=3.77, p=0.152)  
✅ Belief bias confirmed (Δ=13.61 pp, p=0.0152*, d=0.71)  
✅ Syntax > NLU by 22.18 pp (p < 0.0001***, r=0.53)  
✅ Syntax × NLU negative correlation (ρ=-0.611, p=0.0155*)  
✅ Precision × Recall strong positive (ρ=0.851, p=0.0001***)  
✅ Accuracy × Consistency moderate positive (ρ=0.586, p=0.0218*)  

---

## RECOMMENDED ORDER OF CORRECTIONS

1. **FIRST:** Remove all MMLU content (text, figure, table row)
2. **SECOND:** Update McNemar values (786 vs 546, χ²=42.88, N=7200)
3. **THIRD:** Update bias correlation (ρ=-0.632, remove "perfect")
4. **FOURTH:** Update LMArena correlation (ρ=-0.825, p=0.0010, N=12)
5. **FIFTH:** Fix sample size descriptions (N=45 for Wilcoxon, N=7200 for McNemar)
6. **SIXTH:** Ensure consistent "Holm-Bonferroni" terminology

---

## INTERPRETATION REMINDERS

### For Reviewers:
- **Bias correlation (ρ=-0.632):** Negative correlation means better models have SMALLER bias magnitude
  - Bias_Effect = Congruent_Acc - Incongruent_Acc
  - Higher accuracy → bias closer to zero (less interference)
  
- **LMArena correlation (ρ=-0.825):** Negative correlation is CORRECT direction
  - LMArena: Lower rank = Better (1st place = rank 1)
  - Our accuracy: Higher = Better
  - Inverse relationship → negative ρ ✅

- **McNemar (786 vs 546):** Instance-level analysis, not model-level
  - 7,200 total instances evaluated
  - Zero-shot solves 240 MORE instances than few-shot
  - Highly significant redistribution (χ²=42.88, p<0.0001)

---

## FILES TO UPDATE

After making paper corrections, also update these supporting documents:

1. ✅ `CORRECTED_STATISTICAL_TABLE.tex` - Already correct
2. ✅ `STATISTICAL_RESULTS_REFERENCE.md` - Already correct  
3. ✅ `BENCHMARK_CORRECTIONS.md` - Already correct
4. ⏳ `ERRORS_IN_RESULTS_SECTION.md` - Update with resolution status
5. ⏳ `STATISTICAL_ANALYSIS_REPORT.md` - Update with corrected values

---

**END OF CORRECTIONS DOCUMENT**

All changes are based on verified, re-run statistical analyses with bug fixes applied.
Last verification: 2025-12-06
