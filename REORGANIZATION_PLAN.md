# Codebase Reorganization Plan

## Executive Summary

This plan reorganizes the Logic_in_LLMs codebase to create a cleaner, more maintainable structure by:
1. Removing redundant and temporary markdown files (12 files)
2. Consolidating duplicate Python scripts (2 redundancies identified)
3. Organizing scripts into logical subdirectories by purpose
4. Creating clear separation between production code, utilities, and documentation
5. Preserving all protected directories (data/, results/raw_responses/, results/analysis/figures/paper_figures_14_models/)

**Total cleanup**: ~15 files to remove/archive, ~10 files to reorganize into new directory structure

---

## Protected Directories (WILL NOT TOUCH)

✅ **Fully Protected - No Changes:**
- `data/` - All 5 dataset files preserved
- `results/raw_responses/` - All raw API responses preserved
- `results/analysis/figures/paper_figures_14_models/` - All 14-model paper figures preserved

---

## Part 1: Markdown File Cleanup

### Files to REMOVE (9 temporary/redundant files):

1. **ERRORS_IN_RESULTS_SECTION.md** - Historical bug documentation (bugs already fixed)
2. **CORRECTIONS_QUICK_REFERENCE.md** - Temporary checklist (corrections applied)
3. **BENCHMARK_CORRECTIONS.md** - Temporary debugging file (issues resolved)
4. **QUICK_REFERENCE_14_MODELS.md** - Redundant with 14_MODEL_ANALYSIS_SUMMARY.md
5. **MISSING_RUNS_REPORT.md** - Status document (runs completed or outdated)
6. **PAPER_TABLES_FINAL.md** - Working document (tables now auto-generated)
7. **STATISTICAL_RESULTS_REFERENCE.md** - Can be regenerated from scripts
8. **README.md.bak** - Backup file (no longer needed)
9. **LMarena_benchmark.csv** (root) - Duplicate of data/LMarena_benchmark.csv

### Files to KEEP in Root (3 essential files):

1. **README.md** - Main project documentation ✓
2. **LICENSE** - MIT license ✓
3. **.gitignore** - Git configuration ✓

### Files to MOVE to docs/ (3 documentation files):

Create `docs/` directory and move:
1. **BELIEF_BIAS_METRIC_JUSTIFICATION.md** → `docs/methodology/belief_bias_justification.md`
2. **14_MODEL_ANALYSIS_SUMMARY.md** → `docs/analysis/14_models_summary.md`
3. **CORRECTED_STATISTICAL_TABLE.tex** → `docs/paper/corrected_statistical_table.tex`

### Files to ARCHIVE (3 validation/verification files):

Create `docs/archived/` directory and move:
1. **PAPER_CORRECTIONS_REQUIRED.md** → `docs/archived/paper_corrections.md`
2. **VERIFICATION_REPORT.md** → `docs/archived/verification_report.md`
3. **RESULTS_VALIDATION_REPORT.md** → `docs/archived/validation_report.md`

**Note**: Archive these after paper submission if desired; they document quality assurance process.

---

## Part 2: Python Script Reorganization

### Current Structure Issues:

1. **Root directory has 6 analysis scripts** - clutters top level
2. **Redundancy**: `calculate_paper_table1.py` vs `generate_all_tables.py`
3. **No clear organization** between production scripts, utilities, and debugging tools
4. **Two figure generation scripts** with overlapping purposes

### Proposed New Structure:

```
Logic_in_LLMs/
├── scripts/
│   ├── experiments/
│   │   ├── run_experiments.py          (moved from scripts/)
│   │   └── test_small_scale.py         (moved from scripts/)
│   │
│   ├── analysis/
│   │   ├── generate_tables.py          (CONSOLIDATED from root)
│   │   ├── run_statistical_tests.py    (moved from root)
│   │   └── calculate_benchmark_correlations.py (moved from root)
│   │
│   ├── visualization/
│   │   ├── generate_figures.py         (CONSOLIDATED from scripts/)
│   │   └── analyze_results.py          (moved from scripts/)
│   │
│   └── utilities/
│       ├── check_missing_runs.py       (moved from scripts/)
│       └── validate_results.py         (NEW: consolidate debug scripts)
│
├── src/                                 (NO CHANGES - well organized)
│   ├── inference/
│   ├── evaluation/
│   ├── analysis/
│   ├── prompts/
│   └── config.py
│
├── tests/                               (NO CHANGES)
├── data/                                (PROTECTED - NO CHANGES)
├── results/                             (PROTECTED - NO CHANGES)
└── docs/                                (NEW - organized documentation)
```

### Detailed Script Changes:

#### A. CONSOLIDATE: Table Generation Scripts

**Action**: Merge `calculate_paper_table1.py` + `generate_all_tables.py` → `scripts/analysis/generate_tables.py`

**Reasoning**:
- `generate_all_tables.py` (305 lines) is more comprehensive
- `calculate_paper_table1.py` (229 lines) has subset functionality
- Both generate paper tables from raw responses
- Merged version will support both simple and comprehensive table generation

**Implementation**:
1. Keep `generate_all_tables.py` as base
2. Add command-line args: `--table-set [table1|all]` for flexibility
3. Remove `calculate_paper_table1.py`

#### B. CONSOLIDATE: Figure Generation Scripts

**Action**: Merge `generate_figures_14_models.py` + `generate_revamped_figures.py` → `scripts/visualization/generate_figures.py`

**Reasoning**:
- Both generate Plotly/matplotlib figures
- Different model counts (14 vs all)
- Can be parameterized with `--model-count` flag

**Implementation**:
1. Use `generate_revamped_figures.py` as base (more comprehensive)
2. Add command-line args: `--model-count [14|15|all]`
3. Add flag: `--figure-set [paper|full|specific]`
4. Remove redundant script

#### C. CONSOLIDATE: Debug/Validation Scripts

**Action**: Merge `debug_correlation.py` + `check_bias_interpretation.py` → `scripts/utilities/validate_results.py`

**Reasoning**:
- Both are small validation scripts (48-49 lines each)
- Both check statistical calculations
- Can be combined into single validation utility with subcommands

**Implementation**:
1. Create new `validate_results.py` with argparse subcommands:
   - `validate_results.py correlation` (from debug_correlation.py)
   - `validate_results.py bias` (from check_bias_interpretation.py)
2. Remove original scripts

#### D. MOVE: Analysis Scripts from Root

**Action**: Move 3 scripts from root → `scripts/analysis/`

1. `run_statistical_tests.py` → `scripts/analysis/run_statistical_tests.py`
2. `calculate_benchmark_correlations.py` → `scripts/analysis/calculate_benchmark_correlations.py`
3. (Merged table script already here)

**Reasoning**: All analysis/statistics scripts belong together

#### E. MOVE: Scripts into Subdirectories

**Action**: Reorganize existing `scripts/` into purpose-based subdirs

1. `scripts/run_experiments.py` → `scripts/experiments/run_experiments.py`
2. `scripts/test_small_scale.py` → `scripts/experiments/test_small_scale.py`
3. `scripts/analyze_results.py` → `scripts/visualization/analyze_results.py`
4. `scripts/check_missing_runs.py` → `scripts/utilities/check_missing_runs.py`

**Reasoning**: Clear separation by purpose (experiments, analysis, visualization, utilities)

---

## Part 3: Documentation Organization

### Create New docs/ Structure:

```
docs/
├── methodology/
│   └── belief_bias_justification.md    (methods documentation)
│
├── analysis/
│   └── 14_models_summary.md            (analysis summaries)
│
├── paper/
│   └── corrected_statistical_table.tex (LaTeX tables)
│
└── archived/                           (quality assurance history)
    ├── paper_corrections.md
    ├── verification_report.md
    └── validation_report.md
```

---

## Part 4: Configuration Files

### Keep in Root:

- `config.toml` - Main configuration ✓
- `config.toml.example` - Example template ✓
- `requirements.txt` - Dependencies ✓
- `.gitignore` - Git configuration ✓

### No Changes Needed

---

## Part 5: Update Import Statements

### Files Requiring Import Updates:

After moving scripts, update imports in:

1. **Scripts importing from src/**: Update relative paths
   - `scripts/experiments/run_experiments.py`
   - `scripts/experiments/test_small_scale.py`
   - `scripts/analysis/*.py`
   - `scripts/visualization/*.py`

2. **Add proper Python path handling**:
   Each script in subdirectories needs:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent))
   ```

3. **Update README.md**:
   - Update script paths in usage examples
   - Update project structure section

---

## Part 6: Results Directory Cleanup

### Check for Empty/Redundant Files:

Scan `results/` for:
- Empty placeholder CSV files (e.g., `model_rankings.csv`, `correlation_analysis.csv`)
- Duplicate/outdated analysis tables
- **Action**: Remove empty files, keep all generated analysis outputs

**Protected subdirectories** (NO changes):
- `results/raw_responses/` ✓
- `results/analysis/figures/paper_figures_14_models/` ✓

---

## Part 7: Final Directory Tree (After Reorganization)

```
Logic_in_LLMs/
├── README.md                           ✓ Main documentation
├── LICENSE                             ✓ MIT license
├── .gitignore                          ✓ Git config
├── config.toml                         ✓ Configuration
├── config.toml.example                 ✓ Config template
├── requirements.txt                    ✓ Dependencies
│
├── src/                                ✓ Core source code (NO CHANGES)
│   ├── __init__.py
│   ├── config.py
│   ├── inference/
│   ├── evaluation/
│   ├── analysis/
│   └── prompts/
│
├── scripts/                            ✨ REORGANIZED
│   ├── experiments/
│   │   ├── run_experiments.py          (main experiment runner)
│   │   └── test_small_scale.py         (small-scale testing)
│   │
│   ├── analysis/
│   │   ├── generate_tables.py          (consolidated table generation)
│   │   ├── run_statistical_tests.py    (statistical tests)
│   │   └── calculate_benchmark_correlations.py
│   │
│   ├── visualization/
│   │   ├── generate_figures.py         (consolidated figure generation)
│   │   └── analyze_results.py          (analysis orchestration)
│   │
│   └── utilities/
│       ├── check_missing_runs.py       (completeness checker)
│       └── validate_results.py         (validation tools)
│
├── tests/                              ✓ Unit tests (NO CHANGES)
│   ├── test_api_clients.py
│   ├── test_parsing.py
│   └── test_stopping_strategy.py
│
├── data/                               ✅ PROTECTED (NO CHANGES)
│   ├── syllogisms_master_dataset.json
│   ├── syllogisms_main_summary.json
│   ├── syllogisms_main_summary.csv
│   ├── LMarena_benchmark.csv
│   └── MMLU_helm.csv
│
├── results/                            ✅ PROTECTED subdirs
│   ├── raw_responses/                  (PROTECTED)
│   ├── analysis/
│   │   ├── figures/
│   │   │   ├── paper_figures_14_models/ (PROTECTED)
│   │   │   ├── plotly/
│   │   │   └── static/
│   │   └── tables/
│   ├── parsed_results/
│   └── checkpoint.json
│
├── docs/                               ✨ NEW: Organized documentation
│   ├── methodology/
│   │   └── belief_bias_justification.md
│   ├── analysis/
│   │   └── 14_models_summary.md
│   ├── paper/
│   │   └── corrected_statistical_table.tex
│   └── archived/
│       ├── paper_corrections.md
│       ├── verification_report.md
│       └── validation_report.md
│
├── AuthorKit26/                        ✓ Paper submission (NO CHANGES)
│   ├── AnonymousSubmission/
│   ├── CameraReady/
│   ├── ReproducibilityChecklist/
│   └── Copyright/
│
├── lit_rev/                            ✓ Literature PDFs (NO CHANGES)
├── logs/                               ✓ Runtime logs (NO CHANGES)
├── .vscode/                            ✓ VS Code config (NO CHANGES)
└── .claude/                            ✓ Claude config (NO CHANGES)
```

---

## Implementation Steps (Sequential Order)

### Step 1: Backup Check
- Ensure all changes are committed to git
- Create branch: `git checkout -b reorganize-codebase`

### Step 2: Create New Directory Structure
1. Create `docs/` with subdirectories
2. Create `scripts/experiments/`
3. Create `scripts/analysis/`
4. Create `scripts/visualization/`
5. Create `scripts/utilities/`

### Step 3: Remove Redundant Markdown Files
Remove 9 temporary/redundant files from root

### Step 4: Move Documentation Files
Move 3 files to `docs/methodology/`, `docs/analysis/`, `docs/paper/`
Move 3 files to `docs/archived/`

### Step 5: Consolidate Python Scripts
1. Merge table generation scripts → `scripts/analysis/generate_tables.py`
2. Merge figure generation scripts → `scripts/visualization/generate_figures.py`
3. Merge validation scripts → `scripts/utilities/validate_results.py`

### Step 6: Move Python Scripts
Move remaining scripts to appropriate subdirectories

### Step 7: Update Import Statements
Update all moved scripts with correct Python path handling

### Step 8: Update README.md
Update project structure and usage examples

### Step 9: Clean Results Directory
Remove empty placeholder CSV files

### Step 10: Update .gitignore (if needed)
Add any new temporary file patterns

### Step 11: Testing
1. Run `pytest tests/` to verify imports still work
2. Test one script from each subdirectory
3. Verify all protected directories untouched

### Step 12: Commit Changes
```bash
git add .
git commit -m "refactor: reorganize codebase structure

- Remove 9 redundant/temporary markdown files
- Consolidate duplicate Python scripts (table generation, figure generation, validation)
- Organize scripts into purpose-based subdirectories (experiments, analysis, visualization, utilities)
- Create docs/ directory for organized documentation
- Update import statements and README
- Preserve all protected directories (data/, results/raw_responses/, results/analysis/figures/paper_figures_14_models/)"
```

---

## Summary of Changes

### Files to Remove (9):
- ERRORS_IN_RESULTS_SECTION.md
- CORRECTIONS_QUICK_REFERENCE.md
- BENCHMARK_CORRECTIONS.md
- QUICK_REFERENCE_14_MODELS.md
- MISSING_RUNS_REPORT.md
- PAPER_TABLES_FINAL.md
- STATISTICAL_RESULTS_REFERENCE.md
- README.md.bak
- LMarena_benchmark.csv (root duplicate)

### Files to Move (6):
- 3 to docs/
- 3 to docs/archived/

### Scripts to Consolidate (6 → 3):
- calculate_paper_table1.py + generate_all_tables.py → generate_tables.py
- generate_figures_14_models.py + generate_revamped_figures.py → generate_figures.py
- debug_correlation.py + check_bias_interpretation.py → validate_results.py

### Scripts to Reorganize (10):
- Move 10 existing scripts into subdirectories

### New Directories (5):
- docs/methodology/
- docs/analysis/
- docs/paper/
- docs/archived/
- scripts/{experiments, analysis, visualization, utilities}/

### Files to Update (12+):
- All moved scripts (import updates)
- README.md (structure and usage)
- Potentially .gitignore

---

## Risk Assessment

### Low Risk:
- Removing temporary markdown files (can restore from git if needed)
- Moving documentation files (no code dependencies)

### Medium Risk:
- Consolidating scripts (need to preserve all functionality)
- Moving scripts (need to update imports)

### Zero Risk:
- Protected directories completely untouched
- All data and results preserved

### Mitigation:
- All changes in git branch
- Test suite verification after each major change
- Incremental implementation with commit checkpoints

---

## Expected Outcome

### Before:
- 22 files in root directory (cluttered)
- 6 analysis scripts in root
- 13 markdown files (many redundant)
- Flat scripts/ directory

### After:
- 6 files in root directory (clean: README, LICENSE, config files, .gitignore)
- 0 analysis scripts in root (organized in scripts/analysis/)
- 3 essential docs in organized structure
- Hierarchical scripts/ with clear purposes
- 15 fewer redundant files
- Same functionality, better organization

---

## Timeline Estimate

- Directory creation: 2 minutes
- File removal: 3 minutes
- File moves: 5 minutes
- Script consolidation: 15 minutes (careful merging)
- Script moves + import updates: 20 minutes
- README updates: 10 minutes
- Testing: 15 minutes
- **Total: ~70 minutes**

---

## Questions for User (if any)

None - plan is comprehensive and ready for implementation.

All protected directories preserved per requirements.
