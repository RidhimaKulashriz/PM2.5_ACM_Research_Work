# PM2.5 Modeling Audit: Leakage, Split Integrity, and Overfitting

**Analysis date:** 2026-08-15  
**Target:** `pm25`  
**Canonical master dataset:** `data/ml_ready/master_modeling_dataset_v2.csv`

## Executive assessment

The current modeling pipeline is **leakage-safe at the row-key and preprocessing levels**, but the locked holdout is not representative of the master dataset’s month composition. The train/test split has zero overlap on `(station, year, month)`, preserves the full row universe, keeps IIT Delhi train-only, and uses imputation and scaling inside model pipelines. The feature audit found no predictor names that appear to be direct `pm25` or `pm_25` target derivatives, and the train/test schemas match.

The principal risk is therefore **evaluation-design failure rather than direct data leakage**. The locked test partition contains 261 October/November observations and has a mean PM2.5 of 164.67, while training contains only six October/November observations and has a mean of 85.21. This target shift is approximately 79.46 PM2.5 units. Under that holdout, all three models have negative test R² despite positive shuffled cross-validation scores.

A month- and year-aware alternative split was generated without using the target. It assigns approximately 27 test rows per month and keeps all four years represented in both partitions. The resulting target means are closely aligned: 100.65 in training versus 102.91 in testing. On this alternative split, Random Forest reaches test R² 0.920 and LightGBM reaches 0.958, with small train/test R² gaps of 0.058 and 0.039 respectively. These results show that the negative locked-split scores are dominated by composition shift, not by an across-the-board inability to model PM2.5.

> **Decision:** retain the original locked-split results as a stress-test diagnostic, but use a month/season-aware holdout for primary average-performance reporting. Do not describe the original negative test R² values as definitive evidence that the algorithms fail without explicitly labeling the distribution shift.

## 1. Leakage and split-integrity checks

| Check | Result |
|---|---:|
| Master dataset rows | 1,615 |
| Training rows | 1,292 |
| Test rows | 323 |
| Master key uniqueness | Passed |
| Train key uniqueness | Passed |
| Test key uniqueness | Passed |
| Train/test key overlap | 0 |
| Train/test union equals master | Passed |
| All four years in training | Passed |
| All four years in testing | Passed |
| IIT Delhi train-only constraint | Passed |
| Train/test schemas identical | Passed |
| Numeric predictors with target-like names | None found |
| Non-numeric predictor columns silently ignored | None |

The split generator uses station, year, and month for allocation and sampling. It does not use `pm25` to choose rows. The modeling pipeline excludes `station` and `pm25` from predictors, performs median imputation inside each pipeline, and applies standardization inside the Ridge pipeline. This prevents preprocessing statistics from being fitted on the test partition.

## 2. Locked-split overfitting and shift results

| Model | Train R² | Locked-test R² | Absolute R² gap | 5-fold CV R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Ridge | 0.868 | -0.147 | 1.015 | 0.665 ± 0.374 | Least degraded, but still fails on shifted holdout |
| Random Forest | 0.981 | -0.498 | 1.479 | 0.903 ± 0.035 | Strong fit to training distribution; poor shifted-holdout transfer |
| LightGBM | 0.998 | -0.290 | 1.288 | 0.929 ± 0.031 | Highest apparent fit, but negative shifted-holdout R² |

The model-to-model pattern is a clear warning against relying on train R² or shuffled K-fold scores alone. Random Forest and LightGBM have very high training and within-training validation scores, but their locked-test scores are negative. This is consistent with a test distribution that is materially different from the training distribution.

## 3. Alternative month/year-aware split

The alternative split was generated independently of the target and written to `data/modeling/stratified_splits/`; the locked split under `data/modeling/splits/` was not overwritten.

| Property | Locked split | Alternative split |
|---|---:|---:|
| Test rows | 323 | 323 |
| Train PM2.5 mean | 85.21 | 100.65 |
| Test PM2.5 mean | 164.67 | 102.91 |
| Absolute train/test mean difference | 79.46 | 2.26 |
| Test October/November rows | 261 | 54 |
| Test months represented | 9 | 12 |
| All years in both partitions | Yes | Yes |
| IIT Delhi train-only | Yes | Yes |

The alternative split allocates 26–27 test observations per month and then allocates each month’s quota across years. This substantially reduces the target-distribution mismatch while preserving the fixed row count and key-level integrity requirements.

## 4. Alternative-split model performance

| Model | Train R² | Test R² | Absolute R² gap | 5-fold CV R² | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.852 | -1.308 | 2.160 | 0.820 ± 0.025 | 103.91 |
| Random Forest | 0.977 | 0.920 | 0.058 | 0.889 ± 0.020 | 19.37 |
| LightGBM | 0.996 | 0.958 | 0.039 | 0.932 ± 0.022 | 14.06 |

Ridge remains unstable on the alternative split, with a large negative test R² despite a stable positive cross-validation mean. This indicates that the current linear specification is sensitive to the feature distribution or to nonlinear relationships. The tree-based models have much smaller train/test gaps and close agreement between cross-validation and the alternative held-out test, which is a substantially healthier generalization pattern.

## 5. Grouped-validation interpretation

Grouped validation provides an additional guard against overly optimistic random folds. Grouping by station-year produced mean R² of approximately 0.901 for Random Forest and 0.927 for LightGBM. Grouping by station produced mean R² of approximately 0.865 and 0.870, respectively, while Ridge was unstable and negative on average. Grouping by year produced mean R² of approximately 0.854 for Random Forest and 0.874 for LightGBM.

These grouped results should not be interpreted as interchangeable estimands. Station-grouped validation addresses spatial transferability, year-grouped validation addresses temporal transferability, and the original locked holdout measures performance under the existing month-composition stress test. The scientific report should state which estimand each metric represents.

## 6. Required safeguards before final modeling claims

The current implementation should keep the locked split and its diagnostics for reproducibility, but it should add the alternative month/year-aware split and the grouped-validation audit to the repository. The primary performance table should use a split whose month and season composition is explicitly controlled, while the locked split should be reported as a sensitivity or stress-test result.

Before publication, the team should also decide whether spatial transferability is a primary goal. If it is, station-grouped validation should be reported as a separate primary metric rather than mixing it with random K-fold results. Hyperparameter selection should remain inside training-only cross-validation, and the final held-out test partition should be touched once after the split design is frozen.

## Generated audit artifacts

| Artifact | Purpose |
|---|---|
| `src/modeling/audit_overfitting_and_leakage.py` | Reproducible feature, leakage, grouped-validation, and shift audit. |
| `src/features/prepare_and_split_stratified.py` | Generates a target-free month/year-aware alternative split. |
| `src/modeling/evaluate_stratified_split.py` | Evaluates the baseline models on the alternative split. |
| `data/modeling/results/overfitting_leakage_audit.json` | Machine-readable audit results. |
| `data/modeling/results/validation_strategy_comparison.csv` | Comparison across random, station-year, station, and year grouping. |
| `data/modeling/results/stratified_split_model_performance.csv` | Alternative-split model metrics. |
| `data/modeling/results/stratified_split_evaluation.json` | Alternative split summary and metrics. |
| `data/modeling/stratified_splits/` | Alternative train/test CSVs and split manifest. |
