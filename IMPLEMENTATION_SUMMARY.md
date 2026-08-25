# Baseline Model Training Implementation Summary

## Scope completed

The PM2.5 repository now contains a reproducible baseline-modeling workflow for the canonical 1,615-row dataset and its existing 80/20 split. The implementation trains Ridge regression, Random Forest, and LightGBM, evaluates the locked test partition, performs five-fold cross-validation on training data only, saves fitted pipelines, writes feature-importance tables, and validates split integrity.

## Files added or updated

| Path | Change |
|---|---|
| `src/modeling/train_baseline_models.py` | Added the baseline training and evaluation workflow. |
| `src/modeling/validate_data_integrity.py` | Added master/split key, row-count, year-coverage, and IIT Delhi validation. |
| `src/modeling/create_distribution_shift_figure.py` | Added a reproducible monthly distribution-shift figure generator. |
| `reports/model_findings_report.md` | Added the detailed results, diagnosis, interpretation, and recommendations. |
| `requirements.txt` | Added `lightgbm>=4.0.0`. |
| `data/modeling/results/` | Added metrics, validation JSON, distribution tables, feature importance, fitted models, and metadata. |

## Execution commands

```bash
python3 src/modeling/validate_data_integrity.py
python3 src/modeling/train_baseline_models.py
python3 src/modeling/create_distribution_shift_figure.py
```

## Validation status

The integrity validation passed. There are 1,292 training rows and 323 test rows, with no train/test key overlap. The split union equals the canonical master key universe, all four years appear in both partitions, and IIT Delhi is train-only as specified by the repository’s split design.

## Model result status

The locked split produces negative test R² for all three baseline models. Ridge is the strongest of the three on the test partition with test R² of -0.147, followed by LightGBM at -0.290 and Random Forest at -0.498. All three exceed the report’s severe overfitting-gap threshold because the test partition is strongly shifted toward October and November. The findings report therefore classifies these metrics as diagnostic rather than publication-grade final performance.

## Important unresolved issue

The split is balanced by year but not by month or season. The test set contains 261 post-monsoon observations while training contains six. A revised season-stratified or month-stratified holdout should be generated before making final model comparisons or scientific claims.

## GitHub state

The work is currently local in the cloned repository. No pull request has been created and no external submission has been performed.
