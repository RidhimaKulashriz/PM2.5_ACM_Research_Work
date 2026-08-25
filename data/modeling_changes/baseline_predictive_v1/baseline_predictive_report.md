# V3 Baseline Predictive Modeling Report

> **Interpretation boundary:** This package is a predictive baseline for monthly PM₂.₅. It does not estimate a causal effect of vegetation and must not be used as causal evidence. The DML estimands and their dependence-aware intervals remain the separate causal-inference workstream.

## Purpose and immutable inputs

The notebook compares Linear Regression, Random Forest Regressor, and LightGBM Regressor using the frozen 2025-context V3 master dataset and the existing canonical split. The master contains 1,615 station-month rows from 35 stations; the training split contains 1,292 rows and the locked test split contains 323 rows. The train/test keys are disjoint and their union equals the master key universe. The canonical split contains all four years in both partitions, but it is not a spatially independent holdout because most stations are represented in both train and test. IIT Delhi remains train-only because the canonical split contains only one observation for that station.

The input files under `data/modeling_changes/datasets/` and `data/modeling_changes/splits/` were read only. The notebook excludes `station`, `pm25`, row identity, split-membership fields, future/prediction fields, and other target-derived names from the predictors. Numeric environmental, temporal, and coordinate variables are retained. Latitude and longitude provide broad spatial context for prediction and diagnostics; they are not causal variables. Any median imputation or standardization is fitted inside a train-only pipeline.

## Models and evaluation

Linear Regression uses train-fitted median imputation and standardization. Random Forest uses 300 trees with `random_state=42`; LightGBM uses a conservative fixed baseline with `random_state=42`. No large hyperparameter search was conducted. Training-only cross-validation uses five `GroupKFold` folds grouped by station. This is a more conservative training diagnostic, but it is not a spatial generalization assessment. The locked test set is used only for final evaluation after model fitting.

R², RMSE, and MAE are the primary regression metrics. RMSE is useful for showing the effect of large errors, while MAE is easier to interpret as a typical absolute error and is less dominated by extreme observations. Median absolute error is supplementary. Accuracy, precision, and recall are classification metrics and are intentionally not reported as equivalent performance measures for a continuous PM₂.₅ target. A future high-PM₂.₅ alert classifier would need a separately pre-specified and scientifically justified threshold.

| Model | Test R² | Test RMSE (µg/m³) | Test MAE (µg/m³) | Test median AE (µg/m³) |
|---|---:|---:|---:|---:|
| Linear Regression | 0.820184 | 29.096270 | 21.637563 | 17.619091 |
| Random Forest | 0.908961 | 20.703122 | 11.390408 | 6.497000 |
| LightGBM | **0.924285** | **18.880479** | **10.088022** | 6.682582 |

LightGBM is the best locked-test model by RMSE and MAE in this baseline run. Its predictive performance is better than Linear Regression, and Random Forest also improves over Linear Regression. This is a predictive comparison only; it does not demonstrate that any feature, including NDVI, causes PM₂.₅ to change.

The very high tree-model training scores relative to their locked-test scores indicate overfitting or memorization risk that should be considered in future validation. The station-grouped training CV is stable for the tree models, with mean CV RMSE of 25.075591 for Random Forest and 24.826004 for LightGBM, but the Linear Regression grouped-CV diagnostic is unstable because the fold-level target distributions are highly heterogeneous. These facts support retaining the locked-test and grouped-CV results together rather than presenting only the most favorable metric.

## Temporal, residual, and spatial findings

All three models perform worst in 2024 by year-wise RMSE, while post-monsoon is the most difficult season by RMSE. For LightGBM, 2024 RMSE is 30.152834 µg/m³ and post-monsoon RMSE is 39.142764 µg/m³. These are regime-specific diagnostic findings, not evidence that the models are invalid; they indicate that temporal distribution shift and high-pollution episodes deserve further investigation.

The residual analysis retains extreme PM₂.₅ observations rather than removing them. The training 95th percentile is used only as a diagnostic boundary. The high-to-non-high RMSE ratio is 2.107 for Linear Regression, 1.632 for Random Forest, and 1.275 for LightGBM. Thus, all models have more difficulty in the extreme-pollution subgroup, with the largest relative effect for Linear Regression. The largest station-level mean absolute errors under LightGBM occur at Chandni_Chowk, DTU, and Shadipur. These station patterns are descriptive and may reflect local measurement, coverage, or regime differences.

The spatial figure is a station-coordinate map of LightGBM mean absolute test error. It is not an interpolated concentration surface, and no spatial causal claim is made from it. The primary environmental relationship figure shows observed PM₂.₅ against Sentinel-2 NDVI at 1,000 m with a descriptive trend line and year coloring. It is included to summarize the observed sample, not to replace the pre-specified DML analysis.

## Feature-importance caution

Random Forest impurity importance and LightGBM gain importance are reported separately and are not numerically interchangeable. In this run, temporal variables dominate aggregate tree importance, with the largest share associated with temporal features. In particular, `month_cos` is highly influential in both tree models. This is scientifically plausible for seasonal PM₂.₅ prediction but also highlights the risk that prediction is driven by recurring temporal regimes rather than a causal vegetation relationship. The training predictors contain 439 absolute-correlation pairs at or above 0.90, so correlated NDVI/EVI scales, meteorological variables, and related feature families make individual importance rankings unstable. Feature importance is predictive/exploratory and is not a causal effect, treatment ranking, or proof of mechanism.

## Six static figures

The notebook creates exactly six high-resolution PNGs under `results/plots/`: a three-panel metric comparison; observed-versus-predicted scatterplots; residual diagnostics; a station-level spatial error map; Random Forest and LightGBM feature importance; and the descriptive NDVI–PM₂.₅ relationship. The figures were visually reviewed after execution for readable labels, units, non-misleading scales, high-resolution output, and explicit non-causal framing. The visual QA record is `visual_review.md`.

## Limitations and next steps

The year-balanced train/test design is not spatially independent, and station-level repetition can make performance look stronger than performance at unseen monitors. The baseline also uses contemporaneous environmental predictors, so predictive usefulness does not establish temporal precedence. The V3 DML implementation addresses a distinct partially linear causal estimand with station-grouped cross-fitting, dependence-aware intervals, exact lagged-treatment sensitivities, and expanding-time designs. No baseline model should be used to claim that green cover reduces PM₂.₅.

The most defensible next steps are a pre-registered time-window decision, negative-control or lead-treatment placebos, measurement-error analysis for satellite exposure, and a separately specified nonlinear dose-response design if a threshold question remains scientifically justified. These should be evaluated by pre-specified validity and stability rules rather than by selecting a favorable sign or smaller interval.

## Reproducibility

From the repository root, execute the notebook with a Python 3 Jupyter kernel:

```bash
jupyter nbconvert --to notebook --execute notebooks/baseline_regression_models_v3.ipynb --inplace --ExecutePreprocessor.timeout=900
python data/modeling_changes/baseline_predictive_v1/validate_baseline.py
```

For standard notebook execution, use a Python 3 kernel with pandas, NumPy, scikit-learn, matplotlib, seaborn, LightGBM, and Jupyter notebook execution support. The executed notebook and all derived outputs are isolated under `notebooks/baseline_regression_models_v3.ipynb` and `data/modeling_changes/baseline_predictive_v1/`. The `input_hashes.json` file records the SHA-256 hashes of the frozen master, train, and test inputs.

## Artifact inventory

| Artifact | Purpose |
|---|---|
| `notebooks/baseline_regression_models_v3.ipynb` | Executed single-notebook model workflow |
| `results/baseline_model_metrics.csv` | Train, test, and training-only grouped-CV metrics |
| `results/yearly_model_metrics.csv` | Test metrics by year |
| `results/seasonal_model_metrics.csv` | Test metrics by season |
| `results/residual_summary.csv` | Residual and extreme-event diagnostics |
| `results/feature_importance.csv` | RF impurity and LightGBM gain importance |
| `results/findings_report.txt` | Automated findings and guardrails |
| `results/plots/01_...` through `06_...` | Six static research figures |
| `input_hashes.json` | Protected-input integrity record |
| `run_config.json` | Reproducibility and model configuration metadata |
| `visual_review.md` | Static figure quality review |

## References

[1]: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics "scikit-learn regression metrics"
[2]: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data "scikit-learn grouped cross-validation"
[3]: https://lightgbm.readthedocs.io/en/stable/Parameters.html "LightGBM parameters"

The definitions and implementation conventions for R², RMSE, MAE, grouped cross-validation, and LightGBM configuration follow the cited library documentation [1] [2] [3].
