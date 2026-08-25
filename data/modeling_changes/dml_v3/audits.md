# DML V3 Audit Log

## 2026-08-24 — Initial DML implementation and validation

The V3 master dataset and designated train/test split were loaded from `data/modeling_changes/datasets/` and `data/modeling_changes/splits/`. The integrity checks passed: 1,615 master rows, 1,292 training rows, 323 locked test rows, no station-year-month key overlap, and exact train/test key-universe preservation. IIT Delhi remains train-only as documented by the prior split audit.

The primary treatment is `sentinel2_ndvi_mean_1000m`, interpreted as 1,000 m-buffer Sentinel-2 NDVI. Sensitivity treatments are the 500 m Sentinel-2 NDVI and 1,000 m MODIS NDVI means. The outcome is monthly station-level `pm25`.

DML uses 5-fold `GroupKFold` cross-fitting by station and a median-imputation plus `HistGradientBoostingRegressor` nuisance pipeline. Controls were selected before fitting from temporal/spatial variables, ERA5 meteorology, 2025 population density, 2025 road density, and non-vegetation Dynamic World built/water/bare/valid-pixel context. Green-cover proxies, Sentinel-5P NO2/pollution proxies, and contemporaneous MODIS/LST or gradient variables were excluded to avoid treatment-proxy and plausible post-treatment adjustment.

The primary cross-fitted estimate is -21.180373 µg/m³ per one-unit raw NDVI increase, with standard error 5.848786 and 95% influence-function interval [-32.643994, -9.716752]. The standardized effect is -2.419481 µg/m³ per one training-sample standard deviation of the treatment. The 500 m Sentinel-2 sensitivity interval crosses zero, while the MODIS 1,000 m sensitivity estimate is negative with an interval that does not cross zero. These are observational, assumption-dependent estimates and are not by themselves proof of causality.

The external test values are labeled diagnostics rather than fresh causal inference: nuisance models are fit on training data only, then the orthogonal slope is evaluated on the locked test rows. The test diagnostic for the primary treatment is -25.415084.

The validation script confirmed all generated cross-fit files contain 1,292 rows, all five station-held-out fold IDs, and finite nuisance predictions/residuals. The canonical V3 datasets and `baseline_results_v3` were not modified.

## Reproducibility

Run `python data/modeling_changes/dml_v3/run_dml.py` followed by `python data/modeling_changes/dml_v3/validate_dml.py` from the repository root. Generated artifacts are isolated under `data/modeling_changes/dml_v3/`.

## Caveats for the research report

The partially linear DML interpretation requires conditional exchangeability, overlap after adjustment, a well-defined treatment, and sufficiently controlled dependence/measurement processes. The current station-month panel may still contain unmeasured spatial and temporal confounding, treatment measurement error, serial dependence, and exposure/outcome simultaneity. The next methodological improvement should add dependence-robust uncertainty and, if scientifically justified, a pre-treatment exposure design or a spatially grouped sensitivity analysis.

## 2026-08-25 — Immutable-input baseline predictive modeling package

A separate predictive baseline package was added under `data/modeling_changes/baseline_predictive_v1/` using the frozen V3 master dataset and canonical split from `data/modeling_changes/datasets/` and `data/modeling_changes/splits/`. The package does not modify those protected inputs or existing DML/baseline outputs. The audit passed for 1,615 master rows, 1,292 training rows, 323 locked-test rows, exact key-universe preservation, disjoint station-year-month keys, all four years in both partitions, and IIT Delhi remaining train-only. The split files contain one documented split-only `season` field; the master schema otherwise matches.

The executed single-notebook workflow compares Linear Regression, 300-tree Random Forest, and a conservative fixed LightGBM baseline. Linear Regression uses train-fitted median imputation and standardization; tree learners use train-fitted median imputation without unnecessary scaling. Five-fold `GroupKFold` is grouped by station on training rows only. The locked test is used only for final diagnostic evaluation. The continuous target is `pm25`, so R², RMSE, MAE, and supplementary median absolute error are reported. Accuracy, precision, and recall are not presented as regression metrics, and no arbitrary high-PM2.5 classification threshold was introduced.

Locked-test results were: Linear Regression R² 0.820184, RMSE 29.096270, MAE 21.637563; Random Forest R² 0.908961, RMSE 20.703122, MAE 11.390408; and LightGBM R² 0.924285, RMSE 18.880479, MAE 10.088022. LightGBM is therefore the best predictive baseline by locked-test RMSE and MAE, not a causal model. Tree-model train scores are much higher than locked-test scores, so overfitting/memorization risk is explicitly flagged. The training predictors have 439 absolute-correlation pairs at or above 0.90, making individual feature-importance rankings unstable; temporal variables dominate aggregate tree importance in this run.

The package also reports year-wise and season-wise metrics, retains extreme PM2.5 observations, summarizes residuals, identifies high-error stations, records input SHA-256 hashes, and generates exactly six static high-resolution figures. The full notebook executed successfully, the validator passed, all outputs were finite, and the six figures were visually reviewed. Figures and feature importance are descriptive/predictive only and do not establish causality. The full narrative is in `data/modeling_changes/baseline_predictive_v1/baseline_predictive_report.md`; the automated report is `data/modeling_changes/baseline_predictive_v1/results/findings_report.txt`.

## Reproducibility

Execute the notebook with a Python 3 Jupyter kernel after installing its declared dependencies. The notebook is already stored in executed form at `notebooks/baseline_regression_models_v3.ipynb`; derived outputs are isolated under `data/modeling_changes/baseline_predictive_v1/`. The DML headline and time-aware causal sensitivities remain separate and must not be replaced by predictive baseline performance.

## 2026-08-25 — Station-supported spatial surface and threshold screen

A separate package was added under `data/modeling_changes/spatial_threshold_v1/` using the same frozen V3 master, canonical train split, and locked test split. The protected-input hashes match the existing DML configuration exactly: 1,615 master rows, 1,292 training rows, and 323 locked-test rows across 35 fixed-coordinate stations. No canonical dataset, split, baseline result, or existing DML output was modified.

The spatial prediction extension uses the locked LightGBM predictive baseline with five-fold station-grouped training CV and a locked-test diagnostic. Because the V3 repository contains observed station coordinates but no validated grid of environmental covariates, the surface is intentionally station-supported: predictions are shown at observed station coordinates and aggregated by station. No interpolation, IDW, raster extrapolation, or unobserved-grid prediction is performed. The locked-test diagnostic is R² 0.924811, RMSE 18.814873 µg/m³, and MAE 9.973572 µg/m³ across 323 rows and 34 stations.

The threshold extension is a separately specified predictive/associational screen for `sentinel2_ndvi_mean_1000m` using a piecewise-linear Ridge model with 17 candidate training quantiles from 0.10 to 0.90 and five station-grouped training folds. The selected breakpoint is NDVI 0.377235, approximately the 0.75 training quantile. Its locked-test diagnostic is R² 0.742283, RMSE 34.833294 µg/m³, and MAE 25.690121 µg/m³. In 100 station-bootstrap repetitions, the same candidate quantile was selected only 6% of the time, so the frozen stability rule correctly concludes that no stable threshold is identified.

The four static figures, report, input hashes, JSON mirrors, visual review, and validator were generated successfully. The threshold screen is not a causal dose-response estimate and must not be translated into a policy threshold or a required amount of greenery. A continuous PM₂.₅ raster and any causal threshold claim remain separate Version 2 research tasks requiring a validated prediction-grid covariate table, an explicit nonlinear estimand, overlap checks, pre-treatment exposure timing, and a defensible identification strategy.

## Reproducibility

Run `python data/modeling_changes/spatial_threshold_v1/run_spatial_threshold.py` followed by `python data/modeling_changes/spatial_threshold_v1/validate_spatial_threshold.py` from the repository root. Outputs are isolated under `data/modeling_changes/spatial_threshold_v1/`.
