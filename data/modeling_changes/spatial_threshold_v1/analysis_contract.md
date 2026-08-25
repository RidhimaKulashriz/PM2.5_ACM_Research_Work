# V3 Spatial Surface and Threshold Analysis Contract

## Purpose

This package extends the V3 work with two separate descriptive/predictive analyses. It does not alter the DML estimand, replace the locked DML result, or claim that prediction and threshold evidence are causal.

## Protected inputs

The following paths are read-only:

- `data/modeling_changes/datasets/`
- `data/modeling_changes/splits/`
- `reports/baseline_results_v3/`
- Existing DML and baseline outputs already committed under `data/modeling_changes/dml_v3/` and `data/modeling_changes/baseline_predictive_v1/`

The new outputs are isolated under `data/modeling_changes/spatial_threshold_v1/`. SHA-256 hashes for the V3 master, train, and locked-test inputs are recorded.

## Spatial prediction-surface contract

The target is monthly station-level `pm25`. The model is the locked predictive-baseline LightGBM configuration, trained on the canonical V3 training split only. Predictors are numeric environmental, temporal, and coordinate variables with target-derived and split-membership fields excluded. The locked test split is used only for final prediction diagnostics.

Because the V3 repository contains station coordinates but no validated prediction-grid covariate table, the reported surface is **station-supported**: predicted PM₂.₅ values are produced at observed station coordinates and aggregated by station. No interpolation, IDW filling, raster extrapolation, or unobserved-grid prediction is performed. A continuous spatial raster remains a separate future data-engineering task.

The evaluation reports overall and station-level test error, station coverage, coordinate support, and prediction ranges. Spatial figures are descriptive maps of model-supported station predictions, not causal effect maps.

## Threshold-analysis contract

The exposure is `sentinel2_ndvi_mean_1000m`; the outcome is `pm25`. The analysis is a predictive/associational threshold screen, not a causal dose-response estimate. Candidate breakpoints are fixed before test evaluation as the training-treatment quantiles from 0.10 through 0.90 at 0.05 increments. The locked test split is never used to select the breakpoint.

For each candidate breakpoint, a piecewise-linear Ridge model uses the exposure’s below-breakpoint and above-breakpoint components plus the pre-specified numeric V3 control set. Five station-grouped training folds select the breakpoint by mean validation RMSE. The candidate grid, model family, alpha, fold assignment, and selection rule are recorded.

A breakpoint is considered **stable only if** it is selected in a substantial majority of station-grouped folds/bootstrap resamples and is not at a grid boundary. Otherwise the report must state that no stable threshold is identified. Even a stable predictive breakpoint is not a vegetation threshold for intervention: it can reflect confounding, measurement error, seasonality, station structure, or model misspecification.

## Reporting rules

No threshold is selected by visual appeal, causal coefficient sign, p-value, or locked-test performance. Extreme PM₂.₅ observations are retained. No unqualified PM₂.₅ lag, interpolation, IDW, or outcome imputation is introduced. All conclusions distinguish station-supported prediction from continuous spatial generalization and predictive breakpoint evidence from causal dose-response evidence.
