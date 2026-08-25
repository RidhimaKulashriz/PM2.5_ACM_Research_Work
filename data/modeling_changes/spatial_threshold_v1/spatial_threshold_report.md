# V3 Spatial PM₂.₅ Surface and Threshold Analysis Report

## Interpretation boundary

This package is a separate predictive/associational extension. It does not replace the V3 DML estimand, does not create causal effect maps, and does not establish a vegetation intervention threshold.

## Spatial prediction surface

The frozen V3 master contains 35 stations with fixed coordinates spanning 28.470691–28.822836 latitude and 76.933762–77.315809 longitude. It does not contain a validated grid of environmental covariates for unobserved locations. Therefore the “surface” is deliberately **station-supported**: LightGBM predictions are shown at observed station coordinates and summarized by station. No interpolation, IDW filling, raster extrapolation, or unobserved-grid prediction is performed.

The locked-test LightGBM diagnostic has R² **0.924811**, RMSE **18.814873 µg/m³**, and MAE **9.973572 µg/m³** across 323 rows and 34 stations. These are predictive metrics. They should not be interpreted as spatial generalization to a new monitoring network or as causal green-cover effects.

## Threshold screen

The exposure is `sentinel2_ndvi_mean_1000m` and the outcome is `pm25`. Candidate breakpoints were fixed at training quantiles 0.10–0.90 in 0.05 increments. A piecewise-linear Ridge model with the pre-specified control set was selected by five-fold station-grouped training CV; the locked test set was not used to choose the breakpoint. The selected breakpoint is **0.377235** (training quantile **0.75**).

The locked-test piecewise model has R² **0.742283**, RMSE **34.833294 µg/m³**, and MAE **25.690120 µg/m³**. The station-bootstrap same-grid selection share is **0.060**, with a bootstrap breakpoint interval of **[0.155651, 0.489307]**. Under the frozen stability rule, a stable threshold is **not identified**.

Even if a breakpoint is stable, it is a predictive association in this station-month sample. It may reflect confounding, measurement error, seasonality, station structure, or model misspecification. It must not be presented as “the amount of greenery required” or as a causal policy threshold.

## Limitations

The station-supported map is not a continuous PM₂.₅ raster. A continuous surface requires a validated prediction-grid covariate table and a spatially independent evaluation design. The threshold screen is not a nonlinear DML dose-response estimator; a causal threshold claim would require a separate estimand, pre-treatment exposure window, overlap analysis, and identification strategy.

## Reproducibility

```bash
python data/modeling_changes/spatial_threshold_v1/run_spatial_threshold.py
python data/modeling_changes/spatial_threshold_v1/validate_spatial_threshold.py
```

The input contract is frozen in `analysis_contract.md`, and SHA-256 hashes are recorded in `input_hashes.json`. The runner writes canonical CSV tables locally; because the fork routes CSV paths through Git LFS and browser upload cannot create LFS objects, the fork publishes direct JSON review mirrors in `results/README.md`. The JSON mirrors are unchanged serializations of the local tables.
