# Rolling-Origin Time-Aware DML

> This is a temporal sensitivity analysis. Each holdout month is predicted using only strictly earlier calendar months. It is not a replacement for the primary station-grouped DML specification.

The primary treatment is `sentinel2_ndvi_mean_1000m`. The analysis scores 969 rows across 36 chronological holdout months, with a minimum of 12 earlier calendar months in the first training window.

The rolling-origin estimate is -65.311631 with station-clustered 95% interval [-86.982328, -43.640934]. Outcome nuisance RMSE/R² are 28.257072/0.839923; treatment nuisance RMSE/R² are 0.053536/0.784441.

This design reduces temporal leakage but can still be affected by time-varying confounding, station-level dependence, limited early training support, and treatment measurement error. The result must be interpreted jointly with the pre-treatment, spatial-block, cluster-robust, and fold-stability diagnostics.
