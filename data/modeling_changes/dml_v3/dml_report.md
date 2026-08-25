# V3 Double Machine Learning Results

> This is an observational, partially linear DML analysis. The coefficient is interpretable as a causal effect only under the stated identification assumptions; it is not proof of causality from this dataset alone.

## Estimand and design

The primary treatment is `sentinel2_ndvi_mean_1000m` (Sentinel-2 NDVI averaged within the 1,000 m buffer), the outcome is monthly station-level `pm25`, and the estimand is the partially linear average treatment effect per one-unit increase in the raw NDVI treatment. The analysis uses the frozen V3 training split only for cross-fitted estimation. Cross-fitting is grouped by station with 5 folds, so each held-out fold contains stations not used to fit its nuisance models.

The nuisance learners estimate the conditional mean of the outcome and treatment from pre-specified temporal/spatial, ERA5 meteorology, 2025 population, road-density, and non-vegetation built/water/bare land-cover controls. Green-cover proxies, Sentinel-5P NO₂/pollution variables, and contemporaneous MODIS/LST variables are excluded to avoid adjusting for treatment proxies, pollutant proxies, or plausible post-treatment mediators.

## Results

| Treatment | DML estimate | SE | 95% CI | N | N stations | External test diagnostic |
|---|---:|---:|---:|---:|---:|---:|
| `sentinel2_ndvi_mean_1000m` | -21.180373 | 5.848786 | [-32.643994, -9.716752] | 1292 | 35 | -25.415084 |
| `sentinel2_ndvi_mean_500m` | -7.046748 | 4.651392 | [-16.163475, 2.069980] | 1292 | 35 | -27.352185 |
| `modis_ndvi_mean_1000m` | -17.618217 | 6.574982 | [-30.505183, -4.731252] | 1292 | 35 | -19.953420 |

The primary cross-fitted estimate is reported on the raw NDVI scale, so its units are µg/m³ of PM₂.₅ per one-unit NDVI increase. Because NDVI has a bounded, small empirical range, readers should not interpret a one-unit change as a typical real-world intervention. The corresponding estimate per one-standard-deviation increase is included in `dml_summary.csv`.

## Diagnostics and limitations

The cross-fitted nuisance R² values for the primary outcome and treatment models were 0.860 and 0.467, respectively. The residualized treatment standard deviation was 0.083339; this documents treatment overlap after adjustment but does not establish exchangeability.

The 95% interval uses the empirical influence-function variance for the partially linear orthogonal score. It should be treated as model-based uncertainty, not as a correction for unmeasured confounding, temporal dependence, spatial dependence, measurement error, or treatment/outcome simultaneity. IIT Delhi remains train-only by design and is not part of the locked test set.

The external-test column is deliberately labeled a diagnostic: nuisance models are fit on the training split and the orthogonal slope is evaluated on the locked test split. It is not a second independent DML inference procedure and has no confidence interval here.

## Reproducibility

Run `python data/modeling_changes/dml_v3/run_dml.py` from the repository root. All generated artifacts remain in `data/modeling_changes/dml_v3/`; the canonical datasets and `baseline_results_v3` are read-only inputs for this analysis.

## Files

| File | Purpose |
|---|---|
| `dml_summary.csv` | Cross-fitted estimates, uncertainty, standardized effects, and diagnostics |
| `crossfit_observations_<treatment>.csv` | Fold IDs, nuisance predictions, residuals, and orthogonal scores |
| `nuisance_metrics.csv` | Outcome/treatment nuisance-model diagnostics |
| `external_test_diagnostics.csv` | Train-fitted nuisance validation on locked test rows |
| `feature_manifest.json` | Exact treatment/control roles and excluded-variable rationale |
| `dml_config.json` | Dataset hashes, software versions, seed, and model settings |
