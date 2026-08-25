# V3 DML Analysis Contract

## Purpose

This contract freezes the analysis decisions for the strongest defensible V3 implementation before additional sensitivity models are compared. It prevents selecting a specification because it produces a preferred sign, smaller p-value, or narrower interval.

## Primary estimand

The primary estimand is the partially linear average marginal effect of a one-unit increase in `sentinel2_ndvi_mean_1000m` on monthly station-level `pm25`, conditional on the pre-specified temporal, spatial, meteorological, population, road-density, and non-vegetation land-cover controls. The primary analysis is observational and does not establish causality without exchangeability, consistency, positivity, and valid measurement assumptions.

The raw one-unit NDVI scale is retained for reproducibility. A standardized-treatment effect may be reported as a secondary scale conversion but must not replace the raw-scale estimand.

## Protected inputs and outputs

The following paths are read-only for all new analyses:

- `data/modeling_changes/datasets/`
- `data/modeling_changes/splits/`
- `reports/baseline_results_v3/`
- Existing Version 1 DML outputs already committed under `data/modeling_changes/dml_v3/`

New work must use new filenames under `data/modeling_changes/dml_v3/` and must record SHA-256 hashes for the master, train, and locked-test inputs.

## Sample and split contract

The canonical inputs are the 1,615-row V3 master, 1,292-row training split, and 323-row locked-test split. The locked test split is diagnostic only. No test outcome or treatment is used to fit nuisance models or select a primary specification.

Primary cross-fitting uses five station-grouped folds. Temporal sensitivity uses expanding-origin holdouts in chronological order. Lagged treatments must be exact previous calendar-month values constructed within split boundaries; no forward fill, backward fill, interpolation, or cross-split lag is allowed.

## Nuisance learners

The pre-specified primary nuisance learner is `HistGradientBoostingRegressor` with the existing configuration. Learner sensitivity may compare Random Forest and Extra Trees using held-out nuisance prediction loss only. Neither the causal coefficient, interval width, or statistical significance may be used for learner selection.

## Inference contract

The primary uncertainty check for station-month observations is station-clustered inference. Wild cluster bootstrap is a secondary dependence-aware check with its seed and replicate count recorded. Confidence intervals must not be narrowed by trimming, learner selection, or treatment re-scaling unless that rule was fixed before comparison.

## Preferred-specification rule

A preferred specification is selected by the following hierarchy:

1. Satisfies all leakage, split-integrity, finite-value, and control-manifest checks.
2. Uses a scientifically defensible exposure timing definition.
3. Uses dependence-aware uncertainty appropriate for the station panel.
4. Has stable nuisance prediction and adequate residualized-treatment support.
5. Is reported even when its interval includes zero.

If specifications disagree in sign or have unstable fold estimates, the correct conclusion is design sensitivity, not selection of the most favorable result.

## Deferred methods

Nonlinear thresholds, causal forests, interpolation-based imputation, IDW imputation, and ordinary PM2.5 lags are deferred until a separate estimand, leakage audit, and overlap analysis are specified. They must not be introduced merely to improve the apparent result.
