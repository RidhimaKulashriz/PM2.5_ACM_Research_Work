# Attachment-to-Repository Implementation Scope

## Implemented in the V3 DML package

The repository now implements the attachment's core causal workflow through orthogonal residualization, station-grouped cross-fitting, explicit controls, dependence-aware uncertainty, spatial-block sensitivity, exact previous-calendar-month NDVI treatments, expanding time-aware holdouts, continuous validation, metric logging, audit trails, and static diagnostic figures.

The metric audit logs RMSE, MAE, and R2 for both outcome and treatment nuisance models. It also logs a fixed PM2.5 concentration-band agreement metric for presentation readability. The concentration-band metric is descriptive only and is not a substitute for regression evaluation or an official AQI classification.

## Intentionally not applied automatically

Forward-fill, backward-fill, linear interpolation, and inverse-distance interpolation were not applied to PM2.5 or NDVI by default. Filling observed outcomes or treatments can change the estimand and can create artificial temporal or spatial signal. The new missingness audits expose where data are missing so that any imputation decision can be justified in a separate sensitivity specification.

The attachment's proposed 1-hour and 24-hour PM2.5 lags were not inserted as ordinary controls in the primary DML model. A contemporaneous or post-treatment PM2.5 lag can block part of the treatment pathway or create leakage. Such lags should only be added after defining an estimand and a pre-treatment information set.

A threshold or diminishing-returns curve was not claimed from the current partially linear DML estimate. The current estimator targets a constant marginal effect. Threshold extraction requires a separately specified nonlinear dose-response design, overlap checks across the treatment range, and multiple-testing control.

Spatial interpolation of missing sensor outcomes and official AQI accuracy were also not treated as causal-model outputs. The code records a presentation-only PM2.5 band agreement metric and keeps the primary inference in the continuous PM2.5 scale.

## Next scientifically defensible extension

The next extension should compare pre-specified lag windows and nonlinear dose-response models under rolling-origin time-aware validation, with station-clustered or block-resampled inference. Those models should remain sensitivity analyses until the exposure timing, support, and causal estimand are agreed with the research supervisor.
