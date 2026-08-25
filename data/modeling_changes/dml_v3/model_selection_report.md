# Nuisance Learner Benchmark

> This benchmark compares nuisance learners by held-out predictive loss. It does not select a causal estimate by its sign, magnitude, or interval.

The selected outcome learner is `extra_trees` and the selected treatment learner is `hist_gradient_boosting` under the pre-specified minimum-RMSE rule. The resulting sensitivity estimate is -25.996936 with station-clustered 95% interval [-68.265131, 16.271259].

| learner                |   y_rmse |   y_mae |     y_r2 |    t_rmse |     t_mae |     t_r2 |
|:-----------------------|---------:|--------:|---------:|----------:|----------:|---------:|
| hist_gradient_boosting |  25.055  | 16.13   | 0.860111 | 0.0833606 | 0.0639945 | 0.467057 |
| random_forest          |  24.4603 | 16.0759 | 0.866673 | 0.086756  | 0.0632355 | 0.422758 |
| extra_trees            |  24.2032 | 15.6344 | 0.869461 | 0.0843662 | 0.0605121 | 0.454122 |

The benchmark remains a sensitivity analysis. Learner selection based on the same finite sample can still add model-selection uncertainty, so the original pre-specified HistGradientBoosting result remains separately reported.
