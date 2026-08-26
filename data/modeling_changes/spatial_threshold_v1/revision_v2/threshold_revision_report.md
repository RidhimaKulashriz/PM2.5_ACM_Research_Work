# Corrected threshold analysis

This revision uses nested station-grouped cross-validation. Each outer fold selects the breakpoint using only its outer-training stations and four-fold inner grouped CV, then evaluates the segmented and no-break linear models on unseen stations. The locked test is evaluated once after full-training selection.

## Finding

No stable threshold is supported under the frozen rule. The breakpoint is treated as a predictive screen, not a policy or causal threshold.

{
  "exposure": "sentinel2_ndvi_mean_1000m",
  "controls": [
    "latitude",
    "longitude",
    "year",
    "month",
    "month_sin",
    "month_cos",
    "season_encoded",
    "era5_temp_mean",
    "era5_rh_mean",
    "era5_wind_speed_mean",
    "era5_blh_mean",
    "population_density_2025_1000m",
    "road_density_1000m",
    "major_road_density_1000m",
    "dynamicworld_2025_water_frac_1000m",
    "dynamicworld_2025_built_frac_1000m",
    "dynamicworld_2025_bare_frac_1000m"
  ],
  "quantile_grid": [
    0.2,
    0.25,
    0.3,
    0.35,
    0.39999999999999997,
    0.44999999999999996,
    0.49999999999999994,
    0.5499999999999999,
    0.5999999999999999,
    0.6499999999999999,
    0.7,
    0.7499999999999998,
    0.7999999999999998
  ],
  "selected_threshold": 0.40328876778424066,
  "selected_quantile": 0.7999999999999998,
  "bootstrap_modal_quantile": 0.7999999999999998,
  "bootstrap_modal_stability": 0.31,
  "outer_mean_rmse_improvement": -0.16138758355563781,
  "outer_segmented_rmse": 37.10335385026535,
  "outer_linear_rmse": 36.94196626670971,
  "threshold_supported": false,
  "locked_test_r2_segmented": 0.7555030418956294,
  "locked_test_r2_linear": 0.7547186372713847
}

The threshold analysis is predictive/associational and does not establish that changing vegetation causes PM2.5 to change.
