# Visual review of V3 baseline figures

The six generated PNGs were visually reviewed as a contact sheet after notebook execution.

- `01_model_performance.png` uses three separate metric panels, avoiding incompatible R² and error-unit scales. Labels and bar annotations are readable.
- `02_observed_vs_predicted.png` contains one panel per model, a 1:1 reference line, and R²/RMSE labels. The high-PM₂.₅ tail is visibly more dispersed for Linear Regression than for the tree models.
- `03_residual_diagnostics.png` contains one panel per model and a zero-residual reference line. The linear model shows a visibly stronger nonlinear pattern and an extreme negative residual; tree-model residuals are tighter but still have high-pollution outliers.
- `04_spatial_error_map.png` is a coordinate-based station map with an explicit descriptive/non-causal note, labeled axes, a quantitative colorbar, and labels for the highest-error stations.
- `05_feature_importance.png` compares Random Forest impurity importance and LightGBM gain importance, clearly labels both measures, and states that importance is not a causal effect. The different measures are not numerically compared as though they were on the same scale.
- `06_environmental_relationship.png` shows the pre-specified primary Sentinel-2 NDVI versus observed PM₂.₅ relationship, colored by year, with a descriptive trend line and explicit non-causal annotation.

No plot was altered based on whether it produced a favorable sign. Individual figures are high-resolution and suitable for static review or paper drafting, subject to normal publication editing.
The full-resolution performance chart was also checked: the three panels have readable titles, units, and non-misleading separate scales. The full-resolution station error map was checked: the latitude/longitude axes and µg/m³ colorbar are readable, labeled high-error stations are visible, and the descriptive/non-causal warning is present. The map is intentionally a station-coordinate visualization rather than an interpolated surface.
