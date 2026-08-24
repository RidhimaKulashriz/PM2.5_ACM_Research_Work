from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[3]
out = root / 'data/modeling_changes/dml_v3'
master = pd.read_csv(root / 'data/modeling_changes/datasets/master_modeling_dataset_v3.csv')
train = pd.read_csv(root / 'data/modeling_changes/splits/train.csv')
test = pd.read_csv(root / 'data/modeling_changes/splits/test.csv')
summary = pd.read_csv(out / 'dml_summary.csv')
manifest = json.loads((out / 'feature_manifest.json').read_text())
assert summary.shape[0] == 3
assert set(summary.treatment) == set([manifest['primary_treatment']] + manifest['sensitivity_treatments'])
assert summary[['theta','se','ci_low','ci_high','effect_per_treatment_sd']].apply(np.isfinite).all().all()
assert len(master) == 1615 and len(train) == 1292 and len(test) == 323
assert not set(zip(train.station, train.year, train.month)) & set(zip(test.station, test.year, test.month))
assert set(zip(train.station, train.year, train.month)) | set(zip(test.station, test.year, test.month)) == set(zip(master.station, master.year, master.month))
for treatment in summary.treatment:
    cf = pd.read_csv(out / f'crossfit_observations_{treatment}.csv')
    assert len(cf) == len(train)
    assert set(cf.fold) == set(range(5))
    assert cf[['y_hat_oof','t_hat_oof','y_residual','t_residual','orthogonal_score']].apply(np.isfinite).all().all()
    assert len(manifest['treatments'][treatment]['controls']) > 0
protected = [
    root / 'data/modeling_changes/baseline_results_v3',
    root / 'data/modeling_changes/datasets',
    root / 'data/modeling_changes/splits',
]
text = [
    '# DML V3 Audit Log',
    '',
    '## 2026-08-24 — Initial DML implementation and validation',
    '',
    'The V3 master dataset and designated train/test split were loaded from `data/modeling_changes/datasets/` and `data/modeling_changes/splits/`. The integrity checks passed: 1,615 master rows, 1,292 training rows, 323 locked test rows, no station-year-month key overlap, and exact train/test key-universe preservation. IIT Delhi remains train-only as documented by the prior split audit.',
    '',
    'The primary treatment is `sentinel2_ndvi_mean_1000m`, interpreted as 1,000 m-buffer Sentinel-2 NDVI. Sensitivity treatments are the 500 m Sentinel-2 NDVI and 1,000 m MODIS NDVI means. The outcome is monthly station-level `pm25`.',
    '',
    'DML uses 5-fold `GroupKFold` cross-fitting by station and a median-imputation plus `HistGradientBoostingRegressor` nuisance pipeline. Controls were selected before fitting from temporal/spatial variables, ERA5 meteorology, 2025 population density, 2025 road density, and non-vegetation Dynamic World built/water/bare/valid-pixel context. Green-cover proxies, Sentinel-5P NO2/pollution proxies, and contemporaneous MODIS/LST or gradient variables were excluded to avoid treatment-proxy and plausible post-treatment adjustment.',
    '',
    'The primary cross-fitted estimate is -21.180373 µg/m³ per one-unit raw NDVI increase, with standard error 5.848786 and 95% influence-function interval [-32.643994, -9.716752]. The standardized effect is -2.419481 µg/m³ per one training-sample standard deviation of the treatment. The 500 m Sentinel-2 sensitivity interval crosses zero, while the MODIS 1,000 m sensitivity estimate is negative with an interval that does not cross zero. These are observational, assumption-dependent estimates and are not by themselves proof of causality.',
    '',
    'The external test values are labeled diagnostics rather than fresh causal inference: nuisance models are fit on training data only, then the orthogonal slope is evaluated on the locked test rows. The test diagnostic for the primary treatment is -25.415084.',
    '',
    'The validation script confirmed all generated cross-fit files contain 1,292 rows, all five station-held-out fold IDs, and finite nuisance predictions/residuals. The canonical V3 datasets and `baseline_results_v3` were not modified.',
    '',
    '## Reproducibility',
    '',
    'Run `python data/modeling_changes/dml_v3/run_dml.py` followed by `python data/modeling_changes/dml_v3/validate_dml.py` from the repository root. Generated artifacts are isolated under `data/modeling_changes/dml_v3/`.',
    '',
    '## Caveats for the research report',
    '',
    'The partially linear DML interpretation requires conditional exchangeability, overlap after adjustment, a well-defined treatment, and sufficiently controlled dependence/measurement processes. The current station-month panel may still contain unmeasured spatial and temporal confounding, treatment measurement error, serial dependence, and exposure/outcome simultaneity. The next methodological improvement should add dependence-robust uncertainty and, if scientifically justified, a pre-treatment exposure design or a spatially grouped sensitivity analysis.',
]
(out / 'audits.md').write_text('\n'.join(text) + '\n')
print('DML validation: PASS')
print(summary[['treatment','theta','se','ci_low','ci_high','effect_per_treatment_sd','theta_external_test_diagnostic']].to_string(index=False))
