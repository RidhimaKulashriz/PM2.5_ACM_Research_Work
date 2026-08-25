"""Validate the executed V3 baseline-predictive package without changing inputs."""
from pathlib import Path
import hashlib
import json
import math

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'data' / 'modeling_changes' / 'baseline_predictive_v1'
RESULTS = PACKAGE / 'results'
PLOTS = RESULTS / 'plots'
NOTEBOOK = ROOT / 'notebooks' / 'baseline_regression_models_v3.ipynb'

for path in [PACKAGE / 'input_hashes.json', PACKAGE / 'run_config.json', NOTEBOOK,
             RESULTS / 'baseline_model_metrics.csv', RESULTS / 'yearly_model_metrics.csv',
             RESULTS / 'seasonal_model_metrics.csv', RESULTS / 'residual_summary.csv',
             RESULTS / 'feature_importance.csv', RESULTS / 'findings_report.txt']:
    assert path.exists() and path.stat().st_size > 0, f'Missing output: {path}'

plot_names = [f'{i:02d}_{name}.png' for i, name in enumerate([
    'model_performance', 'observed_vs_predicted', 'residual_diagnostics',
    'spatial_error_map', 'feature_importance', 'environmental_relationship'
], start=1)]
assert sorted(p.name for p in PLOTS.glob('*.png')) == sorted(plot_names)
for name in plot_names:
    with Image.open(PLOTS / name) as image:
        assert image.width >= 1200 and image.height >= 900

nb = json.loads(NOTEBOOK.read_text())
assert nb['nbformat'] == 4
code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
assert len(code_cells) == 9 and all(cell.get('execution_count') is not None for cell in code_cells)
assert not any(output.get('ename') for cell in code_cells for output in cell.get('outputs', []) if isinstance(output, dict))

metrics = pd.read_csv(RESULTS / 'baseline_model_metrics.csv')
assert set(metrics['model']) == {'Linear Regression', 'Random Forest', 'LightGBM'}
for column in ['test_R2', 'test_RMSE', 'test_MAE', 'CV_RMSE_mean', 'CV_MAE_mean']:
    assert metrics[column].map(lambda x: math.isfinite(float(x))).all()
assert metrics['test_RMSE'].min() >= 0 and metrics['test_MAE'].min() >= 0
assert set(pd.read_csv(RESULTS / 'yearly_model_metrics.csv')['group'].astype(int)) == {2022, 2023, 2024, 2025}
assert set(pd.read_csv(RESULTS / 'seasonal_model_metrics.csv')['group']) == {'Winter', 'Summer', 'Monsoon', 'Post-monsoon'}
pred = pd.read_csv(RESULTS / 'test_predictions.csv')
assert len(pred) == 323 and pred.filter(regex='predicted_pm25$').shape[1] == 3

hashes = json.loads((PACKAGE / 'input_hashes.json').read_text())
for relative, expected in hashes.items():
    digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    assert digest == expected, f'Protected input changed: {relative}'

findings = (RESULTS / 'findings_report.txt').read_text().lower()
for phrase in ['predictive', 'not causal', 'accuracy, precision, and recall', 'feature importance']:
    assert phrase in findings

print('PASS: executed notebook, exact six plots, finite metrics, required diagnostics, and protected V3 input hashes are valid.')
print('Best model by test RMSE:', metrics.sort_values(['test_RMSE', 'test_MAE', 'model']).iloc[0]['model'])
