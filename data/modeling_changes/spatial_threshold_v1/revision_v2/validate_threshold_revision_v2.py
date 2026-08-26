from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'data/modeling_changes/spatial_threshold_v1/revision_v2'
RES = OUT / 'results'
required = [
    OUT / 'threshold_revision_v2.py',
    OUT / 'threshold_revision_summary.json',
    OUT / 'threshold_revision_report.md',
    OUT / 'input_hashes.json',
    RES / 'outer_fold_results.csv',
    RES / 'outer_oof_predictions.csv',
    RES / 'full_training_threshold_cv.csv',
    RES / 'bootstrap_stability.csv',
    RES / 'locked_test_comparison.csv',
]
for p in required:
    if not p.exists():
        raise AssertionError(f'Missing required file: {p}')

outer = pd.read_csv(RES / 'outer_fold_results.csv')
oof = pd.read_csv(RES / 'outer_oof_predictions.csv')
cv = pd.read_csv(RES / 'full_training_threshold_cv.csv')
boot = pd.read_csv(RES / 'bootstrap_stability.csv')
test = pd.read_csv(RES / 'locked_test_comparison.csv')
summary = json.loads((OUT / 'threshold_revision_summary.json').read_text())

if len(outer) != 5 or sorted(outer.outer_fold.tolist()) != [1, 2, 3, 4, 5]:
    raise AssertionError('Expected five outer station-grouped folds')
if len(oof) != 1292 or len(boot) != 100 or len(test) != 1:
    raise AssertionError('Unexpected output row counts')
if len(cv) < 10:
    raise AssertionError('Candidate threshold grid is unexpectedly short')
for frame, cols in [
    (outer, ['selected_threshold', 'inner_cv_rmse', 'segmented_rmse', 'linear_rmse']),
    (oof, ['observed_pm25', 'segmented_predicted', 'linear_predicted', 'selected_threshold']),
    (cv, ['threshold', 'inner_cv_rmse']),
    (boot, ['selected_threshold', 'selected_quantile', 'segmented_cv_rmse', 'linear_cv_rmse', 'rmse_improvement']),
    (test, ['selected_threshold', 'segmented_r2', 'segmented_rmse', 'segmented_mae', 'linear_r2', 'linear_rmse', 'linear_mae']),
]:
    if not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
        raise AssertionError('Non-finite threshold output values')

if summary['threshold_supported'] is not False:
    raise AssertionError('Unstable threshold was incorrectly marked supported')
if summary['bootstrap_modal_stability'] >= 0.50:
    raise AssertionError('Reported threshold stability unexpectedly exceeds frozen support rule')
if summary['outer_mean_rmse_improvement'] >= 0:
    raise AssertionError('Outer segmented model unexpectedly marked as improvement')
if abs(float(test.loc[0, 'rmse_improvement_linear_minus_segmented']) - (float(test.loc[0, 'linear_rmse']) - float(test.loc[0, 'segmented_rmse']))) > 1e-9:
    raise AssertionError('Locked-test RMSE improvement arithmetic mismatch')

hashes = json.loads((OUT / 'input_hashes.json').read_text())
protected = json.loads((ROOT / 'data/modeling_changes/dml_v3/dml_config.json').read_text())['input_sha256']
for path, digest in hashes.items():
    if protected.get(Path(path).name) != digest:
        raise AssertionError(f'Protected hash mismatch: {path}')

report = (OUT / 'threshold_revision_report.md').read_text().lower()
for phrase in ['nested station-grouped cross-validation', 'no stable threshold', 'not a policy or causal threshold', 'locked test is evaluated once']:
    if phrase not in report:
        raise AssertionError(f'Missing report guardrail: {phrase}')
print('THRESHOLD REVISION VALIDATION PASS')
print('selected threshold:', summary['selected_threshold'])
print('bootstrap modal stability:', summary['bootstrap_modal_stability'])
print('outer segmented minus linear RMSE improvement:', summary['outer_mean_rmse_improvement'])
print('protected input hashes: matched')
