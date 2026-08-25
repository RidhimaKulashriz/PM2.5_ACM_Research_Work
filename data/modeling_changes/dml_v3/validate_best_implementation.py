from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[3]
dml = root / 'data/modeling_changes/dml_v3'
train_path = root / 'data/modeling_changes/splits/train.csv'
train = pd.read_csv(train_path)

rolling = pd.read_csv(dml / 'rolling_time_summary.csv')
rolling_folds = pd.read_csv(dml / 'rolling_time_folds.csv')
rolling_predictions = pd.read_csv(dml / 'rolling_time_predictions_primary.csv')
within = pd.read_csv(dml / 'within_station_time_summary.csv')
within_folds = pd.read_csv(dml / 'within_station_time_folds.csv')
within_predictions = pd.read_csv(dml / 'within_station_time_predictions.csv')
contract = (dml / 'analysis_contract.md').read_text()

assert len(rolling) == 1 and len(within) == 1
assert len(rolling_folds) == 36
assert len(within_folds) == 3
assert len(rolling_predictions) == 969
assert len(within_predictions) == 968
assert rolling_folds['n_train'].min() >= 12 * 20
assert rolling_folds['last_train_period'].max() < (rolling_folds['holdout_year'] * 12 + rolling_folds['holdout_month']).max()
assert within_folds['n_unseen_station_rows_excluded'].sum() == 1
assert within['n_stations'].iloc[0] == 34
assert rolling['n_stations'].iloc[0] == 35
for frame in [rolling, rolling_folds, rolling_predictions, within, within_folds, within_predictions]:
    numeric = frame.select_dtypes(include=np.number).to_numpy()
    assert np.isfinite(numeric).all(), frame.head()
assert set(rolling_predictions['fold']) == set(rolling_folds['fold'])
assert set(within_predictions['fold']) == set(within_folds['fold'])
assert 'causal coefficient' in contract and 'causal coefficient, interval width, or statistical significance may be used for learner selection' in contract
assert 'Protected inputs and outputs' in contract
h = hashlib.sha256(train_path.read_bytes()).hexdigest()
assert h == json.loads((dml / 'rolling_time_config.json').read_text())['input_sha256']['train.csv']
assert h == json.loads((dml / 'within_station_time_config.json').read_text())['input_sha256']['train.csv']
assert all(c in rolling_predictions.columns for c in ['station','year','month','y_residual','t_residual'])
assert all(c in within_predictions.columns for c in ['station','year','month','y_transformed','t_transformed','y_residual','t_residual'])
print('BEST_IMPLEMENTATION_VALIDATION: PASS')
print('rolling_time_summary')
print(rolling.to_string(index=False))
print('within_station_time_summary')
print(within.to_string(index=False))
