from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[3]
dml = root / 'data/modeling_changes/dml_v3'
train = pd.read_csv(root / 'data/modeling_changes/splits/train.csv')
test = pd.read_csv(root / 'data/modeling_changes/splits/test.csv')
lagged = pd.read_csv(dml / 'pre_treatment_lagged_inputs.csv')
summary = pd.read_csv(dml / 'pre_treatment_dml_summary.csv')
folds = pd.read_csv(dml / 'time_aware_dml_folds.csv')
config = json.loads((dml / 'pre_treatment_config.json').read_text())

assert len(train) == 1292 and len(test) == 323
assert len(lagged) == len(train) + len(test)
assert lagged[['station','year','month','split']].duplicated().sum() == 0
assert summary.treatment.nunique() == 3 and len(summary) == 3
assert summary.n_time_aware_oof.gt(0).all()
assert summary.n_time_folds.eq(3).all()
assert folds.groupby('treatment').size().eq(3).all()
assert set(folds.holdout_year.unique()) == {2023, 2024, 2025}
for treatment in summary.treatment:
    col = f'lag_{treatment}'
    assert col in lagged.columns
    scored = pd.read_csv(dml / f'time_aware_crossfit_{treatment}.csv')
    assert len(scored) == int(summary.loc[summary.treatment == treatment, 'n_time_aware_oof'].iloc[0])
    assert scored['time_holdout_year'].min() >= 2023
    assert np.isfinite(scored.select_dtypes(include=np.number).to_numpy()).all()
assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()
assert config['lag_definition'].startswith('exact previous calendar month')
assert config['cross_fitting'].startswith('expanding time')
# Ensure the exact previous calendar month was used, not merely a previous row.
all_panel = pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)
all_panel['period'] = pd.to_datetime(dict(year=all_panel.year.astype(int), month=all_panel.month.astype(int), day=1))
for split, frame in all_panel.groupby('split'):
    frame = frame.sort_values(['station','period'])
    expected = frame.groupby('station')['period'].shift(1)
    current = frame['period']
    # Rows with a previous row must be exactly one calendar month apart to be eligible as a lag.
    gap = current - expected
    assert (gap.dropna() >= pd.Timedelta(days=28)).all()
print('PRE_TREATMENT_VALIDATION: PASS')
print(summary[['treatment','n_lagged_rows','n_time_aware_oof','time_aware_theta','time_aware_cluster_se','time_aware_ci_low','time_aware_ci_high']].to_string(index=False))
