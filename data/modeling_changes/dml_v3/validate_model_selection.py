from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[3]
dml = root / 'data/modeling_changes/dml_v3'
benchmark = pd.read_csv(dml / 'model_selection_benchmark.csv')
result = pd.read_csv(dml / 'model_selection_result.csv')
folds = pd.read_csv(dml / 'model_selection_folds.csv')
config = json.loads((dml / 'model_selection_config.json').read_text())

assert set(benchmark.learner) == {'hist_gradient_boosting', 'random_forest', 'extra_trees'}
assert len(result) == 1
assert len(folds) == 15
assert folds.groupby('learner').size().eq(5).all()
assert result.outcome_learner_selected.iloc[0] == benchmark.loc[benchmark.y_rmse.idxmin(), 'learner']
assert result.treatment_learner_selected.iloc[0] == benchmark.loc[benchmark.t_rmse.idxmin(), 'learner']
assert result.selection_rule.str.contains('minimum cross-fitted nuisance RMSE').iloc[0]
assert config['selection_rule'].startswith('minimum cross-fitted nuisance RMSE')
assert np.isfinite(benchmark.select_dtypes(include=np.number).to_numpy()).all()
assert np.isfinite(result.select_dtypes(include=np.number).to_numpy()).all()
assert benchmark.y_r2.max() > benchmark.y_r2.min()
print('MODEL_SELECTION_VALIDATION: PASS')
print(benchmark.to_string(index=False))
print(result.to_string(index=False))
