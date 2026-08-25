from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[3]
dml = root / 'data/modeling_changes/dml_v3'
train = pd.read_csv(root / 'data/modeling_changes/splits/train.csv')
test = pd.read_csv(root / 'data/modeling_changes/splits/test.csv')
robust = pd.read_csv(dml / 'robustness_summary.csv')
fold = pd.read_csv(dml / 'fold_stability.csv')
station = pd.read_csv(dml / 'station_heterogeneity.csv')
overlap = pd.read_csv(dml / 'overlap_falsification.csv')
learner = pd.read_csv(dml / 'learner_sensitivity.csv')
spatial = pd.read_csv(dml / 'spatial_block_sensitivity.csv')
spatial_folds = pd.read_csv(dml / 'spatial_block_stability.csv')
config = json.loads((dml / 'robustness_config.json').read_text())

assert len(train) == 1292 and len(test) == 323
assert robust.treatment.nunique() == 3 and len(robust) == 3
assert fold.groupby('treatment').size().eq(5).all()
assert station.groupby('treatment').size().eq(35).all()
assert overlap.treatment.nunique() == 3 and len(overlap) == 3
assert learner.shape[0] == 1 and learner.loc[0, 'learner'] == 'random_forest'
assert spatial.shape[0] == 1 and int(spatial.loc[0, 'spatial_block_count']) == 4
assert len(spatial_folds) == 4 and spatial_folds['held_out_block'].nunique() == 4
for name, frame in [('robustness_summary', robust), ('fold_stability', fold), ('overlap_falsification', overlap), ('learner_sensitivity', learner), ('spatial_block_sensitivity', spatial), ('spatial_block_stability', spatial_folds)]:
    numeric = frame.select_dtypes(include=np.number)
    assert np.isfinite(numeric.to_numpy()).all(), f'non-finite robustness value found in {name}'
station_numeric = station.select_dtypes(include=np.number).drop(columns=['t_residual_sd'])
assert np.isfinite(station_numeric.to_numpy()).all(), 'non-finite station heterogeneity estimate'
allowed_station_sd_nan = station['t_residual_sd'].isna() & station['n'].eq(1)
assert station['t_residual_sd'].isna().eq(allowed_station_sd_nan).all(), 'unexpected station residual SD NaN'
assert int(config['cluster_count']) == 35
assert int(config['wild_bootstrap_reps']) == 2000
assert int(config['permutation_reps']) == 1000
for protected in [
    root / 'data/modeling_changes/baseline_results_v3',
    root / 'data/modeling_changes/datasets',
    root / 'data/modeling_changes/splits',
]:
    assert protected.exists()
print('ROBUSTNESS_VALIDATION: PASS')
print(robust[['treatment','theta_robust','cluster_se','cluster_ci_low','cluster_ci_high','wild_bootstrap_ci_low','wild_bootstrap_ci_high']].to_string(index=False))
print(f'fold_rows={len(fold)} station_rows={len(station)} spatial_blocks={len(spatial_folds)}')
