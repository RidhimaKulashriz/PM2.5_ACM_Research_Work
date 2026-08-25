from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DML = ROOT / 'data/modeling_changes/dml_v3'
INPUT = ROOT / 'data/modeling_changes'
TREATMENTS = ['sentinel2_ndvi_mean_1000m', 'sentinel2_ndvi_mean_500m', 'modis_ndvi_mean_1000m']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


missing = pd.read_csv(DML / 'missingness_overview.csv')
by_station = pd.read_csv(DML / 'missingness_by_station.csv')
metrics = pd.read_csv(DML / 'metric_audit_summary.csv')
fold_metrics = pd.read_csv(DML / 'metric_audit_by_fold.csv')
base_config = json.loads((DML / 'dml_config.json').read_text())

assert len(missing) == 2 * len(pd.read_csv(INPUT / 'splits/train.csv', nrows=1).columns)
assert {'train', 'test'} == set(missing.split)
assert set(metrics.treatment) == set(TREATMENTS)
assert set(fold_metrics.treatment) == set(TREATMENTS)
assert fold_metrics.groupby('treatment').size().eq(5).all()
assert metrics.pm25_band_accuracy.between(0, 1).all()
assert np.isfinite(metrics.select_dtypes(include=np.number).to_numpy()).all()
assert np.isfinite(fold_metrics.select_dtypes(include=np.number).to_numpy()).all()
assert metrics.residual_orthogonality_corr.abs().lt(0.25).all()
assert by_station.station.notna().all()
assert len(by_station) >= 30

expected_hashes = base_config['input_sha256']
assert sha256(INPUT / 'datasets/master_modeling_dataset_v3.csv') == expected_hashes['master_modeling_dataset_v3.csv']
assert sha256(INPUT / 'splits/train.csv') == expected_hashes['train.csv']
assert sha256(INPUT / 'splits/test.csv') == expected_hashes['test.csv']

print('ATTACHMENT_AUDIT_VALIDATION: PASS')
print(metrics[['treatment','y_rmse','y_mae','y_r2','t_rmse','t_mae','t_r2','pm25_band_accuracy','residual_orthogonality_corr']].to_string(index=False))
