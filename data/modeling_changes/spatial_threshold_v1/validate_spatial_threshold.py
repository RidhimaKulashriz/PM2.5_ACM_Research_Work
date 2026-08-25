from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/modeling_changes/spatial_threshold_v1'
RESULTS = OUT / 'results'
PLOTS = RESULTS / 'plots'
MASTER = ROOT / 'data/modeling_changes/datasets/master_modeling_dataset_v3.csv'
TRAIN = ROOT / 'data/modeling_changes/splits/train.csv'
TEST = ROOT / 'data/modeling_changes/splits/test.csv'
DML_CONFIG = ROOT / 'data/modeling_changes/dml_v3/dml_config.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    required = [
        OUT / 'analysis_contract.md', OUT / 'run_config.json', OUT / 'input_hashes.json', OUT / 'threshold_summary.json',
        OUT / 'spatial_threshold_report.md', OUT / 'visual_review.md',
        RESULTS / 'surface_training_cv.csv', RESULTS / 'surface_model_metrics.csv', RESULTS / 'surface_test_predictions.csv', RESULTS / 'station_surface_summary.csv',
        RESULTS / 'threshold_cv_results.csv', RESULTS / 'threshold_fold_selection.csv', RESULTS / 'threshold_bootstrap_selection.csv', RESULTS / 'threshold_model_metrics.csv',
    ]
    required += list(RESULTS.glob('*.json'))
    required += list(PLOTS.glob('*.png'))
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f'Missing required outputs: {missing}')

    master = pd.read_csv(MASTER)
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    assert len(master) == 1615 and len(train) == 1292 and len(test) == 323
    assert master.station.nunique() == 35
    assert master.groupby('station')[['latitude', 'longitude']].nunique().max().max() == 1
    assert set(train.station).issubset(set(master.station)) and set(test.station).issubset(set(master.station))

    recorded = json.loads((OUT / 'input_hashes.json').read_text())
    expected = json.loads(DML_CONFIG.read_text())['input_sha256']
    current = {
        'data/modeling_changes/datasets/master_modeling_dataset_v3.csv': sha256(MASTER),
        'data/modeling_changes/splits/train.csv': sha256(TRAIN),
        'data/modeling_changes/splits/test.csv': sha256(TEST),
    }
    assert recorded == current, 'New run hash record does not match current protected inputs.'
    assert current['data/modeling_changes/datasets/master_modeling_dataset_v3.csv'] == expected['master_modeling_dataset_v3.csv']
    assert current['data/modeling_changes/splits/train.csv'] == expected['train.csv']
    assert current['data/modeling_changes/splits/test.csv'] == expected['test.csv']

    surface_cv = pd.read_csv(RESULTS / 'surface_training_cv.csv')
    surface_metrics = pd.read_csv(RESULTS / 'surface_model_metrics.csv')
    pred = pd.read_csv(RESULTS / 'surface_test_predictions.csv')
    station = pd.read_csv(RESULTS / 'station_surface_summary.csv')
    assert len(surface_cv) == 5 and len(pred) == 323 and len(station) <= 35
    assert pred.station.nunique() == len(station)
    assert {'station', 'latitude', 'longitude', 'pm25', 'predicted_pm25', 'residual', 'absolute_error'}.issubset(pred.columns)
    assert np.isfinite(pred.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(surface_metrics[['n', 'r2', 'rmse', 'mae']].to_numpy()).all()
    assert (surface_metrics[['r2']].to_numpy() <= 1.0 + 1e-8).all() and (surface_metrics[['r2']].to_numpy() >= -1.0 - 1e-8).all()
    assert (surface_metrics[['rmse', 'mae']].to_numpy() >= 0).all()
    assert pred[['latitude', 'longitude']].merge(test[['latitude', 'longitude']].drop_duplicates(), on=['latitude', 'longitude'], how='left').shape[0] == 323

    threshold_cv = pd.read_csv(RESULTS / 'threshold_cv_results.csv')
    fold_selection = pd.read_csv(RESULTS / 'threshold_fold_selection.csv')
    boot = pd.read_csv(RESULTS / 'threshold_bootstrap_selection.csv')
    threshold_metrics = pd.read_csv(RESULTS / 'threshold_model_metrics.csv')
    summary = json.loads((OUT / 'threshold_summary.json').read_text())
    assert len(threshold_cv) == 17 and len(fold_selection) == 5 and len(boot) == 100 and len(threshold_metrics) == 1
    assert any(abs(float(summary['selected_training_quantile']) - float(x)) < 1e-8 for x in np.arange(0.10, 0.901, 0.05))
    assert summary['stable_threshold_identified'] is False or summary['bootstrap_selection_share_same_quantile'] >= 0.50
    assert summary['bootstrap_threshold_q025'] <= summary['bootstrap_threshold_q975']
    assert np.isfinite(threshold_cv.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(boot.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(threshold_metrics.select_dtypes(include=[np.number]).to_numpy()).all()
    assert (threshold_metrics[['rmse', 'mae']].to_numpy() >= 0).all()
    assert (threshold_metrics[['r2']].to_numpy() <= 1.0 + 1e-8).all() and (threshold_metrics[['r2']].to_numpy() >= -1.0 - 1e-8).all()
    assert set(fold_selection['fold']) == {1, 2, 3, 4, 5}
    assert set(boot['bootstrap']) == set(range(1, 101))

    config = json.loads((OUT / 'run_config.json').read_text())
    assert config['spatial_support'].startswith('observed station coordinates only')
    assert config['threshold_selection'] == 'minimum mean station-grouped training CV RMSE'
    contract = (OUT / 'analysis_contract.md').read_text()
    report = (OUT / 'spatial_threshold_report.md').read_text()
    for phrase in ['No interpolation', 'not a causal', 'locked test', 'stable threshold is']:
        assert phrase.lower() in (contract + report).lower()
    assert len(list(PLOTS.glob('*.png'))) == 4
    print('PASS: spatial-surface and threshold-analysis outputs, hashes, safeguards, and protected-input checks validated.')


if __name__ == '__main__':
    main()
