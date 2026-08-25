"""Pre-specified nuisance-learner benchmark for the primary V3 DML treatment.

Learners are compared by held-out nuisance prediction loss under the same
station-grouped folds. The causal coefficient is not used to select a learner.
This is a sensitivity benchmark, not permission to choose the most favorable
causal estimate.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
DML = ROOT / 'data/modeling_changes/dml_v3'
INPUT = ROOT / 'data/modeling_changes/splits/train.csv'
TARGET = 'pm25'
GROUP = 'station'
TREATMENT = 'sentinel2_ndvi_mean_1000m'
SEED = 42
N_FOLDS = 5


def controls() -> List[str]:
    manifest = json.loads((DML / 'feature_manifest.json').read_text())
    return manifest['treatments'][TREATMENT]['controls']


def pipeline(kind: str, seed: int) -> Pipeline:
    if kind == 'hist_gradient_boosting':
        estimator = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=1.0, random_state=seed)
    elif kind == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=8, max_features='sqrt', random_state=seed, n_jobs=-1)
    elif kind == 'extra_trees':
        estimator = ExtraTreesRegressor(n_estimators=250, max_depth=16, min_samples_leaf=8, max_features=1.0, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', estimator)])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'mae': float(mean_absolute_error(actual, predicted)),
        'r2': float(r2_score(actual, predicted)),
    }


def cluster_interval(station: np.ndarray, t_resid: np.ndarray, y_resid: np.ndarray) -> Dict[str, float]:
    theta = float(np.dot(t_resid, y_resid) / np.dot(t_resid, t_resid))
    psi = t_resid * (y_resid - theta * t_resid)
    denom_mean = float(np.mean(t_resid ** 2))
    influence = psi / denom_mean
    grouped = pd.DataFrame({'station': station.astype(str), 'influence': influence}).groupby('station')['influence'].sum()
    n = len(station)
    g = len(grouped)
    variance = (g / (g - 1)) * float(np.sum(grouped.to_numpy() ** 2)) / (n ** 2)
    se = float(np.sqrt(max(variance, 0.0)))
    return {'theta': theta, 'cluster_se': se, 'cluster_ci_low': theta - 1.96 * se, 'cluster_ci_high': theta + 1.96 * se, 'n_clusters': g}


def main() -> None:
    frame = pd.read_csv(INPUT)
    x = frame[controls()]
    y = frame[TARGET].to_numpy(float)
    t = frame[TREATMENT].to_numpy(float)
    groups = frame[GROUP].astype(str).to_numpy()
    candidates = ['hist_gradient_boosting', 'random_forest', 'extra_trees']
    splitter = GroupKFold(n_splits=N_FOLDS)
    y_predictions: Dict[str, np.ndarray] = {k: np.full(len(frame), np.nan) for k in candidates}
    t_predictions: Dict[str, np.ndarray] = {k: np.full(len(frame), np.nan) for k in candidates}
    fold_rows = []
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x, y, groups)):
        for kind in candidates:
            ym = pipeline(kind, SEED + fold)
            tm = pipeline(kind, SEED + 100 + fold)
            ym.fit(x.iloc[fit_idx], y[fit_idx])
            tm.fit(x.iloc[fit_idx], t[fit_idx])
            y_predictions[kind][hold_idx] = ym.predict(x.iloc[hold_idx])
            t_predictions[kind][hold_idx] = tm.predict(x.iloc[hold_idx])
            fold_rows.append({'fold': int(fold), 'learner': kind, 'n_train': int(len(fit_idx)), 'n_holdout': int(len(hold_idx)), 'n_train_stations': int(np.unique(groups[fit_idx]).size), 'n_holdout_stations': int(np.unique(groups[hold_idx]).size)})
    benchmark = []
    for kind in candidates:
        ym = metrics(y, y_predictions[kind])
        tm = metrics(t, t_predictions[kind])
        benchmark.append({'learner': kind, 'y_rmse': ym['rmse'], 'y_mae': ym['mae'], 'y_r2': ym['r2'], 't_rmse': tm['rmse'], 't_mae': tm['mae'], 't_r2': tm['r2']})
    benchmark_df = pd.DataFrame(benchmark)
    y_kind = str(benchmark_df.loc[benchmark_df.y_rmse.idxmin(), 'learner'])
    t_kind = str(benchmark_df.loc[benchmark_df.t_rmse.idxmin(), 'learner'])
    y_resid = y - y_predictions[y_kind]
    t_resid = t - t_predictions[t_kind]
    robust = cluster_interval(groups, t_resid, y_resid)
    result = {'treatment': TREATMENT, 'outcome_learner_selected': y_kind, 'treatment_learner_selected': t_kind, 'selection_rule': 'minimum cross-fitted nuisance RMSE; causal estimate not used', **robust, 'y_rmse_selected': float(benchmark_df.loc[benchmark_df.learner == y_kind, 'y_rmse'].iloc[0]), 't_rmse_selected': float(benchmark_df.loc[benchmark_df.learner == t_kind, 't_rmse'].iloc[0]), 'y_r2_selected': float(benchmark_df.loc[benchmark_df.learner == y_kind, 'y_r2'].iloc[0]), 't_r2_selected': float(benchmark_df.loc[benchmark_df.learner == t_kind, 't_r2'].iloc[0])}
    benchmark_df.to_csv(DML / 'model_selection_benchmark.csv', index=False)
    pd.DataFrame([result]).to_csv(DML / 'model_selection_result.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(DML / 'model_selection_folds.csv', index=False)
    (DML / 'model_selection_config.json').write_text(json.dumps({'seed': SEED, 'n_folds': N_FOLDS, 'fold_group': GROUP, 'treatment': TREATMENT, 'candidates': candidates, 'selection_rule': 'minimum cross-fitted nuisance RMSE separately for outcome and treatment; never select on theta or interval'}, indent=2), encoding='utf-8')
    lines = [
        '# Nuisance Learner Benchmark',
        '',
        '> This benchmark compares nuisance learners by held-out predictive loss. It does not select a causal estimate by its sign, magnitude, or interval.',
        '',
        f'The selected outcome learner is `{y_kind}` and the selected treatment learner is `{t_kind}` under the pre-specified minimum-RMSE rule. The resulting sensitivity estimate is {robust["theta"]:.6f} with station-clustered 95% interval [{robust["cluster_ci_low"]:.6f}, {robust["cluster_ci_high"]:.6f}].',
        '',
        benchmark_df.to_markdown(index=False),
        '',
        'The benchmark remains a sensitivity analysis. Learner selection based on the same finite sample can still add model-selection uncertainty, so the original pre-specified HistGradientBoosting result remains separately reported.',
    ]
    (DML / 'model_selection_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(benchmark_df.to_string(index=False))
    print(pd.DataFrame([result]).to_string(index=False))
    print('MODEL_SELECTION_BENCHMARK: PASS')


if __name__ == '__main__':
    main()
