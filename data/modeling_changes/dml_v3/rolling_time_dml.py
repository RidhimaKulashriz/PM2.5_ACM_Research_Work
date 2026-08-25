"""Rolling-origin, time-aware DML sensitivity for the primary V3 treatment.

Each holdout month is predicted using observations from strictly earlier
calendar months. This tests temporal generalization without using future rows.
The design is a sensitivity analysis; the pre-specified station-grouped DML
remains separately reported.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_dml import INPUT_DIR, TARGET, GROUP, PRIMARY_TREATMENT, choose_controls, make_model

DML = Path(__file__).resolve().parent
TRAIN_PATH = INPUT_DIR / 'splits' / 'train.csv'
SEED = 42
MIN_TRAIN_MONTHS = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def month_index(frame: pd.DataFrame) -> pd.Series:
    return frame['year'].astype(int) * 12 + frame['month'].astype(int)


def cluster_summary(frame: pd.DataFrame) -> Dict[str, float]:
    t = frame['t_residual'].to_numpy(float)
    y = frame['y_residual'].to_numpy(float)
    theta = float(np.dot(t, y) / np.dot(t, t))
    psi = t * (y - theta * t)
    denom = float(np.mean(t ** 2))
    influence = psi / denom
    by_station = pd.DataFrame({'station': frame[GROUP].astype(str), 'influence': influence}).groupby('station')['influence'].sum()
    n = len(frame)
    g = len(by_station)
    variance = (g / (g - 1)) * float(np.sum(by_station.to_numpy() ** 2)) / (n ** 2)
    se = float(np.sqrt(max(variance, 0.0)))
    return {'theta': theta, 'cluster_se': se, 'cluster_ci_low': theta - 1.96 * se, 'cluster_ci_high': theta + 1.96 * se, 'n': int(n), 'n_stations': int(g), 'n_folds': int(frame['fold'].nunique())}


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    controls = choose_controls(train, PRIMARY_TREATMENT)
    train = train.copy()
    train['_month_index'] = month_index(train)
    periods = sorted(train.loc[train['_month_index'] > train['_month_index'].min(), '_month_index'].unique())
    periods = [int(p) for p in periods if int(p) - int(train['_month_index'].min()) >= MIN_TRAIN_MONTHS]
    y = train[TARGET].to_numpy(float)
    t = train[PRIMARY_TREATMENT].to_numpy(float)
    y_hat = np.full(len(train), np.nan)
    t_hat = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1, dtype=int)
    fold_rows = []
    fold_id = 0
    for period in periods:
        fit_idx = np.flatnonzero(train['_month_index'].to_numpy() < period)
        hold_idx = np.flatnonzero(train['_month_index'].to_numpy() == period)
        if len(fit_idx) == 0 or len(hold_idx) == 0:
            continue
        ym = make_model(seed_offset=1000 + fold_id)
        tm = make_model(seed_offset=2000 + fold_id)
        ym.fit(train.iloc[fit_idx][controls], y[fit_idx])
        tm.fit(train.iloc[fit_idx][controls], t[fit_idx])
        y_hat[hold_idx] = ym.predict(train.iloc[hold_idx][controls])
        t_hat[hold_idx] = tm.predict(train.iloc[hold_idx][controls])
        fold_ids[hold_idx] = fold_id
        held = train.iloc[hold_idx]
        fit_periods = train.iloc[fit_idx]['_month_index']
        fold_rows.append({
            'fold': fold_id,
            'holdout_year': int(held['year'].iloc[0]),
            'holdout_month': int(held['month'].iloc[0]),
            'n_train': int(len(fit_idx)),
            'n_holdout': int(len(hold_idx)),
            'n_train_stations': int(train.iloc[fit_idx][GROUP].nunique()),
            'n_holdout_stations': int(held[GROUP].nunique()),
            'first_train_period': int(fit_periods.min()),
            'last_train_period': int(fit_periods.max()),
        })
        fold_id += 1
    valid = fold_ids >= 0
    assert valid.any(), 'No rolling-origin holdout rows'
    scored = train.loc[valid, [GROUP, 'year', 'month', TARGET, PRIMARY_TREATMENT]].copy()
    scored['fold'] = fold_ids[valid]
    scored['y_hat_oof'] = y_hat[valid]
    scored['t_hat_oof'] = t_hat[valid]
    scored['y_residual'] = y[valid] - y_hat[valid]
    scored['t_residual'] = t[valid] - t_hat[valid]
    scored['orthogonal_score'] = scored['t_residual'] * (scored['y_residual'] - scored['t_residual'] * (np.dot(scored['t_residual'], scored['y_residual']) / np.dot(scored['t_residual'], scored['t_residual'])))
    result = cluster_summary(scored)
    result.update({
        'treatment': PRIMARY_TREATMENT,
        'first_holdout_year': int(scored['year'].min()),
        'last_holdout_year': int(scored['year'].max()),
        'y_rmse_oof': float(np.sqrt(mean_squared_error(scored[TARGET], scored['y_hat_oof']))),
        'y_mae_oof': float(mean_absolute_error(scored[TARGET], scored['y_hat_oof'])),
        'y_r2_oof': float(r2_score(scored[TARGET], scored['y_hat_oof'])),
        't_rmse_oof': float(np.sqrt(mean_squared_error(scored[PRIMARY_TREATMENT], scored['t_hat_oof']))),
        't_mae_oof': float(mean_absolute_error(scored[PRIMARY_TREATMENT], scored['t_hat_oof'])),
        't_r2_oof': float(r2_score(scored[PRIMARY_TREATMENT], scored['t_hat_oof'])),
        'min_train_months': MIN_TRAIN_MONTHS,
    })
    fold_df = pd.DataFrame(fold_rows)
    scored.to_csv(DML / 'rolling_time_predictions_primary.csv', index=False)
    fold_df.to_csv(DML / 'rolling_time_folds.csv', index=False)
    pd.DataFrame([result]).to_csv(DML / 'rolling_time_summary.csv', index=False)
    (DML / 'rolling_time_config.json').write_text(json.dumps({'seed': SEED, 'group': GROUP, 'treatment': PRIMARY_TREATMENT, 'min_train_months': MIN_TRAIN_MONTHS, 'rule': 'fit only on rows with strictly earlier calendar month than the holdout month', 'input_sha256': {'train.csv': sha256(TRAIN_PATH)}}, indent=2), encoding='utf-8')
    lines = [
        '# Rolling-Origin Time-Aware DML',
        '',
        '> This is a temporal sensitivity analysis. Each holdout month is predicted using only strictly earlier calendar months. It is not a replacement for the primary station-grouped DML specification.',
        '',
        f"The primary treatment is `{PRIMARY_TREATMENT}`. The analysis scores {len(scored)} rows across {len(fold_df)} chronological holdout months, with a minimum of {MIN_TRAIN_MONTHS} earlier calendar months in the first training window.",
        '',
        f"The rolling-origin estimate is {result['theta']:.6f} with station-clustered 95% interval [{result['cluster_ci_low']:.6f}, {result['cluster_ci_high']:.6f}]. Outcome nuisance RMSE/R² are {result['y_rmse_oof']:.6f}/{result['y_r2_oof']:.6f}; treatment nuisance RMSE/R² are {result['t_rmse_oof']:.6f}/{result['t_r2_oof']:.6f}.",
        '',
        'This design reduces temporal leakage but can still be affected by time-varying confounding, station-level dependence, limited early training support, and treatment measurement error. The result must be interpreted jointly with the pre-treatment, spatial-block, cluster-robust, and fold-stability diagnostics.',
    ]
    (DML / 'rolling_time_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(pd.DataFrame([result]).to_string(index=False))
    print(f'ROLLING_TIME_DML: PASS ({len(fold_df)} holdout months)')


if __name__ == '__main__':
    main()
