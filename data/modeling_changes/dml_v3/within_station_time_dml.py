"""Within-station expanding time-aware DML sensitivity.

For each held-out year, station means are learned from earlier years only and
applied to both training and holdout rows. This is a sensitivity design for
reducing time-invariant station-level differences without leaking holdout
outcomes or treatments.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_dml import INPUT_DIR, TARGET, GROUP, PRIMARY_TREATMENT, choose_controls, make_model

DML = Path(__file__).resolve().parent
TRAIN_PATH = INPUT_DIR / 'splits' / 'train.csv'
MIN_TRAIN_YEARS = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cluster_summary(frame: pd.DataFrame) -> Dict[str, float]:
    t = frame['t_residual'].to_numpy(float)
    y = frame['y_residual'].to_numpy(float)
    theta = float(np.dot(t, y) / np.dot(t, t))
    psi = t * (y - theta * t)
    influence = psi / float(np.mean(t ** 2))
    grouped = pd.DataFrame({'station': frame[GROUP].astype(str), 'influence': influence}).groupby('station')['influence'].sum()
    n, g = len(frame), len(grouped)
    variance = (g / (g - 1)) * float(np.sum(grouped.to_numpy() ** 2)) / (n ** 2)
    se = float(np.sqrt(max(variance, 0.0)))
    return {'theta': theta, 'cluster_se': se, 'cluster_ci_low': theta - 1.96 * se, 'cluster_ci_high': theta + 1.96 * se, 'n': n, 'n_stations': g, 'n_folds': int(frame['fold'].nunique())}


def demean_by_training_station(fit: pd.DataFrame, hold: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    means = fit.groupby(GROUP)[columns].mean()
    unseen = int((~hold[GROUP].isin(means.index)).sum())
    if unseen:
        raise AssertionError(f'Holdout contains {unseen} station rows absent from earlier training data')
    fit_mean_array = means.reindex(fit[GROUP]).to_numpy(dtype=float)
    hold_mean_array = means.reindex(hold[GROUP]).to_numpy(dtype=float)
    fit_out = pd.DataFrame(fit[columns].to_numpy(dtype=float) - fit_mean_array, columns=columns, index=fit.index)
    hold_out = pd.DataFrame(hold[columns].to_numpy(dtype=float) - hold_mean_array, columns=columns, index=hold.index)
    return fit_out, hold_out, unseen


def main() -> None:
    raw = pd.read_csv(TRAIN_PATH)
    controls = choose_controls(raw, PRIMARY_TREATMENT)
    columns = [TARGET, PRIMARY_TREATMENT] + controls
    years = sorted(raw.year.astype(int).unique())
    y_hat = np.full(len(raw), np.nan)
    t_hat = np.full(len(raw), np.nan)
    y_trans = np.full(len(raw), np.nan)
    t_trans = np.full(len(raw), np.nan)
    fold_ids = np.full(len(raw), -1, dtype=int)
    rows = []
    fold = 0
    for year in years:
        earlier = [y for y in years if y < year]
        if len(earlier) < MIN_TRAIN_YEARS:
            continue
        fit_idx = np.flatnonzero(raw.year.to_numpy(int) < year)
        candidate_hold_idx = np.flatnonzero(raw.year.to_numpy(int) == year)
        fit = raw.iloc[fit_idx]
        known_station_mask = raw.iloc[candidate_hold_idx][GROUP].isin(fit[GROUP].unique()).to_numpy()
        hold_idx = candidate_hold_idx[known_station_mask]
        hold = raw.iloc[hold_idx]
        unseen = int((~known_station_mask).sum())
        fit_dm, hold_dm, _ = demean_by_training_station(fit, hold, columns)
        x_fit = fit_dm[controls].copy()
        x_hold = hold_dm[controls].copy()
        keep = [c for c in controls if np.nanstd(x_fit[c].to_numpy(float)) > 1e-10]
        assert keep, 'No within-station varying controls remain'
        y_fit = fit_dm[TARGET].to_numpy(float)
        t_fit = fit_dm[PRIMARY_TREATMENT].to_numpy(float)
        y_hold = hold_dm[TARGET].to_numpy(float)
        t_hold = hold_dm[PRIMARY_TREATMENT].to_numpy(float)
        ym = make_model(seed_offset=3000 + fold)
        tm = make_model(seed_offset=4000 + fold)
        ym.fit(x_fit[keep], y_fit)
        tm.fit(x_fit[keep], t_fit)
        y_hat[hold_idx] = ym.predict(x_hold[keep])
        t_hat[hold_idx] = tm.predict(x_hold[keep])
        y_trans[hold_idx] = y_hold
        t_trans[hold_idx] = t_hold
        fold_ids[hold_idx] = fold
        rows.append({'fold': fold, 'holdout_year': int(year), 'train_years': ','.join(map(str, earlier)), 'n_train': len(fit_idx), 'n_holdout': len(hold_idx), 'n_train_stations': fit[GROUP].nunique(), 'n_holdout_stations': hold[GROUP].nunique(), 'n_unseen_station_rows_excluded': unseen, 'n_within_controls': len(keep)})
        fold += 1
    valid = fold_ids >= 0
    scored = raw.loc[valid, [GROUP, 'year', 'month', TARGET, PRIMARY_TREATMENT]].copy()
    scored['fold'] = fold_ids[valid]
    scored['y_transformed'] = y_trans[valid]
    scored['t_transformed'] = t_trans[valid]
    scored['y_hat_oof'] = y_hat[valid]
    scored['t_hat_oof'] = t_hat[valid]
    scored['y_residual'] = scored['y_transformed'] - scored['y_hat_oof']
    scored['t_residual'] = scored['t_transformed'] - scored['t_hat_oof']
    scored['orthogonal_score'] = scored['t_residual'] * scored['y_residual']
    result = cluster_summary(scored)
    result.update({'treatment': PRIMARY_TREATMENT, 'first_holdout_year': int(scored.year.min()), 'last_holdout_year': int(scored.year.max()), 'y_rmse_oof': float(np.sqrt(mean_squared_error(scored.y_transformed, scored.y_hat_oof))), 'y_mae_oof': float(mean_absolute_error(scored.y_transformed, scored.y_hat_oof)), 'y_r2_oof': float(r2_score(scored.y_transformed, scored.y_hat_oof)), 't_rmse_oof': float(np.sqrt(mean_squared_error(scored.t_transformed, scored.t_hat_oof))), 't_mae_oof': float(mean_absolute_error(scored.t_transformed, scored.t_hat_oof)), 't_r2_oof': float(r2_score(scored.t_transformed, scored.t_hat_oof))})
    scored.to_csv(DML / 'within_station_time_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(DML / 'within_station_time_folds.csv', index=False)
    pd.DataFrame([result]).to_csv(DML / 'within_station_time_summary.csv', index=False)
    (DML / 'within_station_time_config.json').write_text(json.dumps({'treatment': PRIMARY_TREATMENT, 'group': GROUP, 'min_train_years': MIN_TRAIN_YEARS, 'rule': 'station means learned from earlier years only and applied to holdout rows', 'input_sha256': {'train.csv': sha256(TRAIN_PATH)}}, indent=2), encoding='utf-8')
    lines = ['# Within-Station Expanding Time DML', '', '> This is a sensitivity design. Station means are learned from earlier years only and applied to later holdouts; no holdout outcome or treatment is used in the transformation.', '', f"The design scores {len(scored)} rows across {len(rows)} annual holdouts. The within-station estimate is {result['theta']:.6f} with station-clustered 95% interval [{result['cluster_ci_low']:.6f}, {result['cluster_ci_high']:.6f}].", '', f"Outcome nuisance RMSE/R² are {result['y_rmse_oof']:.6f}/{result['y_r2_oof']:.6f}; treatment nuisance RMSE/R² are {result['t_rmse_oof']:.6f}/{result['t_r2_oof']:.6f}.", '', 'This design removes training-period station means but does not solve time-varying confounding, measurement error, spatial spillovers, or simultaneity. It is not a replacement for the primary station-grouped DML estimate.']
    (DML / 'within_station_time_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(pd.DataFrame([result]).to_string(index=False))
    print(f'WITHIN_STATION_TIME_DML: PASS ({len(rows)} annual holdouts)')


if __name__ == '__main__':
    main()
