from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[3]
dml = root / 'data/modeling_changes/dml_v3'
treatment = 'sentinel2_ndvi_mean_1000m'
df = pd.read_csv(dml / f'crossfit_observations_{treatment}.csv')
df['influence_contribution'] = df['t_residual'] * df['y_residual']
rows = []
for fold, g in df.groupby('fold', sort=True):
    theta = (g['t_residual'] * g['y_residual']).sum() / (g['t_residual'] ** 2).sum()
    rows.append({
        'fold': int(fold), 'n': len(g), 'stations': g.station.nunique(),
        'theta': theta, 'mean_t_residual': g.t_residual.mean(),
        'sd_t_residual': g.t_residual.std(ddof=1),
        'mean_y_residual': g.y_residual.mean(),
        'influence_abs_p99': g.influence_contribution.abs().quantile(.99),
        'influence_abs_max': g.influence_contribution.abs().max(),
    })
print('FOLD_DIAGNOSTICS')
print(pd.DataFrame(rows).to_string(index=False))
print('\nFOLD_STATIONS')
print(df.groupby('fold')['station'].agg(lambda s: ', '.join(sorted(s.unique()))).to_string())
print('\nTOP_INFLUENCE_ROWS')
print(df.loc[df['influence_contribution'].abs().nlargest(15).index, ['station','year','month','fold','t_residual','y_residual','influence_contribution']].to_string(index=False))
