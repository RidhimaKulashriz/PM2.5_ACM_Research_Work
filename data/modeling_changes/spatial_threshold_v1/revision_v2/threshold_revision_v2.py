from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)
SEED = 42
N_BOOT = 100
ROOT = Path(__file__).resolve().parents[4]
MASTER = ROOT / 'data/modeling_changes/datasets/master_modeling_dataset_v3.csv'
TRAIN = ROOT / 'data/modeling_changes/splits/train.csv'
TEST = ROOT / 'data/modeling_changes/splits/test.csv'
OUT = ROOT / 'data/modeling_changes/spatial_threshold_v1/revision_v2'
RES = OUT / 'results'
RES.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

def fit_pipe(X, y):
    m = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler()), ('model', Ridge(alpha=10.0))])
    m.fit(X, y)
    return m

def design(df, exposure, controls, threshold=None):
    out = pd.DataFrame(index=df.index)
    out['exposure'] = pd.to_numeric(df[exposure], errors='coerce')
    if threshold is not None: out['hinge_above_threshold'] = np.maximum(out['exposure'] - float(threshold), 0.0)
    for c in controls: out[c] = pd.to_numeric(df[c], errors='coerce')
    return out

def threshold_candidates(frame, exposure, quantiles):
    vals = [float(frame[exposure].quantile(float(q))) for q in quantiles]
    return [(float(q), v) for q, v in zip(quantiles, vals) if np.isfinite(v)]

def select_threshold(frame, y, groups, exposure, controls, quantiles, n_splits=4):
    candidates = threshold_candidates(frame, exposure, quantiles)
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for q, threshold in candidates:
        fold_scores = []
        X = design(frame, exposure, controls, threshold)
        for tr_idx, va_idx in gkf.split(X, y, groups):
            model = fit_pipe(X.iloc[tr_idx], y.iloc[tr_idx])
            fold_scores.append(rmse(y.iloc[va_idx], model.predict(X.iloc[va_idx])))
        scores.append({'quantile': q, 'threshold': threshold, 'inner_cv_rmse': float(np.mean(fold_scores)), 'inner_cv_rmse_std': float(np.std(fold_scores, ddof=1))})
    return min(scores, key=lambda r: (r['inner_cv_rmse'], r['quantile'])), pd.DataFrame(scores)


def evaluate_outer(frame, y, groups, exposure, controls, quantiles):
    outer = GroupKFold(n_splits=5)
    rows, oof = [], []
    for fold, (tr_idx, va_idx) in enumerate(outer.split(frame, y, groups), 1):
        tr, va = frame.iloc[tr_idx], frame.iloc[va_idx]
        yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
        gt = groups.iloc[tr_idx]
        best, candidates_df = select_threshold(tr, yt, gt, exposure, controls, quantiles, n_splits=4)
        Xseg_tr = design(tr, exposure, controls, best['threshold']); Xseg_va = design(va, exposure, controls, best['threshold'])
        Xlin_tr = design(tr, exposure, controls, None); Xlin_va = design(va, exposure, controls, None)
        seg = fit_pipe(Xseg_tr, yt); lin = fit_pipe(Xlin_tr, yt)
        pseg, plin = seg.predict(Xseg_va), lin.predict(Xlin_va)
        rows.append({'outer_fold': fold, 'selected_quantile': best['quantile'], 'selected_threshold': best['threshold'], 'inner_cv_rmse': best['inner_cv_rmse'], 'segmented_rmse': rmse(yv, pseg), 'linear_rmse': rmse(yv, plin), 'segmented_mae': float(mean_absolute_error(yv, pseg)), 'linear_mae': float(mean_absolute_error(yv, plin)), 'rmse_improvement_linear_minus_segmented': rmse(yv, plin) - rmse(yv, pseg), 'n_validation': len(va), 'stations_validation': int(groups.iloc[va_idx].nunique())})
        oof.extend(pd.DataFrame({'outer_fold': fold, 'observed_pm25': yv.to_numpy(), 'segmented_predicted': pseg, 'linear_predicted': plin, 'selected_threshold': best['threshold']}, index=va.index).to_dict('records'))
    return pd.DataFrame(rows), pd.DataFrame(oof)

master, train, test = pd.read_csv(MASTER), pd.read_csv(TRAIN), pd.read_csv(TEST)
if (len(master), len(train), len(test)) != (1615, 1292, 323): raise AssertionError('Unexpected V3 row counts')
key = lambda d: set(zip(d.station.astype(str), d.year.astype(int), d.month.astype(int)))
if key(train) & key(test) or key(train) | key(test) != key(master): raise AssertionError('V3 key integrity failed')
if 'IIT_Delhi' in set(test.station): raise AssertionError('IIT_Delhi must remain train-only')

exposure = 'sentinel2_ndvi_mean_1000m'
preferred = ['latitude','longitude','year','month','month_sin','month_cos','season_encoded','era5_temp_mean','era5_rh_mean','era5_wind_speed_mean','era5_blh_mean','population_density_2025_1000m','road_density_1000m','major_road_density_1000m','dynamicworld_2025_water_frac_1000m','dynamicworld_2025_built_frac_1000m','dynamicworld_2025_bare_frac_1000m']
controls = [c for c in preferred if c in train.columns and c not in {exposure, 'pm25'}]
quantiles = np.arange(0.20, 0.801, 0.05)
y = train.pm25.astype(float); groups = train.station.astype(str)
outer_rows, outer_oof = evaluate_outer(train, y, groups, exposure, controls, quantiles)

# Full training selection, then locked-test evaluation after selection is frozen.
selected, full_cv = select_threshold(train, y, groups, exposure, controls, quantiles, n_splits=5)
Xseg_train, Xseg_test = design(train, exposure, controls, selected['threshold']), design(test, exposure, controls, selected['threshold'])
Xlin_train, Xlin_test = design(train, exposure, controls, None), design(test, exposure, controls, None)
seg, lin = fit_pipe(Xseg_train, y), fit_pipe(Xlin_train, y)
pseg, plin = seg.predict(Xseg_test), lin.predict(Xlin_test)

# Station bootstrap: repeat selection and evaluate training grouped-CV delta only.
rng = np.random.default_rng(SEED); stations = train.station.drop_duplicates().tolist(); boot_rows = []
for b in range(1, N_BOOT + 1):
    sampled = rng.choice(stations, size=len(stations), replace=True)
    boot = pd.concat([train[train.station == s] for s in sampled], ignore_index=True)
    by = boot.pm25.astype(float); bg = boot.station.astype(str)
    best, _ = select_threshold(boot, by, bg, exposure, controls, quantiles, n_splits=5)
    Xs, Xl = design(boot, exposure, controls, best['threshold']), design(boot, exposure, controls, None)
    gkf = GroupKFold(n_splits=5); seg_scores=[]; lin_scores=[]
    for tr_idx, va_idx in gkf.split(boot, by, bg):
        ms, ml = fit_pipe(Xs.iloc[tr_idx], by.iloc[tr_idx]), fit_pipe(Xl.iloc[tr_idx], by.iloc[tr_idx])
        seg_scores.append(rmse(by.iloc[va_idx], ms.predict(Xs.iloc[va_idx])))
        lin_scores.append(rmse(by.iloc[va_idx], ml.predict(Xl.iloc[va_idx])))
    boot_rows.append({'bootstrap': b, 'selected_quantile': best['quantile'], 'selected_threshold': best['threshold'], 'segmented_cv_rmse': np.mean(seg_scores), 'linear_cv_rmse': np.mean(lin_scores), 'rmse_improvement': np.mean(lin_scores)-np.mean(seg_scores)})
bootstrap = pd.DataFrame(boot_rows)

outer_rows.to_csv(RES / 'outer_fold_results.csv', index=False)
outer_oof.to_csv(RES / 'outer_oof_predictions.csv', index=False)
full_cv.to_csv(RES / 'full_training_threshold_cv.csv', index=False)
bootstrap.to_csv(RES / 'bootstrap_stability.csv', index=False)
pd.DataFrame([{
    'split': 'locked_test', 'n': len(test), 'selected_threshold': selected['threshold'], 'selected_quantile': selected['quantile'],
    'segmented_r2': r2_score(test.pm25, pseg), 'segmented_rmse': rmse(test.pm25, pseg), 'segmented_mae': mean_absolute_error(test.pm25, pseg),
    'linear_r2': r2_score(test.pm25, plin), 'linear_rmse': rmse(test.pm25, plin), 'linear_mae': mean_absolute_error(test.pm25, plin),
    'rmse_improvement_linear_minus_segmented': rmse(test.pm25, plin)-rmse(test.pm25, pseg), 'controls': len(controls)
}]).to_csv(RES / 'locked_test_comparison.csv', index=False)

hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [MASTER, TRAIN, TEST]}
(OUT / 'input_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
mode_q = float(bootstrap.selected_quantile.mode().iloc[0]); stability = float((bootstrap.selected_quantile == mode_q).mean())
summary = {'exposure': exposure, 'controls': controls, 'quantile_grid': list(map(float, quantiles)), 'selected_threshold': selected['threshold'], 'selected_quantile': selected['quantile'], 'bootstrap_modal_quantile': mode_q, 'bootstrap_modal_stability': stability, 'outer_mean_rmse_improvement': float(outer_rows.rmse_improvement_linear_minus_segmented.mean()), 'outer_segmented_rmse': float(outer_rows.segmented_rmse.mean()), 'outer_linear_rmse': float(outer_rows.linear_rmse.mean()), 'threshold_supported': bool(stability >= 0.50 and outer_rows.rmse_improvement_linear_minus_segmented.mean() > 0), 'locked_test_r2_segmented': float(r2_score(test.pm25, pseg)), 'locked_test_r2_linear': float(r2_score(test.pm25, plin))}
(OUT / 'threshold_revision_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(OUT / 'threshold_revision_report.md').write_text('# Corrected threshold analysis\n\nThis revision uses nested station-grouped cross-validation. Each outer fold selects the breakpoint using only its outer-training stations and four-fold inner grouped CV, then evaluates the segmented and no-break linear models on unseen stations. The locked test is evaluated once after full-training selection.\n\n## Finding\n\n' + ('A stable threshold is supported under the frozen rule.' if summary['threshold_supported'] else 'No stable threshold is supported under the frozen rule. The breakpoint is treated as a predictive screen, not a policy or causal threshold.') + '\n\n' + json.dumps(summary, indent=2) + '\n\nThe threshold analysis is predictive/associational and does not establish that changing vegetation causes PM2.5 to change.\n')
print(json.dumps(summary, indent=2))
