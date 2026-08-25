from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore', category=FutureWarning)
RANDOM_STATE = 42
N_BOOTSTRAP = 100

ROOT = Path(__file__).resolve().parents[3]
MASTER_PATH = ROOT / 'data/modeling_changes/datasets/master_modeling_dataset_v3.csv'
TRAIN_PATH = ROOT / 'data/modeling_changes/splits/train.csv'
TEST_PATH = ROOT / 'data/modeling_changes/splits/test.csv'
OUT = ROOT / 'data/modeling_changes/spatial_threshold_v1'
RESULTS = OUT / 'results'
PLOTS = RESULTS / 'plots'
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(y, pred):
    return {
        'n': int(len(y)),
        'r2': float(r2_score(y, pred)),
        'rmse': float(np.sqrt(mean_squared_error(y, pred))),
        'mae': float(mean_absolute_error(y, pred)),
    }


def make_lgbm():
    return LGBMRegressor(
        objective='regression',
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )


def make_surface_pipeline(features):
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('model', make_lgbm()),
    ])


def make_piecewise_pipeline(features):
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('model', Ridge(alpha=1.0)),
    ])


def add_piecewise(df, threshold, exposure, controls):
    x = pd.to_numeric(df[exposure], errors='coerce')
    out = pd.DataFrame(index=df.index)
    out['exposure'] = x
    out['above_threshold'] = np.maximum(x - threshold, 0.0)
    for col in controls:
        out[col] = pd.to_numeric(df[col], errors='coerce')
    return out


def write_json_mirror(df, path):
    path.write_text(json.dumps(df.to_dict(orient='records'), indent=2, ensure_ascii=False, default=str) + '\n')


def main():
    master = pd.read_csv(MASTER_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if len(master) != 1615 or len(train) != 1292 or len(test) != 323:
        raise AssertionError('Unexpected V3 row counts; refusing to continue.')
    if set(train.station.unique()) - set(master.station.unique()) or set(test.station.unique()) - set(master.station.unique()):
        raise AssertionError('Split station universe is not contained in master.')

    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [MASTER_PATH, TRAIN_PATH, TEST_PATH]}
    (OUT / 'input_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')

    protected = {'station', 'pm25', 'season'}
    numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    surface_features = [c for c in numeric if c not in protected]
    if not surface_features:
        raise AssertionError('No numeric surface predictors available.')

    X_train = train[surface_features]
    y_train = train['pm25'].astype(float)
    X_test = test[surface_features]
    y_test = test['pm25'].astype(float)

    # Training-only station-grouped CV for the station-supported predictive model.
    cv_rows = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=train['station']), start=1):
        model = make_surface_pipeline(surface_features)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[va_idx])
        row = metrics(y_train.iloc[va_idx], pred)
        row.update({'fold': fold, 'n_train': int(len(tr_idx)), 'n_validation': int(len(va_idx)), 'stations_validation': int(train.iloc[va_idx]['station'].nunique())})
        cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS / 'surface_training_cv.csv', index=False)

    surface_model = make_surface_pipeline(surface_features)
    surface_model.fit(X_train, y_train)
    pred_test = surface_model.predict(X_test)
    overall = metrics(y_test, pred_test)
    overall.update({'model': 'LightGBM', 'split': 'locked_test', 'features': int(len(surface_features)), 'stations_test': int(test['station'].nunique())})
    cv_mean = {'model': 'LightGBM', 'split': 'training_grouped_cv_mean', **{k: float(cv_df[k].mean()) for k in ['r2', 'rmse', 'mae']}, 'n': int(cv_df['n'].sum())}
    pd.DataFrame([overall, cv_mean]).to_csv(RESULTS / 'surface_model_metrics.csv', index=False)

    predictions = test[['station', 'year', 'month', 'latitude', 'longitude', 'pm25']].copy()
    predictions['predicted_pm25'] = pred_test
    predictions['residual'] = predictions['pm25'] - predictions['predicted_pm25']
    predictions['absolute_error'] = predictions['residual'].abs()
    predictions.to_csv(RESULTS / 'surface_test_predictions.csv', index=False)
    station_summary = predictions.groupby('station', as_index=False).agg(
        latitude=('latitude', 'first'), longitude=('longitude', 'first'), n_test=('pm25', 'size'),
        observed_pm25_mean=('pm25', 'mean'), predicted_pm25_mean=('predicted_pm25', 'mean'),
        mae=('absolute_error', 'mean'), rmse=('residual', lambda x: float(np.sqrt(np.mean(np.square(x))))),
    )
    station_summary.to_csv(RESULTS / 'station_surface_summary.csv', index=False)

    # Descriptive station-supported map. No interpolation or raster extrapolation.
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    sc = ax.scatter(station_summary['longitude'], station_summary['latitude'], c=station_summary['predicted_pm25_mean'], s=95, cmap='viridis', edgecolor='black', linewidth=0.35)
    for _, r in station_summary.iterrows():
        ax.annotate(str(r['station'])[:14], (r['longitude'], r['latitude']), xytext=(3, 3), textcoords='offset points', fontsize=5.8)
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('Predicted PM₂.₅ (µg/m³)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Station-supported predicted PM₂.₅ surface\nLightGBM; observed station coordinates only — not interpolated')
    ax.text(0.01, 0.01, 'Descriptive predictive map; no unobserved-grid or causal inference.', transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.7'))
    fig.tight_layout(); fig.savefig(PLOTS / '01_station_supported_surface.png', bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    sc = ax.scatter(station_summary['longitude'], station_summary['latitude'], c=station_summary['mae'], s=95, cmap='magma', edgecolor='black', linewidth=0.35)
    for _, r in station_summary.iterrows():
        ax.annotate(str(r['station'])[:14], (r['longitude'], r['latitude']), xytext=(3, 3), textcoords='offset points', fontsize=5.8)
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('Station mean absolute error (µg/m³)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Station-level predictive error map\nDescriptive diagnostic, not a spatial causal effect map')
    fig.tight_layout(); fig.savefig(PLOTS / '02_station_error_map.png', bbox_inches='tight'); plt.close(fig)

    # Threshold screen: predeclared treatment quantile grid, training-only grouped CV.
    exposure = 'sentinel2_ndvi_mean_1000m'
    if exposure not in train.columns:
        raise AssertionError(f'Missing threshold exposure: {exposure}')
    preferred_controls = [
        'latitude', 'longitude', 'year', 'month_sin', 'month_cos', 'season_encoded',
        'era5_temp_mean', 'era5_rh_mean', 'era5_wind_speed_mean', 'era5_blh_mean',
        'population_density_2025_1000m', 'road_density_1000m', 'major_road_density_1000m',
        'dynamicworld_2025_water_frac_1000m', 'dynamicworld_2025_built_frac_1000m',
        'dynamicworld_2025_bare_frac_1000m',
    ]
    threshold_controls = [c for c in preferred_controls if c in train.columns]
    quantiles = np.arange(0.10, 0.901, 0.05)
    candidate_values = [float(train[exposure].quantile(q)) for q in quantiles]
    threshold_rows = []
    fold_selected = []
    for threshold in candidate_values:
        Xpw = add_piecewise(train, threshold, exposure, threshold_controls)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xpw, y_train, groups=train['station']), start=1):
            model = make_piecewise_pipeline(list(Xpw.columns))
            model.fit(Xpw.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = model.predict(Xpw.iloc[va_idx])
            score = metrics(y_train.iloc[va_idx], pred)
            fold_scores.append(score)
        threshold_rows.append({'threshold': threshold, 'threshold_quantile': float(np.interp(threshold, candidate_values, quantiles)), 'cv_rmse': float(np.mean([s['rmse'] for s in fold_scores])), 'cv_mae': float(np.mean([s['mae'] for s in fold_scores])), 'cv_r2': float(np.mean([s['r2'] for s in fold_scores]))})
    threshold_cv = pd.DataFrame(threshold_rows).sort_values(['cv_rmse', 'threshold']).reset_index(drop=True)
    selected_threshold = float(threshold_cv.iloc[0]['threshold'])
    selected_q = float(threshold_cv.iloc[0]['threshold_quantile'])

    # Selection stability across the five validation training subsets.
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_train, groups=train['station']), start=1):
        fold_train = train.iloc[tr_idx]
        fold_y = y_train.iloc[tr_idx]
        fold_candidates = [float(fold_train[exposure].quantile(q)) for q in quantiles]
        scores = []
        for threshold in fold_candidates:
            Xpw = add_piecewise(fold_train, threshold, exposure, threshold_controls)
            inner = GroupKFold(n_splits=4)
            rmses = []
            for it, iv in inner.split(Xpw, fold_y, groups=fold_train['station']):
                model = make_piecewise_pipeline(list(Xpw.columns)); model.fit(Xpw.iloc[it], fold_y.iloc[it]); p = model.predict(Xpw.iloc[iv]); rmses.append(np.sqrt(mean_squared_error(fold_y.iloc[iv], p)))
            scores.append(float(np.mean(rmses)))
        best = int(np.argmin(scores))
        fold_selected.append({'fold': fold, 'selected_threshold': fold_candidates[best], 'selected_quantile': float(quantiles[best]), 'inner_cv_rmse': scores[best]})
    pd.DataFrame(fold_selected).to_csv(RESULTS / 'threshold_fold_selection.csv', index=False)

    # Locked test evaluation after threshold selection.
    Xpw_train = add_piecewise(train, selected_threshold, exposure, threshold_controls)
    Xpw_test = add_piecewise(test, selected_threshold, exposure, threshold_controls)
    threshold_model = make_piecewise_pipeline(list(Xpw_train.columns)); threshold_model.fit(Xpw_train, y_train); threshold_pred_test = threshold_model.predict(Xpw_test)
    threshold_metrics = metrics(y_test, threshold_pred_test)
    threshold_metrics.update({'model': 'Piecewise Ridge', 'split': 'locked_test', 'selected_threshold': selected_threshold, 'selected_training_quantile': selected_q, 'controls': int(len(threshold_controls))})
    pd.DataFrame([threshold_metrics]).to_csv(RESULTS / 'threshold_model_metrics.csv', index=False)

    # Station bootstrap of the predeclared candidate-selection procedure.
    rng = np.random.default_rng(RANDOM_STATE)
    stations = train['station'].drop_duplicates().tolist()
    bootstrap_rows = []
    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        boot = pd.concat([train[train['station'] == s] for s in sampled], ignore_index=True)
        boot_y = boot['pm25'].astype(float)
        boot_candidates = [float(boot[exposure].quantile(q)) for q in quantiles]
        boot_scores = []
        for threshold in boot_candidates:
            Xpw = add_piecewise(boot, threshold, exposure, threshold_controls)
            inner = GroupKFold(n_splits=5)
            rmses = []
            for it, iv in inner.split(Xpw, boot_y, groups=boot['station']):
                if len(set(boot.iloc[it]['station']) & set(boot.iloc[iv]['station'])) > 0:
                    raise AssertionError('Bootstrap grouped fold overlap detected.')
                model = make_piecewise_pipeline(list(Xpw.columns)); model.fit(Xpw.iloc[it], boot_y.iloc[it]); p = model.predict(Xpw.iloc[iv]); rmses.append(np.sqrt(mean_squared_error(boot_y.iloc[iv], p)))
            boot_scores.append(float(np.mean(rmses)))
        best = int(np.argmin(boot_scores))
        bootstrap_rows.append({'bootstrap': b + 1, 'selected_threshold': boot_candidates[best], 'selected_quantile': float(quantiles[best]), 'cv_rmse': boot_scores[best]})
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(RESULTS / 'threshold_bootstrap_selection.csv', index=False)

    threshold_cv.to_csv(RESULTS / 'threshold_cv_results.csv', index=False)
    # Threshold diagnostic plot.
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.plot(threshold_cv['threshold'], threshold_cv['cv_rmse'], marker='o', color='#155e75')
    ax.axvline(selected_threshold, color='#b91c1c', linestyle='--', label=f'selected = {selected_threshold:.3f}')
    ax.set_xlabel('Candidate NDVI breakpoint (training quantile grid)')
    ax.set_ylabel('Mean grouped-CV RMSE (µg/m³)')
    ax.set_title('Pre-specified predictive threshold screen\nSelection uses training station-grouped CV only')
    ax.legend(frameon=False)
    ax.text(0.01, 0.02, 'A breakpoint is not a causal vegetation threshold.', transform=ax.transAxes, fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / '03_threshold_cv.png', bbox_inches='tight'); plt.close(fig)

    # Relationship plot with selected piecewise fitted curve at median controls.
    grid = np.linspace(float(train[exposure].min()), float(train[exposure].max()), 200)
    base = {c: float(pd.to_numeric(train[c], errors='coerce').median()) for c in threshold_controls}
    curve = pd.DataFrame({exposure: grid, **base})
    curve_pw = add_piecewise(curve, selected_threshold, exposure, threshold_controls)
    curve['predicted_pm25'] = threshold_model.predict(curve_pw)
    bins = pd.qcut(train[exposure], q=10, duplicates='drop')
    binned = train.assign(_bin=bins).groupby('_bin', observed=True).agg(ndvi_mean=(exposure, 'mean'), pm25_mean=('pm25', 'mean'), pm25_sd=('pm25', 'std'), n=('pm25', 'size')).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180)
    ax.errorbar(binned['ndvi_mean'], binned['pm25_mean'], yerr=1.96 * binned['pm25_sd'] / np.sqrt(binned['n']), fmt='o', color='#334155', capsize=3, label='Observed decile mean ± 95% SE')
    ax.plot(curve[exposure], curve['predicted_pm25'], color='#0f766e', linewidth=2.2, label='Piecewise Ridge fitted curve')
    ax.axvline(selected_threshold, color='#b91c1c', linestyle='--', label=f'selected breakpoint = {selected_threshold:.3f}')
    ax.set_xlabel('Sentinel-2 NDVI mean, 1,000 m')
    ax.set_ylabel('PM₂.₅ (µg/m³)')
    ax.set_title('Descriptive NDVI–PM₂.₅ threshold screen\nControls held at training medians for visualization')
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, 0.02, 'Associational/predictive diagnostic; not an intervention threshold or causal dose-response.', transform=ax.transAxes, fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / '04_threshold_relationship.png', bbox_inches='tight'); plt.close(fig)

    stable_share = float((bootstrap_df['selected_quantile'] == selected_q).mean())
    at_boundary = bool(selected_q in {0.10, 0.90})
    bootstrap_q025 = float(bootstrap_df['selected_threshold'].quantile(0.025))
    bootstrap_q975 = float(bootstrap_df['selected_threshold'].quantile(0.975))
    threshold_summary = {
        'exposure': exposure,
        'selected_threshold': selected_threshold,
        'selected_training_quantile': selected_q,
        'candidate_quantiles': quantiles.tolist(),
        'bootstrap_replicates': N_BOOTSTRAP,
        'bootstrap_selection_share_same_quantile': stable_share,
        'bootstrap_threshold_q025': bootstrap_q025,
        'bootstrap_threshold_q975': bootstrap_q975,
        'selected_at_grid_boundary': at_boundary,
        'stable_threshold_rule': 'same-grid-quantile share >= 0.50 and not at 0.10/0.90 boundary',
        'stable_threshold_identified': bool(stable_share >= 0.50 and not at_boundary),
        'locked_test_metrics': threshold_metrics,
    }
    (OUT / 'threshold_summary.json').write_text(json.dumps(threshold_summary, indent=2, default=float) + '\n')

    report = f'''# V3 Spatial PM₂.₅ Surface and Threshold Analysis Report\n\n## Interpretation boundary\n\nThis package is a separate predictive/associational extension. It does not replace the V3 DML estimand, does not create causal effect maps, and does not establish a vegetation intervention threshold.\n\n## Spatial prediction surface\n\nThe frozen V3 master contains {master.station.nunique()} stations with fixed coordinates spanning {master.latitude.min():.6f}–{master.latitude.max():.6f} latitude and {master.longitude.min():.6f}–{master.longitude.max():.6f} longitude. It does not contain a validated grid of environmental covariates for unobserved locations. Therefore the “surface” is deliberately **station-supported**: LightGBM predictions are shown at observed station coordinates and summarized by station. No interpolation, IDW filling, raster extrapolation, or unobserved-grid prediction is performed.\n\nThe locked-test LightGBM diagnostic has R² **{overall["r2"]:.6f}**, RMSE **{overall["rmse"]:.6f} µg/m³**, and MAE **{overall["mae"]:.6f} µg/m³** across {overall["n"]} rows and {test.station.nunique()} stations. These are predictive metrics. They should not be interpreted as spatial generalization to a new monitoring network or as causal green-cover effects.\n\n## Threshold screen\n\nThe exposure is `{exposure}` and the outcome is `pm25`. Candidate breakpoints were fixed at training quantiles 0.10–0.90 in 0.05 increments. A piecewise-linear Ridge model with the pre-specified control set was selected by five-fold station-grouped training CV; the locked test set was not used to choose the breakpoint. The selected breakpoint is **{selected_threshold:.6f}** (training quantile **{selected_q:.2f}**).\n\nThe locked-test piecewise model has R² **{threshold_metrics["r2"]:.6f}**, RMSE **{threshold_metrics["rmse"]:.6f} µg/m³**, and MAE **{threshold_metrics["mae"]:.6f} µg/m³**. The station-bootstrap same-grid selection share is **{stable_share:.3f}**, with a bootstrap breakpoint interval of **[{bootstrap_q025:.6f}, {bootstrap_q975:.6f}]**. Under the frozen stability rule, a stable threshold is **{'identified' if threshold_summary['stable_threshold_identified'] else 'not identified'}**.\n\nEven if a breakpoint is stable, it is a predictive association in this station-month sample. It may reflect confounding, measurement error, seasonality, station structure, or model misspecification. It must not be presented as “the amount of greenery required” or as a causal policy threshold.\n\n## Limitations\n\nThe station-supported map is not a continuous PM₂.₅ raster. A continuous surface requires a validated prediction-grid covariate table and a spatially independent evaluation design. The threshold screen is not a nonlinear DML dose-response estimator; a causal threshold claim would require a separate estimand, pre-treatment exposure window, overlap analysis, and identification strategy.\n\n## Reproducibility\n\n```bash\npython data/modeling_changes/spatial_threshold_v1/run_spatial_threshold.py\npython data/modeling_changes/spatial_threshold_v1/validate_spatial_threshold.py\n```\n\nThe input contract is frozen in `analysis_contract.md`, and SHA-256 hashes are recorded in `input_hashes.json`.\n'''
    (OUT / 'spatial_threshold_report.md').write_text(report)

    config = {
        'random_state': RANDOM_STATE,
        'surface_model': 'LightGBMRegressor',
        'surface_features_count': len(surface_features),
        'surface_training_cv': 'GroupKFold(n_splits=5, groups=station)',
        'threshold_exposure': exposure,
        'threshold_controls': threshold_controls,
        'threshold_candidate_quantiles': quantiles.tolist(),
        'threshold_selection': 'minimum mean station-grouped training CV RMSE',
        'threshold_model': 'piecewise Ridge with x and max(x-c,0)',
        'threshold_bootstrap_replicates': N_BOOTSTRAP,
        'spatial_support': 'observed station coordinates only; no interpolation or raster extrapolation',
    }
    (OUT / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    # JSON review mirrors for all tables; CSV remains the local notebook output.
    for csv_path in RESULTS.glob('*.csv'):
        write_json_mirror(pd.read_csv(csv_path), csv_path.with_suffix('.json'))

    print(json.dumps({'surface_test': overall, 'threshold_test': threshold_metrics, 'selected_threshold': selected_threshold, 'selected_quantile': selected_q, 'bootstrap_same_quantile_share': stable_share, 'stable_threshold_identified': threshold_summary['stable_threshold_identified']}, indent=2))


if __name__ == '__main__':
    main()
