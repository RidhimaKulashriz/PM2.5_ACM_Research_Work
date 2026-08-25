"""Double Machine Learning analysis for the V3 PM2.5 panel.

This script intentionally writes only under data/modeling_changes/dml_v3/.
The locked V3 master/train/test inputs and baseline results are never modified.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "pm25"
GROUP = "station"
PRIMARY_TREATMENT = "sentinel2_ndvi_mean_1000m"
SENSITIVITY_TREATMENTS = [
    "sentinel2_ndvi_mean_500m",
    "modis_ndvi_mean_1000m",
]

ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT / "data" / "modeling_changes"
MASTER_PATH = INPUT_DIR / "datasets" / "master_modeling_dataset_v3.csv"
TRAIN_PATH = INPUT_DIR / "splits" / "train.csv"
TEST_PATH = INPUT_DIR / "splits" / "test.csv"
OUT_DIR = INPUT_DIR / "dml_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Treatment and confounder roles are defined before model fitting. All green-cover
# indicators are withheld from X when estimating any one vegetation treatment to
# avoid adjusting for close proxies of the treatment itself.
TIME_SPACE_CONTROLS = {
    "year", "month", "month_sin", "month_cos", "season_encoded",
    "latitude", "longitude",
}
CONTROL_PREFIXES = (
    "era5_",
    "population_density_2025_",
    "road_density_",
    "major_road_density_",
    "dynamicworld_2025_built_frac_",
    "dynamicworld_2025_water_frac_",
    "dynamicworld_2025_bare_frac_",
    "dynamicworld_2025_valid_pixels_",
)
GREEN_TOKENS = (
    "ndvi", "evi", "ndwi", "green", "vegetation", "trees_frac",
    "grass_frac", "crops_frac", "shrub_and_scrub_frac",
    "flooded_vegetation_frac",
)
POLLUTION_TOKENS = ("no2", "s5p", "pollution", "pm10", "aod")
POST_TREATMENT_TOKENS = (
    "modis_lst", "gradient_lst", "gradient_no2", "lst_day", "lst_night",
    "diurnal_range",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(seed_offset: int = 0) -> Pipeline:
    """Create a deterministic nonlinear nuisance learner."""
    learner = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=RANDOM_SEED + seed_offset,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", learner),
    ])


def load_and_validate() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(MASTER_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(master) == 1615, f"Unexpected master rows: {len(master)}"
    assert len(train) == 1292, f"Unexpected train rows: {len(train)}"
    assert len(test) == 323, f"Unexpected test rows: {len(test)}"
    for name, frame in {"master": master, "train": train, "test": test}.items():
        assert {"station", "year", "month", TARGET}.issubset(frame.columns), name
        assert frame[["station", "year", "month"]].duplicated().sum() == 0, name
        assert frame[TARGET].notna().all(), f"{name} has missing outcome"
        numeric = frame.select_dtypes(include=np.number)
        assert np.isfinite(numeric.to_numpy()).all(), f"{name} has non-finite values"
    key = lambda d: set(zip(d.station.astype(str), d.year.astype(int), d.month.astype(int)))
    assert key(train).isdisjoint(key(test)), "Train/test key overlap"
    assert key(train) | key(test) == key(master), "Train/test universe mismatch"
    assert train.station.nunique() == 35 and test.station.nunique() == 34
    assert (train.station == "IIT_Delhi").sum() == 1
    assert (test.station == "IIT_Delhi").sum() == 0
    return master, train, test


def choose_controls(frame: pd.DataFrame, treatment: str) -> List[str]:
    controls: List[str] = []
    for col in frame.columns:
        lower = col.lower()
        if col in {TARGET, GROUP, treatment, "date", "year_month", "season"}:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if any(token in lower for token in GREEN_TOKENS):
            continue
        if any(token in lower for token in POLLUTION_TOKENS):
            continue
        if any(token in lower for token in POST_TREATMENT_TOKENS):
            continue
        if col in TIME_SPACE_CONTROLS or lower.startswith(CONTROL_PREFIXES):
            controls.append(col)
    required = sorted(TIME_SPACE_CONTROLS - {"year", "month"})
    missing_required = [c for c in required if c not in controls]
    assert not missing_required, f"Missing required controls: {missing_required}"
    assert controls, "No controls selected"
    return controls


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_fit(
    train: pd.DataFrame, treatment: str, controls: List[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Cross-fit E[Y|X] and E[T|X] with station-held-out folds."""
    x = train[controls].copy()
    y = train[TARGET].to_numpy(dtype=float)
    t = train[treatment].to_numpy(dtype=float)
    groups = train[GROUP].astype(str).to_numpy()
    y_oof = np.full(len(train), np.nan)
    t_oof = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1, dtype=int)
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(x, y, groups)):
        y_model = make_model(seed_offset=fold)
        t_model = make_model(seed_offset=100 + fold)
        y_model.fit(x.iloc[fit_idx], y[fit_idx])
        t_model.fit(x.iloc[fit_idx], t[fit_idx])
        y_oof[holdout_idx] = y_model.predict(x.iloc[holdout_idx])
        t_oof[holdout_idx] = t_model.predict(x.iloc[holdout_idx])
        fold_ids[holdout_idx] = fold
    assert np.isfinite(y_oof).all() and np.isfinite(t_oof).all()
    y_resid = y - y_oof
    t_resid = t - t_oof
    denominator = float(np.dot(t_resid, t_resid))
    theta = float(np.dot(t_resid, y_resid) / denominator)
    influence = t_resid * (y_resid - theta * t_resid)
    variance = float(np.mean(influence ** 2) / (np.mean(t_resid ** 2) ** 2) / len(train))
    se = float(np.sqrt(max(variance, 0.0)))
    output = train[[GROUP, "year", "month", TARGET, treatment]].copy()
    output["fold"] = fold_ids
    output["y_hat_oof"] = y_oof
    output["t_hat_oof"] = t_oof
    output["y_residual"] = y_resid
    output["t_residual"] = t_resid
    output["orthogonal_score"] = influence
    nuisance = {
        "y_rmse": metrics(y, y_oof)["rmse"],
        "y_mae": metrics(y, y_oof)["mae"],
        "y_r2": metrics(y, y_oof)["r2"],
        "t_rmse": metrics(t, t_oof)["rmse"],
        "t_mae": metrics(t, t_oof)["mae"],
        "t_r2": metrics(t, t_oof)["r2"],
        "t_residual_sd": float(np.std(t_resid, ddof=1)),
        "y_residual_sd": float(np.std(y_resid, ddof=1)),
        "theta": theta,
        "se": se,
        "ci_low": theta - 1.96 * se,
        "ci_high": theta + 1.96 * se,
        "n": int(len(train)),
        "n_stations": int(train[GROUP].nunique()),
        "n_folds": N_SPLITS,
    }
    return output, nuisance


def external_test_diagnostic(
    train: pd.DataFrame, test: pd.DataFrame, treatment: str, controls: List[str]
) -> Dict[str, float]:
    """Estimate an external orthogonal-slope diagnostic on the locked test set.

    Nuisance models are fit on train only. This is a validation diagnostic, not a
    second causal estimate with a fresh confidence interval.
    """
    y_model = make_model(seed_offset=900)
    t_model = make_model(seed_offset=901)
    y_model.fit(train[controls], train[TARGET].to_numpy(dtype=float))
    t_model.fit(train[controls], train[treatment].to_numpy(dtype=float))
    y = test[TARGET].to_numpy(dtype=float)
    t = test[treatment].to_numpy(dtype=float)
    y_resid = y - y_model.predict(test[controls])
    t_resid = t - t_model.predict(test[controls])
    theta = float(np.dot(t_resid, y_resid) / np.dot(t_resid, t_resid))
    return {
        "n_test": int(len(test)),
        "n_test_stations": int(test[GROUP].nunique()),
        "theta_external_test_diagnostic": theta,
        "outcome_nuisance_rmse_test": metrics(y, y - y_resid)["rmse"],
        "treatment_nuisance_rmse_test": metrics(t, t - t_resid)["rmse"],
        "test_t_residual_sd": float(np.std(t_resid, ddof=1)),
    }


def write_report(summary: pd.DataFrame, controls_by_treatment: Dict[str, List[str]],
                 validation: pd.DataFrame) -> None:
    primary = summary.iloc[0]
    lines = [
        "# V3 Double Machine Learning Results",
        "",
        "> This is an observational, partially linear DML analysis. The coefficient is interpretable as a causal effect only under the stated identification assumptions; it is not proof of causality from this dataset alone.",
        "",
        "## Estimand and design",
        "",
        f"The primary treatment is `{PRIMARY_TREATMENT}` (Sentinel-2 NDVI averaged within the 1,000 m buffer), the outcome is monthly station-level `pm25`, and the estimand is the partially linear average treatment effect per one-unit increase in the raw NDVI treatment. The analysis uses the frozen V3 training split only for cross-fitted estimation. Cross-fitting is grouped by station with {N_SPLITS} folds, so each held-out fold contains stations not used to fit its nuisance models.",
        "",
        "The nuisance learners estimate the conditional mean of the outcome and treatment from pre-specified temporal/spatial, ERA5 meteorology, 2025 population, road-density, and non-vegetation built/water/bare land-cover controls. Green-cover proxies, Sentinel-5P NO₂/pollution variables, and contemporaneous MODIS/LST variables are excluded to avoid adjusting for treatment proxies, pollutant proxies, or plausible post-treatment mediators.",
        "",
        "## Results",
        "",
        "| Treatment | DML estimate | SE | 95% CI | N | N stations | External test diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['treatment']}` | {row['theta']:.6f} | {row['se']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {int(row['n'])} | {int(row['n_stations'])} | {row['theta_external_test_diagnostic']:.6f} |"
        )
    lines += [
        "",
        "The primary cross-fitted estimate is reported on the raw NDVI scale, so its units are µg/m³ of PM₂.₅ per one-unit NDVI increase. Because NDVI has a bounded, small empirical range, readers should not interpret a one-unit change as a typical real-world intervention. The corresponding estimate per one-standard-deviation increase is included in `dml_summary.csv`.",
        "",
        "## Diagnostics and limitations",
        "",
        f"The cross-fitted nuisance R² values for the primary outcome and treatment models were {primary['y_r2']:.3f} and {primary['t_r2']:.3f}, respectively. The residualized treatment standard deviation was {primary['t_residual_sd']:.6f}; this documents treatment overlap after adjustment but does not establish exchangeability.",
        "",
        "The 95% interval uses the empirical influence-function variance for the partially linear orthogonal score. It should be treated as model-based uncertainty, not as a correction for unmeasured confounding, temporal dependence, spatial dependence, measurement error, or treatment/outcome simultaneity. IIT Delhi remains train-only by design and is not part of the locked test set.",
        "",
        "The external-test column is deliberately labeled a diagnostic: nuisance models are fit on the training split and the orthogonal slope is evaluated on the locked test split. It is not a second independent DML inference procedure and has no confidence interval here.",
        "",
        "## Reproducibility",
        "",
        "Run `python data/modeling_changes/dml_v3/run_dml.py` from the repository root. All generated artifacts remain in `data/modeling_changes/dml_v3/`; the canonical datasets and `baseline_results_v3` are read-only inputs for this analysis.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `dml_summary.csv` | Cross-fitted estimates, uncertainty, standardized effects, and diagnostics |",
        "| `crossfit_observations_<treatment>.csv` | Fold IDs, nuisance predictions, residuals, and orthogonal scores |",
        "| `nuisance_metrics.csv` | Outcome/treatment nuisance-model diagnostics |",
        "| `external_test_diagnostics.csv` | Train-fitted nuisance validation on locked test rows |",
        "| `feature_manifest.json` | Exact treatment/control roles and excluded-variable rationale |",
        "| `dml_config.json` | Dataset hashes, software versions, seed, and model settings |",
    ]
    (OUT_DIR / "dml_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    master, train, test = load_and_validate()
    treatments = [PRIMARY_TREATMENT] + SENSITIVITY_TREATMENTS
    missing_treatments = [t for t in treatments if t not in train.columns]
    assert not missing_treatments, f"Missing treatment columns: {missing_treatments}"
    summary_rows = []
    nuisance_rows = []
    external_rows = []
    manifest = {
        "outcome": TARGET,
        "grouping_variable": GROUP,
        "primary_treatment": PRIMARY_TREATMENT,
        "sensitivity_treatments": SENSITIVITY_TREATMENTS,
        "treatment_role": "green-cover exposure; all estimates are observational and assumption-dependent",
        "control_selection": {
            "included": ["time_space", "ERA5 meteorology", "2025 population density", "2025 road density", "non-vegetation Dynamic World built/water/bare/valid-pixel context"],
            "excluded": ["all green-cover/vegetation proxies", "Sentinel-5P NO2 and pollution proxies", "contemporaneous MODIS/LST and gradient variables", "station identifier and outcome-derived fields"],
        },
        "treatments": {},
    }
    for treatment in treatments:
        controls = choose_controls(train, treatment)
        cf, nuisance = cross_fit(train, treatment, controls)
        cf.to_csv(OUT_DIR / f"crossfit_observations_{treatment}.csv", index=False)
        external = external_test_diagnostic(train, test, treatment, controls)
        nuisance_rows.append({"treatment": treatment, **nuisance, "n_controls": len(controls)})
        external_rows.append({"treatment": treatment, **external})
        theta = nuisance["theta"]
        t_sd = float(train[treatment].std(ddof=1))
        summary_rows.append({
            "treatment": treatment,
            "theta": theta,
            "se": nuisance["se"],
            "ci_low": nuisance["ci_low"],
            "ci_high": nuisance["ci_high"],
            "effect_per_treatment_sd": theta * t_sd,
            "treatment_mean": float(train[treatment].mean()),
            "treatment_sd": t_sd,
            "treatment_min": float(train[treatment].min()),
            "treatment_max": float(train[treatment].max()),
            **nuisance,
            "n": nuisance["n"],
            "n_stations": nuisance["n_stations"],
            **external,
        })
        manifest["treatments"][treatment] = {
            "role": "primary" if treatment == PRIMARY_TREATMENT else "sensitivity",
            "controls": controls,
            "n_controls": len(controls),
        }
    summary = pd.DataFrame(summary_rows)
    nuisance = pd.DataFrame(nuisance_rows)
    external = pd.DataFrame(external_rows)
    summary.to_csv(OUT_DIR / "dml_summary.csv", index=False)
    nuisance.to_csv(OUT_DIR / "nuisance_metrics.csv", index=False)
    external.to_csv(OUT_DIR / "external_test_diagnostics.csv", index=False)
    (OUT_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = {
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "cross_fitting_group": GROUP,
        "nuisance_learner": "HistGradientBoostingRegressor inside median-imputation pipeline",
        "nuisance_settings": {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 15, "l2_regularization": 1.0},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "master_rows": len(master),
        "train_rows": len(train),
        "test_rows": len(test),
        "input_sha256": {
            "master_modeling_dataset_v3.csv": file_sha256(MASTER_PATH),
            "train.csv": file_sha256(TRAIN_PATH),
            "test.csv": file_sha256(TEST_PATH),
        },
    }
    (OUT_DIR / "dml_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(summary, manifest["treatments"], external)
    print(summary[["treatment", "theta", "se", "ci_low", "ci_high", "effect_per_treatment_sd", "theta_external_test_diagnostic"]].to_string(index=False))
    print(f"Wrote DML outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
