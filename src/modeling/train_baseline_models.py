"""Train publication-ready baseline models on the locked PM2.5 split.

The station identifier is intentionally excluded from predictors. This prevents
memorization of station-specific target levels and keeps the baseline focused on
measured environmental and spatial predictors. The master dataset and split CSVs
are read-only inputs; all generated outputs go under data/modeling/results/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/modeling/splits/train.csv"
TEST_PATH = ROOT / "data/modeling/splits/test.csv"
RESULTS_DIR = ROOT / "data/modeling/results"
RANDOM_SEED = 42
TARGET = "pm25"
IDENTIFIER_COLS = ["station", TARGET]
NON_PREDICTIVE_COLS = {"station", TARGET}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET not in train or TARGET not in test:
        raise ValueError(f"Both split files must contain the target column {TARGET!r}")

    feature_cols = [
        c for c in train.columns
        if c not in NON_PREDICTIVE_COLS
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric predictors found after excluding identifiers and target")

    x_train = train[feature_cols].copy()
    x_test = test[feature_cols].copy()
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)
    return x_train, x_test, y_train, y_test


def make_models() -> dict[str, object]:
    ridge = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=18,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lightgbm = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=600,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=8,
                    min_child_samples=20,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.2,
                    reg_lambda=1.0,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    return {"Ridge": ridge, "Random Forest": random_forest, "LightGBM": lightgbm}


def evaluate_model(name: str, model: object, x_train: pd.DataFrame, x_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series, cv: KFold) -> tuple[dict, object]:
    cv_scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        n_jobs=1,
        return_train_score=False,
    )
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    metrics = {
        "model": name,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "overfitting_indicator_abs_r2_gap": float(abs(train_r2 - test_r2)),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
        "cv_r2_std": float(np.std(cv_scores["test_r2"], ddof=1)),
        "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
        "cv_mae_std": float(np.std(-cv_scores["test_mae"], ddof=1)),
        "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
        "cv_rmse_std": float(np.std(-cv_scores["test_rmse"], ddof=1)),
    }
    return metrics, model


def save_feature_importance(name: str, model: object, feature_names: list[str]) -> None:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        return
    table = pd.DataFrame({"feature": feature_names, "importance": values})
    table["model"] = name
    table.sort_values("importance", ascending=False).to_csv(
        RESULTS_DIR / f"{name.lower().replace(' ', '_')}_feature_importance.csv", index=False
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = load_data()
    models = make_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    feature_names = list(x_train.columns)

    all_metrics = []
    fitted_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        metrics, fitted = evaluate_model(name, model, x_train, x_test, y_train, y_test, cv)
        all_metrics.append(metrics)
        fitted_models[name] = fitted
        save_feature_importance(name, fitted, feature_names)
        print(
            f"  test R2={metrics['test_r2']:.4f}; test RMSE={metrics['test_rmse']:.4f}; "
            f"gap={metrics['overfitting_indicator_abs_r2_gap']:.4f}"
        )

    summary = pd.DataFrame(all_metrics).sort_values("test_r2", ascending=False)
    summary.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False)
    with (RESULTS_DIR / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "target": TARGET,
                "feature_count": len(feature_names),
                "features": feature_names,
                "excluded_columns": sorted(NON_PREDICTIVE_COLS),
                "cv": "5-fold shuffled KFold on training data only",
                "master_dataset_untouched": True,
            },
            handle,
            indent=2,
        )
    for name, model in fitted_models.items():
        joblib.dump(model, RESULTS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    print("\nModel comparison:")
    print(summary[["model", "train_r2", "test_r2", "test_rmse", "overfitting_indicator_abs_r2_gap", "cv_r2_mean", "cv_r2_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
