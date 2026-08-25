from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from train_baseline_models import make_models

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/modeling/stratified_splits/train.csv"
TEST_PATH = ROOT / "data/modeling/stratified_splits/test.csv"
RESULTS_DIR = ROOT / "data/modeling/results"
TARGET = "pm25"
NON_PREDICTIVE_COLS = {"station", TARGET}
SEED = 42


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = [
        c for c in train.columns
        if c not in NON_PREDICTIVE_COLS and pd.api.types.is_numeric_dtype(train[c])
    ]
    x_train = train[features]
    x_test = test[features]
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for name, model in make_models().items():
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
        rows.append(
            {
                "model": name,
                "train_r2": float(r2_score(y_train, train_pred)),
                "test_r2": float(r2_score(y_test, test_pred)),
                "train_mae": float(mean_absolute_error(y_train, train_pred)),
                "test_mae": float(mean_absolute_error(y_test, test_pred)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "abs_r2_gap": float(abs(r2_score(y_train, train_pred) - r2_score(y_test, test_pred))),
                "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
                "cv_r2_std": float(np.std(cv_scores["test_r2"], ddof=1)),
                "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
                "cv_rmse_std": float(np.std(-cv_scores["test_rmse"], ddof=1)),
            }
        )

    summary = {
        "split": "month_year_stratified_alternative",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_target_mean": float(y_train.mean()),
        "test_target_mean": float(y_test.mean()),
        "target_mean_difference": float(y_test.mean() - y_train.mean()),
        "models": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "stratified_split_model_performance.csv", index=False)
    with (RESULTS_DIR / "stratified_split_evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
