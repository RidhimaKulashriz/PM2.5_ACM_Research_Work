from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, cross_validate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.modeling.train_baseline_models import TARGET, NON_PREDICTIVE_COLS, load_data, make_models

TRAIN_PATH = ROOT / "data/modeling/splits/train.csv"
TEST_PATH = ROOT / "data/modeling/splits/test.csv"
RESULTS_DIR = ROOT / "data/modeling/results"
SEED = 42


def score_models(x_train, y_train, groups, split_name, splitter):
    rows = []
    for name, model in make_models().items():
        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=splitter,
            groups=groups,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "validation": split_name,
                "model": name,
                "folds": int(len(scores["test_r2"])),
                "r2_mean": float(np.mean(scores["test_r2"])),
                "r2_std": float(np.std(scores["test_r2"], ddof=1)),
                "r2_min": float(np.min(scores["test_r2"])),
                "r2_max": float(np.max(scores["test_r2"])),
                "mae_mean": float(-np.mean(scores["test_mae"])),
                "rmse_mean": float(-np.mean(scores["test_rmse"])),
                "rmse_std": float(np.std(-scores["test_rmse"], ddof=1)),
            }
        )
    return rows


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    master = pd.read_csv(ROOT / "data/ml_ready/master_modeling_dataset_v2.csv")
    x_train, x_test, y_train, y_test = load_data()

    feature_cols = list(x_train.columns)
    suspicious_target_features = [c for c in feature_cols if "pm25" in c.lower() or "pm_25" in c.lower()]
    nonnumeric = [c for c in train.columns if c not in NON_PREDICTIVE_COLS and not pd.api.types.is_numeric_dtype(train[c])]
    test_only = sorted(set(test.columns) - set(train.columns))
    train_only = sorted(set(train.columns) - set(test.columns))

    station_year_groups = train["station"].astype(str) + "::" + train["year"].astype(str)
    station_groups = train["station"].astype(str)
    year_groups = train["year"]

    rows = []
    rows.extend(score_models(x_train, y_train, None, "random_kfold_5", KFold(n_splits=5, shuffle=True, random_state=SEED)))
    rows.extend(score_models(x_train, y_train, station_year_groups, "group_station_year_5", GroupKFold(n_splits=5)))
    rows.extend(score_models(x_train, y_train, station_groups, "group_station_5", GroupKFold(n_splits=5)))
    rows.extend(score_models(x_train, y_train, year_groups, "group_year_4", GroupKFold(n_splits=4)))

    split_shift = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_target_mean": float(train[TARGET].mean()),
        "test_target_mean": float(test[TARGET].mean()),
        "target_mean_difference": float(test[TARGET].mean() - train[TARGET].mean()),
        "train_target_std": float(train[TARGET].std()),
        "test_target_std": float(test[TARGET].std()),
        "test_month_counts": {str(k): int(v) for k, v in test["month"].value_counts().sort_index().items()},
        "train_month_counts": {str(k): int(v) for k, v in train["month"].value_counts().sort_index().items()},
        "test_station_count": int(test["station"].nunique()),
        "train_station_count": int(train["station"].nunique()),
    }

    audit = {
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "suspicious_target_feature_names": suspicious_target_features,
        "ignored_nonnumeric_columns": nonnumeric,
        "train_only_columns": train_only,
        "test_only_columns": test_only,
        "master_columns_match_splits": set(master.columns) == set(train.columns) == set(test.columns),
        "master_key_unique": bool(not master.duplicated(["station", "year", "month"]).any()),
        "train_test_key_overlap": int(
            len(
                set(map(tuple, train[["station", "year", "month"]].to_numpy())).intersection(
                    set(map(tuple, test[["station", "year", "month"]].to_numpy()))
                )
            )
        ),
        "validation_results": rows,
        "split_shift": split_shift,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "validation_strategy_comparison.csv", index=False)
    with (RESULTS_DIR / "overfitting_leakage_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

