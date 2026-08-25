"""Validate the canonical PM2.5 modeling dataset and locked train/test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data/ml_ready/master_modeling_dataset_v2.csv"
TRAIN_PATH = ROOT / "data/modeling/splits/train.csv"
TEST_PATH = ROOT / "data/modeling/splits/test.csv"
RESULTS_DIR = ROOT / "data/modeling/results"

KEY_COLS = ["station", "year", "month"]
REQUIRED_YEARS = {2022, 2023, 2024, 2025}


def season(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5, 6}:
        return "Summer"
    if month in {7, 8, 9}:
        return "Monsoon"
    if month in {10, 11}:
        return "Post-monsoon"
    return "Unknown"


def main() -> None:
    master = pd.read_csv(MASTER_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    validations = {
        "master_rows_1615": len(master) == 1615,
        "train_rows_1292": len(train) == 1292,
        "test_rows_323": len(test) == 323,
        "master_keys_unique": not master.duplicated(KEY_COLS).any(),
        "train_keys_unique": not train.duplicated(KEY_COLS).any(),
        "test_keys_unique": not test.duplicated(KEY_COLS).any(),
        "no_train_test_key_overlap": not bool(
            set(map(tuple, train[KEY_COLS].to_numpy())).intersection(
                set(map(tuple, test[KEY_COLS].to_numpy()))
            )
        ),
        "split_union_equals_master": set(map(tuple, pd.concat([train, test])[KEY_COLS].to_numpy()))
        == set(map(tuple, master[KEY_COLS].to_numpy())),
        "train_years_complete": set(train["year"]) == REQUIRED_YEARS,
        "test_years_complete": set(test["year"]) == REQUIRED_YEARS,
        "iit_delhi_train_only": "IIT_Delhi" in set(train["station"])
        and "IIT_Delhi" not in set(test["station"]),
        "target_present": "pm25" in train.columns and "pm25" in test.columns,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "validations": validations,
        "all_passed": all(validations.values()),
        "master_shape": list(master.shape),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_summary": {
            "master_mean": float(master["pm25"].mean()),
            "train_mean": float(train["pm25"].mean()),
            "test_mean": float(test["pm25"].mean()),
            "master_std": float(master["pm25"].std()),
            "train_std": float(train["pm25"].std()),
            "test_std": float(test["pm25"].std()),
        },
    }

    train_for_analysis = train.copy(deep=True)
    test_for_analysis = test.copy(deep=True)
    train_for_analysis["split"] = "train"
    test_for_analysis["split"] = "test"
    season_table = pd.concat(
        [train_for_analysis, test_for_analysis],
        ignore_index=True,
    )
    season_table["season"] = season_table["month"].map(season)
    season_summary = (
        season_table.groupby(["split", "season"], as_index=False)
        .agg(rows=("pm25", "size"), mean_pm25=("pm25", "mean"), median_pm25=("pm25", "median"))
        .sort_values(["split", "season"])
    )
    season_summary.to_csv(RESULTS_DIR / "season_distribution_analysis.csv", index=False)

    year_summary = (
        season_table.groupby(["split", "year"], as_index=False)
        .agg(rows=("pm25", "size"), mean_pm25=("pm25", "mean"), median_pm25=("pm25", "median"))
        .sort_values(["split", "year"])
    )
    year_summary.to_csv(RESULTS_DIR / "year_distribution_analysis.csv", index=False)

    station_summary = (
        season_table.groupby(["split", "station"], as_index=False)
        .agg(rows=("pm25", "size"), mean_pm25=("pm25", "mean"))
        .sort_values(["split", "station"])
    )
    station_summary.to_csv(RESULTS_DIR / "station_distribution_analysis.csv", index=False)

    with (RESULTS_DIR / "data_integrity_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit("Data-integrity validation failed")


if __name__ == "__main__":
    main()
