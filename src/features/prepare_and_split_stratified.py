from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/ml_ready/master_modeling_dataset_v2.csv"
OUTPUT = ROOT / "data/modeling/stratified_splits"
SEED = 42
TARGET_TEST = 323
KEYS = ["station", "year", "month"]


def largest_remainder_allocation(counts: pd.Series, total: int) -> pd.Series:
    if total < 0 or total > int(counts.sum()):
        raise ValueError("Requested allocation is outside the available row count")
    raw = counts.astype(float) * (total / counts.sum())
    allocation = np.floor(raw).astype(int)
    remainder = raw - allocation
    left = int(total - allocation.sum())
    order = sorted(counts.index, key=lambda idx: (-remainder.loc[idx], str(idx)))
    for idx in order[:left]:
        allocation.loc[idx] += 1
    return allocation.astype(int)


def main() -> None:
    df = pd.read_csv(INPUT)
    if len(df) != 1615:
        raise ValueError(f"Expected 1615 master rows, found {len(df)}")
    if df.duplicated(KEYS).any():
        raise ValueError("Master dataset contains duplicate station-year-month keys")

    iit_mask = df["station"].eq("IIT_Delhi")
    if int(iit_mask.sum()) != 1:
        raise ValueError("Expected exactly one IIT_Delhi row")
    pool = df.loc[~iit_mask].copy()
    rng = np.random.default_rng(SEED)

    month_counts = pool.groupby("month").size().sort_index()
    month_quota = largest_remainder_allocation(month_counts, TARGET_TEST)
    selected_indices: list[int] = []

    for month, month_total in month_quota.items():
        month_pool = pool.loc[pool["month"].eq(month)]
        year_counts = month_pool.groupby("year").size().sort_index()
        year_quota = largest_remainder_allocation(year_counts, int(month_total))
        for year, quota in year_quota.items():
            group = month_pool.loc[month_pool["year"].eq(year)]
            if quota:
                chosen = rng.choice(group.index.to_numpy(), size=int(quota), replace=False)
                selected_indices.extend(int(i) for i in chosen)

    test = df.loc[sorted(selected_indices)].copy()
    train = df.drop(index=selected_indices).copy()

    validations = {
        "master_rows": len(df) == 1615,
        "train_rows": len(train) == 1292,
        "test_rows": len(test) == 323,
        "no_key_overlap": not bool(
            set(map(tuple, train[KEYS].to_numpy())).intersection(set(map(tuple, test[KEYS].to_numpy())))
        ),
        "union_equals_master": set(map(tuple, pd.concat([train, test])[KEYS].to_numpy()))
        == set(map(tuple, df[KEYS].to_numpy())),
        "all_years_in_train": set(train["year"]) == {2022, 2023, 2024, 2025},
        "all_years_in_test": set(test["year"]) == {2022, 2023, 2024, 2025},
        "iit_delhi_train_only": "IIT_Delhi" in set(train["station"]) and "IIT_Delhi" not in set(test["station"]),
    }
    if not all(validations.values()):
        raise RuntimeError(f"Alternative split validation failed: {validations}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT / "train.csv", index=False)
    test.to_csv(OUTPUT / "test.csv", index=False)

    diagnostics = {
        "seed": SEED,
        "target_test_rows": TARGET_TEST,
        "validations": validations,
        "train_month_counts": {str(k): int(v) for k, v in train["month"].value_counts().sort_index().items()},
        "test_month_counts": {str(k): int(v) for k, v in test["month"].value_counts().sort_index().items()},
        "train_year_counts": {str(k): int(v) for k, v in train["year"].value_counts().sort_index().items()},
        "test_year_counts": {str(k): int(v) for k, v in test["year"].value_counts().sort_index().items()},
        "train_target_mean": float(train["pm25"].mean()),
        "test_target_mean": float(test["pm25"].mean()),
        "train_target_std": float(train["pm25"].std()),
        "test_target_std": float(test["pm25"].std()),
    }
    with (OUTPUT / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
