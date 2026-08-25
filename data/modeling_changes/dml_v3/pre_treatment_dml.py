"""Pre-treatment, time-aware DML sensitivity for the frozen V3 panel.

Lagged treatments are constructed within each split and only when the exact
previous calendar month exists in that split. Nuisance models are fit on
strictly earlier years and evaluated on later-year holdouts. Outputs are
isolated under the DML workspace and do not alter base results.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
DML_DIR = ROOT / "data" / "modeling_changes" / "dml_v3"
INPUT_DIR = ROOT / "data" / "modeling_changes"
TRAIN_PATH = INPUT_DIR / "splits" / "train.csv"
TEST_PATH = INPUT_DIR / "splits" / "test.csv"
TARGET = "pm25"
GROUP = "station"
TREATMENTS = [
    "sentinel2_ndvi_mean_1000m",
    "sentinel2_ndvi_mean_500m",
    "modis_ndvi_mean_1000m",
]
HOLDOUT_YEARS = [2023, 2024, 2025]
SEED = 42

sys.path.insert(0, str(DML_DIR))
from robustness_checks import cluster_se  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_controls(treatment: str) -> List[str]:
    manifest = json.loads((DML_DIR / "feature_manifest.json").read_text())
    controls = manifest["treatments"][treatment]["controls"]
    assert controls
    return controls


def add_exact_lags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["period"] = pd.to_datetime(dict(year=result["year"].astype(int), month=result["month"].astype(int), day=1))
    result = result.sort_values([GROUP, "period"]).reset_index(drop=True)
    key = result[[GROUP, "period"]].copy()
    for treatment in TREATMENTS:
        lag_lookup = result[[GROUP, "period", treatment]].copy()
        lag_lookup["period"] = lag_lookup["period"] + pd.offsets.MonthBegin(1)
        lag_lookup = lag_lookup.rename(columns={treatment: f"lag_{treatment}"})
        result = result.merge(lag_lookup, on=[GROUP, "period"], how="left", validate="one_to_one")
    result["lag_source_split"] = result.get("split", "unknown")
    return result


def model(seed: int) -> Pipeline:
    learner = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", learner)])


def run_treatment(train: pd.DataFrame, treatment: str, controls: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    lagged = f"lag_{treatment}"
    usable = train[train[lagged].notna() & train[TARGET].notna()].copy()
    y_oof = np.full(len(usable), np.nan)
    t_oof = np.full(len(usable), np.nan)
    fold_rows = []
    for holdout_year in HOLDOUT_YEARS:
        hold_mask = usable["year"].eq(holdout_year)
        fit_mask = usable["year"].lt(holdout_year)
        fit_idx = np.flatnonzero(fit_mask.to_numpy())
        hold_idx = np.flatnonzero(hold_mask.to_numpy())
        if len(fit_idx) < 40 or len(hold_idx) < 10:
            continue
        ym = model(SEED + holdout_year)
        tm = model(SEED + 100 + holdout_year)
        x_fit = usable.iloc[fit_idx][controls]
        x_hold = usable.iloc[hold_idx][controls]
        y_fit = usable.iloc[fit_idx][TARGET].to_numpy(float)
        t_fit = usable.iloc[fit_idx][lagged].to_numpy(float)
        ym.fit(x_fit, y_fit)
        tm.fit(x_fit, t_fit)
        y_oof[hold_idx] = ym.predict(x_hold)
        t_oof[hold_idx] = tm.predict(x_hold)
        fold_rows.append({
            "treatment": treatment,
            "holdout_year": int(holdout_year),
            "train_year_max": int(holdout_year - 1),
            "n_train": int(len(fit_idx)),
            "n_holdout": int(len(hold_idx)),
            "n_train_stations": int(usable.iloc[fit_idx][GROUP].nunique()),
            "n_holdout_stations": int(usable.iloc[hold_idx][GROUP].nunique()),
        })
    valid = np.isfinite(y_oof) & np.isfinite(t_oof)
    scored = usable.loc[valid, [GROUP, "year", "month", TARGET, lagged]].copy()
    scored["time_holdout_year"] = usable.loc[valid, "year"].astype(int).to_numpy()
    scored["y_prediction"] = y_oof[valid]
    scored["t_prediction"] = t_oof[valid]
    scored["y_residual"] = scored[TARGET].to_numpy(float) - scored["y_prediction"].to_numpy(float)
    scored["t_residual"] = scored[lagged].to_numpy(float) - scored["t_prediction"].to_numpy(float)
    scored["orthogonal_score"] = scored["t_residual"] * scored["y_residual"]
    scored["treatment"] = treatment
    assert len(scored) > 0
    robust = cluster_se(scored.rename(columns={GROUP: "station"}), treatment)
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "treatment": treatment,
        "lagged_column": lagged,
        "n_lagged_rows": int(len(usable)),
        "n_time_aware_oof": int(len(scored)),
        "time_aware_theta": float(robust["theta"]),
        "time_aware_cluster_se": float(robust["cluster_se"]),
        "time_aware_ci_low": float(robust["cluster_ci_low"]),
        "time_aware_ci_high": float(robust["cluster_ci_high"]),
        "n_clusters": int(robust["n_clusters"]),
        "n_time_folds": int(len(fold_df)),
        "lagged_treatment_mean": float(usable[lagged].mean()),
        "lagged_treatment_sd": float(usable[lagged].std(ddof=1)),
    }
    return scored, fold_df, summary


def main() -> None:
    train = pd.read_csv(TRAIN_PATH).assign(split="train")
    test = pd.read_csv(TEST_PATH).assign(split="test")
    train_lagged = add_exact_lags(train)
    test_lagged = add_exact_lags(test)
    combined_lagged = pd.concat([train_lagged, test_lagged], ignore_index=True)
    lag_cols = ["station", "year", "month", "split", "period"] + [f"lag_{t}" for t in TREATMENTS]
    combined_lagged[lag_cols].to_csv(DML_DIR / "pre_treatment_lagged_inputs.csv", index=False)

    summaries, fold_frames = [], []
    for treatment in TREATMENTS:
        controls = load_manifest_controls(treatment)
        scored, folds, summary = run_treatment(train_lagged, treatment, controls)
        scored.to_csv(DML_DIR / f"time_aware_crossfit_{treatment}.csv", index=False)
        folds.to_csv(DML_DIR / f"time_aware_folds_{treatment}.csv", index=False)
        summaries.append(summary)
        fold_frames.append(folds)
    pd.DataFrame(summaries).to_csv(DML_DIR / "pre_treatment_dml_summary.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(DML_DIR / "time_aware_dml_folds.csv", index=False)
    config = {
        "seed": SEED,
        "holdout_years": HOLDOUT_YEARS,
        "lag_definition": "exact previous calendar month within each split; no cross-split lag lookup",
        "cross_fitting": "expanding time: fit on years earlier than holdout year",
        "input_sha256": {"train.csv": sha256(TRAIN_PATH), "test.csv": sha256(TEST_PATH)},
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    (DML_DIR / "pre_treatment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Pre-treatment time-aware DML complete")


if __name__ == "__main__":
    main()
