"""Attachment-driven missingness and metric audit for the V3 DML package.

The PM2.5 band accuracy metric is presentation-only classification accuracy
from fixed concentration bands; it is not a replacement for regression
metrics or a formal AQI computation.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[3]
DML = ROOT / "data" / "modeling_changes" / "dml_v3"
INPUT = ROOT / "data" / "modeling_changes"
TREATMENTS = [
    "sentinel2_ndvi_mean_1000m",
    "sentinel2_ndvi_mean_500m",
    "modis_ndvi_mean_1000m",
]
TARGET = "pm25"
# Concentration bands used only to make model-output diagnostics presentation-friendly.
PM25_BAND_EDGES = np.array([30.0, 60.0, 90.0, 120.0, 250.0])
PM25_BAND_LABELS = ["good", "satisfactory", "moderate", "poor", "very_poor", "severe"]


def domain(column: str) -> str:
    c = column.lower()
    if c in {"station", "year", "month", "pm25"}:
        return "outcome_and_keys"
    if "ndvi" in c or "evi" in c or "ndwi" in c:
        return "vegetation_treatment_or_proxy"
    if "s5p" in c or "no2" in c:
        return "pollution_proxy"
    if "era5" in c or "lst" in c or any(x in c for x in ["temp", "rh_", "wind", "blh"]):
        return "meteorology"
    if "dynamicworld" in c or "population" in c or "road" in c or "spatial" in c or "lat" in c or "lon" in c:
        return "land_use_and_spatial"
    if "month_" in c or "season" in c or "year" in c:
        return "temporal"
    if "gradient" in c:
        return "derived_gradient"
    return "other"


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": safe_r2(y_true, y_pred),
    }


def band_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = np.digitize(y_true, PM25_BAND_EDGES)
    predicted = np.digitize(y_pred, PM25_BAND_EDGES)
    return float(np.mean(actual == predicted))


def write_missingness() -> None:
    train = pd.read_csv(INPUT / "splits/train.csv")
    test = pd.read_csv(INPUT / "splits/test.csv")
    rows: List[Dict[str, object]] = []
    for split, frame in [("train", train), ("test", test)]:
        for column in frame.columns:
            missing_n = int(frame[column].isna().sum())
            rows.append({
                "split": split,
                "variable": column,
                "domain": domain(column),
                "n_rows": int(len(frame)),
                "missing_n": missing_n,
                "missing_pct": float(100.0 * missing_n / len(frame)),
                "nonmissing_n": int(frame[column].notna().sum()),
            })
    overview = pd.DataFrame(rows).sort_values(["split", "domain", "missing_pct"], ascending=[True, True, False])
    overview.to_csv(DML / "missingness_overview.csv", index=False)

    station_rows: List[Dict[str, object]] = []
    selected = [TARGET] + TREATMENTS
    for split, frame in [("train", train), ("test", test)]:
        for station, group in frame.groupby("station", dropna=False):
            row: Dict[str, object] = {"split": split, "station": station, "n_rows": int(len(group))}
            for col in selected:
                row[f"{col}_missing_pct"] = float(100.0 * group[col].isna().mean())
            station_rows.append(row)
    pd.DataFrame(station_rows).sort_values(["split", "station"]).to_csv(DML / "missingness_by_station.csv", index=False)


def write_metrics() -> None:
    rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for treatment in TREATMENTS:
        path = DML / f"crossfit_observations_{treatment}.csv"
        frame = pd.read_csv(path)
        for fold, part in frame.groupby("fold", sort=True):
            y = part[TARGET].to_numpy(float)
            yhat = part["y_hat_oof"].to_numpy(float)
            t = part[treatment].to_numpy(float)
            that = part["t_hat_oof"].to_numpy(float)
            ym = regression_metrics(y, yhat)
            tm = regression_metrics(t, that)
            row: Dict[str, object] = {
                "treatment": treatment,
                "fold": int(fold),
                "n_rows": int(len(part)),
                "y_rmse": ym["rmse"],
                "y_mae": ym["mae"],
                "y_r2": ym["r2"],
                "t_rmse": tm["rmse"],
                "t_mae": tm["mae"],
                "t_r2": tm["r2"],
                "pm25_band_accuracy": band_accuracy(y, yhat),
                "residual_orthogonality_corr": float(np.corrcoef(part["t_residual"], part["y_residual"])[0, 1]),
                "mean_orthogonal_score": float(part["orthogonal_score"].mean()),
            }
            rows.append(row)
        y = frame[TARGET].to_numpy(float)
        yhat = frame["y_hat_oof"].to_numpy(float)
        t = frame[treatment].to_numpy(float)
        that = frame["t_hat_oof"].to_numpy(float)
        ym = regression_metrics(y, yhat)
        tm = regression_metrics(t, that)
        summaries.append({
            "treatment": treatment,
            "n_rows": int(len(frame)),
            "n_folds": int(frame["fold"].nunique()),
            "y_rmse": ym["rmse"],
            "y_mae": ym["mae"],
            "y_r2": ym["r2"],
            "t_rmse": tm["rmse"],
            "t_mae": tm["mae"],
            "t_r2": tm["r2"],
            "pm25_band_accuracy": band_accuracy(y, yhat),
            "residual_orthogonality_corr": float(np.corrcoef(frame["t_residual"], frame["y_residual"])[0, 1]),
            "mean_orthogonal_score": float(frame["orthogonal_score"].mean()),
        })
    pd.DataFrame(rows).sort_values(["treatment", "fold"]).to_csv(DML / "metric_audit_by_fold.csv", index=False)
    pd.DataFrame(summaries).to_csv(DML / "metric_audit_summary.csv", index=False)
    (DML / "metric_definitions.json").write_text(json.dumps({
        "regression_metrics": ["RMSE", "MAE", "R2"],
        "pm25_band_accuracy": "Presentation-only agreement of predicted and observed PM2.5 concentration bands.",
        "pm25_band_edges_ug_m3": [0, 30, 60, 90, 120, 250, "infinity"],
        "causal_interpretation": "None; these are nuisance prediction and diagnostic metrics, not causal effect metrics.",
    }, indent=2), encoding="utf-8")


def main() -> None:
    write_missingness()
    write_metrics()
    print("ATTACHMENT_AUDITS: PASS")
    print(pd.read_csv(DML / "metric_audit_summary.csv").to_string(index=False))
    print("missingness_rows", len(pd.read_csv(DML / "missingness_overview.csv")))


if __name__ == "__main__":
    main()
