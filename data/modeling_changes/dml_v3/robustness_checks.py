"""Robustness diagnostics for the V3 station-month DML analysis.

The script reads the locked V3 inputs and existing cross-fitted residual files,
then writes only additional artifacts below data/modeling_changes/dml_v3/.
It does not alter canonical datasets or baseline results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
DML_DIR = ROOT / "data" / "modeling_changes" / "dml_v3"
INPUT_DIR = ROOT / "data" / "modeling_changes"
TRAIN_PATH = INPUT_DIR / "splits" / "train.csv"
TEST_PATH = INPUT_DIR / "splits" / "test.csv"
TARGET = "pm25"
GROUP = "station"
PRIMARY = "sentinel2_ndvi_mean_1000m"
TREATMENTS = [PRIMARY, "sentinel2_ndvi_mean_500m", "modis_ndvi_mean_1000m"]
SEED = 42
N_FOLDS = 5
N_WILD_BOOTSTRAPS = 2000
N_PERMUTATIONS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) == 1292 and len(test) == 323
    assert train[TARGET].notna().all() and test[TARGET].notna().all()
    return train, test


def controls_from_manifest(treatment: str) -> List[str]:
    manifest = json.loads((DML_DIR / "feature_manifest.json").read_text())
    controls = manifest["treatments"][treatment]["controls"]
    assert controls
    return controls


def cluster_se(cf: pd.DataFrame, treatment: str) -> Dict[str, float]:
    t_resid = cf["t_residual"].to_numpy(float)
    y_resid = cf["y_residual"].to_numpy(float)
    theta = float(np.dot(t_resid, y_resid) / np.dot(t_resid, t_resid))
    psi = t_resid * (y_resid - theta * t_resid)
    denom_mean = float(np.mean(t_resid ** 2))
    influence = psi / denom_mean
    grouped = pd.DataFrame({"group": cf[GROUP].astype(str), "influence": influence}).groupby("group")["influence"].sum()
    n = len(cf)
    g = len(grouped)
    variance = (g / (g - 1)) * float(np.sum(grouped.to_numpy() ** 2)) / (n ** 2)
    se = float(np.sqrt(max(variance, 0.0)))
    return {
        "theta": theta,
        "cluster_se": se,
        "cluster_ci_low": theta - 1.96 * se,
        "cluster_ci_high": theta + 1.96 * se,
        "n_clusters": g,
    }


def wild_cluster_bootstrap(cf: pd.DataFrame, treatment: str) -> Dict[str, float]:
    rng = np.random.default_rng(SEED + 500)
    t_resid = cf["t_residual"].to_numpy(float)
    y_resid = cf["y_residual"].to_numpy(float)
    theta = float(np.dot(t_resid, y_resid) / np.dot(t_resid, t_resid))
    psi = t_resid * (y_resid - theta * t_resid)
    denom = float(np.dot(t_resid, t_resid))
    groups, codes = np.unique(cf[GROUP].astype(str).to_numpy(), return_inverse=True)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_WILD_BOOTSTRAPS, len(groups)))
    signed_scores = signs[:, codes] * psi[None, :]
    boot_theta = theta + signed_scores.sum(axis=1) / denom
    return {
        "wild_bootstrap_reps": N_WILD_BOOTSTRAPS,
        "wild_bootstrap_se": float(np.std(boot_theta, ddof=1)),
        "wild_bootstrap_ci_low": float(np.quantile(boot_theta, 0.025)),
        "wild_bootstrap_ci_high": float(np.quantile(boot_theta, 0.975)),
    }


def fold_and_station_stability(cf: pd.DataFrame, treatment: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def estimate(frame: pd.DataFrame) -> float:
        t = frame["t_residual"].to_numpy(float)
        y = frame["y_residual"].to_numpy(float)
        return float(np.dot(t, y) / np.dot(t, t)) if np.dot(t, t) > 0 else np.nan
    fold_rows = []
    for fold_id, x in cf.groupby("fold", sort=True):
        fold_rows.append({
            "treatment": treatment,
            "fold": int(fold_id),
            "n": len(x),
            "n_stations": x[GROUP].nunique(),
            "theta_fold": estimate(x),
            "t_residual_sd": x["t_residual"].std(ddof=1),
        })
    fold = pd.DataFrame(fold_rows)
    station_rows = []
    for station_id, x in cf.groupby(GROUP, sort=True):
        station_rows.append({
            "treatment": treatment,
            "station": str(station_id),
            "n": len(x),
            "theta_station": estimate(x),
            "t_residual_sd": x["t_residual"].std(ddof=1),
        })
    station = pd.DataFrame(station_rows)
    return fold, station


def overlap_and_permutation(cf: pd.DataFrame, treatment: str) -> Dict[str, float]:
    residual = cf["t_residual"].to_numpy(float)
    rng = np.random.default_rng(SEED + 700)
    by_group = [idx.to_numpy() for _, idx in cf.groupby(GROUP).groups.items()]
    y_resid = cf["y_residual"].to_numpy(float)
    null = np.empty(N_PERMUTATIONS)
    for b in range(N_PERMUTATIONS):
        permuted = residual.copy()
        for idx in by_group:
            permuted[idx] = residual[rng.permutation(idx)]
        null[b] = np.dot(permuted, y_resid) / np.dot(permuted, permuted)
    return {
        "t_residual_min": float(np.min(residual)),
        "t_residual_p01": float(np.quantile(residual, 0.01)),
        "t_residual_p05": float(np.quantile(residual, 0.05)),
        "t_residual_median": float(np.median(residual)),
        "t_residual_p95": float(np.quantile(residual, 0.95)),
        "t_residual_p99": float(np.quantile(residual, 0.99)),
        "t_residual_max": float(np.max(residual)),
        "permutation_reps": N_PERMUTATIONS,
        "permutation_null_mean": float(np.mean(null)),
        "permutation_null_p025": float(np.quantile(null, 0.025)),
        "permutation_null_p975": float(np.quantile(null, 0.975)),
        "permutation_observed_percentile": float(np.mean(null <= (np.dot(residual, y_resid) / np.dot(residual, residual)))),
    }


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=1.0, random_state=seed,
        )
    elif kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=8,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def spatial_block_sensitivity(train: pd.DataFrame, treatment: str, controls: List[str]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Cross-fit nuisance models while holding out geographic station blocks."""
    station_coords = train[[GROUP, "latitude", "longitude"]].drop_duplicates(GROUP).copy()
    lat_mid = station_coords["latitude"].median()
    lon_mid = station_coords["longitude"].median()
    station_coords["spatial_block"] = (
        (station_coords["latitude"] >= lat_mid).astype(int).astype(str)
        + "_"
        + (station_coords["longitude"] >= lon_mid).astype(int).astype(str)
    )
    frame = train.merge(station_coords[[GROUP, "spatial_block"]], on=GROUP, how="left", validate="many_to_one")
    x = frame[controls]
    y = frame[TARGET].to_numpy(float)
    t = frame[treatment].to_numpy(float)
    groups = frame["spatial_block"].to_numpy()
    n_groups = len(np.unique(groups))
    assert n_groups >= 3
    y_hat = np.full(len(frame), np.nan)
    t_hat = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=n_groups)
    fold_rows = []
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x, y, groups)):
        ym = make_model("hist_gradient_boosting", SEED + 200 + fold)
        tm = make_model("hist_gradient_boosting", SEED + 300 + fold)
        ym.fit(x.iloc[fit_idx], y[fit_idx])
        tm.fit(x.iloc[fit_idx], t[fit_idx])
        y_hat[hold_idx] = ym.predict(x.iloc[hold_idx])
        t_hat[hold_idx] = tm.predict(x.iloc[hold_idx])
        fold_rows.append({
            "treatment": treatment,
            "spatial_block_fold": fold,
            "held_out_block": str(groups[hold_idx][0]),
            "n": len(hold_idx),
            "n_stations": frame.iloc[hold_idx][GROUP].nunique(),
        })
    y_resid = y - y_hat
    t_resid = t - t_hat
    theta = float(np.dot(t_resid, y_resid) / np.dot(t_resid, t_resid))
    result = {
        "treatment": treatment,
        "spatial_block_theta": theta,
        "spatial_block_y_r2_oof": float(1 - np.sum(y_resid ** 2) / np.sum((y - y.mean()) ** 2)),
        "spatial_block_t_r2_oof": float(1 - np.sum(t_resid ** 2) / np.sum((t - t.mean()) ** 2)),
        "spatial_block_t_residual_sd": float(np.std(t_resid, ddof=1)),
        "spatial_block_count": n_groups,
    }
    fold_output = pd.DataFrame(fold_rows)
    return result, fold_output


def learner_sensitivity(train: pd.DataFrame, treatment: str, controls: List[str]) -> Dict[str, float]:
    x = train[controls]
    y = train[TARGET].to_numpy(float)
    t = train[treatment].to_numpy(float)
    groups = train[GROUP].astype(str).to_numpy()
    y_hat = np.full(len(train), np.nan)
    t_hat = np.full(len(train), np.nan)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x, y, groups)):
        ym = make_model("random_forest", SEED + fold)
        tm = make_model("random_forest", SEED + 100 + fold)
        ym.fit(x.iloc[fit_idx], y[fit_idx])
        tm.fit(x.iloc[fit_idx], t[fit_idx])
        y_hat[hold_idx] = ym.predict(x.iloc[hold_idx])
        t_hat[hold_idx] = tm.predict(x.iloc[hold_idx])
    y_resid = y - y_hat
    t_resid = t - t_hat
    theta = float(np.dot(t_resid, y_resid) / np.dot(t_resid, t_resid))
    pseudo = pd.DataFrame({GROUP: train[GROUP], "t_residual": t_resid, "y_residual": y_resid})
    robust = cluster_se(pseudo, treatment)
    return {
        "learner": "random_forest",
        "theta": theta,
        "cluster_se": robust["cluster_se"],
        "cluster_ci_low": robust["cluster_ci_low"],
        "cluster_ci_high": robust["cluster_ci_high"],
        "y_r2_oof": float(1 - np.sum(y_resid ** 2) / np.sum((y - y.mean()) ** 2)),
        "t_r2_oof": float(1 - np.sum(t_resid ** 2) / np.sum((t - t.mean()) ** 2)),
    }


def write_report(summary: pd.DataFrame, fold: pd.DataFrame, station: pd.DataFrame,
                 overlap: pd.DataFrame, learner: Dict[str, float], spatial: Dict[str, float]) -> None:
    p = summary.loc[summary.treatment == PRIMARY].iloc[0]
    lines = [
        "# V3 DML Robustness and Validation Report",
        "",
        "> These diagnostics strengthen the first-pass DML analysis but do not remove the need for a credible causal identification strategy. All results remain observational and assumption-dependent.",
        "",
        "## Robust uncertainty",
        "",
        f"For the primary Sentinel-2 1,000 m NDVI treatment, the point estimate is {p['theta_robust']:.6f} µg/m³ per raw NDVI unit. The original influence-function standard error is {p['se']:.6f}; the station-clustered standard error is {p['cluster_se']:.6f}, with 95% interval [{p['cluster_ci_low']:.6f}, {p['cluster_ci_high']:.6f}]. A {int(p['wild_bootstrap_reps'])}-replicate wild cluster bootstrap gives interval [{p['wild_bootstrap_ci_low']:.6f}, {p['wild_bootstrap_ci_high']:.6f}].",
        "",
        "The clustered and wild-bootstrap procedures account for within-station dependence more directly than an observation-independent standard error. With only 35 station clusters, these intervals should still be interpreted cautiously.",
        "",
        "## Stability and falsification",
        "",
        f"The five station-held-out fold estimates range from {fold.loc[fold.treatment == PRIMARY, 'theta_fold'].min():.6f} to {fold.loc[fold.treatment == PRIMARY, 'theta_fold'].max():.6f}; the fold-level standard deviation is {fold.loc[fold.treatment == PRIMARY, 'theta_fold'].std(ddof=1):.6f}. Station-level slopes are exploratory because many stations have limited repeated observations; the complete table is in `station_heterogeneity.csv`.",
        "",
        f"The within-station permutation falsification distribution for the primary residualized treatment has a 2.5%–97.5% null interval of [{overlap.loc[overlap.treatment == PRIMARY, 'permutation_null_p025'].iloc[0]:.6f}, {overlap.loc[overlap.treatment == PRIMARY, 'permutation_null_p975'].iloc[0]:.6f}]. This is a design check under broken treatment/outcome alignment, not a test of unmeasured confounding.",
        "",
        "## Spatial-block sensitivity",
        "",
        f"Holding out deterministic geographic station blocks defined by the median latitude and longitude gives a primary-treatment estimate of {spatial['spatial_block_theta']:.6f}. This is a sensitivity design with {int(spatial['spatial_block_count'])} blocks, not a replacement for a pre-treatment or quasi-experimental design.",
        "",
        "## Nuisance-learner sensitivity",

        "",
        f"Replacing the primary HistGradientBoosting nuisance learners with random-forest nuisance learners gives an estimate of {learner['theta']:.6f} with station-clustered 95% interval [{learner['cluster_ci_low']:.6f}, {learner['cluster_ci_high']:.6f}]. This checks whether the headline result is driven only by one flexible learner family.",
        "",
        "## Interpretation",
        "",
        "The overlap table reports the empirical distribution of the residualized treatment, which is relevant to whether the treatment remains informative after adjustment. It is not a formal proof of positivity. The station and fold tables are stability diagnostics rather than independent causal estimates. The current panel can still suffer from unmeasured spatial confounding, serial dependence, treatment measurement error, and exposure/outcome simultaneity. A stronger next design would use a clearly pre-treatment exposure window or a defensible quasi-experimental source of variation, together with dependence-aware inference.",
        "",
        "## Generated artifacts",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `robustness_summary.csv` | Original, clustered, and bootstrap uncertainty for all treatments |",
        "| `fold_stability.csv` | Station-held-out fold estimates |",
        "| `station_heterogeneity.csv` | Exploratory station-specific residualized slopes |",
        "| `overlap_falsification.csv` | Residualized-treatment support and within-station permutation nulls |",
        "| `learner_sensitivity.csv` | Random-forest nuisance learner sensitivity for the primary treatment |",
        "| `spatial_block_sensitivity.csv` | Geographic-block DML sensitivity summary |",
        "| `spatial_block_stability.csv` | Geographic block holdout fold composition |",
        "| `robustness_config.json` | Seeds, replicate counts, and input hashes |",
        "| `robustness_report.md` | Human-readable methods, results, and caveats |",
    ]
    (DML_DIR / "robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    train, _ = load_inputs()
    robust_rows, overlap_rows, fold_frames, station_frames = [], [], [], []
    for treatment in TREATMENTS:
        cf = pd.read_csv(DML_DIR / f"crossfit_observations_{treatment}.csv")
        cluster = cluster_se(cf, treatment)
        wild = wild_cluster_bootstrap(cf, treatment)
        fold, station = fold_and_station_stability(cf, treatment)
        overlap = overlap_and_permutation(cf, treatment)
        row = {"treatment": treatment, **cluster, **wild, **overlap}
        robust_rows.append(row)
        overlap_rows.append({"treatment": treatment, **overlap})
        fold_frames.append(fold)
        station_frames.append(station)
    baseline = pd.read_csv(DML_DIR / "dml_summary.csv")
    robust = pd.DataFrame(robust_rows).merge(baseline[["treatment", "theta", "se", "ci_low", "ci_high", "effect_per_treatment_sd"]], on="treatment", suffixes=("_robust", "_baseline"))
    fold = pd.concat(fold_frames, ignore_index=True)
    station = pd.concat(station_frames, ignore_index=True)
    controls = controls_from_manifest(PRIMARY)
    spatial, spatial_folds = spatial_block_sensitivity(train, PRIMARY, controls)
    learner = learner_sensitivity(train, PRIMARY, controls)
    robust.to_csv(DML_DIR / "robustness_summary.csv", index=False)
    fold.to_csv(DML_DIR / "fold_stability.csv", index=False)
    station.to_csv(DML_DIR / "station_heterogeneity.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(DML_DIR / "overlap_falsification.csv", index=False)
    pd.DataFrame([learner]).to_csv(DML_DIR / "learner_sensitivity.csv", index=False)
    spatial_folds.to_csv(DML_DIR / "spatial_block_stability.csv", index=False)
    pd.DataFrame([spatial]).to_csv(DML_DIR / "spatial_block_sensitivity.csv", index=False)
    config = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "wild_bootstrap_reps": N_WILD_BOOTSTRAPS,
        "permutation_reps": N_PERMUTATIONS,
        "grouping": GROUP,
        "input_sha256": {"train.csv": sha256(TRAIN_PATH), "test.csv": sha256(TEST_PATH)},
        "primary_treatment": PRIMARY,
        "cluster_count": int(train[GROUP].nunique()),
    }
    (DML_DIR / "robustness_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(robust, fold, station, pd.DataFrame(overlap_rows), learner, spatial)
    print(robust[["treatment", "theta_baseline", "se", "cluster_se", "cluster_ci_low", "cluster_ci_high", "wild_bootstrap_ci_low", "wild_bootstrap_ci_high"]].to_string(index=False))
    print("Robustness checks complete")


if __name__ == "__main__":
    main()
