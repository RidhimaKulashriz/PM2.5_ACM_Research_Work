"""
PHASE 0 — TEMPORAL CONTEXTUAL UPDATE
====================================

Research:
Urban Green Cover Thresholds for PM2.5 Mitigation:
A Spatial Causal Machine Learning Framework for Delhi NCR

Purpose
-------
Create an isolated V3 modelling branch in which the previously static
2021 contextual layers are replaced with genuinely time-aligned 2025
contextual layers wherever an authoritative 2025 product is available.

2025 LAND COVER:
    Google Dynamic World V1
    10 m
    2025 annual mode composite

2025 POPULATION:
    JRC GHSL P2023A GHS_POP
    100 m
    2025 population projection

2025 ROAD INFRASTRUCTURE:
    Existing validated feat_osm_roads.csv
    Q1-2025 OSM PBF snapshot

IMPORTANT
---------
V1 and V2 canonical datasets are READ ONLY.

This script will FAIL rather than fabricate, approximate, rename,
or silently substitute unavailable data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime

import ee
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

V2_PATH = ROOT / "data" / "ml_ready" / "master_modeling_dataset_v2.csv"

OSM_PATH = ROOT / "data" / "03_features" / "feat_osm_roads.csv"

OUT_ROOT = ROOT / "data" / "modeling_changes"
FEATURE_DIR = OUT_ROOT / "features"
VALIDATION_DIR = OUT_ROOT / "validation"
DATASET_DIR = OUT_ROOT / "datasets"
SPLIT_DIR = OUT_ROOT / "splits"
REPORT_DIR = OUT_ROOT / "reports"


# ============================================================
# CONSTANTS
# ============================================================

GEE_PROJECT = "delhi-pm25-research"

BUFFER_SCALES = [100, 250, 500, 1000]

YEARS = {2022, 2023, 2024, 2025}

TARGET = "pm25"

KEYS = ["station", "year", "month"]

EXPECTED_ROWS = 1615
EXPECTED_STATIONS = 35

TRAIN_ROWS = 1292
TEST_ROWS = 323

SEED = 42


# ============================================================
# GEE DATASETS
# ============================================================

DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"

GHSL_POPULATION = "JRC/GHSL/P2023A/GHS_POP"

DYNAMIC_WORLD_CLASSES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

for path in [
    OUT_ROOT,
    FEATURE_DIR,
    VALIDATION_DIR,
    DATASET_DIR,
    SPLIT_DIR,
    REPORT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def initialise_gee() -> None:
    print("Initialising Google Earth Engine...")

    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        print("Normal initialization failed; attempting authentication...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)

    print("Google Earth Engine initialized successfully.")


def assert_v2_integrity(df: pd.DataFrame) -> None:

    required = [
        "station",
        "year",
        "month",
        "latitude",
        "longitude",
        TARGET,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"V2 is missing required columns: {missing}"
        )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"V2 row count changed: expected {EXPECTED_ROWS}, got {len(df)}"
        )

    if df["station"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError(
            f"Station count changed: expected {EXPECTED_STATIONS}, "
            f"got {df['station'].nunique()}"
        )

    if df.duplicated(KEYS).any():
        raise RuntimeError(
            "V2 contains duplicate station-year-month keys."
        )

    if df[TARGET].isna().any():
        raise RuntimeError(
            "V2 contains missing PM2.5 target values."
        )

    if not set(df["year"].unique()).issubset(YEARS):
        raise RuntimeError(
            "V2 contains years outside 2022-2025."
        )

    coords = (
        df.groupby("station")[["latitude", "longitude"]]
        .nunique()
    )

    if (coords > 1).any().any():
        raise RuntimeError(
            "Station coordinates are not stable across observations."
        )


# ============================================================
# DYNAMIC WORLD 2025
# ============================================================

def build_dynamic_world_2025(stations: pd.DataFrame) -> pd.DataFrame:

    print("\n============================================================")
    print("2025 LAND COVER — GOOGLE DYNAMIC WORLD")
    print("============================================================")

    collection = (
        ee.ImageCollection(DYNAMIC_WORLD)
        .filterDate("2025-01-01", "2026-01-01")
    )

    image_count = collection.size().getInfo()

    print(f"Dynamic World 2025 images: {image_count}")

    if image_count == 0:
        raise RuntimeError(
            "NO Dynamic World images were found for 2025. "
            "Stopping rather than using fallback data."
        )

    # Annual modal land-cover class
    annual_label = (
        collection
        .select("label")
        .reduce(ee.Reducer.mode())
    )

    records = []

    for i, row in stations.iterrows():

        station = row["station"]
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        point = ee.Geometry.Point([lon, lat])

        record = {"station": station}

        print(
            f"Processing Dynamic World: "
            f"{station} ({i + 1}/{len(stations)})"
        )

        for radius in BUFFER_SCALES:

            buffer = point.buffer(radius)

            histogram = annual_label.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=buffer,
                scale=10,
                maxPixels=100_000_000,
                bestEffort=False,
            ).getInfo()

            histogram = histogram.get("label_mode")

            if histogram is None:
                raise RuntimeError(
                    f"No Dynamic World pixels returned for "
                    f"{station}, {radius} m."
                )

            histogram = {
                int(float(k)): int(v)
                for k, v in histogram.items()
            }

            total_pixels = sum(histogram.values())

            if total_pixels <= 0:
                raise RuntimeError(
                    f"Zero Dynamic World pixels for "
                    f"{station}, {radius} m."
                )

            for class_id, class_name in DYNAMIC_WORLD_CLASSES.items():

                fraction = (
                    histogram.get(class_id, 0)
                    / total_pixels
                )

                record[
                    f"dynamicworld_2025_{class_name}_frac_{radius}m"
                ] = float(fraction)

            record[
                f"dynamicworld_2025_valid_pixels_{radius}m"
            ] = int(total_pixels)

        records.append(record)

    result = pd.DataFrame(records)

    # Validate fractions
    fraction_cols = [
        c for c in result.columns
        if "_frac_" in c
    ]

    for col in fraction_cols:

        if result[col].isna().any():
            raise RuntimeError(
                f"Missing Dynamic World values in {col}"
            )

        if ((result[col] < 0) | (result[col] > 1)).any():
            raise RuntimeError(
                f"Invalid fraction in {col}"
            )

    result.to_csv(
        FEATURE_DIR / "feat_dynamicworld_2025.csv",
        index=False
    )

    validation = {
        "dataset": "Google Dynamic World V1",
        "year": 2025,
        "image_count": image_count,
        "resolution_m": 10,
        "stations": int(result["station"].nunique()),
        "rows": len(result),
        "missing_values": int(result.isna().sum().sum()),
        "minimum_fraction": float(result[fraction_cols].min().min()),
        "maximum_fraction": float(result[fraction_cols].max().max()),
        "status": "PASS",
    }

    pd.DataFrame([validation]).to_csv(
        VALIDATION_DIR / "dynamicworld_2025_validation.csv",
        index=False
    )

    return result


# ============================================================
# GHSL 2025 POPULATION
# ============================================================

def build_population_2025(stations: pd.DataFrame) -> pd.DataFrame:

    print("\n============================================================")
    print("2025 POPULATION — JRC GHSL P2023A")
    print("============================================================")

    population_image = ee.Image(
        "JRC/GHSL/P2023A/GHS_POP/2025"
    ).select("population_count")

    # Test that the image actually exists and is queryable.
    try:
        population_image.projection().getInfo()
    except Exception as exc:
        raise RuntimeError(
            "GHSL 2025 population image could not be accessed."
        ) from exc

    records = []

    for i, row in stations.iterrows():

        station = row["station"]
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        point = ee.Geometry.Point([lon, lat])

        record = {"station": station}

        print(
            f"Processing GHSL population: "
            f"{station} ({i + 1}/{len(stations)})"
        )

        for radius in [250, 500, 1000]:

            buffer = point.buffer(radius)

            result = population_image.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=buffer,
                scale=100,
                maxPixels=10_000_000,
                bestEffort=False,
            ).getInfo()

            total_population = result.get("population_count")

            if total_population is None:
                raise RuntimeError(
                    f"No GHSL population value returned for "
                    f"{station}, {radius} m."
                )

            area_km2 = np.pi * (radius / 1000) ** 2

            density = float(total_population) / area_km2

            if density < 0:
                raise RuntimeError(
                    f"Negative population density for {station}, {radius}m."
                )

            record[
                f"population_density_2025_{radius}m"
            ] = density

        records.append(record)

    result = pd.DataFrame(records)

    density_cols = [
        c for c in result.columns
        if c.startswith("population_density_2025_")
    ]

    if result[density_cols].isna().any().any():
        raise RuntimeError(
            "Missing GHSL 2025 population values."
        )

    if (result[density_cols] < 0).any().any():
        raise RuntimeError(
            "Negative GHSL population density detected."
        )

    result.to_csv(
        FEATURE_DIR / "feat_population_2025_ghsl.csv",
        index=False
    )

    validation = {
        "dataset": "JRC/GHSL/P2023A/GHS_POP",
        "year": 2025,
        "resolution_m": 100,
        "stations": int(result["station"].nunique()),
        "rows": len(result),
        "missing_values": int(result.isna().sum().sum()),
        "minimum_density": float(result[density_cols].min().min()),
        "maximum_density": float(result[density_cols].max().max()),
        "status": "PASS",
    }

    pd.DataFrame([validation]).to_csv(
        VALIDATION_DIR / "population_2025_validation.csv",
        index=False
    )

    return result


# ============================================================
# OSM 2025
# ============================================================

def load_osm_2025() -> pd.DataFrame:

    print("\n============================================================")
    print("2025 OSM ROAD INFRASTRUCTURE")
    print("============================================================")

    if not OSM_PATH.exists():
        raise RuntimeError(
            f"Existing OSM feature file not found: {OSM_PATH}"
        )

    osm = pd.read_csv(OSM_PATH)

    required = [
        "station",
        "year",
        "month",
        "road_density_100m",
        "road_density_250m",
        "road_density_500m",
        "road_density_1000m",
        "major_road_density_1000m",
    ]

    missing = [c for c in required if c not in osm.columns]

    if missing:
        raise RuntimeError(
            f"OSM feature file missing columns: {missing}"
        )

    # Keep one station-level record.
    station_cols = [
        "station",
        "road_density_100m",
        "road_density_250m",
        "road_density_500m",
        "road_density_1000m",
        "major_road_density_1000m",
    ]

    station_osm = (
        osm[station_cols]
        .drop_duplicates()
        .copy()
    )

    if station_osm["station"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError(
            "OSM station count does not match V2."
        )

    if station_osm.duplicated("station").any():
        raise RuntimeError(
            "OSM file contains multiple static records per station."
        )

    density_cols = [
        "road_density_100m",
        "road_density_250m",
        "road_density_500m",
        "road_density_1000m",
        "major_road_density_1000m",
    ]

    if station_osm[density_cols].isna().any().any():
        raise RuntimeError(
            "Missing OSM feature values."
        )

    if (station_osm[density_cols] < 0).any().any():
        raise RuntimeError(
            "Negative OSM density detected."
        )

    invalid_major = (
        station_osm["major_road_density_1000m"]
        > station_osm["road_density_1000m"] + 1e-9
    )

    if invalid_major.any():
        raise RuntimeError(
            "Major-road density exceeds total road density."
        )

    station_osm.to_csv(
        FEATURE_DIR / "feat_osm_roads_2025.csv",
        index=False
    )

    validation = {
        "dataset": "Existing validated OSM feature table",
        "snapshot": "Q1-2025 PBF according to project provenance",
        "stations": int(station_osm["station"].nunique()),
        "rows": len(station_osm),
        "missing_values": int(station_osm.isna().sum().sum()),
        "status": "PASS",
    }

    pd.DataFrame([validation]).to_csv(
        VALIDATION_DIR / "osm_2025_validation.csv",
        index=False
    )

    return station_osm


# ============================================================
# BUILD V3
# ============================================================

def build_v3(
    v2: pd.DataFrame,
    dynamicworld: pd.DataFrame,
    population: pd.DataFrame,
    osm: pd.DataFrame,
) -> pd.DataFrame:

    print("\n============================================================")
    print("ASSEMBLING MASTER MODELING DATASET V3")
    print("============================================================")

    v2_core = v2.copy()

    # Remove the old 2021 contextual layers.
    old_columns = [
        c for c in v2_core.columns
        if c.startswith("worldcover_2021_")
        or c.startswith("worldpop_density_")
        or c.startswith("worldpop_")
        or c.startswith("road_density_")
        or c == "major_road_density_1000m"
    ]

    print("Removing old contextual columns:")
    for col in old_columns:
        print(f"  - {col}")

    v2_core = v2_core.drop(
        columns=old_columns,
        errors="ignore"
    )

    # Station-level joins.
    v3 = (
        v2_core
        .merge(dynamicworld, on="station", how="left", validate="many_to_one")
        .merge(population, on="station", how="left", validate="many_to_one")
        .merge(osm, on="station", how="left", validate="many_to_one")
    )

    # Preserve exact key universe.
    if len(v3) != len(v2):
        raise RuntimeError(
            "V3 row count changed during feature integration."
        )

    if not np.array_equal(
        v3[KEYS].astype(str).to_numpy(),
        v2[KEYS].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "V3 station/year/month universe changed."
        )

    if not np.allclose(
        v3[TARGET].to_numpy(),
        v2[TARGET].to_numpy(),
        equal_nan=True,
    ):
        raise RuntimeError(
            "PM2.5 target changed during V3 construction."
        )

    if v3.duplicated(KEYS).any():
        raise RuntimeError(
            "Duplicate station-year-month keys in V3."
        )

    # Ensure no accidental 2021 contextual columns remain.
    forbidden = [
        c for c in v3.columns
        if "worldcover_2021" in c.lower()
        or "worldpop_" in c.lower()
    ]

    if forbidden:
        raise RuntimeError(
            f"Old 2021 contextual columns remain in V3: {forbidden}"
        )

    # Ensure no obvious missing values in newly created contextual columns.
    new_contextual = (
        [
            c for c in dynamicworld.columns
            if c != "station"
        ]
        +
        [
            c for c in population.columns
            if c != "station"
        ]
        +
        [
            c for c in osm.columns
            if c != "station"
        ]
    )

    missing_new = v3[new_contextual].isna().sum()

    if missing_new.sum() > 0:
        raise RuntimeError(
            "Missing values detected in new contextual features:\n"
            f"{missing_new[missing_new > 0]}"
        )

    output_path = DATASET_DIR / "master_modeling_dataset_v3.csv"

    v3.to_csv(output_path, index=False)

    audit = {
        "v2_rows": len(v2),
        "v3_rows": len(v3),
        "v2_columns": len(v2.columns),
        "v3_columns": len(v3.columns),
        "v2_stations": int(v2.station.nunique()),
        "v3_stations": int(v3.station.nunique()),
        "removed_2021_contextual_columns": old_columns,
        "new_contextual_columns": new_contextual,
        "pm25_unchanged": True,
        "key_universe_unchanged": True,
        "missing_new_features": int(missing_new.sum()),
        "status": "PASS",
    }

    with open(
        REPORT_DIR / "v3_construction_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit, f, indent=2)

    print(f"\nV3 saved to:\n{output_path}")

    return v3


# ============================================================
# SPLIT
# ============================================================

def largest_remainder(counts: pd.Series, total: int) -> pd.Series:

    raw = counts.astype(float) * total / counts.sum()

    allocation = np.floor(raw).astype(int)

    remaining = int(total - allocation.sum())

    remainders = raw - allocation

    order = sorted(
        counts.index,
        key=lambda x: (-remainders.loc[x], str(x))
    )

    for key in order[:remaining]:
        allocation.loc[key] += 1

    return allocation


def split_v3(df: pd.DataFrame):

    print("\n============================================================")
    print("GENERATING YEAR × MONTH STRATIFIED SPLIT")
    print("============================================================")

    rng = np.random.default_rng(SEED)

    singleton = df["station"].eq("IIT_Delhi")

    train_singleton = df[singleton].copy()
    pool = df[~singleton].copy()

    quotas_by_ym = largest_remainder(
        pool.groupby(["year", "month"]).size(),
        TEST_ROWS,
    )

    selected = []

    for (year, month), quota in quotas_by_ym.items():

        subset = pool[
            (pool["year"] == year)
            & (pool["month"] == month)
        ]

        if quota > len(subset):
            raise RuntimeError(
                "Requested test quota exceeds available observations."
            )

        selected.extend(
            rng.choice(
                subset.index.to_numpy(),
                size=int(quota),
                replace=False,
            ).tolist()
        )

    test = df.loc[selected].copy()

    train = df.drop(index=selected).copy()

    # IIT Delhi must remain in training.
    if "IIT_Delhi" not in set(train.station):
        raise RuntimeError(
            "IIT_Delhi was not retained in training."
        )

    if "IIT_Delhi" in set(test.station):
        raise RuntimeError(
            "IIT_Delhi appeared in test."
        )

    # Key-based validation.
    train_keys = set(
        map(tuple, train[KEYS].astype(str).to_numpy())
    )

    test_keys = set(
        map(tuple, test[KEYS].astype(str).to_numpy())
    )

    master_keys = set(
        map(tuple, df[KEYS].astype(str).to_numpy())
    )

    overlap = train_keys & test_keys
    union = train_keys | test_keys

    if overlap:
        raise RuntimeError(
            f"Train/test key overlap detected: {len(overlap)}"
        )

    if union != master_keys:
        raise RuntimeError(
            "Train/test key union does not equal V3."
        )

    if len(train) != TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {TRAIN_ROWS} train rows, got {len(train)}"
        )

    if len(test) != TEST_ROWS:
        raise RuntimeError(
            f"Expected {TEST_ROWS} test rows, got {len(test)}"
        )

    if set(train.year) != YEARS:
        raise RuntimeError(
            "Train does not contain all study years."
        )

    if set(test.year) != YEARS:
        raise RuntimeError(
            "Test does not contain all study years."
        )

    # Every non-singleton year-month must appear in test.
    eligible_ym = (
        pool.groupby(["year", "month"])
        .size()
        .index
    )

    test_ym = set(
        test.set_index(["year", "month"]).index
    )

    missing_ym = [
        ym for ym in eligible_ym
        if ym not in test_ym
    ]

    if missing_ym:
        raise RuntimeError(
            f"Missing year-month strata in test: {missing_ym}"
        )

    train_out = train.sort_values(KEYS).reset_index(drop=True)
    test_out = test.sort_values(KEYS).reset_index(drop=True)

    train_out.to_csv(
        SPLIT_DIR / "train.csv",
        index=False
    )

    test_out.to_csv(
        SPLIT_DIR / "test.csv",
        index=False
    )

    diagnostics = pd.concat(
        [
            train_out.assign(split="train"),
            test_out.assign(split="test"),
        ],
        ignore_index=True,
    )

    diagnostics["season"] = diagnostics["month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Summer",
            4: "Summer",
            5: "Summer",
            6: "Summer",
            7: "Monsoon",
            8: "Monsoon",
            9: "Monsoon",
            10: "Post-monsoon",
            11: "Post-monsoon",
        }
    )

    distribution_records = []

    for dimension in [
        "year",
        "month",
        "season",
        "station",
    ]:

        totals = diagnostics[dimension].value_counts()

        train_counts = (
            diagnostics[diagnostics.split == "train"]
            [dimension]
            .value_counts()
        )

        test_counts = (
            diagnostics[diagnostics.split == "test"]
            [dimension]
            .value_counts()
        )

        for group in totals.index:

            total = int(totals[group])
            tr = int(train_counts.get(group, 0))
            te = int(test_counts.get(group, 0))

            distribution_records.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "total": total,
                    "train": tr,
                    "test": te,
                    "train_fraction": tr / total,
                    "test_fraction": te / total,
                }
            )

    pd.DataFrame(distribution_records).to_csv(
        SPLIT_DIR / "distribution_diagnostics.csv",
        index=False
    )

    manifest = {
        "method": "Year x Month stratified sampling",
        "seed": SEED,
        "master_rows": len(df),
        "train_rows": len(train_out),
        "test_rows": len(test_out),
        "train_fraction": len(train_out) / len(df),
        "test_fraction": len(test_out) / len(df),
        "train_stations": int(train.station.nunique()),
        "test_stations": int(test.station.nunique()),
        "train_years": sorted(map(int, train.year.unique())),
        "test_years": sorted(map(int, test.year.unique())),
        "IIT_Delhi_train_only": True,
        "key_overlap": len(overlap),
        "union_equals_master": True,
        "target_used_for_selection": False,
        "status": "PASS",
    }

    with open(
        SPLIT_DIR / "split_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(manifest, f, indent=2)

    pd.DataFrame(
        [
            {
                "check": "master_rows",
                "status": "PASS",
                "value": len(df),
            },
            {
                "check": "train_rows",
                "status": "PASS",
                "value": len(train_out),
            },
            {
                "check": "test_rows",
                "status": "PASS",
                "value": len(test_out),
            },
            {
                "check": "key_overlap",
                "status": "PASS",
                "value": len(overlap),
            },
            {
                "check": "union_equals_master",
                "status": "PASS",
                "value": True,
            },
            {
                "check": "IIT_Delhi_train_only",
                "status": "PASS",
                "value": True,
            },
            {
                "check": "all_years_in_train",
                "status": "PASS",
                "value": True,
            },
            {
                "check": "all_years_in_test",
                "status": "PASS",
                "value": True,
            },
            {
                "check": "all_year_month_strata_in_test",
                "status": "PASS",
                "value": True,
            },
        ]
    ).to_csv(
        SPLIT_DIR / "validation_report.csv",
        index=False
    )

    return train_out, test_out


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 0 — CLEAN 2025 CONTEXTUAL UPDATE")
    print("=" * 70)

    # ---------------------------
    # Load V2
    # ---------------------------

    if not V2_PATH.exists():

        raise FileNotFoundError(
            f"Canonical V2 not found: {V2_PATH}"
        )

    v2 = pd.read_csv(V2_PATH)

    print(
        f"Loaded V2: {len(v2)} rows, "
        f"{len(v2.columns)} columns."
    )

    assert_v2_integrity(v2)

    # ---------------------------
    # GEE
    # ---------------------------

    initialise_gee()

    # ---------------------------
    # Station table
    # ---------------------------

    stations = (
        v2[
            [
                "station",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates()
        .sort_values("station")
        .reset_index(drop=True)
    )

    if len(stations) != EXPECTED_STATIONS:
        raise RuntimeError(
            "Unexpected station count."
        )

    # ---------------------------
    # Actual 2025 extraction
    # ---------------------------

    dynamicworld = build_dynamic_world_2025(
        stations
    )

    population = build_population_2025(
        stations
    )

    osm = load_osm_2025()

    # ---------------------------
    # Build V3
    # ---------------------------

    v3 = build_v3(
        v2,
        dynamicworld,
        population,
        osm,
    )

    # ---------------------------
    # Split
    # ---------------------------

    train, test = split_v3(v3)

    # ---------------------------
    # Final summary
    # ---------------------------

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE — ALL DATA ARE ACTUAL SOURCED VALUES")
    print("=" * 70)

    print(f"V2 rows: {len(v2)}")
    print(f"V3 rows: {len(v3)}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    print("\n2025 contextual layers:")
    print("✓ Dynamic World 2025 — genuine GEE extraction")
    print("✓ GHSL 2025 population — genuine GEE extraction")
    print("✓ OSM 2025 — existing validated project feature table")

    print("\n2021 contextual layers removed:")
    print("✓ ESA WorldCover 2021 removed")
    print("✓ WorldPop 2021-era features removed")

    print("\nCanonical datasets modified:")
    print("NO")

    print("\nOutput:")
    print(
        DATASET_DIR / "master_modeling_dataset_v3.csv"
    )
    print(
        SPLIT_DIR / "train.csv"
    )
    print(
        SPLIT_DIR / "test.csv"
    )

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()