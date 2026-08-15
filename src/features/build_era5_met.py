#!/usr/bin/env python3
"""
ERA5 / ERA5-Land Monthly Meteorological Feature Pipeline
==========================================================

Project:
    Urban Green Cover Thresholds for PM2.5 Mitigation (Delhi NCR)

Current scope:
    Integrate ONLY ERA5 / ERA5-Land into the existing
    master_modelling_dataset.csv.

Important:
    - Existing 35-station ML-ready master is treated as immutable.
    - No stations are added or removed.
    - No other datasets are integrated in this step.
    - No 100/250/500/1000 m buffers are used for ERA5.
    - ERA5-Land and ERA5 are processed at their native spatial scales.
    - Monthly aggregation is performed server-side in GEE.
    - Only ~35 station records per month are transferred to Python.

Existing master:
    data/ml_ready/master_modeling_dataset.csv

Output:
    data/03_features/feat_era5_met.csv

Validation:
    data/05_validation/era5_validation_summary.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_DATASET_PATH = Path(
    "data/ml_ready/master_modeling_dataset.csv"
)

OUTPUT_FEATURE_PATH = Path(
    "data/03_features/feat_era5_met.csv"
)

VALIDATION_PATH = Path(
    "data/05_validation/era5_validation_summary.csv"
)

GEE_PROJECT = "delhi-pm25-research"

START_YEAR = 2022
END_YEAR = 2025

# Require at least 80% of expected hourly observations
# for a station-month to be considered valid.
MIN_COMPLETENESS = 0.80

# Native approximate spatial scales.
ERA5_LAND_SCALE = 11132   # ~0.1 degree / ~11 km
ERA5_SCALE = 27830       # ~0.25 degree / ~28 km


# =============================================================================
# DATASETS
# =============================================================================

ERA5_LAND_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_COLLECTION = "ECMWF/ERA5/HOURLY"

ERA5_LAND_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

ERA5_BANDS = [
    "boundary_layer_height"
]


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("ERA5_Pipeline")


# =============================================================================
# EARTH ENGINE INITIALIZATION
# =============================================================================

def initialize_gee() -> None:
    """Initialize Earth Engine using the explicit Cloud project."""

    logger.info(
        "Initializing Google Earth Engine using project: %s",
        GEE_PROJECT,
    )

    try:
        ee.Initialize(
            project=GEE_PROJECT
        )

        # Small connectivity check.
        ee.Number(1).getInfo()

        logger.info(
            "Google Earth Engine initialized successfully."
        )

    except Exception as exc:
        logger.warning(
            "Initial GEE initialization failed: %s",
            exc,
        )

        logger.info(
            "Starting Earth Engine authentication..."
        )

        ee.Authenticate()

        ee.Initialize(
            project=GEE_PROJECT
        )

        ee.Number(1).getInfo()

        logger.info(
            "Google Earth Engine authenticated and initialized."
        )


# =============================================================================
# MASTER DATASET
# =============================================================================

def load_master_dataset() -> pd.DataFrame:
    """Load and validate the existing immutable master dataset."""

    if not MASTER_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n"
            f"{MASTER_DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(
        MASTER_DATASET_PATH,
        low_memory=False,
    )

    required = {
        "station",
        "year",
        "month",
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Check station coordinates.
    coordinate_counts = (
        df.groupby("station")[
            ["latitude", "longitude"]
        ]
        .nunique()
    )

    unstable = coordinate_counts[
        (coordinate_counts["latitude"] > 1)
        | (coordinate_counts["longitude"] > 1)
    ]

    if not unstable.empty:
        raise ValueError(
            "Station coordinates are not stable for:\n"
            + unstable.to_string()
        )

    # Check station-month uniqueness.
    duplicate_count = df.duplicated(
        subset=["station", "year", "month"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Master dataset contains {duplicate_count} "
            "duplicate station-year-month rows."
        )

    logger.info(
        "Loaded %d rows.",
        len(df),
    )

    logger.info(
        "Loaded %d unique stations.",
        df["station"].nunique(),
    )

    logger.info(
        "Loaded %d station-month observations.",
        len(df),
    )

    return df


# =============================================================================
# STATION GEOMETRIES
# =============================================================================

def build_station_collection(
    master_df: pd.DataFrame,
) -> ee.FeatureCollection:
    """Create an Earth Engine FeatureCollection from master station coordinates."""

    station_df = (
        master_df[
            [
                "station",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates(subset=["station"])
        .reset_index(drop=True)
    )

    features = []

    for _, row in station_df.iterrows():

        geometry = ee.Geometry.Point(
            [
                float(row["longitude"]),
                float(row["latitude"]),
            ]
        )

        features.append(
            ee.Feature(
                geometry,
                {
                    "station": str(
                        row["station"]
                    )
                },
            )
        )

    return ee.FeatureCollection(features)


# =============================================================================
# ERA5-LAND HOURLY TRANSFORMATION
# =============================================================================

def transform_era5_land_hourly(
    image: ee.Image,
) -> ee.Image:
    """
    Calculate the physical variables at hourly resolution.

    Important:
        RH and wind speed are calculated BEFORE monthly aggregation.
    """

    temperature_c = (
        image
        .select("temperature_2m")
        .subtract(273.15)
        .rename("temperature_c")
    )

    dewpoint_c = (
        image
        .select("dewpoint_temperature_2m")
        .subtract(273.15)
        .rename("dewpoint_c")
    )

    # August-Roche-Magnus formulation.
    a = 17.625
    b = 243.04

    saturation_term = (
        temperature_c
        .multiply(a)
        .divide(
            temperature_c.add(b)
        )
        .exp()
    )

    actual_term = (
        dewpoint_c
        .multiply(a)
        .divide(
            dewpoint_c.add(b)
        )
        .exp()
    )

    relative_humidity = (
        actual_term
        .divide(saturation_term)
        .multiply(100.0)
        .clamp(0, 100)
        .rename("relative_humidity")
    )

    u10 = image.select(
        "u_component_of_wind_10m"
    )

    v10 = image.select(
        "v_component_of_wind_10m"
    )

    wind_speed = (
        u10.pow(2)
        .add(v10.pow(2))
        .sqrt()
        .rename("wind_speed")
    )

    return ee.Image.cat(
        temperature_c,
        relative_humidity,
        wind_speed,
    )


# =============================================================================
# BUILD ONE MONTH
# =============================================================================

def build_monthly_features(
    year: int,
    month: int,
    station_fc: ee.FeatureCollection,
) -> ee.FeatureCollection:
    """
    Compute one station-month table on the GEE server.

    This is deliberately monthly rather than yearly so that no large
    FeatureCollection needs to be materialized at once.
    """

    start = ee.Date.fromYMD(
        year,
        month,
        1,
    )

    end = start.advance(
        1,
        "month",
    )

    # -------------------------------------------------------------------------
    # ERA5-Land
    # -------------------------------------------------------------------------

    land_hourly = (
        ee.ImageCollection(
            ERA5_LAND_COLLECTION
        )
        .filterDate(start, end)
        .select(ERA5_LAND_BANDS)
        .map(transform_era5_land_hourly)
    )

    land_monthly = (
        land_hourly
        .mean()
        .resample("bilinear")
    )

    land_counts = (
        land_hourly
        .count()
        .rename(
            [
                "valid_temp_hours",
                "valid_rh_hours",
                "valid_wind_hours",
            ]
        )
    )

    land_image = ee.Image.cat(
        land_monthly,
        land_counts,
    )

    # -------------------------------------------------------------------------
    # ERA5 BLH
    # -------------------------------------------------------------------------

    era5_hourly = (
        ee.ImageCollection(
            ERA5_COLLECTION
        )
        .filterDate(start, end)
        .select(ERA5_BANDS)
    )

    era5_monthly = (
        era5_hourly
        .mean()
        .resample("bilinear")
        .rename("boundary_layer_height")
    )

    era5_counts = (
        era5_hourly
        .count()
        .rename("valid_blh_hours")
    )

    era5_image = ee.Image.cat(
        era5_monthly,
        era5_counts,
    )

    # -------------------------------------------------------------------------
    # Extract ERA5-Land at its native approximate scale.
    # -------------------------------------------------------------------------

    land_sample = (
        land_image
        .reduceRegions(
            collection=station_fc,
            reducer=ee.Reducer.first(),
            scale=ERA5_LAND_SCALE,
            tileScale=4,
        )
        .map(
            lambda feature: feature.set(
                {
                    "year": year,
                    "month": month,
                }
            )
        )
    )

    # -------------------------------------------------------------------------
    # Extract ERA5 BLH at its native approximate scale.
    # -------------------------------------------------------------------------

    era5_sample = (
        era5_image
        .reduceRegions(
            collection=station_fc,
            reducer=ee.Reducer.first(),
            scale=ERA5_SCALE,
            tileScale=4,
        )
        .map(
            lambda feature: feature.set(
                {
                    "year": year,
                    "month": month,
                }
            )
        )
    )

    # -------------------------------------------------------------------------
    # Join the two monthly station collections.
    # -------------------------------------------------------------------------

    join_filter = ee.Filter.And(
        ee.Filter.equals(
            leftField="station",
            rightField="station",
        ),
        ee.Filter.equals(
            leftField="year",
            rightField="year",
        ),
        ee.Filter.equals(
            leftField="month",
            rightField="month",
        ),
    )

    joined = ee.Join.inner().apply(
        land_sample,
        era5_sample,
        join_filter,
    )

    def merge_pair(pair: ee.Feature) -> ee.Feature:

        land_feature = ee.Feature(
            pair.get("primary")
        )

        era5_feature = ee.Feature(
            pair.get("secondary")
        )

        return (
            land_feature
            .set(
                "era5_blh_mean",
                era5_feature.get(
                    "boundary_layer_height"
                ),
            )
            .set(
                "valid_blh_hours",
                era5_feature.get(
                    "valid_blh_hours"
                ),
            )
        )

    return ee.FeatureCollection(
        joined
    ).map(merge_pair)


# =============================================================================
# FETCH ONE MONTH
# =============================================================================

def fetch_month(
    year: int,
    month: int,
    station_fc: ee.FeatureCollection,
) -> pd.DataFrame:
    """Fetch one monthly station table from GEE."""

    monthly_fc = build_monthly_features(
        year,
        month,
        station_fc,
    )

    df = ee.data.computeFeatures(
        {
            "expression": monthly_fc,
            "fileFormat": "PANDAS_DATAFRAME",
        }
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Geometry isn't needed locally.
    if ".geo" in df.columns:
        df = df.drop(
            columns=[".geo"]
        )

    return df


# =============================================================================
# FETCH ALL YEARS
# =============================================================================

def fetch_all_months(
    station_fc: ee.FeatureCollection,
) -> pd.DataFrame:
    """
    Process 2022-2025 month by month.

    Only the small monthly station table is transferred to the laptop.
    """

    frames = []

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        logger.info(
            "=============================================="
        )

        logger.info(
            "Processing year %d",
            year,
        )

        logger.info(
            "=============================================="
        )

        for month in range(
            1,
            13,
        ):

            logger.info(
                "Extracting %04d-%02d ...",
                year,
                month,
            )

            monthly_df = fetch_month(
                year,
                month,
                station_fc,
            )

            if monthly_df.empty:

                logger.warning(
                    "No data returned for %04d-%02d.",
                    year,
                    month,
                )

                continue

            frames.append(
                monthly_df
            )

            logger.info(
                "Retrieved %d station records.",
                len(monthly_df),
            )

    if not frames:
        raise RuntimeError(
            "No ERA5 data were returned from Earth Engine."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =============================================================================
# LOCAL QUALITY CONTROL
# =============================================================================

def clean_and_validate(
    df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert units, apply completeness checks and create one compact
    validation summary.
    """

    df = df.copy()

    # -------------------------------------------------------------------------
    # Rename / convert temperature.
    # -------------------------------------------------------------------------

    df["era5_temp_mean"] = (
        pd.to_numeric(
            df["temperature_c"],
            errors="coerce",
        )
    )

    df["era5_rh_mean"] = (
        pd.to_numeric(
            df["relative_humidity"],
            errors="coerce",
        )
    )

    df["era5_wind_speed_mean"] = (
        pd.to_numeric(
            df["wind_speed"],
            errors="coerce",
        )
    )

    df["era5_blh_mean"] = (
        pd.to_numeric(
            df["era5_blh_mean"],
            errors="coerce",
        )
    )

    # -------------------------------------------------------------------------
    # Expected hourly counts.
    # -------------------------------------------------------------------------

    dates = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-01"
    )

    df["expected_hours"] = (
        dates.dt.days_in_month
        * 24
    )

    # -------------------------------------------------------------------------
    # Numeric valid-hour counts.
    # -------------------------------------------------------------------------

    count_columns = [
        "valid_temp_hours",
        "valid_rh_hours",
        "valid_wind_hours",
        "valid_blh_hours",
    ]

    for col in count_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Completeness ratios.
    # -------------------------------------------------------------------------

    df["temp_completeness"] = (
        df["valid_temp_hours"]
        / df["expected_hours"]
    )

    df["rh_completeness"] = (
        df["valid_rh_hours"]
        / df["expected_hours"]
    )

    df["wind_completeness"] = (
        df["valid_wind_hours"]
        / df["expected_hours"]
    )

    df["blh_completeness"] = (
        df["valid_blh_hours"]
        / df["expected_hours"]
    )

    # -------------------------------------------------------------------------
    # Variable-specific invalidation.
    # -------------------------------------------------------------------------

    variables_and_completeness = {
        "era5_temp_mean": "temp_completeness",
        "era5_rh_mean": "rh_completeness",
        "era5_wind_speed_mean": "wind_completeness",
        "era5_blh_mean": "blh_completeness",
    }

    for variable, completeness_column in (
        variables_and_completeness.items()
    ):

        invalid = (
            df[completeness_column]
            < MIN_COMPLETENESS
        )

        df.loc[
            invalid,
            variable
        ] = np.nan

    # -------------------------------------------------------------------------
    # Physical sanity checks.
    # -------------------------------------------------------------------------

    physical_ranges = {
        "era5_temp_mean": (
            -10.0,
            60.0,
        ),
        "era5_rh_mean": (
            0.0,
            100.0,
        ),
        "era5_wind_speed_mean": (
            0.0,
            60.0,
        ),
        "era5_blh_mean": (
            0.0,
            10000.0,
        ),
    }

    physical_violations = {}

    for variable, (
        lower,
        upper,
    ) in physical_ranges.items():

        values = df[
            variable
        ].dropna()

        violations = (
            (values < lower)
            | (values > upper)
        )

        physical_violations[
            variable
        ] = int(
            violations.sum()
        )

        # Do not silently change outliers here.
        # They are reported in validation only.

    # -------------------------------------------------------------------------
    # Check station-month uniqueness.
    # -------------------------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "station",
            "year",
            "month",
        ]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"ERA5 feature extraction produced "
            f"{duplicate_count} duplicate station-month rows."
        )

    # -------------------------------------------------------------------------
    # Align with master.
    # -------------------------------------------------------------------------

    features = df[
        [
            "station",
            "year",
            "month",
            "era5_temp_mean",
            "era5_rh_mean",
            "era5_wind_speed_mean",
            "era5_blh_mean",
        ]
    ].copy()

    merged = master_df.merge(
        features,
        on=[
            "station",
            "year",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(
        master_df
    ):
        raise ValueError(
            "CRITICAL: ERA5 merge changed the "
            "number of master rows."
        )

    # -------------------------------------------------------------------------
    # Compact validation summary.
    # -------------------------------------------------------------------------

    validation_rows = []

    for variable in [
        "era5_temp_mean",
        "era5_rh_mean",
        "era5_wind_speed_mean",
        "era5_blh_mean",
    ]:

        values = (
            merged[variable]
            .dropna()
        )

        if values.empty:
            actual_min = np.nan
            actual_max = np.nan
        else:
            actual_min = values.min()
            actual_max = values.max()

        validation_rows.append(
            {
                "variable": variable,
                "rows_in_master": len(merged),
                "valid_values": int(
                    values.count()
                ),
                "missing_values": int(
                    merged[variable].isna().sum()
                ),
                "missing_percent": (
                    merged[variable].isna().mean()
                    * 100
                ),
                "actual_min": actual_min,
                "actual_max": actual_max,
                "physical_violations": (
                    physical_violations
                    .get(
                        variable,
                        0,
                    )
                ),
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    # Add overall metadata to the same single validation file.
    metadata_rows = pd.DataFrame(
        [
            {
                "variable": "__PIPELINE__",
                "rows_in_master": len(
                    master_df
                ),
                "valid_values": (
                    merged[
                        "era5_temp_mean"
                    ]
                    .notna()
                    .sum()
                ),
                "missing_values": (
                    merged[
                        "era5_temp_mean"
                    ]
                    .isna()
                    .sum()
                ),
                "missing_percent": (
                    merged[
                        "era5_temp_mean"
                    ]
                    .isna()
                    .mean()
                    * 100
                ),
                "actual_min": np.nan,
                "actual_max": np.nan,
                "physical_violations": (
                    0
                ),
            }
        ]
    )

    validation = pd.concat(
        [
            metadata_rows,
            validation,
        ],
        ignore_index=True,
    )

    return (
        merged,
        validation,
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

def save_feature_table(
    merged_df: pd.DataFrame,
) -> None:
    """
    Save only the ERA5 feature table.
    The original master is never overwritten.
    """

    final_columns = [
        "station",
        "year",
        "month",
        "era5_temp_mean",
        "era5_rh_mean",
        "era5_wind_speed_mean",
        "era5_blh_mean",
    ]

    features = merged_df[
        final_columns
    ].copy()

    # Final key check.
    if features.duplicated(
        subset=[
            "station",
            "year",
            "month",
        ]
    ).any():

        raise ValueError(
            "Final ERA5 feature table contains duplicate keys."
        )

    features.to_csv(
        OUTPUT_FEATURE_PATH,
        index=False,
    )

    logger.info(
        "Saved ERA5 features to: %s",
        OUTPUT_FEATURE_PATH.resolve(),
    )

    logger.info(
        "Final feature rows: %d",
        len(features),
    )

    logger.info(
        "Final stations: %d",
        features["station"].nunique(),
    )


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline() -> None:

    logger.info(
        "=" * 70
    )

    logger.info(
        "ERA5 / ERA5-LAND PIPELINE START"
    )

    logger.info(
        "=" * 70
    )

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------

    OUTPUT_FEATURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # GEE
    # -------------------------------------------------------------------------

    initialize_gee()

    # -------------------------------------------------------------------------
    # Master
    # -------------------------------------------------------------------------

    master_df = load_master_dataset()

    # Explicitly report the station population.
    logger.info(
        "Using exactly the stations already present "
        "in the current ML master."
    )

    logger.info(
        "Station count: %d",
        master_df["station"].nunique(),
    )

    # -------------------------------------------------------------------------
    # Station collection
    # -------------------------------------------------------------------------

    station_fc = build_station_collection(
        master_df
    )

    # -------------------------------------------------------------------------
    # GEE extraction
    # -------------------------------------------------------------------------

    raw_features = fetch_all_months(
        station_fc
    )

    logger.info(
        "Total monthly station records retrieved: %d",
        len(raw_features),
    )

    # -------------------------------------------------------------------------
    # Local processing + validation
    # -------------------------------------------------------------------------

    merged_df, validation_df = (
        clean_and_validate(
            raw_features,
            master_df,
        )
    )

    # -------------------------------------------------------------------------
    # Validation output
    # -------------------------------------------------------------------------

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    logger.info(
        "Saved validation summary to: %s",
        VALIDATION_PATH.resolve(),
    )

    # -------------------------------------------------------------------------
    # Save feature table
    # -------------------------------------------------------------------------

    save_feature_table(
        merged_df
    )

    # -------------------------------------------------------------------------
    # Final checks
    # -------------------------------------------------------------------------

    assert (
        len(merged_df)
        == len(master_df)
    ), (
        "Master row count changed."
    )

    assert (
        merged_df[
            "station"
        ].nunique()
        == master_df[
            "station"
        ].nunique()
    ), (
        "Station population changed."
    )

    assert (
        merged_df[
            [
                "station",
                "year",
                "month",
            ]
        ]
        .equals(
            master_df[
                [
                    "station",
                    "year",
                    "month",
                ]
            ]
        )
    ), (
        "Station-month keys changed."
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "ERA5 / ERA5-LAND PIPELINE COMPLETED"
    )

    logger.info(
        "Master dataset was NOT modified."
    )

    logger.info(
        "Stations retained: %d",
        master_df["station"].nunique(),
    )

    logger.info(
        "Rows retained: %d",
        len(master_df),
    )

    logger.info(
        "=" * 70
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:
        run_pipeline()

    except KeyboardInterrupt:

        logger.warning(
            "Pipeline interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        logger.exception(
            "ERA5 pipeline failed: %s",
            exc,
        )

        sys.exit(1)