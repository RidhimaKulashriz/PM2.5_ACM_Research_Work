#!/usr/bin/env python3
"""
Read-only audit of CPCB stations missing from the ML-ready dataset.

Project:
    Urban Green Cover Thresholds for PM2.5 Mitigation:
    A Spatial Causal Machine Learning Framework for Delhi NCR

Purpose
-------
Trace why CPCB stations present in the CPCB master are absent from the
current ML-ready station-month dataset.

This script DOES NOT:
    - modify any data
    - modify any scripts
    - rerun the satellite pipeline
    - create a new modelling dataset
    - overwrite existing files

It DOES:
    1. Audit CPCB master data.
    2. Audit the ML-ready dataset.
    3. Identify missing stations.
    4. Quantify CPCB hourly/monthly coverage.
    5. Search repository scripts for the final matching pipeline.
    6. Search intermediate data products for station presence.
    7. Identify likely exclusion mechanisms.
    8. Produce diagnostic CSV and TXT reports.

Expected files
--------------
data/processed/master/cpcb_pm25_master.csv
data/ml_ready/master_modeling_dataset.csv

Optional intermediate locations
--------------------------------
data/processed/
data/intermediate/
data/features/
data/ml_ready/
src/
scripts/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CPCB_MASTER = (
    PROJECT_ROOT
    / "acm slot 11"
    / "data"
    / "processed"
    / "master"
    / "cpcb_pm25_master.csv"
)

ML_MASTER = (
    PROJECT_ROOT
    / "acm slot 11"
    / "data"
    / "ml_ready"
    / "master_modeling_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "acm slot 11"
    / "data"
    / "05_validation"
    / "station_audit"
)

REPORT_TXT = OUTPUT_DIR / "missing_station_audit_report.txt"
DIAGNOSIS_CSV = OUTPUT_DIR / "missing_station_diagnosis.csv"
CPCB_COVERAGE_CSV = OUTPUT_DIR / "missing_station_cpcb_coverage.csv"
INTERMEDIATE_TRACE_CSV = OUTPUT_DIR / "missing_station_intermediate_trace.csv"
PIPELINE_SCAN_CSV = OUTPUT_DIR / "pipeline_script_scan.csv"

EXPECTED_MISSING_STATIONS = {
    "CPRI_Mathura_Road",
    "Commonwealth_Sports_Complex",
    "IGNOU_Maidan_Garhi",
    "JNU",
    "Lodhi_Road",
    "NSIT_Dwarka",
    "NSUT_Jaffarpur",
    "Pusa",
    "Talkatora_Garden",
}

# Search only likely data-processing/code locations.
SEARCH_DIRECTORIES = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "intermediate",
    PROJECT_ROOT / "data" / "features",
    PROJECT_ROOT / "data" / "ml_ready",
]

# File types we will inspect.
CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".ps1",
    ".bat",
    ".txt",
}

DATA_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".pkl",
    ".pickle",
}

# -------------------------------------------------------------------------
# Heuristics for detecting relevant pipeline logic.
# -------------------------------------------------------------------------

PIPELINE_PATTERNS = {
    "merge": re.compile(
        r"\.merge\s*\(|pd\.merge\s*\(|join\s*\(",
        re.IGNORECASE,
    ),
    "inner_join": re.compile(
        r"(how\s*=\s*[\"']inner[\"']|join\s*\(\s*[\"']inner[\"'])",
        re.IGNORECASE,
    ),
    "left_join": re.compile(
        r"(how\s*=\s*[\"']left[\"']|join\s*\(\s*[\"']left[\"'])",
        re.IGNORECASE,
    ),
    "dropna": re.compile(
        r"\.dropna\s*\(",
        re.IGNORECASE,
    ),
    "drop_duplicates": re.compile(
        r"\.drop_duplicates\s*\(",
        re.IGNORECASE,
    ),
    "station_filter": re.compile(
        r"(station|site).*?(isin|query|filter|==)",
        re.IGNORECASE,
    ),
    "feature_completeness": re.compile(
        r"(missing|completeness|valid|coverage|nan|isna)",
        re.IGNORECASE,
    ),
    "matched_samples": re.compile(
        r"(matched samples|dataset matched|total matched)",
        re.IGNORECASE,
    ),
    "inner": re.compile(
        r"\binner\b",
        re.IGNORECASE,
    ),
}


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str) -> None:
    print(f"[AUDIT] {message}")


# =============================================================================
# BASIC FILE CHECKS
# =============================================================================

def ensure_files_exist() -> None:
    missing = []

    if not CPCB_MASTER.exists():
        missing.append(str(CPCB_MASTER))

    if not ML_MASTER.exists():
        missing.append(str(ML_MASTER))

    if missing:
        raise FileNotFoundError(
            "Required files were not found:\n"
            + "\n".join(missing)
        )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_cpcb() -> pd.DataFrame:
    log(f"Loading CPCB master: {CPCB_MASTER}")

    df = pd.read_csv(
        CPCB_MASTER,
        low_memory=False,
    )

    required = {
        "station",
        "year",
        "Timestamp",
        "PM.2.5",
    }

    # Handle common alternate PM2.5 naming.
    if "PM.2.5" not in df.columns and "PM2.5" in df.columns:
        df = df.rename(columns={"PM2.5": "PM.2.5"})

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CPCB master missing columns: {sorted(missing)}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    return df


def load_ml_master() -> pd.DataFrame:
    log(f"Loading ML master: {ML_MASTER}")

    df = pd.read_csv(
        ML_MASTER,
        low_memory=False,
    )

    required = {
        "station",
        "year",
        "month",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"ML master missing columns: {sorted(missing)}"
        )

    return df


# =============================================================================
# STATION-LEVEL COMPARISON
# =============================================================================

def get_station_sets(
    cpcb: pd.DataFrame,
    ml: pd.DataFrame,
) -> tuple[set[str], set[str], set[str], set[str]]:
    cpcb_stations = set(
        cpcb["station"].dropna().astype(str).unique()
    )

    ml_stations = set(
        ml["station"].dropna().astype(str).unique()
    )

    missing = cpcb_stations - ml_stations
    retained = cpcb_stations & ml_stations

    return (
        cpcb_stations,
        ml_stations,
        missing,
        retained,
    )


# =============================================================================
# DUPLICATE AUDIT
# =============================================================================

def duplicate_audit(
    cpcb: pd.DataFrame,
) -> dict:
    exact_duplicates = int(
        cpcb.duplicated().sum()
    )

    station_timestamp_duplicates = int(
        cpcb.duplicated(
            subset=["station", "Timestamp"]
        ).sum()
    )

    station_year_timestamp_duplicates = int(
        cpcb.duplicated(
            subset=[
                "station",
                "year",
                "Timestamp",
            ]
        ).sum()
    )

    return {
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_station_timestamp": station_timestamp_duplicates,
        "duplicate_station_year_timestamp": (
            station_year_timestamp_duplicates
        ),
    }


# =============================================================================
# CPCB COVERAGE AUDIT
# =============================================================================

def calculate_cpcb_coverage(
    cpcb: pd.DataFrame,
    stations: Iterable[str],
) -> pd.DataFrame:

    subset = cpcb[
        cpcb["station"].astype(str).isin(stations)
    ].copy()

    subset["month"] = subset["Timestamp"].dt.month

    # Valid PM2.5 means non-null numeric values.
    subset["PM.2.5"] = pd.to_numeric(
        subset["PM.2.5"],
        errors="coerce",
    )

    subset["pm25_valid"] = subset["PM.2.5"].notna()

    monthly = (
        subset
        .groupby(
            [
                "station",
                "year",
                "month",
            ],
            dropna=False,
        )
        .agg(
            hourly_records=(
                "Timestamp",
                "count",
            ),
            pm25_valid_hours=(
                "pm25_valid",
                "sum",
            ),
        )
        .reset_index()
    )

    monthly["expected_hours"] = (
        pd.to_datetime(
            monthly["year"].astype(str)
            + "-"
            + monthly["month"].astype(str)
            + "-01"
        ).dt.days_in_month
        * 24
    )

    monthly["pm25_coverage_pct"] = (
        monthly["pm25_valid_hours"]
        / monthly["expected_hours"]
        * 100
    )

    # Monthly quality categories.
    monthly["coverage_class"] = pd.cut(
        monthly["pm25_coverage_pct"],
        bins=[
            -np.inf,
            0,
            25,
            50,
            75,
            90,
            100,
            np.inf,
        ],
        labels=[
            "0%",
            "0-25%",
            "25-50%",
            "50-75%",
            "75-90%",
            "90-100%",
            "100%+",
        ],
        include_lowest=True,
    )

    return monthly


def summarize_station_coverage(
    monthly: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        monthly
        .groupby("station")
        .agg(
            station_years=(
                "year",
                "nunique",
            ),
            station_months_present=(
                "month",
                "count",
            ),
            mean_monthly_pm25_coverage_pct=(
                "pm25_coverage_pct",
                "mean",
            ),
            median_monthly_pm25_coverage_pct=(
                "pm25_coverage_pct",
                "median",
            ),
            min_monthly_pm25_coverage_pct=(
                "pm25_coverage_pct",
                "min",
            ),
            months_with_any_pm25=(
                "pm25_valid_hours",
                lambda x: int((x > 0).sum()),
            ),
            months_with_ge_75pct_pm25=(
                "pm25_coverage_pct",
                lambda x: int((x >= 75).sum()),
            ),
            months_with_ge_80pct_pm25=(
                "pm25_coverage_pct",
                lambda x: int((x >= 80).sum()),
            ),
            months_with_ge_90pct_pm25=(
                "pm25_coverage_pct",
                lambda x: int((x >= 90).sum()),
            ),
        )
        .reset_index()
    )

    return summary


# =============================================================================
# RESEARCH-LEVEL STATION CLASSIFICATION
# =============================================================================

def classify_station(
    station: str,
    summary_row: pd.Series,
) -> tuple[str, str, str]:

    station_months = int(
        summary_row.get(
            "station_months_present",
            0,
        )
    )

    months_ge_80 = int(
        summary_row.get(
            "months_with_ge_80pct_pm25",
            0,
        )
    )

    mean_coverage = float(
        summary_row.get(
            "mean_monthly_pm25_coverage_pct",
            0,
        )
    )

    # Five stations in the current file have 2025 only and zero PM2.5.
    if station_months == 12 and mean_coverage == 0:
        return (
            "EXCLUDE",
            "No valid PM2.5 observations in the available study-period records.",
            "Do not integrate unless an independent validated CPCB source restores genuine observations.",
        )

    # Strong ground-data coverage.
    if (
        station in {
            "CPRI_Mathura_Road",
            "NSIT_Dwarka",
            "Pusa",
        }
        and months_ge_80 >= 30
    ):
        return (
            "RECOVER IF MULTIMODAL FEATURES AVAILABLE",
            "Strong CPCB PM2.5 coverage; exclusion is unlikely to be explained by CPCB outcome availability alone.",
            "Trace satellite matching and feature-generation stages before restoring.",
        )

    # Lodhi Road has substantial missingness.
    if station == "Lodhi_Road":
        return (
            "EXCLUDE FROM PRIMARY MODEL; SENSITIVITY ONLY",
            "Substantial and temporally irregular PM2.5 missingness, especially in 2022-2024.",
            "Do not add to the primary model until an explicit station-month completeness rule supports it.",
        )

    return (
        "INVESTIGATE",
        "CPCB coverage alone does not establish the exclusion mechanism.",
        "Check intermediate feature products and the final matching script.",
    )


# =============================================================================
# REPOSITORY SCRIPT SCAN
# =============================================================================

def iter_existing_files(
    roots: Iterable[Path],
    extensions: set[str],
):
    seen: set[Path] = set()

    for root in roots:

        if not root.exists():
            continue

        if root.is_file():

            if root.suffix.lower() in extensions:
                resolved = root.resolve()

                if resolved not in seen:
                    seen.add(resolved)
                    yield root

            continue

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)

            yield path


def scan_pipeline_scripts() -> pd.DataFrame:

    records = []

    for path in iter_existing_files(
        SEARCH_DIRECTORIES,
        CODE_EXTENSIONS,
    ):

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        hits = []

        for label, pattern in PIPELINE_PATTERNS.items():
            if pattern.search(text):
                hits.append(label)

        # Stronger signal: script mentions target files.
        mentions_ml = (
            "master_modeling_dataset"
            in text
            or "master_modelling_dataset"
            in text
        )

        mentions_cpcb = (
            "cpcb_pm25_master"
            in text
            or "cleaned_cpcb_monthly"
            in text
        )

        if hits or mentions_ml or mentions_cpcb:
            records.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "mentions_master_modeling": mentions_ml,
                    "mentions_cpcb": mentions_cpcb,
                    "detected_patterns": (
                        ", ".join(sorted(hits))
                    ),
                }
            )

    return pd.DataFrame(records)


# =============================================================================
# INTERMEDIATE DATA TRACE
# =============================================================================

def read_station_names_from_file(
    path: Path,
) -> tuple[set[str], str]:

    try:

        if path.suffix.lower() == ".csv":

            # Read only station if possible.
            sample = pd.read_csv(
                path,
                usecols=lambda c: c == "station",
                low_memory=False,
            )

            if "station" not in sample.columns:
                return set(), "no_station_column"

            return (
                set(
                    sample["station"]
                    .dropna()
                    .astype(str)
                    .unique()
                ),
                "csv_station_column",
            )

        if path.suffix.lower() == ".parquet":

            df = pd.read_parquet(
                path,
                columns=["station"],
            )

            return (
                set(
                    df["station"]
                    .dropna()
                    .astype(str)
                    .unique()
                ),
                "parquet_station_column",
            )

    except Exception as exc:

        return set(), f"read_error:{exc}"

    return set(), "unsupported"


def trace_intermediate_data(
    missing_stations: set[str],
) -> pd.DataFrame:

    records = []

    # Only inspect actual data products.
    for path in iter_existing_files(
        SEARCH_DIRECTORIES,
        DATA_EXTENSIONS,
    ):

        # Never treat the two known endpoint files as intermediate.
        if path.resolve() in {
            CPCB_MASTER.resolve(),
            ML_MASTER.resolve(),
        }:
            continue

        # Skip huge/raw files where possible.
        try:
            size_mb = path.stat().st_size / (
                1024 * 1024
            )
        except OSError:
            continue

        if size_mb > 250:
            # Avoid accidentally loading massive raw datasets.
            records.append(
                {
                    "file": str(
                        path.relative_to(PROJECT_ROOT)
                    ),
                    "file_size_mb": round(
                        size_mb,
                        2,
                    ),
                    "trace_status": "skipped_large_file",
                    "stations_found": np.nan,
                    "missing_stations_present": "",
                    "missing_count": np.nan,
                }
            )
            continue

        stations, status = (
            read_station_names_from_file(
                path
            )
        )

        if not stations:
            continue

        present_missing = (
            missing_stations & stations
        )

        records.append(
            {
                "file": str(
                    path.relative_to(PROJECT_ROOT)
                ),
                "file_size_mb": round(
                    size_mb,
                    2,
                ),
                "trace_status": status,
                "stations_found": len(
                    stations
                ),
                "missing_stations_present": (
                    "; ".join(
                        sorted(
                            present_missing
                        )
                    )
                ),
                "missing_count": len(
                    present_missing
                ),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# LIKELY EXCLUSION MECHANISM
# =============================================================================

def infer_exclusion_mechanism(
    station: str,
    cpcb_summary: pd.Series,
    intermediate_trace: pd.DataFrame,
    pipeline_scan: pd.DataFrame,
) -> tuple[str, str]:

    # CPCB-only evidence.
    months_ge_80 = int(
        cpcb_summary.get(
            "months_with_ge_80pct_pm25",
            0,
        )
    )

    mean_coverage = float(
        cpcb_summary.get(
            "mean_monthly_pm25_coverage_pct",
            0,
        )
    )

    if mean_coverage == 0:
        return (
            "CPCB outcome unavailable",
            "The station cannot contribute a PM2.5 outcome from the available CPCB records.",
        )

    # If intermediate files contain the station, it survived at least
    # one preprocessing stage.
    files_containing_station = []

    if not intermediate_trace.empty:

        for _, row in intermediate_trace.iterrows():

            raw_present = row.get(
                "missing_stations_present",
                "",
            )

            if isinstance(
                raw_present,
                str,
            ) and station in raw_present.split("; "):

                files_containing_station.append(
                    str(row["file"])
                )

    if files_containing_station:

        return (
            "Lost after at least one intermediate stage",
            "Station is present in one or more intermediate feature files; inspect downstream merge/filter logic.",
        )

    # Strong CPCB stations disappearing from every detected intermediate
    # feature file strongly suggests the satellite processing stage did not
    # produce a compatible record, or station naming/coordinates failed.
    if (
        station in {
            "CPRI_Mathura_Road",
            "NSIT_Dwarka",
            "Pusa",
        }
        and months_ge_80 >= 30
    ):
        return (
            "Likely multimodal matching / satellite-stage exclusion",
            "CPCB coverage is strong, so CPCB PM2.5 availability alone does not explain exclusion.",
        )

    if station == "Lodhi_Road":
        return (
            "Likely CPCB temporal coverage + downstream completeness filtering",
            "Ground observations are substantially incomplete and could fail a monthly completeness or merge criterion.",
        )

    return (
        "Undetermined from available repository artifacts",
        "The repository must contain the relevant intermediate stage output to identify the exact filter.",
    )


# =============================================================================
# REPORT WRITING
# =============================================================================

def write_text_report(
    *,
    cpcb: pd.DataFrame,
    ml: pd.DataFrame,
    cpcb_stations: set[str],
    ml_stations: set[str],
    missing_stations: set[str],
    retained_stations: set[str],
    duplicate_results: dict,
    coverage_summary: pd.DataFrame,
    diagnosis: pd.DataFrame,
    pipeline_scan: pd.DataFrame,
    intermediate_trace: pd.DataFrame,
) -> None:

    lines: list[str] = []

    lines.append(
        "CPCB → ML-READY STATION EXCLUSION AUDIT"
    )
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "Project: Urban Green Cover Thresholds for "
        "PM2.5 Mitigation: A Spatial Causal Machine "
        "Learning Framework for Delhi NCR."
    )
    lines.append("")
    lines.append(
        f"CPCB master rows: {len(cpcb):,}"
    )
    lines.append(
        f"CPCB unique stations: {len(cpcb_stations)}"
    )
    lines.append(
        f"ML-ready rows: {len(ml):,}"
    )
    lines.append(
        f"ML-ready unique stations: {len(ml_stations)}"
    )
    lines.append(
        f"Stations missing from ML-ready: {len(missing_stations)}"
    )
    lines.append(
        f"Stations retained: {len(retained_stations)}"
    )
    lines.append("")

    lines.append("DUPLICATE AUDIT")
    lines.append("-" * 80)

    for key, value in duplicate_results.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")
    lines.append("MISSING STATIONS")
    lines.append("-" * 80)

    for station in sorted(missing_stations):
        lines.append(station)

    lines.append("")
    lines.append("STATION DIAGNOSIS")
    lines.append("-" * 80)

    if not diagnosis.empty:

        columns = [
            "station",
            "decision",
            "cpcb_status",
            "satellite_status",
            "exclusion_mechanism",
            "reason",
            "recommended_action",
        ]

        available = [
            c
            for c in columns
            if c in diagnosis.columns
        ]

        lines.append(
            diagnosis[
                available
            ].to_string(index=False)
        )

    lines.append("")
    lines.append("PIPELINE SCRIPTS DETECTED")
    lines.append("-" * 80)

    if pipeline_scan.empty:
        lines.append(
            "No relevant pipeline scripts were detected."
        )
    else:
        lines.append(
            pipeline_scan.to_string(
                index=False
            )
        )

    lines.append("")
    lines.append("INTERMEDIATE DATA TRACE")
    lines.append("-" * 80)

    if intermediate_trace.empty:
        lines.append(
            "No usable intermediate station-level data files were detected."
        )
    else:
        lines.append(
            intermediate_trace.to_string(
                index=False
            )
        )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 80)

    lines.append(
        "This audit does not modify the repository."
    )

    lines.append(
        "A station with strong CPCB PM2.5 availability "
        "should not be excluded solely because it was "
        "absent from the final ML-ready table."
    )

    lines.append(
        "For CPRI_Mathura_Road, NSIT_Dwarka and Pusa, "
        "the next decision should depend on whether the "
        "existing Sentinel-2, Sentinel-5P and MODIS "
        "feature-generation pipeline can produce valid "
        "station-month records for those stations."
    )

    lines.append(
        "Stations with zero valid PM2.5 throughout their "
        "available study-period records should not be "
        "restored to a PM2.5 regression dataset."
    )

    REPORT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# BUILD DIAGNOSIS TABLE
# =============================================================================

def build_diagnosis(
    missing_stations: set[str],
    coverage_summary: pd.DataFrame,
    intermediate_trace: pd.DataFrame,
    pipeline_scan: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for station in sorted(missing_stations):

        station_row = coverage_summary[
            coverage_summary["station"] == station
        ]

        if station_row.empty:
            cpcb_status = "No CPCB summary available"
            summary_row = pd.Series(dtype=object)
        else:
            summary_row = station_row.iloc[0]

            months_ge_80 = int(
                summary_row[
                    "months_with_ge_80pct_pm25"
                ]
            )

            mean_coverage = float(
                summary_row[
                    "mean_monthly_pm25_coverage_pct"
                ]
            )

            if mean_coverage == 0:
                cpcb_status = (
                    "No valid PM2.5"
                )
            elif months_ge_80 >= 30:
                cpcb_status = (
                    "Strong"
                )
            elif months_ge_80 >= 12:
                cpcb_status = (
                    "Moderate"
                )
            else:
                cpcb_status = (
                    "Poor / irregular"
                )

        decision, reason, recommended = classify_station(
            station,
            summary_row,
        )

        exclusion_mechanism, exclusion_reason = (
            infer_exclusion_mechanism(
                station,
                summary_row,
                intermediate_trace,
                pipeline_scan,
            )
        )

        # Intermediate status.
        station_files = []

        if not intermediate_trace.empty:

            for _, trace_row in intermediate_trace.iterrows():

                present = trace_row.get(
                    "missing_stations_present",
                    "",
                )

                if (
                    isinstance(present, str)
                    and station
                    in present.split("; ")
                ):
                    station_files.append(
                        trace_row["file"]
                    )

        if station_files:
            satellite_status = (
                "Present in intermediate data: "
                + "; ".join(
                    station_files[:5]
                )
            )
        else:
            satellite_status = (
                "Not found in detected intermediate station-level files"
            )

        rows.append(
            {
                "station": station,
                "cpcb_status": cpcb_status,
                "station_years": (
                    summary_row.get(
                        "station_years",
                        np.nan,
                    )
                ),
                "station_months_present": (
                    summary_row.get(
                        "station_months_present",
                        np.nan,
                    )
                ),
                "mean_monthly_pm25_coverage_pct": (
                    summary_row.get(
                        "mean_monthly_pm25_coverage_pct",
                        np.nan,
                    )
                ),
                "months_ge_80pct_pm25": (
                    summary_row.get(
                        "months_with_ge_80pct_pm25",
                        np.nan,
                    )
                ),
                "satellite_status": satellite_status,
                "exclusion_mechanism": exclusion_mechanism,
                "exclusion_reason": exclusion_reason,
                "decision": decision,
                "recommended_action": recommended,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("Starting read-only station exclusion audit.")

    ensure_files_exist()

    # -------------------------------------------------------------------------
    # Load endpoint datasets.
    # -------------------------------------------------------------------------

    cpcb = load_cpcb()
    ml = load_ml_master()

    # -------------------------------------------------------------------------
    # Station sets.
    # -------------------------------------------------------------------------

    (
        cpcb_stations,
        ml_stations,
        missing_stations,
        retained_stations,
    ) = get_station_sets(
        cpcb,
        ml,
    )

    # -------------------------------------------------------------------------
    # Duplicate audit.
    # -------------------------------------------------------------------------

    duplicate_results = duplicate_audit(
        cpcb
    )

    # -------------------------------------------------------------------------
    # CPCB coverage.
    # -------------------------------------------------------------------------

    coverage_monthly = calculate_cpcb_coverage(
        cpcb,
        missing_stations,
    )

    coverage_summary = summarize_station_coverage(
        coverage_monthly
    )

    coverage_monthly.to_csv(
        CPCB_COVERAGE_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Scan code.
    # -------------------------------------------------------------------------

    log("Scanning repository pipeline scripts.")

    pipeline_scan = scan_pipeline_scripts()

    pipeline_scan.to_csv(
        PIPELINE_SCAN_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Trace intermediate station-level products.
    # -------------------------------------------------------------------------

    log(
        "Tracing station presence in intermediate data products."
    )

    intermediate_trace = trace_intermediate_data(
        missing_stations
    )

    intermediate_trace.to_csv(
        INTERMEDIATE_TRACE_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Build diagnosis.
    # -------------------------------------------------------------------------

    diagnosis = build_diagnosis(
        missing_stations,
        coverage_summary,
        intermediate_trace,
        pipeline_scan,
    )

    diagnosis.to_csv(
        DIAGNOSIS_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Write report.
    # -------------------------------------------------------------------------

    write_text_report(
        cpcb=cpcb,
        ml=ml,
        cpcb_stations=cpcb_stations,
        ml_stations=ml_stations,
        missing_stations=missing_stations,
        retained_stations=retained_stations,
        duplicate_results=duplicate_results,
        coverage_summary=coverage_summary,
        diagnosis=diagnosis,
        pipeline_scan=pipeline_scan,
        intermediate_trace=intermediate_trace,
    )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"CPCB stations       : {len(cpcb_stations)}")
    print(f"ML-ready stations   : {len(ml_stations)}")
    print(f"Missing stations    : {len(missing_stations)}")
    print()

    print(
        "Missing stations:"
    )

    for station in sorted(
        missing_stations
    ):
        print(f"  - {station}")

    print()
    print(
        "Duplicate checks:"
    )

    for key, value in duplicate_results.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        f"Diagnosis CSV       : {DIAGNOSIS_CSV}"
    )

    print(
        f"CPCB coverage CSV   : {CPCB_COVERAGE_CSV}"
    )

    print(
        f"Pipeline scan CSV   : {PIPELINE_SCAN_CSV}"
    )

    print(
        f"Intermediate trace  : {INTERMEDIATE_TRACE_CSV}"
    )

    print(
        f"Text report         : {REPORT_TXT}"
    )

    print()
    print(
        "NO DATA OR SCRIPTS WERE MODIFIED."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nAudit interrupted by user."
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nAUDIT FAILED: {exc}"
        )
        raise SystemExit(1)