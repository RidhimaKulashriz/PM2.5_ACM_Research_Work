"""
01_audit_cpcb.py

CPCB Raw Dataset Audit
----------------------

Purpose:
    Audit the raw CPCB station-year files WITHOUT modifying them.

Checks:
    - Missing station/year folders
    - Missing CSV files
    - Empty CSV files
    - Unreadable/corrupted files
    - Row and column counts
    - Timestamp availability
    - PM2.5 / PM10 availability
    - Duplicate rows
    - Missing values
    - Date coverage
    - Schema variations
    - Missing expected variables

Outputs:
    data/processed/reports/
        cpcb_file_inventory.csv
        cpcb_column_audit.csv
        cpcb_missing_data_report.csv
        cpcb_station_year_summary.csv
        cpcb_quality_report.html
"""

from pathlib import Path
import logging
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CPCB"
REPORTS_DIR = PROJECT_ROOT / "data" / "processed" / "reports"

EXPECTED_YEARS = ["2022", "2023", "2024", "2025"]


EXPECTED_VARIABLES = [
    "Timestamp",
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "SO2",
    "CO",
    "Ozone",
    "Benzene",
    "Toluene",
    "Xylene",
    "O Xylene",
    "Eth-Benzene",
    "MP-Xylene",
    "AT",
    "RH",
    "WS",
    "WD",
    "RF",
    "TOT-RF",
    "SR",
    "BP",
    "VWS",
]


MISSING_TOKENS = {
    "",
    "NA",
    "N/A",
    "NULL",
    "NONE",
    "NAN",
    "-",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """
    Convert raw CPCB column names into a comparable
    standardized representation.

    This does NOT modify the raw data.
    """

    col = str(column).strip()

    # Fix common encoding artifacts
    col = col.replace("Âµ", "µ")
    col = col.replace("Â", "")
    col = col.replace("â", "")

    # Remove units in parentheses
    col = re.sub(r"\s*\([^)]*\)", "", col)

    # Normalize whitespace
    col = re.sub(r"\s+", " ", col).strip()

    # Lowercase for matching
    lower = col.lower()

    mappings = {
        "timestamp": "Timestamp",
        "date": "Timestamp",
        "datetime": "Timestamp",
        "date time": "Timestamp",
        "from date": "Timestamp",

        "pm2.5": "PM2.5",
        "pm25": "PM2.5",
        "pm 2.5": "PM2.5",

        "pm10": "PM10",
        "pm 10": "PM10",

        "no": "NO",
        "no2": "NO2",
        "nox": "NOx",
        "nh3": "NH3",
        "so2": "SO2",
        "co": "CO",
        "ozone": "Ozone",
        "o3": "Ozone",

        "benzene": "Benzene",
        "toluene": "Toluene",
        "xylene": "Xylene",
        "o xylene": "O Xylene",
        "eth-benzene": "Eth-Benzene",
        "mp-xylene": "MP-Xylene",

        "at": "AT",
        "rh": "RH",
        "ws": "WS",
        "wd": "WD",
        "rf": "RF",
        "tot-rf": "TOT-RF",
        "sr": "SR",
        "bp": "BP",
        "vws": "VWS",
    }

    return mappings.get(lower, col)


# ============================================================
# TIMESTAMP DETECTION
# ============================================================

def find_timestamp_column(columns):

    for column in columns:

        normalized = normalize_column_name(column)

        if normalized == "Timestamp":
            return column

    return None


# ============================================================
# POLLUTANT DETECTION
# ============================================================

def find_pollutant_column(columns, pollutant):

    for column in columns:

        normalized = normalize_column_name(column)

        if normalized == pollutant:
            return column

    return None


# ============================================================
# MISSING VALUE COUNT
# ============================================================

def count_missing(series):

    missing = series.isna()

    string_missing = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .isin(MISSING_TOKENS)
    )

    return int((missing | string_missing).sum())


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_pipeline():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():

        logging.error(
            f"Raw CPCB directory not found:\n{RAW_DIR}"
        )

        return

    station_folders = sorted(
        [
            folder
            for folder in RAW_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    logging.info(
        f"Found {len(station_folders)} station directories."
    )

    file_inventory = []
    column_audits = []
    missing_reports = []
    station_summaries = []

    # ========================================================
    # STATION LOOP
    # ========================================================

    for station_path in station_folders:

        station_name = station_path.name

        logging.info(
            f"Auditing station: {station_name}"
        )

        # ====================================================
        # YEAR LOOP
        # ====================================================

        for year in EXPECTED_YEARS:

            year_dir = station_path / year

            expected_csv = (
                year_dir /
                f"{year}_hourly.csv"
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Initialize these BEFORE any conditional.
            # ------------------------------------------------

            pm25_col = None
            pm10_col = None
            ts_col = None

            # Default values
            status = "available"

            row_count = 0
            column_count = 0
            duplicate_rows = 0

            columns_found = []

            timestamp_found = False

            first_timestamp = None
            last_timestamp = None

            pm25_valid = 0
            pm10_valid = 0

            pm25_completeness = 0.0
            pm10_completeness = 0.0

            # =================================================
            # FILE EXISTENCE CHECK
            # =================================================

            if not year_dir.exists():

                status = "MISSING_YEAR_FOLDER"

            elif not expected_csv.exists():

                status = "MISSING_CSV"

            elif expected_csv.stat().st_size == 0:

                status = "EMPTY_CSV_0BYTES"

            else:

                # =================================================
                # READ FILE
                # =================================================

                try:

                    df = pd.read_csv(
                        expected_csv,
                        encoding_errors="replace",
                        low_memory=False,
                    )

                    if df.empty:

                        status = "EMPTY_CSV_NO_ROWS"

                    else:

                        row_count = len(df)
                        column_count = len(df.columns)

                        columns_found = list(df.columns)

                        # -----------------------------------------
                        # Duplicate rows
                        # -----------------------------------------

                        duplicate_rows = int(
                            df.duplicated().sum()
                        )

                        # -----------------------------------------
                        # Timestamp
                        # -----------------------------------------

                        ts_col = find_timestamp_column(
                            columns_found
                        )

                        if ts_col is not None:

                            timestamp_found = True

                            parsed_ts = pd.to_datetime(
                                df[ts_col],
                                errors="coerce"
                            )

                            valid_ts = parsed_ts.dropna()

                            if not valid_ts.empty:

                                first_timestamp = str(
                                    valid_ts.min()
                                )

                                last_timestamp = str(
                                    valid_ts.max()
                                )

                        # -----------------------------------------
                        # PM2.5
                        # -----------------------------------------

                        pm25_col = find_pollutant_column(
                            columns_found,
                            "PM2.5"
                        )

                        if pm25_col is not None:

                            pm25_numeric = pd.to_numeric(
                                df[pm25_col],
                                errors="coerce"
                            )

                            pm25_valid = int(
                                pm25_numeric.notna().sum()
                            )

                            pm25_completeness = round(
                                pm25_valid /
                                row_count *
                                100,
                                2
                            )

                        # -----------------------------------------
                        # PM10
                        # -----------------------------------------

                        pm10_col = find_pollutant_column(
                            columns_found,
                            "PM10"
                        )

                        if pm10_col is not None:

                            pm10_numeric = pd.to_numeric(
                                df[pm10_col],
                                errors="coerce"
                            )

                            pm10_valid = int(
                                pm10_numeric.notna().sum()
                            )

                            pm10_completeness = round(
                                pm10_valid /
                                row_count *
                                100,
                                2
                            )

                        # -----------------------------------------
                        # Column-level missingness
                        # -----------------------------------------

                        for column in columns_found:

                            missing_count = count_missing(
                                df[column]
                            )

                            missing_reports.append({

                                "station": station_name,
                                "year": year,

                                "raw_column": column,

                                "normalized_column":
                                    normalize_column_name(
                                        column
                                    ),

                                "total_rows": row_count,

                                "missing_count":
                                    missing_count,

                                "missing_percent":
                                    round(
                                        missing_count /
                                        row_count *
                                        100,
                                        2
                                    ),

                            })

                except Exception as error:

                    status = (
                        f"CORRUPTED_CSV: {error}"
                    )

            # =================================================
            # COLUMN AUDIT
            # =================================================

            normalized_columns = [
                normalize_column_name(column)
                for column in columns_found
            ]

            missing_expected = [
                variable
                for variable in EXPECTED_VARIABLES
                if variable not in normalized_columns
            ]

            extra_columns = [
                variable
                for variable in normalized_columns
                if variable not in EXPECTED_VARIABLES
            ]

            column_audits.append({

                "station": station_name,

                "year": year,

                "status": status,

                "timestamp_found":
                    timestamp_found,

                "pm25_found":
                    pm25_col is not None,

                "pm10_found":
                    pm10_col is not None,

                "column_count":
                    column_count,

                "missing_expected_columns":
                    "; ".join(missing_expected),

                "extra_columns":
                    "; ".join(extra_columns),

            })

            # =================================================
            # FILE INVENTORY
            # =================================================

            file_inventory.append({

                "station": station_name,

                "year": year,

                "status": status,

                "file_path":
                    str(
                        expected_csv.relative_to(
                            PROJECT_ROOT
                        )
                    )
                    if expected_csv.exists()
                    else None,

                "file_size_bytes":
                    expected_csv.stat().st_size
                    if expected_csv.exists()
                    else 0,

                "row_count":
                    row_count,

                "column_count":
                    column_count,

                "duplicate_rows":
                    duplicate_rows,

            })

            # =================================================
            # STATION-YEAR SUMMARY
            # =================================================

            station_summaries.append({

                "station": station_name,

                "year": year,

                "status": status,

                "row_count":
                    row_count,

                "first_timestamp":
                    first_timestamp,

                "last_timestamp":
                    last_timestamp,

                "pm25_valid_obs":
                    pm25_valid,

                "pm25_completeness_pct":
                    pm25_completeness,

                "pm10_valid_obs":
                    pm10_valid,

                "pm10_completeness_pct":
                    pm10_completeness,

            })

    # ========================================================
    # EXPORT REPORTS
    # ========================================================

    df_inventory = pd.DataFrame(
        file_inventory
    )

    df_columns = pd.DataFrame(
        column_audits
    )

    df_missing = pd.DataFrame(
        missing_reports
    )

    df_summary = pd.DataFrame(
        station_summaries
    )

    df_inventory.to_csv(
        REPORTS_DIR /
        "cpcb_file_inventory.csv",
        index=False
    )

    df_columns.to_csv(
        REPORTS_DIR /
        "cpcb_column_audit.csv",
        index=False
    )

    df_missing.to_csv(
        REPORTS_DIR /
        "cpcb_missing_data_report.csv",
        index=False
    )

    df_summary.to_csv(
        REPORTS_DIR /
        "cpcb_station_year_summary.csv",
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CPCB DATASET AUDIT COMPLETE"
    )

    print("=" * 70)

    print(
        f"Stations: {len(station_folders)}"
    )

    print(
        f"Expected station-years: "
        f"{len(station_folders) * len(EXPECTED_YEARS)}"
    )

    print(
        "\nFile status:"
    )

    print(
        df_inventory["status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nPM2.5 availability:"
    )

    print(
        df_columns["pm25_found"]
        .value_counts()
        .to_string()
    )

    print(
        "\nPM10 availability:"
    )

    print(
        df_columns["pm10_found"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nReports saved to:\n"
        f"{REPORTS_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    audit_pipeline()