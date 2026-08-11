"""
01_prepare_stations.py
----------------------

Creates a validated CPCB station coordinate master table
from the official CPCB/OGD real-time air-quality dataset.

INPUT
-----
data/raw/station_metadata/cpcb_realtime_aqi.csv

OUTPUT
------
data/processed/satellite/station_master.csv
data/processed/satellite/station_coordinate_audit.csv

The script:

1. Reads the CPCB OGD dataset.
2. Filters Delhi stations.
3. Extracts unique station coordinates.
4. Checks coordinate consistency within each station.
5. Reads the station names from the existing CPCB raw folders.
6. Normalizes station names for matching.
7. Matches OGD station names to your CPCB folders.
8. Flags unmatched/ambiguous stations.
9. Creates the final station_master.csv.

IMPORTANT:
The script NEVER modifies raw CPCB files.
"""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OGD_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "station_metadata"
    / "cpcb_realtime_aqi.csv"
)

RAW_CPCB_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CPCB"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "satellite"
)

MASTER_OUTPUT = OUTPUT_DIR / "station_master.csv"
AUDIT_OUTPUT = OUTPUT_DIR / "station_coordinate_audit.csv"


# =========================================================
# NAME NORMALIZATION
# =========================================================

def normalize_station_name(name):
    """
    Converts different representations of the same station
    into a comparable string.

    Examples:

    'Anand Vihar, Delhi'
        ->
    'anandvihar'

    'Anand_Vihar'
        ->
    'anandvihar'

    'IGI Airport Terminal - 3, New Delhi - IMD'
        ->
    'igiairportterminal3'
    """

    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    # Remove encoding/unicode artifacts
    name = unicodedata.normalize("NFKD", name)

    # Remove organization suffixes
    suffix_patterns = [
        r"\s*-\s*cpcb.*$",
        r"\s*-\s*dpcc.*$",
        r"\s*-\s*imd.*$",
        r"\s*-\s*delhi.*$",
        r"\s*,\s*new delhi.*$",
        r"\s*,\s*delhi.*$",
    ]

    for pattern in suffix_patterns:
        name = re.sub(pattern, "", name)

    # Standardize common variations
    replacements = {
        "sector 8": "sector8",
        "sector-8": "sector8",
        "sector_8": "sector8",

        "terminal 3": "terminal3",
        "terminal-3": "terminal3",
        "terminal_3": "terminal3",

        "t3": "terminal3",

        "r k puram": "rkpuram",
        "r.k. puram": "rkpuram",
        "r_k_puram": "rkpuram",

        "siri fort": "sirifort",
        "siri_fort": "sirifort",

        "mandir marg": "mandirmarg",
        "mandir_marg": "mandirmarg",

        "jawaharlal nehru stadium": "jawaharlalnehrustadium",

        "major dhyan chand national stadium":
            "majordhyanchandnationalstadium",

        "north campus du": "northcampusdu",
        "north campus, du": "northcampusdu",

        "dr karni singh shooting range":
            "drkarnisinghshootingrange",

        "crri mathura road":
            "crrimathuraroad",

        "dwarka sector 8":
            "dwarkasector8",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Remove punctuation, whitespace and underscores
    name = re.sub(r"[^a-z0-9]", "", name)

    return name


# =========================================================
# READ CPCB OGD DATA
# =========================================================

def read_ogd_dataset():

    if not OGD_FILE.exists():
        raise FileNotFoundError(
            f"\nCPCB OGD file not found:\n{OGD_FILE}\n\n"
            "Place your downloaded CPCB CSV here and name it "
            "'cpcb_realtime_aqi.csv'."
        )

    print("\nReading CPCB OGD dataset...")

    # sep=None automatically detects comma/tab separated files
    df = pd.read_csv(
        OGD_FILE,
        sep=None,
        engine="python",
    )

    # Standardize column names
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    required = {
        "state",
        "city",
        "station",
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    print(f"Total OGD rows: {len(df):,}")

    return df


# =========================================================
# EXTRACT DELHI STATIONS
# =========================================================

def extract_delhi_stations(df):

    print("\nFiltering Delhi stations...")

    # Handle variations such as Delhi / Delhi NCR
    state_mask = (
        df["state"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("delhi")
    )

    delhi = df[state_mask].copy()

    print(f"Delhi rows: {len(delhi):,}")

    # Convert coordinates
    delhi["latitude"] = pd.to_numeric(
        delhi["latitude"],
        errors="coerce"
    )

    delhi["longitude"] = pd.to_numeric(
        delhi["longitude"],
        errors="coerce"
    )

    # Remove invalid coordinates
    delhi = delhi[
        delhi["latitude"].between(-90, 90)
        & delhi["longitude"].between(-180, 180)
    ].copy()

    return delhi


# =========================================================
# BUILD UNIQUE STATION COORDINATES
# =========================================================

def build_coordinate_table(delhi):

    print("\nBuilding station coordinate table...")

    records = []

    for station_name, group in delhi.groupby("station"):

        group = group.dropna(
            subset=["latitude", "longitude"]
        )

        if group.empty:
            continue

        # Unique coordinate pairs
        coordinate_counts = (
            group[
                ["latitude", "longitude"]
            ]
            .value_counts()
            .reset_index(name="count")
        )

        # Most frequently observed coordinate
        best = coordinate_counts.iloc[0]

        lat = float(best["latitude"])
        lon = float(best["longitude"])

        unique_coordinates = len(coordinate_counts)

        if unique_coordinates == 1:
            status = "CONSISTENT"

        elif unique_coordinates <= 3:
            status = "MINOR_COORDINATE_VARIATION"

        else:
            status = "COORDINATE_CONFLICT"

        records.append({

            "ogd_station": station_name,

            "latitude": lat,
            "longitude": lon,

            "coordinate_records": len(group),

            "unique_coordinate_pairs":
                unique_coordinates,

            "dominant_coordinate_count":
                int(best["count"]),

            "coordinate_status":
                status,

        })

    result = pd.DataFrame(records)

    return result


# =========================================================
# DISCOVER YOUR CPCB FOLDERS
# =========================================================

def discover_cpcb_stations():

    if not RAW_CPCB_DIR.exists():
        raise FileNotFoundError(
            f"CPCB raw directory not found:\n{RAW_CPCB_DIR}"
        )

    stations = sorted([
        folder.name
        for folder in RAW_CPCB_DIR.iterdir()
        if folder.is_dir()
    ])

    print(
        f"\nCPCB folders discovered: {len(stations)}"
    )

    return stations


# =========================================================
# MATCH YOUR FOLDERS TO OGD STATIONS
# =========================================================

def match_stations(
    folder_stations,
    coordinate_table
):

    print("\nMatching CPCB folders to OGD stations...")

    coordinate_table = coordinate_table.copy()

    coordinate_table["normalized_name"] = (
        coordinate_table["ogd_station"]
        .apply(normalize_station_name)
    )

    lookup = {}

    for _, row in coordinate_table.iterrows():

        key = row["normalized_name"]

        if key:

            lookup.setdefault(
                key,
                []
            ).append(row)

    records = []

    for folder_station in folder_stations:

        normalized = normalize_station_name(
            folder_station
        )

        matches = lookup.get(
            normalized,
            []
        )

        if len(matches) == 1:

            row = matches[0]

            records.append({

                "station": folder_station,

                "ogd_station": row["ogd_station"],

                "latitude": row["latitude"],

                "longitude": row["longitude"],

                "coordinate_status":
                    row["coordinate_status"],

                "coordinate_source":
                    "CPCB OGD",

                "coordinate_source_url":
                    "https://www.data.gov.in/resource/"
                    "real-time-air-quality-index-various-locations",

                "match_status":
                    "MATCHED",

                "match_method":
                    "NORMALIZED_EXACT",

                "unique_coordinate_pairs":
                    row["unique_coordinate_pairs"],

            })

        elif len(matches) > 1:

            records.append({

                "station": folder_station,

                "ogd_station":
                    "; ".join(
                        x["ogd_station"]
                        for x in matches
                    ),

                "latitude": np.nan,

                "longitude": np.nan,

                "coordinate_status":
                    "AMBIGUOUS",

                "coordinate_source":
                    "CPCB OGD",

                "coordinate_source_url":
                    "https://www.data.gov.in/resource/"
                    "real-time-air-quality-index-various-locations",

                "match_status":
                    "AMBIGUOUS",

                "match_method":
                    "NORMALIZED_EXACT",

                "unique_coordinate_pairs":
                    np.nan,

            })

        else:

            records.append({

                "station": folder_station,

                "ogd_station": "",

                "latitude": np.nan,

                "longitude": np.nan,

                "coordinate_status":
                    "NOT_FOUND",

                "coordinate_source":
                    "",

                "coordinate_source_url":
                    "",

                "match_status":
                    "NOT_FOUND",

                "match_method":
                    "",

                "unique_coordinate_pairs":
                    np.nan,

            })

    return pd.DataFrame(records)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)
    print("CPCB STATION COORDINATE MASTER BUILDER")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 1. Read OGD dataset
    # -----------------------------------------------------

    df = read_ogd_dataset()

    # -----------------------------------------------------
    # 2. Filter Delhi
    # -----------------------------------------------------

    delhi = extract_delhi_stations(df)

    # -----------------------------------------------------
    # 3. Build coordinate table
    # -----------------------------------------------------

    coordinate_table = build_coordinate_table(
        delhi
    )

    print(
        f"\nUnique Delhi OGD stations found: "
        f"{len(coordinate_table)}"
    )

    # -----------------------------------------------------
    # 4. Discover stations in your CPCB folders
    # -----------------------------------------------------

    folder_stations = discover_cpcb_stations()

    # -----------------------------------------------------
    # 5. Match
    # -----------------------------------------------------

    station_master = match_stations(
        folder_stations,
        coordinate_table
    )

    # -----------------------------------------------------
    # 6. Save
    # -----------------------------------------------------

    station_master.to_csv(
        MASTER_OUTPUT,
        index=False
    )

    # -----------------------------------------------------
    # 7. Create audit
    # -----------------------------------------------------

    audit = station_master.copy()

    audit["needs_manual_review"] = (
        audit["match_status"]
        != "MATCHED"
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False
    )

    # -----------------------------------------------------
    # 8. Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("RESULT")
    print("=" * 75)

    print(
        "\nMatching status:"
    )

    print(
        station_master[
            "match_status"
        ].value_counts()
    )

    print(
        "\nCoordinate quality:"
    )

    print(
        station_master[
            "coordinate_status"
        ].value_counts()
    )

    # -----------------------------------------------------
    # 9. Display unmatched
    # -----------------------------------------------------

    unmatched = station_master[
        station_master["match_status"]
        != "MATCHED"
    ]

    if not unmatched.empty:

        print(
            "\n" + "=" * 75
        )

        print(
            "STATIONS REQUIRING REVIEW"
        )

        print(
            "=" * 75
        )

        print(
            unmatched[
                [
                    "station",
                    "ogd_station",
                    "match_status"
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "\nAll CPCB folders matched successfully!"
        )

    print("\n" + "=" * 75)
    print("FILES CREATED")
    print("=" * 75)

    print(
        f"\nStation master:\n{MASTER_OUTPUT}"
    )

    print(
        f"\nCoordinate audit:\n{AUDIT_OUTPUT}"
    )

    print(
        "\nIMPORTANT:"
        "\nOnly MATCHED stations should proceed "
        "automatically to GEE."
        "\nNOT_FOUND and AMBIGUOUS stations must "
        "be reviewed before satellite extraction."
    )


if __name__ == "__main__":
    main()