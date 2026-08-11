"""
02_clean_cpcb.py
----------------
Processes raw CPCB CSVs into cleaned, standardized files without modifying original inputs.

Standardization rules:
 - Fixes encoding artifacts (e.g. Âµg/mÂ³)
 - Maps column variations to standard parameter names
 - Parses timestamps to datetime, sorts chronologically, removes duplicates
 - Converts numeric columns safely while keeping genuine missing values as NaN
 - Creates quality flag columns for suspicious/extreme pollutant levels (does NOT drop them)
 - Outputs clean station-year CSVs and a merged master dataset.
"""

import os
import re
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CPCB"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATION_COMBINED_DIR = PROCESSED_DIR / "station_combined"
MASTER_DIR = PROCESSED_DIR / "master"
REPORTS_DIR = PROCESSED_DIR / "reports"

COLUMN_MAPPING = {
    'pm2.5': 'PM2.5', 'pm25': 'PM2.5', 'pm2_5': 'PM2.5',
    'pm10': 'PM10',
    'no': 'NO', 'no2': 'NO2', 'nox': 'NOx', 'nh3': 'NH3',
    'so2': 'SO2', 'co': 'CO', 'ozone': 'Ozone', 'o3': 'Ozone',
    'benzene': 'Benzene', 'toluene': 'Toluene', 'xylene': 'Xylene',
    'at': 'AT', 'rh': 'RH', 'ws': 'WS', 'wd': 'WD', 'rf': 'RF',
    'tot-rf': 'TOT-RF', 'sr': 'SR', 'bp': 'BP', 'vws': 'VWS'
}

def clean_column_header(col_name: str) -> str:
    """Repairs encoding and maps to standard parameter string."""
    cleaned = re.sub(r'Â|µ|â|€|â|™', '', str(col_name))
    cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned).strip()
    c_lower = cleaned.lower()
    return COLUMN_MAPPING.get(c_lower, cleaned)

def clean_station_files():
    STATION_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    cleaning_summaries = []
    all_clean_dfs = []

    if not RAW_DIR.exists():
        logging.error(f"Directory not found: {RAW_DIR}")
        return

    for station_folder in RAW_DIR.iterdir():
        if not station_folder.is_dir():
            continue

        station_name = station_folder.name

        for year_folder in station_folder.iterdir():
            if not year_folder.is_dir():
                continue

            year_str = year_folder.name
            csv_file = year_folder / f"{year_str}_hourly.csv"

            if not csv_file.exists() or csv_file.stat().st_size == 0:
                continue

            try:
                # Load raw data handling encodings
                df_raw = pd.read_csv(csv_file, encoding_errors='replace', low_memory=False)
                if df_raw.empty:
                    continue

                raw_rows = len(df_raw)

                # Rename columns
                df_clean = df_raw.copy()
                df_clean.columns = [clean_column_header(c) for c in df_clean.columns]

                # Identify Timestamp Column
                ts_col = next((c for c in df_clean.columns if any(k in c.lower() for k in ['timestamp', 'date', 'datetime', 'time'])), None)

                if ts_col:
                    df_clean['Timestamp'] = pd.to_datetime(df_clean[ts_col], errors='coerce')
                    if ts_col != 'Timestamp':
                        df_clean.drop(columns=[ts_col], inplace=True, errors='ignore')
                else:
                    logging.warning(f"No timestamp column found for {station_name} {year_str}")
                    continue

                # Remove records with unparseable timestamps
                df_clean = df_clean.dropna(subset=['Timestamp'])

                # Convert string 'NA', 'None', etc. to NaN
                df_clean.replace(['NA', 'N/A', 'None', 'null', 'nan', 'NaN', ' '], np.nan, inplace=True)

                # Convert pollutant/met variables to numeric
                ignore_cols = {'Timestamp', 'station', 'year'}
                for col in df_clean.columns:
                    if col not in ignore_cols:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                # Remove exact duplicate rows
                init_len = len(df_clean)
                df_clean = df_clean.drop_duplicates(subset=['Timestamp'] + [c for c in df_clean.columns if c not in ignore_cols])
                dups_removed = init_len - len(df_clean)

                # Sort chronologically
                df_clean = df_clean.sort_values('Timestamp').reset_index(drop=True)

                # Add Metadata
                df_clean['station'] = station_name
                df_clean['year'] = int(year_str)

                # Flag extreme or questionable observations (RESEARCH RULE: FLAG, DO NOT DELETE)
                if 'PM2.5' in df_clean.columns:
                    df_clean['flag_pm25_negative'] = df_clean['PM2.5'] < 0
                    df_clean['flag_pm25_extreme'] = df_clean['PM2.5'] > 1000  # Extreme spike threshold for Delhi
                
                if 'PM10' in df_clean.columns:
                    df_clean['flag_pm10_negative'] = df_clean['PM10'] < 0
                    df_clean['flag_pm10_extreme'] = df_clean['PM10'] > 1500
                
                if 'PM2.5' in df_clean.columns and 'PM10' in df_clean.columns:
                    df_clean['flag_pm10_less_than_pm25'] = df_clean['PM10'] < df_clean['PM2.5']

                # Output single cleaned station-year file
                out_filename = f"{station_name}_{year_str}_clean.csv"
                out_path = STATION_COMBINED_DIR / out_filename
                df_clean.to_csv(out_path, index=False)

                # Compute Metrics for summary
                clean_rows = len(df_clean)
                pm25_valid = df_clean['PM2.5'].notna().sum() if 'PM2.5' in df_clean.columns else 0
                pm25_miss_pct = round(((clean_rows - pm25_valid) / clean_rows) * 100, 2) if clean_rows > 0 else 100.0
                pm10_valid = df_clean['PM10'].notna().sum() if 'PM10' in df_clean.columns else 0
                pm10_miss_pct = round(((clean_rows - pm10_valid) / clean_rows) * 100, 2) if clean_rows > 0 else 100.0

                cleaning_summaries.append({
                    'station': station_name,
                    'year': year_str,
                    'raw_rows': raw_rows,
                    'clean_rows': clean_rows,
                    'duplicates_removed': dups_removed,
                    'pm25_valid': pm25_valid,
                    'pm25_missing_percent': pm25_miss_pct,
                    'pm10_valid': pm10_valid,
                    'pm10_missing_percent': pm10_miss_pct,
                    'date_start': df_clean['Timestamp'].min(),
                    'date_end': df_clean['Timestamp'].max(),
                    'columns_available': ";".join([c for c in df_clean.columns if not c.startswith('flag_')])
                })

                all_clean_dfs.append(df_clean)

            except Exception as e:
                logging.error(f"Error processing {station_name} {year_str}: {e}")

    # Export Cleaning Summary
    df_summary = pd.DataFrame(cleaning_summaries)
    df_summary.to_csv(REPORTS_DIR / "cpcb_cleaning_summary.csv", index=False)

    # Build Master Unified Dataset
    if all_clean_dfs:
        logging.info("Building merged master dataset...")
        master_df = pd.concat(all_clean_dfs, ignore_index=True)
        
        # Ensure optimal column order
        priority_cols = ['station', 'year', 'Timestamp', 'PM2.5', 'PM10']
        other_cols = [c for c in master_df.columns if c not in priority_cols]
        master_df = master_df[priority_cols + other_cols]

        master_path = MASTER_DIR / "cpcb_pm25_master.csv"
        master_df.to_csv(master_path, index=False)
        logging.info(f"Master file successfully written to {master_path} with {len(master_df)} records.")

if __name__ == "__main__":
    clean_station_files()