"""
03_eda_cpcb.py
--------------
Exploratory Data Analysis for the cleaned CPCB Delhi dataset.

Reads:
    data/processed/master/cpcb_pm25_master.csv
    data/processed/reports/cpcb_station_year_summary.csv

Outputs:
    data/processed/reports/figures/

Visualizations:
    1. Data coverage by station
    2. Station-wise mean PM2.5
    3. Station-wise mean PM10
    4. Monthly PM2.5 trend
    5. Seasonal PM2.5
    6. PM2.5 vs PM10
    7. Station-year coverage heatmap
"""

from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master"
    / "cpcb_pm25_master.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reports"
    / "cpcb_station_year_summary.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reports"
    / "figures"
    / "eda"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{MASTER_FILE}"
        )

    logging.info("Loading master CPCB dataset...")

    df = pd.read_csv(
        MASTER_FILE,
        low_memory=False
    )

    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column not found.")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    if "PM2.5" in df.columns:
        df["PM2.5"] = pd.to_numeric(
            df["PM2.5"],
            errors="coerce"
        )

    if "PM10" in df.columns:
        df["PM10"] = pd.to_numeric(
            df["PM10"],
            errors="coerce"
        )

    df = df.dropna(subset=["Timestamp"])

    df["year"] = df["Timestamp"].dt.year
    df["month"] = df["Timestamp"].dt.month

    logging.info(
        f"Loaded {len(df):,} observations."
    )

    logging.info(
        f"Stations: {df['station'].nunique()}"
    )

    logging.info(
        f"Date range: "
        f"{df['Timestamp'].min()} → {df['Timestamp'].max()}"
    )

    return df


# ============================================================
# 1. DATA COVERAGE BY STATION
# ============================================================

def plot_station_coverage(df):

    coverage = (
        df.groupby("station")
        .size()
        .sort_values()
    )

    plt.figure(figsize=(10, 12))

    coverage.plot(kind="barh")

    plt.title("CPCB Observations by Monitoring Station")
    plt.xlabel("Number of observations")
    plt.ylabel("Monitoring station")

    plt.tight_layout()

    output = FIGURES_DIR / "01_station_data_coverage.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# 2. STATION-WISE MEAN PM2.5
# ============================================================

def plot_station_pm25(df):

    if "PM2.5" not in df.columns:
        logging.warning("PM2.5 column not found.")
        return

    station_pm25 = (
        df.groupby("station")["PM2.5"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(10, 12))

    station_pm25.plot(kind="barh")

    plt.title("Mean PM2.5 Concentration by Monitoring Station")
    plt.xlabel("Mean PM2.5")
    plt.ylabel("Monitoring station")

    plt.tight_layout()

    output = FIGURES_DIR / "02_station_mean_pm25.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# 3. STATION-WISE MEAN PM10
# ============================================================

def plot_station_pm10(df):

    if "PM10" not in df.columns:
        logging.warning("PM10 column not found.")
        return

    station_pm10 = (
        df.groupby("station")["PM10"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(10, 12))

    station_pm10.plot(kind="barh")

    plt.title("Mean PM10 Concentration by Monitoring Station")
    plt.xlabel("Mean PM10")
    plt.ylabel("Monitoring station")

    plt.tight_layout()

    output = FIGURES_DIR / "03_station_mean_pm10.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# 4. MONTHLY PM2.5 TREND
# ============================================================

def plot_monthly_pm25(df):

    if "PM2.5" not in df.columns:
        return

    monthly = (
        df.groupby("month")["PM2.5"]
        .mean()
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    plt.xticks(
        range(1, 13),
        month_names
    )

    plt.title("Monthly Mean PM2.5 Concentration")
    plt.xlabel("Month")
    plt.ylabel("Mean PM2.5")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = FIGURES_DIR / "04_monthly_pm25_trend.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# 5. SEASONAL PM2.5
# ============================================================

def assign_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Pre-Monsoon"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post-Monsoon"


def plot_seasonal_pm25(df):

    if "PM2.5" not in df.columns:
        return

    df = df.copy()

    df["season"] = df["month"].apply(assign_season)

    season_order = [
        "Winter",
        "Pre-Monsoon",
        "Monsoon",
        "Post-Monsoon"
    ]

    seasonal = (
        df.groupby("season")["PM2.5"]
        .mean()
        .reindex(season_order)
    )

    plt.figure(figsize=(9, 6))

    seasonal.plot(
        kind="bar"
    )

    plt.title("Seasonal Mean PM2.5 Concentration")
    plt.xlabel("Season")
    plt.ylabel("Mean PM2.5")

    plt.xticks(rotation=0)

    plt.tight_layout()

    output = FIGURES_DIR / "05_seasonal_pm25.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# 6. PM2.5 VS PM10
# ============================================================

def plot_pm25_pm10(df):

    if "PM2.5" not in df.columns:
        return

    if "PM10" not in df.columns:
        return

    subset = df[
        ["PM2.5", "PM10"]
    ].dropna()

    # Avoid extremely large values dominating the visualization
    subset = subset[
        (subset["PM2.5"] >= 0) &
        (subset["PM10"] >= 0)
    ]

    plt.figure(figsize=(8, 7))

    plt.scatter(
        subset["PM10"],
        subset["PM2.5"],
        alpha=0.15,
        s=8
    )

    plt.title("Relationship Between PM2.5 and PM10")

    plt.xlabel("PM10")
    plt.ylabel("PM2.5")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = FIGURES_DIR / "06_pm25_vs_pm10.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# 7. STATION-YEAR COVERAGE HEATMAP
# ============================================================

def plot_station_year_heatmap(df):

    coverage = (
        df.groupby(
            ["station", "year"]
        )
        .size()
        .unstack(fill_value=0)
    )

    coverage_binary = (
        coverage > 0
    ).astype(int)

    coverage_binary = coverage_binary.sort_index()

    plt.figure(
        figsize=(
            9,
            max(10, len(coverage_binary) * 0.3)
        )
    )

    plt.imshow(
        coverage_binary.values,
        aspect="auto"
    )

    plt.colorbar(
        label="Data available (1 = Yes, 0 = No)"
    )

    plt.xticks(
        range(len(coverage_binary.columns)),
        coverage_binary.columns
    )

    plt.yticks(
        range(len(coverage_binary.index)),
        coverage_binary.index
    )

    plt.xlabel("Year")
    plt.ylabel("Monitoring station")

    plt.title(
        "CPCB Station-Year Data Coverage"
    )

    plt.tight_layout()

    output = FIGURES_DIR / "07_station_year_coverage.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(f"Saved: {output}")


# ============================================================
# MAIN
# ============================================================

def main():

    logging.info("=" * 60)
    logging.info("CPCB EXPLORATORY DATA ANALYSIS")
    logging.info("=" * 60)

    df = load_data()

    plot_station_coverage(df)

    plot_station_pm25(df)

    plot_station_pm10(df)

    plot_monthly_pm25(df)

    plot_seasonal_pm25(df)

    plot_pm25_pm10(df)

    plot_station_year_heatmap(df)

    logging.info("=" * 60)
    logging.info("EDA COMPLETE")
    logging.info(
        f"Figures saved to: {FIGURES_DIR}"
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()