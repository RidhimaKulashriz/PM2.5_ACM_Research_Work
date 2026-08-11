"""
03_visualize_cpcb.py
--------------------
Generates exploratory publication-quality static figures from the cleaned CPCB master dataset.
Outputs are saved into data/processed/reports/figures/
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_CSV = PROJECT_ROOT / "data" / "processed" / "master" / "cpcb_pm25_master.csv"
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "reports" / "figures"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Pre-Monsoon'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

def run_visualization_pipeline():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_CSV.exists():
        logging.error(f"Master dataset not found at {MASTER_CSV}. Run cleaning script first.")
        return

    logging.info("Loading master dataset...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Month'] = df['Timestamp'].dt.month
    df['Month_Year'] = df['Timestamp'].dt.to_period('M')
    df['Season'] = df['Month'].apply(get_season)

    # 1. Station-wise Mean PM2.5
    plt.figure(figsize=(10, 8))
    st_pm25 = df.groupby('station')['PM2.5'].mean().sort_values()
    plt.barh(st_pm25.index, st_pm25.values, color='#c53030')
    plt.axvline(60, color='black', linestyle='--', label='CPCB National Standard (60 µg/m³)')
    plt.xlabel('Mean PM2.5 (µg/m³)')
    plt.title('Station-wise Mean PM2.5 Concentrations (Delhi)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_station_mean_pm25.png", dpi=300)
    plt.close()

    # 2. Station-wise Mean PM10
    plt.figure(figsize=(10, 8))
    st_pm10 = df.groupby('station')['PM10'].mean().sort_values()
    plt.barh(st_pm10.index, st_pm10.values, color='#dd6b20')
    plt.axvline(100, color='black', linestyle='--', label='CPCB National Standard (100 µg/m³)')
    plt.xlabel('Mean PM10 (µg/m³)')
    plt.title('Station-wise Mean PM10 Concentrations (Delhi)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_station_mean_pm10.png", dpi=300)
    plt.close()

    # 3. PM2.5 Distribution
    plt.figure(figsize=(8, 5))
    valid_pm25 = df['PM2.5'].dropna()
    valid_pm25 = valid_pm25[(valid_pm25 >= 0) & (valid_pm25 <= 800)]  # Filter visual outliers
    plt.hist(valid_pm25, bins=50, color='#2b6cb0', edgecolor='black', alpha=0.7)
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('Frequency (Hours)')
    plt.title('Distribution of Hourly PM2.5 Observations')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_pm25_distribution.png", dpi=300)
    plt.close()

    # 4. PM10 Distribution
    plt.figure(figsize=(8, 5))
    valid_pm10 = df['PM10'].dropna()
    valid_pm10 = valid_pm10[(valid_pm10 >= 0) & (valid_pm10 <= 1200)]
    plt.hist(valid_pm10, bins=50, color='#319795', edgecolor='black', alpha=0.7)
    plt.xlabel('PM10 (µg/m³)')
    plt.ylabel('Frequency (Hours)')
    plt.title('Distribution of Hourly PM10 Observations')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_pm10_distribution.png", dpi=300)
    plt.close()

    # 5. PM2.5 vs PM10 Scatter Plot
    plt.figure(figsize=(7, 7))
    sample_df = df.dropna(subset=['PM2.5', 'PM10']).sample(n=min(10000, len(df)), random_state=42)
    plt.scatter(sample_df['PM10'], sample_df['PM2.5'], alpha=0.2, color='#805ad5', s=10)
    max_val = min(sample_df['PM10'].max(), 1000)
    plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')
    plt.xlabel('PM10 (µg/m³)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.title('PM2.5 vs PM10 Concentration Relationship (Sampled n=10,000)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_pm25_vs_pm10_scatter.png", dpi=300)
    plt.close()

    # 6 & 7. Monthly PM2.5 and PM10 Trends
    monthly = df.groupby('Month_Year')[['PM2.5', 'PM10']].mean().reset_index()
    monthly['Month_Year_Str'] = monthly['Month_Year'].astype(str)

    plt.figure(figsize=(12, 5))
    plt.plot(monthly['Month_Year_Str'], monthly['PM2.5'], marker='o', color='#c53030', label='Monthly Mean PM2.5')
    plt.xticks(rotation=45)
    plt.ylabel('Concentration (µg/m³)')
    plt.title('Monthly Mean PM2.5 Trend (2022–2025)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_monthly_pm25_trend.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly['Month_Year_Str'], monthly['PM10'], marker='s', color='#dd6b20', label='Monthly Mean PM10')
    plt.xticks(rotation=45)
    plt.ylabel('Concentration (µg/m³)')
    plt.title('Monthly Mean PM10 Trend (2022–2025)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_monthly_pm10_trend.png", dpi=300)
    plt.close()

    # 8 & 9. Seasonal PM2.5 & PM10 Comparisons
    season_order = ['Pre-Monsoon', 'Monsoon', 'Post-Monsoon', 'Winter']
    
    plt.figure(figsize=(8, 5))
    seasonal_pm25 = df.groupby('Season')['PM2.5'].mean().reindex(season_order)
    plt.bar(seasonal_pm25.index, seasonal_pm25.values, color=['#e53e3e', '#319795', '#d69e2e', '#2b6cb0'])
    plt.ylabel('Mean PM2.5 (µg/m³)')
    plt.title('Seasonal Comparison of PM2.5 Levels in Delhi')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_seasonal_pm25.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    seasonal_pm10 = df.groupby('Season')['PM10'].mean().reindex(season_order)
    plt.bar(seasonal_pm10.index, seasonal_pm10.values, color=['#e53e3e', '#319795', '#d69e2e', '#2b6cb0'])
    plt.ylabel('Mean PM10 (µg/m³)')
    plt.title('Seasonal Comparison of PM10 Levels in Delhi')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_seasonal_pm10.png", dpi=300)
    plt.close()

    # 10. Data Completeness by Station (% valid PM2.5)
    plt.figure(figsize=(10, 8))
    completeness = df.groupby('station')['PM2.5'].apply(lambda x: (x.notna().sum() / len(x)) * 100).sort_values()
    plt.barh(completeness.index, completeness.values, color='#3182ce')
    plt.xlabel('PM2.5 Completeness (%)')
    plt.title('PM2.5 Data Completeness Percentage by Station')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_station_completeness.png", dpi=300)
    plt.close()

    # 11. Number of Observations by Station and Year
    plt.figure(figsize=(12, 8))
    obs_count = df.groupby(['station', 'year']).size().unstack(fill_value=0)
    obs_count.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Blues')
    plt.ylabel('Number of Observations')
    plt.title('Observation Counts per Station by Year')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_observations_by_station_year.png", dpi=300)
    plt.close()

    logging.info(f"All 11 visual figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    run_visualization_pipeline()