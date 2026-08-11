import pandas as pd
import numpy as np
from pathlib import Path

def engineer_features():
    base_dir = Path(__file__).resolve().parent.parent.parent
    in_file = base_dir / 'data' / 'processed' / 'pipeline' / '02_satellite_qc_imputed.csv'
    out_file = base_dir / 'data' / 'processed' / 'pipeline' / '03_engineered_features.csv'

    print("\n--- PHASE 9: FEATURE ENGINEERING ---")
    df = pd.read_csv(in_file)

    # 1. Cyclical Time Features (Month)
    print("Engineering cyclical temporal features...")
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 2. Spatial Gradients (Macro vs Micro Environment)
    # Example: How much hotter/greener is the 1km region compared to the immediate 100m station area?
    print("Engineering spatial gradient features (1000m - 100m)...")
    
    # Safely calculate gradients if the columns survived Phase 8 QC
    if 'modis_ndvi_mean_1000m' in df.columns and 'modis_ndvi_mean_100m' in df.columns:
        df['gradient_ndvi_1000_100'] = df['modis_ndvi_mean_1000m'] - df['modis_ndvi_mean_100m']
        
    if 'modis_lst_day_mean_c_1000m' in df.columns and 'modis_lst_day_mean_c_100m' in df.columns:
        df['gradient_lst_day_1000_100'] = df['modis_lst_day_mean_c_1000m'] - df['modis_lst_day_mean_c_100m']

    if 's5p_no2_trop_mean_1000m' in df.columns and 's5p_no2_trop_mean_100m' in df.columns:
        df['gradient_no2_1000_100'] = df['s5p_no2_trop_mean_1000m'] - df['s5p_no2_trop_mean_100m']

    # 3. Season Encodings
    seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 1} # 1:Winter, 2:Summer, 3:Monsoon, 4:Post-Monsoon
    df['season_encoded'] = df['month'].map(seasons)

    df.to_csv(out_file, index=False)
    print(f"[SUCCESS] Engineered features saved to {out_file}")

if __name__ == "__main__":
    engineer_features()