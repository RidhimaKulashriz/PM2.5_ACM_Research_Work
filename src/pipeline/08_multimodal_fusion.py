import pandas as pd
from pathlib import Path

def pivot_satellite_data(df, dataset_prefix):
    """Pivots buffer sizes into columns for a flat ML structure."""
    # Identify feature columns (ignore metadata)
    meta_cols = ['station', 'latitude', 'longitude', 'year', 'month', 'buffer_m', 'source_dataset', 'spatial_scale_m', 'temporal_aggregation']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Pivot
    pivoted = df.pivot_table(
        index=['station', 'year', 'month', 'latitude', 'longitude'],
        columns='buffer_m',
        values=feature_cols,
        aggfunc='first'
    )
    
    # Flatten MultiIndex columns (e.g., 'modis_ndvi_mean' and 500 -> 'modis_ndvi_mean_500m')
    pivoted.columns = [f"{col}_{int(buffer)}m" for col, buffer in pivoted.columns]
    return pivoted.reset_index()

def fuse_multimodal_data():
    base_dir = Path(__file__).resolve().parent.parent.parent
    sat_dir = base_dir / 'data' / 'processed' / 'satellite'
    out_dir = base_dir / 'data' / 'processed' / 'pipeline'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- PHASE 7: MULTIMODAL DATA FUSION ---")
    
    # Load all satellite datasets
    files = {
        's2': sat_dir / 'sentinel2' / 'sentinel2_station_monthly_features.csv',
        's5p': sat_dir / 'sentinel5p' / 'sentinel5p_no2_station_monthly_features.csv',
        'modis_veg': sat_dir / 'modis_veg' / 'modis_veg_station_monthly_features.csv',
        'modis_lst': sat_dir / 'modis_lst' / 'modis_lst_station_monthly_features.csv'
    }

    dfs = {}
    for name, filepath in files.items():
        if filepath.exists():
            print(f"Loading and pivoting {name}...")
            df = pd.read_csv(filepath)
            dfs[name] = pivot_satellite_data(df, name)
        else:
            print(f"[WARN] {filepath.name} not found. Skipping.")

    if not dfs:
        raise ValueError("No satellite datasets found to merge.")

    # Merge all pivoted datasets on spatial-temporal keys
    print("Merging datasets into unified master table...")
    master_df = list(dfs.values())[0]
    keys = ['station', 'year', 'month', 'latitude', 'longitude']
    
    for name, df in list(dfs.items())[1:]:
        master_df = pd.merge(master_df, df, on=keys, how='outer')

    out_path = out_dir / '01_multimodal_satellite_master.csv'
    master_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Fused data saved to {out_path} with shape {master_df.shape}")

if __name__ == "__main__":
    fuse_multimodal_data()