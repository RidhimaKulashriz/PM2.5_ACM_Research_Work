import pandas as pd
from pathlib import Path

def prepare_ml_dataset():
    base_dir = Path(__file__).resolve().parent.parent.parent
    features_file = base_dir / 'data' / 'processed' / 'pipeline' / '03_engineered_features.csv'
    ml_dir = base_dir / 'data' / 'ml_ready'
    ml_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- PHASE 10: TARGET INTEGRATION & SPLIT ---")
    df_features = pd.read_csv(features_file)

    # List of candidate paths where CPCB target data might live
    possible_cpcb_paths = [
        base_dir / 'data' / 'processed' / 'cpcb'/ 'cleaned_cpcb_monthly.csv',
        base_dir / 'data' / 'processed' / 'cpcb' / 'cpcb_monthly.csv',
        base_dir / 'data' / 'processed' / 'cpcb_monthly.csv',
        base_dir / 'data' / 'processed' / 'cleaned_cpcb.csv',
        base_dir / 'data' / 'cpcb_monthly.csv'
    ]

    cpcb_file = None
    for path in possible_cpcb_paths:
        if path.exists():
            cpcb_file = path
            break

    if not cpcb_file:
        print("\n[!] COULD NOT FIND CPCB TARGET FILE.")
        print("Please verify where your cleaned monthly PM2.5 CSV is saved in the data/ folder.")
        df_features.to_csv(ml_dir / 'X_features_master.csv', index=False)
        print(f"Saved feature matrix (without targets) to {ml_dir / 'X_features_master.csv'}")
        return

    print(f"Found CPCB Target File at: {cpcb_file}")
    df_cpcb = pd.read_csv(cpcb_file)

    # Standardize target column name
    pm25_col = [c for c in df_cpcb.columns if 'pm25' in c.lower() or 'pm2.5' in c.lower()]
    if not pm25_col:
        raise ValueError(f"Could not find a PM2.5 target column in {cpcb_file}. Columns found: {list(df_cpcb.columns)}")
    
    df_cpcb = df_cpcb.rename(columns={pm25_col[0]: 'pm25'})

    # Ensure required join keys exist
    join_keys = ['station', 'year', 'month']
    for key in join_keys:
        if key not in df_cpcb.columns:
            raise KeyError(f"Missing required join key '{key}' in {cpcb_file}")

    print("Merging satellite features with CPCB PM2.5 Targets...")
    df_merged = pd.merge(df_features, df_cpcb[['station', 'year', 'month', 'pm25']], 
                         on=join_keys, 
                         how='inner')
                         
    df_merged = df_merged.dropna(subset=['pm25'])

    # Out-of-Time Train/Test Split (<=2023 Train, >=2024 Test)
    train = df_merged[df_merged['year'] <= 2023]
    test = df_merged[df_merged['year'] >= 2024]
    
    print(f"\n[SUCCESS] Dataset Matched Successfully:")
    print(f"  • Total Matched Samples : {len(df_merged)}")
    print(f"  • Train Set (<= 2023)  : {len(train)} rows")
    print(f"  • Test Set  (>= 2024)  : {len(test)} rows")
    
    df_merged.to_csv(ml_dir / 'master_modeling_dataset.csv', index=False)
    train.to_csv(ml_dir / 'train_set.csv', index=False)
    test.to_csv(ml_dir / 'test_set.csv', index=False)
    print(f"\nML-ready files saved in: {ml_dir}")

if __name__ == "__main__":
    prepare_ml_dataset()