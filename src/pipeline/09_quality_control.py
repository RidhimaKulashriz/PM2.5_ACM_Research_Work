import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

def run_quality_control():
    base_dir = Path(__file__).resolve().parent.parent.parent
    in_file = base_dir / 'data' / 'processed' / 'pipeline' / '01_multimodal_satellite_master.csv'
    out_file = base_dir / 'data' / 'processed' / 'pipeline' / '02_satellite_qc_imputed.csv'

    print("\n--- PHASE 8: QUALITY CONTROL & IMPUTATION ---")
    df = pd.read_csv(in_file)
    initial_rows, initial_cols = df.shape
    
    # 1. Drop highly sparse features (>30% missing)
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.30].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} features with >30% missing data: {cols_to_drop}")

    # 2. Impute remaining missing values using KNN
    # We only impute numeric features, keeping metadata untouched
    meta_cols = ['station', 'year', 'month']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    print("Running KNN Imputation (k=5) for remaining gaps...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # Fit and transform only on the feature columns
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # 3. Final Validation
    assert df.isnull().sum().sum() == 0, "Null values remain after imputation!"
    
    df.to_csv(out_file, index=False)
    print(f"[SUCCESS] QC complete. Shape transformed from ({initial_rows}, {initial_cols}) to {df.shape}")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    run_quality_control()