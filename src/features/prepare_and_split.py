import pandas as pd
import numpy as np
import os
import sys

def main():
    # =========================================================================
    # 1. CONFIGURATION & SETUP
    # =========================================================================
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    input_file = "data/ml_ready/master_modeling_dataset_v2.csv"
    out_dir = "data/modeling/splits"
    os.makedirs(out_dir, exist_ok=True)
    
    # =========================================================================
    # 2. LOAD AND VALIDATE BASE DATASET
    # =========================================================================
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not locate {input_file}")
        sys.exit(1)
        
    if len(df) != 1615:
        print(f"CRITICAL ERROR: Expected 1615 rows, found {len(df)}")
        sys.exit(1)
        
    key_cols = ['station', 'year', 'month']
    if df.duplicated(subset=key_cols).any():
        print("CRITICAL ERROR: Duplicates found in base dataset keys.")
        sys.exit(1)
        
    # =========================================================================
    # 3. IDENTIFY & ISOLATE IIT_DELHI
    # =========================================================================
    iit_delhi_idx = df[df['station'] == 'IIT_Delhi'].index
    if len(iit_delhi_idx) != 1:
        print(f"CRITICAL ERROR: Expected exactly 1 IIT_Delhi row, found {len(iit_delhi_idx)}")
        sys.exit(1)
        
    # Exclude IIT_Delhi from the sampling pool (it will be forced into TRAIN later)
    pool_df = df.drop(index=iit_delhi_idx)
    
    # =========================================================================
    # 4. HIERARCHICAL QUOTA ALLOCATION (323 EXACT TEST ROWS)
    # =========================================================================
    group_counts = pool_df.groupby(['station', 'year']).size().reset_index(name='n')
    
    target_test = 323
    ratio = target_test / len(pool_df)
    
    group_counts['ideal_test'] = group_counts['n'] * ratio
    group_counts['base_test'] = np.floor(group_counts['ideal_test']).astype(int)
    
    # Ensure minimum 1 observation if the group has at least 2 rows
    group_counts['current_test'] = np.where(
        (group_counts['base_test'] == 0) & (group_counts['n'] >= 2),
        1,
        group_counts['base_test']
    )
    
    group_counts['fractional_remainder'] = group_counts['ideal_test'] - group_counts['base_test']
    
    # Resolve any difference between current allocation and exact target
    remaining_test = target_test - group_counts['current_test'].sum()
    
    if remaining_test > 0:
        group_counts = group_counts.sort_values(
            by=['fractional_remainder', 'station', 'year'],
            ascending=[False, True, True]
        )
        for idx in group_counts.index:
            if remaining_test <= 0:
                break
            if group_counts.loc[idx, 'current_test'] < group_counts.loc[idx, 'n']:
                group_counts.loc[idx, 'current_test'] += 1
                remaining_test -= 1
                
    elif remaining_test < 0:
        group_counts = group_counts.sort_values(
            by=['fractional_remainder', 'station', 'year'],
            ascending=[True, True, True]
        )
        for idx in group_counts.index:
            if remaining_test >= 0:
                break
            if group_counts.loc[idx, 'current_test'] > 0:
                group_counts.loc[idx, 'current_test'] -= 1
                remaining_test += 1
                
    # =========================================================================
    # 5. REPRODUCIBLE STRATIFIED SAMPLING
    # =========================================================================
    test_indices = []
    group_counts = group_counts.sort_values(['station', 'year'])
    
    for _, row in group_counts.iterrows():
        st = row['station']
        yr = row['year']
        k = int(row['current_test'])
        
        group_idx = pool_df[(pool_df['station'] == st) & (pool_df['year'] == yr)].index
        sampled = pd.Series(group_idx).sample(n=k, random_state=RANDOM_SEED, replace=False)
        test_indices.extend(sampled.tolist())
        
    # Split using pure indices (prevents dataframe fragmentation)
    test_df = df.loc[test_indices].copy()
    train_df = df.drop(index=test_indices).copy() # Inherently includes IIT_Delhi
    
    # =========================================================================
    # 6. HARD VALIDATION
    # =========================================================================
    len_train = len(train_df)
    len_test = len(test_df)
    
    train_keys = set(train_df.set_index(key_cols).index)
    test_keys = set(test_df.set_index(key_cols).index)
    base_keys = set(df.set_index(key_cols).index)
    
    train_stations = set(train_df['station'].unique())
    test_stations = set(test_df['station'].unique())
    train_years = set(train_df['year'].unique())
    test_years = set(test_df['year'].unique())
    req_years = {2022, 2023, 2024, 2025}
    
    validations = {
        "train rows == 1292": (len_train == 1292),
        "test rows == 323": (len_test == 323),
        "total rows == 1615": (len_train + len_test == 1615),
        "train fraction == 0.80": np.isclose(len_train/1615, 0.8),
        "test fraction == 0.20": np.isclose(len_test/1615, 0.2),
        "no train/test key overlap": (len(train_keys.intersection(test_keys)) == 0),
        "union exactly equals V2": (train_keys.union(test_keys) == base_keys),
        "train has no duplicate keys": not train_df.duplicated(subset=key_cols).any(),
        "test has no duplicate keys": not test_df.duplicated(subset=key_cols).any(),
        "train contains all four years": (train_years == req_years),
        "test contains all four years": (test_years == req_years),
        "train contains all 35 stations": (len(train_stations) == 35),
        "test contains exactly 34 stations": (len(test_stations) == 34),
        "IIT_Delhi is train-only": ("IIT_Delhi" in train_stations) and ("IIT_Delhi" not in test_stations)
    }
    
    failed = False
    for condition, passed in validations.items():
        if not passed:
            print(f"CRITICAL VALIDATION FAILED: {condition}")
            failed = True
            
    if failed:
        sys.exit(1)
        
    # =========================================================================
    # 7. OUTPUT GENERATION (Files & Manifests)
    # =========================================================================
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    
    # 7.1 Split Manifest
    manifest = pd.DataFrame([{
        "random_seed": RANDOM_SEED,
        "total_rows": len(df),
        "train_rows": len_train,
        "test_rows": len_test,
        "train_fraction": len_train / len(df),
        "test_fraction": len_test / len(df),
        "train_station_count": len(train_stations),
        "test_station_count": len(test_stations),
        "train_years": list(sorted(train_years)),
        "test_years": list(sorted(test_years)),
        "train_only_stations": "IIT_Delhi",
        "target_used_for_split": "FALSE"
    }])
    manifest.to_csv(os.path.join(out_dir, "split_manifest.csv"), index=False)
    
    # 7.2 Station Diagnostics
    st_train = train_df.groupby('station').size().reset_index(name='train')
    st_test = test_df.groupby('station').size().reset_index(name='test')
    diag_st = df.groupby('station').size().reset_index(name='total')
    diag_st = diag_st.merge(st_train, on='station', how='left').merge(st_test, on='station', how='left').fillna(0)
    
    diag_st['train'] = diag_st['train'].astype(int)
    diag_st['test'] = diag_st['test'].astype(int)
    
    diag_st['status'] = 'OK'
    diag_st.loc[diag_st['station'] == 'IIT_Delhi', 'status'] = 'TRAIN_ONLY_INSUFFICIENT_OBSERVATIONS'
    diag_st.to_csv(os.path.join(out_dir, "station_split_diagnostics.csv"), index=False)
    
    # 7.3 Year Diagnostics
    yr_train = train_df.groupby('year').size().reset_index(name='train')
    yr_test = test_df.groupby('year').size().reset_index(name='test')
    diag_yr = df.groupby('year').size().reset_index(name='total')
    diag_yr = diag_yr.merge(yr_train, on='year', how='left').merge(yr_test, on='year', how='left').fillna(0)
    
    diag_yr['train'] = diag_yr['train'].astype(int)
    diag_yr['test'] = diag_yr['test'].astype(int)
    diag_yr['train_fraction'] = diag_yr['train'] / diag_yr['total']
    diag_yr['test_fraction'] = diag_yr['test'] / diag_yr['total']
    diag_yr.to_csv(os.path.join(out_dir, "year_split_diagnostics.csv"), index=False)
    
    # 7.4 Season Diagnostics
    def get_season(m):
        if m in [12, 1, 2]: return 'Winter'
        elif m in [3, 4, 5, 6]: return 'Summer'
        elif m in [7, 8, 9]: return 'Monsoon'
        elif m in [10, 11]: return 'Post-monsoon'
        return 'Unknown'
        
    df['_season'] = df['month'].apply(get_season)
    train_df['_season'] = train_df['month'].apply(get_season)
    test_df['_season'] = test_df['month'].apply(get_season)
    
    se_train = train_df.groupby('_season').size().reset_index(name='train')
    se_test = test_df.groupby('_season').size().reset_index(name='test')
    diag_se = df.groupby('_season').size().reset_index(name='total')
    
    diag_se = diag_se.rename(columns={'_season': 'season'})
    se_train = se_train.rename(columns={'_season': 'season'})
    se_test = se_test.rename(columns={'_season': 'season'})
    
    diag_se = diag_se.merge(se_train, on='season', how='left').merge(se_test, on='season', how='left').fillna(0)
    diag_se['train'] = diag_se['train'].astype(int)
    diag_se['test'] = diag_se['test'].astype(int)
    diag_se.to_csv(os.path.join(out_dir, "season_split_diagnostics.csv"), index=False)
    
    # =========================================================================
    # 8. FINAL TERMINAL REPORT
    # =========================================================================
    print("============================================================")
    print("FINAL SPLIT RESULT")
    print("============================================================")
    print(f"V2 rows: 1615\n")
    print(f"Train rows: {len_train}")
    print(f"Test rows: {len_test}\n")
    print(f"Train fraction: {len_train/len(df):.6f}")
    print(f"Test fraction: {len_test/len(df):.6f}\n")
    print(f"Train stations: {len(train_stations)}")
    print(f"Test stations: {len(test_stations)}\n")
    print(f"Train years: {sorted(list(train_years))}")
    print(f"Test years: {sorted(list(test_years))}\n")
    print(f"Train-only stations: IIT_Delhi")
    print(f"Key overlap: 0")
    print(f"Row universe preserved: TRUE\n")
    print(f"FINAL STATUS:\nPASS")

if __name__ == "__main__":
    main()