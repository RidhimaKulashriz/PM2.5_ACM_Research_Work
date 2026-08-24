"""
Urban Green Cover Thresholds for PM2.5 Mitigation: A Spatial Causal Machine Learning Framework
Primary Train/Test Split Generator
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Hardcoded constants per project requirements
INPUT_DATASET = os.path.join('data', 'ml_ready', 'master_modeling_dataset_v2.csv')
OUTPUT_DIR = os.path.join('data', 'modeling_final')
TARGET_TRAIN = 1292
TARGET_TEST = 323
TARGET_TOTAL = 1615
STATION_COUNT = 35
SEED = 42

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5, 6]: return 'Summer'
    elif month in [7, 8, 9]: return 'Monsoon'
    elif month in [10, 11]: return 'Post-monsoon'
    return 'Unknown'

def generate_split(df, seed):
    """Generates the Year x Month stratified split using Largest-Remainder Quota."""
    # Isolate singleton
    singleton_mask = df['station'] == 'IIT_Delhi'
    train_singleton = df[singleton_mask].copy()
    eligible_df = df[~singleton_mask].copy()
    
    # Shuffle eligible rows deterministically
    eligible_df = eligible_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Stratification quotas
    strata = eligible_df.groupby(['year', 'month']).size().reset_index(name='count')
    total_eligible = strata['count'].sum()
    
    strata['ideal'] = (strata['count'] / total_eligible) * TARGET_TEST
    strata['floor'] = np.floor(strata['ideal']).astype(int)
    
    # Require at least 1 for non-degenerate
    strata['alloc'] = strata.apply(lambda x: max(1, x['floor']) if x['count'] > 0 else 0, axis=1)
    strata['alloc'] = strata[['alloc', 'count']].min(axis=1) # Never exceed available count
    
    remaining = TARGET_TEST - strata['alloc'].sum()
    
    # Largest Remainder allocation for remaining test slots
    strata['remainder'] = strata['ideal'] - strata['floor']
    strata.loc[strata['alloc'] > strata['floor'], 'remainder'] = -1.0 # Deprioritize already bumped
    strata.loc[strata['alloc'] >= strata['count'], 'remainder'] = -1.0 # Deprioritize maxed out
    
    if remaining > 0:
        strata = strata.sort_values('remainder', ascending=False)
        for idx in strata.index:
            if remaining <= 0: break
            if strata.loc[idx, 'alloc'] < strata.loc[idx, 'count']:
                strata.loc[idx, 'alloc'] += 1
                remaining -= 1
    elif remaining < 0:
        strata = strata.sort_values('remainder', ascending=True)
        for idx in strata.index:
            if remaining >= 0: break
            if strata.loc[idx, 'alloc'] > 1:
                strata.loc[idx, 'alloc'] -= 1
                remaining += 1
                
    # Assign rows based on quotas
    test_indices = []
    train_indices = []
    
    for _, row in strata.iterrows():
        y, m, alloc = int(row['year']), int(row['month']), int(row['alloc'])
        stratum_rows = eligible_df[(eligible_df['year'] == y) & (eligible_df['month'] == m)]
        
        test_indices.extend(stratum_rows.index[:alloc].tolist())
        train_indices.extend(stratum_rows.index[alloc:].tolist())
        
    test_df = eligible_df.loc[test_indices].copy()
    train_eligible_df = eligible_df.loc[train_indices].copy()
    
    # Combine train sets
    train_df = pd.concat([train_singleton, train_eligible_df], ignore_index=True)
    
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing split files")
    args = parser.parse_args()

    reports = []
    
    def log_val(check, status, observed, requirement):
        reports.append({'check': check, 'status': status, 'observed': observed, 'requirement': requirement})
        if status == 'FAIL':
            print(f"CRITICAL FAILURE: {check}\nObserved: {observed} | Required: {requirement}")
            sys.exit(1)

    # 1. Output File Safety Check
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    expected_outputs = ['train.csv', 'test.csv', 'split_manifest.csv', 'validation_report.csv', 'distribution_diagnostics.csv']
    if not args.overwrite:
        for file in expected_outputs:
            if os.path.exists(os.path.join(OUTPUT_DIR, file)):
                print(f"FAIL: {file} already exists. Use --overwrite to replace.")
                sys.exit(1)

    # 2. Input Dataset Check
    if not os.path.exists(INPUT_DATASET):
        log_val("V2 file exists", "FAIL", "Missing", INPUT_DATASET)
    log_val("V2 file exists", "PASS", "Found", INPUT_DATASET)

    # Read Read-Only Input
    v2_df = pd.read_csv(INPUT_DATASET)
    
    # 3. Pre-Split Data Integrity Validations
    log_val("V2 row count", "PASS" if len(v2_df) == TARGET_TOTAL else "FAIL", len(v2_df), TARGET_TOTAL)
    log_val("V2 station count", "PASS" if v2_df['station'].nunique() == STATION_COUNT else "FAIL", v2_df['station'].nunique(), STATION_COUNT)
    
    dupe_keys = v2_df.duplicated(subset=['station', 'year', 'month']).sum()
    log_val("Duplicate keys in V2", "PASS" if dupe_keys == 0 else "FAIL", dupe_keys, 0)
    
    missing_pm25 = v2_df['pm25'].isna().sum()
    log_val("Missing PM2.5", "PASS" if missing_pm25 == 0 else "FAIL", missing_pm25, 0)
    
    numeric_cols = v2_df.select_dtypes(include=[np.number]).columns
    inf_vals = np.isinf(v2_df[numeric_cols]).sum().sum()
    log_val("Infinite numeric values", "PASS" if inf_vals == 0 else "FAIL", inf_vals, 0)
    
    valid_years = {2022, 2023, 2024, 2025}
    obs_years = set(v2_df['year'].unique())
    log_val("Valid years", "PASS" if obs_years.issubset(valid_years) else "FAIL", list(obs_years), list(valid_years))
    
    obs_months = set(v2_df['month'].unique())
    log_val("Valid months", "PASS" if obs_months.issubset(set(range(1, 13))) else "FAIL", min(obs_months), "1-12")
    
    unstable_coords = (v2_df.groupby('station')[['latitude', 'longitude']].nunique() > 1).sum().sum()
    log_val("Stable coordinates", "PASS" if unstable_coords == 0 else "FAIL", unstable_coords, 0)

    # 4. Reproducibility Check
    train_rep1, test_rep1 = generate_split(v2_df, SEED)
    train_rep2, test_rep2 = generate_split(v2_df, SEED)
    keys1 = set(train_rep1.apply(lambda x: f"{x['station']}_{x['year']}_{x['month']}", axis=1))
    keys2 = set(train_rep2.apply(lambda x: f"{x['station']}_{x['year']}_{x['month']}", axis=1))
    is_reproducible = keys1 == keys2
    log_val("Reproducibility check", "PASS" if is_reproducible else "FAIL", is_reproducible, True)

    train_df, test_df = train_rep1, test_rep1

    # 5. Post-Split Metric Validations
    log_val("Train rows", "PASS" if len(train_df) == TARGET_TRAIN else "FAIL", len(train_df), TARGET_TRAIN)
    log_val("Test rows", "PASS" if len(test_df) == TARGET_TEST else "FAIL", len(test_df), TARGET_TEST)
    log_val("Train + Test union", "PASS" if len(train_df) + len(test_df) == TARGET_TOTAL else "FAIL", len(train_df) + len(test_df), TARGET_TOTAL)

    train_keys = set(train_df.apply(lambda x: f"{x['station']}_{x['year']}_{x['month']}", axis=1))
    test_keys = set(test_df.apply(lambda x: f"{x['station']}_{x['year']}_{x['month']}", axis=1))
    v2_keys = set(v2_df.apply(lambda x: f"{x['station']}_{x['year']}_{x['month']}", axis=1))
    
    overlap = len(train_keys.intersection(test_keys))
    log_val("Train/test overlap", "PASS" if overlap == 0 else "FAIL", overlap, 0)
    
    union_match = (train_keys.union(test_keys) == v2_keys)
    log_val("Union differs from V2 universe", "PASS" if union_match else "FAIL", "Match" if union_match else "Mismatch", "Match")

    log_val("Duplicate keys in Train", "PASS" if len(train_keys) == len(train_df) else "FAIL", len(train_df) - len(train_keys), 0)
    log_val("Duplicate keys in Test", "PASS" if len(test_keys) == len(test_df) else "FAIL", len(test_df) - len(test_keys), 0)

    for y in [2022, 2023, 2024, 2025]:
        log_val(f"{y} in train", "PASS" if y in train_df['year'].values else "FAIL", True, True)
        log_val(f"{y} in test", "PASS" if y in test_df['year'].values else "FAIL", True, True)

    # Eligible year-month check (non-zero test obs for every year/month present in eligible universe)
    eligible_ym = v2_df[v2_df['station'] != 'IIT_Delhi'].groupby(['year', 'month']).size()
    test_ym = test_df.groupby(['year', 'month']).size()
    zero_test_strata = sum((test_ym.reindex(eligible_ym.index, fill_value=0) == 0))
    log_val("Any eligible stratum zero test obs", "PASS" if zero_test_strata == 0 else "FAIL", zero_test_strata, 0)

    log_val("IIT_Delhi in test", "PASS" if 'IIT_Delhi' not in test_df['station'].values else "FAIL", 'IIT_Delhi' in test_df['station'].values, False)
    
    schema_match = list(train_df.columns) == list(v2_df.columns) and list(test_df.columns) == list(v2_df.columns)
    log_val("Output schemas identical to V2", "PASS" if schema_match else "FAIL", schema_match, True)

    # Diagnostics logic
    train_stations = set(train_df['station'].unique())
    test_stations = set(test_df['station'].unique())
    train_only = list(train_stations - test_stations)
    test_only = list(test_stations - train_stations)
    
    # 6. Save Files Securely
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    pd.DataFrame(reports).to_csv(os.path.join(OUTPUT_DIR, 'validation_report.csv'), index=False)
    
    # Generate distribution diagnostics
    v2_df['year_month'] = v2_df['year'].astype(str) + '-' + v2_df['month'].astype(str).str.zfill(2)
    train_df['year_month'] = train_df['year'].astype(str) + '-' + train_df['month'].astype(str).str.zfill(2)
    test_df['year_month'] = test_df['year'].astype(str) + '-' + test_df['month'].astype(str).str.zfill(2)
    
    v2_df['season'] = v2_df['month'].apply(get_season)
    train_df['season'] = train_df['month'].apply(get_season)
    test_df['season'] = test_df['month'].apply(get_season)

    dist_records = []
    dimensions = ['year', 'month', 'year_month', 'season', 'station']
    for dim in dimensions:
        t_counts = v2_df[dim].value_counts()
        tr_counts = train_df[dim].value_counts()
        te_counts = test_df[dim].value_counts()
        
        for val in t_counts.index:
            t = int(t_counts.get(val, 0))
            tr = int(tr_counts.get(val, 0))
            te = int(te_counts.get(val, 0))
            dist_records.append({
                'dimension': dim,
                'group': val,
                'total': t,
                'train': tr,
                'test': te,
                'train_fraction': round(tr / t, 4) if t > 0 else 0,
                'test_fraction': round(te / t, 4) if t > 0 else 0
            })
    pd.DataFrame(dist_records).to_csv(os.path.join(OUTPUT_DIR, 'distribution_diagnostics.csv'), index=False)

    # Generate Manifest
    manifest_data = [{
        'input_dataset': INPUT_DATASET,
        'output_directory': OUTPUT_DIR,
        'random_seed': SEED,
        'v2_rows': TARGET_TOTAL,
        'train_rows': TARGET_TRAIN,
        'test_rows': TARGET_TEST,
        'train_fraction': 0.800000,
        'test_fraction': 0.200000,
        'v2_station_count': STATION_COUNT,
        'train_station_count': len(train_stations),
        'test_station_count': len(test_stations),
        'train_years': "[2022, 2023, 2024, 2025]",
        'test_years': "[2022, 2023, 2024, 2025]",
        'train_only_stations': str(train_only),
        'test_only_stations': str(test_only),
        'target_used_for_split': False,
        'singleton_station_handling': "IIT_Delhi forced to train-only because n=1",
        'split_method': "Exact 80:20 year-month stratified largest-remainder allocation",
        'validation_status': "PASS"
    }]
    pd.DataFrame(manifest_data).to_csv(os.path.join(OUTPUT_DIR, 'split_manifest.csv'), index=False)

    # 7. Post-Write Verification
    try:
        val_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'train.csv'))
        val_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'test.csv'))
        assert len(val_train) == TARGET_TRAIN
        assert len(val_test) == TARGET_TEST
        assert len(set(val_train.columns).symmetric_difference(set(v2_df.columns.drop(['year_month', 'season'])))) == 0
        assert 'IIT_Delhi' not in val_test['station'].values
    except AssertionError:
        print("CRITICAL FAILURE: Post-write verification failed. Disk files corrupted.")
        sys.exit(1)

    # 8. Terminal Output
    print("================================================================")
    print("FINAL YEAR × MONTH STRATIFIED SPLIT")
    print("================================================================\n")
    print(f"V2 rows:\n{TARGET_TOTAL}\n")
    print(f"Train rows:\n{TARGET_TRAIN}\n")
    print(f"Test rows:\n{TARGET_TEST}\n")
    print(f"Train fraction:\n0.800000\n")
    print(f"Test fraction:\n0.200000\n")
    print(f"Train years:\n[2022, 2023, 2024, 2025]\n")
    print(f"Test years:\n[2022, 2023, 2024, 2025]\n")
    print(f"Train stations:\n{len(train_stations)}\n")
    print(f"Test stations:\n{len(test_stations)}\n")
    print(f"Train-only stations:\n{train_only}\n")
    print(f"Test-only stations:\n{test_only}\n")
    print(f"IIT_Delhi in train:\nTRUE\n")
    print(f"IIT_Delhi in test:\nFALSE\n")
    print(f"Key overlap:\n0\n")
    print(f"Exact row-universe preserved:\nTRUE\n")
    print(f"Every year-month represented in test:\nTRUE\n")
    print(f"Reproducibility:\nTRUE\n")
    print(f"V2 modified:\nFALSE\n")
    print("FINAL STATUS:\nPASS\n")
    
if __name__ == "__main__":
    main()