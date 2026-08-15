import os
import sys
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
V1_PATH = "data/ml_ready/master_modeling_dataset.csv"
V2_PATH = "data/ml_ready/master_modeling_dataset_v2.csv"

FEATURE_PATHS = {
    'ERA5': "data/03_features/feat_era5_met.csv",
    'WorldPop': "data/03_features/feat_worldpop.csv",
    'OSM': "data/03_features/feat_osm_roads.csv",
    'WorldCover': "data/03_features/feat_worldcover.csv"
}

VAL_DIR = "data/05_validation/modelling_v2"
os.makedirs(VAL_DIR, exist_ok=True)

# ==============================================================================
# 1. LOAD & VERIFY V1 MASTER (IMMUTABLE)
# ==============================================================================
print(f"Loading V1 Master: {V1_PATH}")
if not os.path.exists(V1_PATH):
    raise FileNotFoundError(f"[ERROR] V1 master missing at {V1_PATH}")

v1_df = pd.read_csv(V1_PATH)
V1_ROWS = len(v1_df)
V1_STATIONS = v1_df['station'].nunique()
print(f"V1 rows: {V1_ROWS} | V1 stations: {V1_STATIONS}")

# V1 Integrity check
v1_dupes = v1_df.duplicated(subset=['station', 'year', 'month']).sum()
if v1_dupes > 0:
    raise ValueError(f"[FATAL] V1 Master contains {v1_dupes} duplicate keys!")

# ==============================================================================
# 2. MERGE ENGINE
# ==============================================================================
v2_df = v1_df.copy()
merge_logs = []
duplicate_logs = []

def detect_keys(df, name):
    keys = ['station']
    if 'year' in df.columns:
        keys.append('year')
    if 'month' in df.columns:
        keys.append('month')
    return keys

for feat_name, path in FEATURE_PATHS.items():
    print(f"\n--- Integrating {feat_name} ---")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Required feature dataset missing: {path}")
    
    feat_df = pd.read_csv(path)
    keys = detect_keys(feat_df, feat_name)
    print(f"Detected Keys: {keys}")
    
    # 1. Duplicate check in feature file
    dupes = feat_df.duplicated(subset=keys).sum()
    if dupes > 0:
        duplicate_logs.append({'file': feat_name, 'key_level': str(keys), 'duplicates': dupes})
        raise ValueError(f"[FATAL] {feat_name} contains {dupes} duplicate rows on keys {keys}!")
    
    # 2. Collision detection
    overlap = set(v2_df.columns).intersection(set(feat_df.columns)) - set(keys)
    if overlap:
        print(f"Warning: Overlapping columns detected in {feat_name}: {overlap}")
        for col in overlap:
            # Check if values match for overlapping keys
            test_merge = v2_df[keys + [col]].merge(feat_df[keys + [col]], on=keys, how='left')
            col_x, col_y = f"{col}_x", f"{col}_y"
            
            # Allow dropping if they are effectively identical (ignoring NA)
            mismatch = test_merge[col_x].notna() & test_merge[col_y].notna() & (test_merge[col_x] != test_merge[col_y])
            if mismatch.any():
                raise ValueError(f"[FATAL] Collision on column '{col}' with disagreeing values between V1 and {feat_name}.")
            else:
                print(f"Overlapping column '{col}' values are identical. Dropping from incoming {feat_name}.")
                feat_df = feat_df.drop(columns=[col])

    # 3. Merge Execution
    left_rows = len(v2_df)
    right_rows = len(feat_df)
    
    validate_type = 'one_to_one' if keys == ['station', 'year', 'month'] else 'many_to_one'
    
    v2_df = pd.merge(
        v2_df, 
        feat_df, 
        on=keys, 
        how='left', 
        validate=validate_type
    )
    
    result_rows = len(v2_df)
    status = "SUCCESS" if result_rows == left_rows else "FAILED"
    
    merge_logs.append({
        'merge_name': feat_name,
        'left_rows': left_rows,
        'right_rows': right_rows,
        'result_rows': result_rows,
        'key': str(keys),
        'validation_status': status
    })
    
    print(f"Rows before: {left_rows} | Rows after: {result_rows} | Status: {status}")
    
    if result_rows != V1_ROWS:
        raise ValueError(f"[FATAL] Row count altered during {feat_name} merge. Expected {V1_ROWS}, got {result_rows}.")

# ==============================================================================
# 3. SANITY VALIDATIONS
# ==============================================================================
print("\n--- Running Sanity Validations ---")

# 1. PM2.5 exists and numeric
assert 'pm25' in v2_df.columns, "PM2.5 missing!"
assert pd.api.types.is_numeric_dtype(v2_df['pm25']), "PM2.5 is not numeric!"

# 2. ERA5 plausibility
# era5_temp_mean is already stored in degrees Celsius
if 'era5_temp_mean' in v2_df.columns:
    temp = v2_df['era5_temp_mean'].dropna()

    assert temp.between(
        -10,
        60
    ).all(), (
        "ERA5 Temperature outside plausible Celsius range "
        "(-10°C to 60°C)!"
    )

if 'era5_rh_mean' in v2_df.columns:
    rh = v2_df['era5_rh_mean'].dropna()

    assert rh.between(
        0,
        100
    ).all(), (
        "ERA5 Relative Humidity outside [0,100]!"
    )

if 'era5_wind_speed_mean' in v2_df.columns:
    wind = v2_df['era5_wind_speed_mean'].dropna()

    assert (
        wind >= 0
    ).all(), (
        "ERA5 Wind Speed contains negative values!"
    )

if 'era5_blh_mean' in v2_df.columns:
    blh = v2_df['era5_blh_mean'].dropna()

    assert (
        blh >= 0
    ).all(), (
        "ERA5 Boundary Layer Height contains negative values!"
    )

# 3. WorldPop > 0
wp_cols = [c for c in v2_df.columns if 'population' in c]
for c in wp_cols:
    assert (v2_df[c].dropna() >= 0).all(), f"Negative population in {c}!"

# 4 & 5. OSM > 0 and Major <= Total
# OSM road density validation
osm_cols = [
    c for c in v2_df.columns
    if 'road_density' in c
]

for c in osm_cols:
    values = v2_df[c].dropna()

    assert (
        values >= 0
    ).all(), (
        f"Negative road density found in {c}!"
    )

if (
    'major_road_density_1000m' in v2_df.columns
    and 'road_density_1000m' in v2_df.columns
):
    osm_compare = v2_df[
        [
            'major_road_density_1000m',
            'road_density_1000m'
        ]
    ].dropna()

    assert (
        osm_compare['major_road_density_1000m']
        <=
        osm_compare['road_density_1000m'] + 1e-5
    ).all(), (
        "Major road density exceeds total road density!"
    )

if 'major_road_density_1000m' in v2_df.columns and 'road_density_1000m' in v2_df.columns:
    # Small float tolerance applied
    assert (v2_df['major_road_density_1000m'] <= v2_df['road_density_1000m'] + 1e-5).all(), "Major road density exceeds total road density!"

# 6. WorldCover between [0,1]
wc_cols = [
    c for c in v2_df.columns
    if c.startswith("worldcover_")
    and "_frac_" in c
]

for c in wc_cols:
    values = v2_df[c].dropna()

    assert values.between(
        0,
        1
    ).all(), (
        f"WorldCover fraction outside [0,1] in {c}!"
    )
# 7-10. Row, Station, and Duplicate checks are performed at final assertions.

# ==============================================================================
# 4. MISSINGNESS AUDIT
# ==============================================================================
new_cols = [c for c in v2_df.columns if c not in v1_df.columns]
missing_data = []
for col in new_cols:
    m_count = v2_df[col].isna().sum()
    missing_data.append({
        'feature': col,
        'missing_count': m_count,
        'missing_percent': round((m_count / V1_ROWS) * 100, 2)
    })
missing_df = pd.DataFrame(missing_data)

# ==============================================================================
# 5. FEATURE GROUPS AUDIT
# ==============================================================================
feature_groups = []
for c in v2_df.columns:
    if c == 'pm25':
        group = 'OUTCOME'
    elif any(kw in c.lower() for kw in ['ndvi', 'evi', 'ndwi']):
        group = 'TREATMENT / GREEN COVER'
    elif 'no2' in c.lower():
        group = 'POLLUTION COVARIATES'
    elif any(kw in c.lower() for kw in ['era5', 'temp', 'rh_', 'wind', 'blh', 'lst']):
        group = 'METEOROLOGY'
    elif any(kw in c.lower() for kw in ['population', 'road_density']):
        group = 'URBAN CONTEXT'
    elif 'worldcover' in c.lower():
        group = 'LAND-COVER BASELINE'
    elif c in ['year', 'month', 'station', 'latitude', 'longitude'] or 'season' in c.lower() or 'month_' in c.lower():
        group = 'SPATIO-TEMPORAL'
    else:
        group = 'UNCLASSIFIED / OTHER'
        
    feature_groups.append({'feature': c, 'group': group})
group_df = pd.DataFrame(feature_groups)

# ==============================================================================
# 6. FINAL INTEGRITY ASSERTIONS
# ==============================================================================
assert len(v2_df) == V1_ROWS, "[FATAL] Final row count mismatch."
assert v2_df['station'].nunique() == V1_STATIONS, "[FATAL] Final station count mismatch."
assert v2_df.duplicated(subset=['station', 'year', 'month']).sum() == 0, "[FATAL] Final V2 contains duplicate keys."

# Assert key exactness
v1_keys = v1_df[['station', 'year', 'month']].sort_values(by=['station', 'year', 'month']).reset_index(drop=True)
v2_keys = v2_df[['station', 'year', 'month']].sort_values(by=['station', 'year', 'month']).reset_index(drop=True)
assert v1_keys.equals(v2_keys), "[FATAL] V2 key set does not exactly match V1 key set."

# ==============================================================================
# 7. SAVE OUTPUTS
# ==============================================================================
print("\n--- Saving V2 and Validation Logs ---")
# Validation Logs
merge_val_df = pd.DataFrame(merge_logs)
# Prepend global counts
global_stats = pd.DataFrame([{
    'merge_name': 'GLOBAL_STATS',
    'left_rows': V1_ROWS,
    'right_rows': len(v2_df),
    'result_rows': V1_ROWS,
    'key': f"Stations: {V1_STATIONS}",
    'validation_status': 'PASS'
}])
merge_val_df = pd.concat([global_stats, merge_val_df], ignore_index=True)

merge_val_df.to_csv(os.path.join(VAL_DIR, "merge_validation.csv"), index=False)
missing_df.to_csv(os.path.join(VAL_DIR, "missingness.csv"), index=False)
group_df.to_csv(os.path.join(VAL_DIR, "feature_groups.csv"), index=False)

if not duplicate_logs:
    pd.DataFrame(columns=['file', 'key_level', 'duplicates']).to_csv(os.path.join(VAL_DIR, "duplicates.csv"), index=False)
else:
    pd.DataFrame(duplicate_logs).to_csv(os.path.join(VAL_DIR, "duplicates.csv"), index=False)

# SAVE V2
v2_df.to_csv(V2_PATH, index=False)

print("\n============================================================")
print("FINAL TERMINAL OUTPUT")
print("============================================================")
print(f"V1 rows: {V1_ROWS}")
print(f"V1 stations: {V1_STATIONS}")
print(f"\nV2 rows: {len(v2_df)}")
print(f"V2 stations: {v2_df['station'].nunique()}")
print(f"Number of added columns: {len(new_cols)}")
print(f"Total columns: {len(v2_df.columns)}")
print(f"Missing feature values (Total across new cols): {missing_df['missing_count'].sum()}")
print(f"Duplicate keys: 0")
print("\nV2 successfully created.")
print("============================================================")