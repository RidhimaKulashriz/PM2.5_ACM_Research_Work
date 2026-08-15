import os
import sys
import ee
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MASTER_PATH = "data/ml_ready/master_modeling_dataset.csv"
OUTPUT_FEAT_PATH = "data/03_features/feat_worldcover.csv"
OUTPUT_VAL_PATH = "data/05_validation/worldcover_validation.csv"
GEE_PROJECT = "delhi-pm25-research"

# WorldCover Config
ASSET_ID = "ESA/WorldCover/v200"
BUFFERS = [100, 250, 500, 1000]
# GEE frequencyHistogram string keys mapped to class names
TARGET_CLASSES = {
    '50': 'built',
    '30': 'grass',
    '40': 'cropland',
    '80': 'water'
}

# Directories
os.makedirs("data/03_features", exist_ok=True)
os.makedirs("data/05_validation", exist_ok=True)

# ==============================================================================
# 1. INITIALIZE GEE
# ==============================================================================
print(f"Initializing Earth Engine with project '{GEE_PROJECT}'...")
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception as e:
    print(f"EE Initialization failed. Please authenticate using `earthengine authenticate`.")
    raise e

# ==============================================================================
# 2. LOAD MASTER DATASET
# ==============================================================================
print(f"Loading master dataset from {MASTER_PATH}...")
master_df = pd.read_csv(MASTER_PATH)
unique_stations = master_df[['station', 'latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
print(f"Master rows: {len(master_df)} | Unique stations: {len(unique_stations)}")

# ==============================================================================
# 3. GEE EXTRACTION (SERVER-SIDE)
# ==============================================================================
print("Extracting WorldCover 2021 composition from Earth Engine...")

# ==============================================================================
# LOAD ESA WORLDCOVER 2021 v200
# ==============================================================================

print(
    f"Loading ESA WorldCover from '{ASSET_ID}'..."
)

worldcover_collection = ee.ImageCollection(
    ASSET_ID
)

image_count = worldcover_collection.size().getInfo()

print(
    f"WorldCover collection image count: {image_count}"
)

if image_count == 0:
    raise RuntimeError(
        f"Earth Engine returned zero images for {ASSET_ID}. "
        "Check the asset ID and GEE access."
    )

worldcover = (
    worldcover_collection
    .first()
    .select("Map")
)

print(
    "WorldCover 2021 image loaded successfully."
)

# Create EE FeatureCollection of stations
features = []
for _, row in unique_stations.iterrows():
    point = ee.Geometry.Point([row['longitude'], row['latitude']])
    feat = ee.Feature(point, {'station': row['station']})
    features.append(feat)
station_fc = ee.FeatureCollection(features)

station_results = []

for radius in BUFFERS:
    print(f"Processing {radius}m buffer...")
    
    # Buffer the stations
    # Note: ee.Geometry.buffer(radius) defaults to projection units (meters in web mercator/spherical)
    buffered_fc = station_fc.map(lambda f: f.buffer(radius))
    
    # Calculate frequency histogram of classes
    histograms = worldcover.reduceRegions(
        collection=buffered_fc,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=10,
        crs='EPSG:32643',
        tileScale=4,
        maxPixelsPerRegion=100_000,
    )
    
    # Fetch to client
    result_list = histograms.getInfo()['features']
    
    for item in result_list:
        station_name = item['properties']['station']
        histogram = item['properties'].get('histogram', {})
        
        # Calculate total valid pixels (sum of all classes present, not just target classes)
        total_pixels = sum(histogram.values())
        
        # Calculate fractions
        buffer_data = {'station': station_name}
        for code, name in TARGET_CLASSES.items():
            col_name = f'worldcover_2021_{name}_frac_{radius}m'
            
            if total_pixels == 0:
                buffer_data[col_name] = pd.NA
            else:
                class_pixels = histogram.get(code, 0)
                buffer_data[col_name] = class_pixels / total_pixels
                
        # Store total pixels for validation
        buffer_data[f'_valid_pixels_{radius}m'] = total_pixels
                
        # Append or update station record
        existing = next((r for r in station_results if r['station'] == station_name), None)
        if existing:
            existing.update(buffer_data)
        else:
            station_results.append(buffer_data)

features_df = pd.DataFrame(station_results)

# Drop tracking pixel counts after calculation is done, keeping them only for logging if needed
# We keep them out of the final broadcast to maintain clean ML features
pixel_cols = [c for c in features_df.columns if '_valid_pixels_' in c]
valid_pixel_df = features_df[['station'] + pixel_cols].copy()
features_df = features_df.drop(columns=pixel_cols)

print("\nEarth Engine extraction complete.")

# ==============================================================================
# 4. BROADCAST TO PANEL
# ==============================================================================
print("Broadcasting static 2021 WorldCover baseline to station-month panel...")
panel_keys = master_df[['station', 'year', 'month']].copy()
final_out_df = pd.merge(
    panel_keys, 
    features_df, 
    on='station', 
    how='left', 
    validate='many_to_one'
)

# ==============================================================================
# 5. VALIDATION
# ==============================================================================
print("Running validations...")
val_records = []

# Basic Counts
val_records.append({'metric': 'master_row_count', 'value': len(master_df)})
val_records.append({'metric': 'output_row_count', 'value': len(final_out_df)})
val_records.append({'metric': 'master_station_count', 'value': master_df['station'].nunique()})
val_records.append({'metric': 'output_station_count', 'value': final_out_df['station'].nunique()})

# Keys
dupes = final_out_df.duplicated(subset=['station', 'year', 'month']).sum()
val_records.append({'metric': 'duplicate_station_year_month_count', 'value': dupes})

# Missing Data
missing_feats = final_out_df.isna().sum().sum()
val_records.append({'metric': 'missing_feature_count', 'value': missing_feats})

# Fraction Range
frac_cols = [c for c in final_out_df.columns if 'frac' in c]
val_records.append({'metric': 'minimum_fraction', 'value': final_out_df[frac_cols].min().min()})
val_records.append({'metric': 'maximum_fraction', 'value': final_out_df[frac_cols].max().max()})

# Out of bounds check
out_of_bounds = ((final_out_df[frac_cols] < 0) | (final_out_df[frac_cols] > 1)).sum().sum()
val_records.append({'metric': 'fraction_values_outside_[0,1]', 'value': out_of_bounds})

# Complete stations
feature_cols = [
    c for c in features_df.columns
    if "_frac_" in c
]

complete_stations = (
    features_df[feature_cols]
    .notna()
    .all(axis=1)
    .sum()
)
val_records.append({'metric': 'stations_with_complete_WorldCover_features', 'value': complete_stations})

# Asset/Metadata
val_records.append({'metric': 'WorldCover_asset_ID', 'value': ASSET_ID})
val_records.append({'metric': 'WorldCover_year', 'value': '2021'})
val_records.append({'metric': 'WorldCover_spatial_resolution', 'value': '10m'})
val_records.append({'metric': 'GEE_project', 'value': GEE_PROJECT})
val_records.append({'metric': 'CRS_used_for_station_buffers', 'value': 'EPSG:4326/spherical_meters (EE Native)'})

# Number of Valid Pixels (Using max 1000m buffer from tracked stats)
total_1000m_pixels = valid_pixel_df['_valid_pixels_1000m'].sum()
val_records.append({'metric': 'number_of_valid_pixels_used_1000m_buffers', 'value': total_1000m_pixels})

val_records.append({'metric': 'Scientific_Limitation_Note', 
                    'value': 'ESA WorldCover 2021 is used as a static pre-study land-cover baseline. It does not represent land-cover changes during 2022-2025.'})

val_df = pd.DataFrame(val_records)

# ==============================================================================
# 6. INTEGRITY ASSERTS & SAVE
# ==============================================================================
# Strict asserts exactly as requested
assert len(final_out_df) == len(master_df), "Output length does not match Master!"
assert final_out_df['station'].nunique() == master_df['station'].nunique(), "Output station count does not match Master!"
assert dupes == 0, "Duplicate station-year-month keys detected!"
assert out_of_bounds == 0, "Fraction values exist outside [0, 1] interval!"

# Save
final_out_df.to_csv(OUTPUT_FEAT_PATH, index=False)
val_df.to_csv(OUTPUT_VAL_PATH, index=False)

print("\nSUCCESS! Dataset frozen.")
print(f"1. Saved Features: {OUTPUT_FEAT_PATH} ({final_out_df.shape[0]} rows)")
print(f"2. Saved Validation: {OUTPUT_VAL_PATH}")
print("\nValidation Summary:")
print(val_df.to_string(index=False))