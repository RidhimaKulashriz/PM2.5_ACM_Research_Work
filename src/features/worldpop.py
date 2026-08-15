import pandas as pd
import numpy as np
import ee
import os
import math

# ---------------------------------------------------------
# 1. Initialize Google Earth Engine
# ---------------------------------------------------------
try:
    ee.Initialize(project="delhi-pm25-research")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="delhi-pm25-research")

# ---------------------------------------------------------
# 2. File Paths
# ---------------------------------------------------------
MASTER_PATH = "data/ml_ready/master_modeling_dataset.csv"
OUTPUT_FEAT_PATH = "data/03_features/feat_worldpop.csv"
OUTPUT_VAL_PATH = "data/05_validation/worldpop_validation.csv"

os.makedirs("data/03_features", exist_ok=True)
os.makedirs("data/05_validation", exist_ok=True)

# ---------------------------------------------------------
# 3. Load Master and Extract Unique Stations
# ---------------------------------------------------------
print("Loading master dataset...")
master_df = pd.read_csv(MASTER_PATH)

# Extract the 35 unique stations with their coordinates
unique_stations = master_df[['station', 'latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
print(f"Found {len(unique_stations)} unique stations.")

# ---------------------------------------------------------
# 4. GEE WorldPop Extraction Setup
# ---------------------------------------------------------
# We use the 2020 UN WPP-adjusted population count (100m)
# as a static baseline for the spatial population pressure.
print("Querying WorldPop dataset via Google Earth Engine...")
worldpop_img = ee.ImageCollection("WorldPop/GP/100m/pop") \
    .filter(ee.Filter.eq('country', 'IND')) \
    .filter(ee.Filter.eq('year', 2020)) \
    .first()

# Define scales and their areas in km2
buffers = [250, 500, 1000]
buffer_areas_km2 = {
    250: math.pi * (0.25 ** 2),
    500: math.pi * (0.50 ** 2),
    1000: math.pi * (1.00 ** 2)
}

# ---------------------------------------------------------
# 5. Extract Data per Station
# ---------------------------------------------------------
results = []

for idx, row in unique_stations.iterrows():
    station = row['station']
    lat = row['latitude']
    lon = row['longitude']
    
    point = ee.Geometry.Point([lon, lat])
    
    station_data = {'station': station}
    
    for radius in buffers:
        try:
            # Buffer the point
            geom = point.buffer(radius)
            
            # Sum the population count within the buffer
            stats = worldpop_img.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geom,
                scale=100,
                maxPixels=1e9
            ).getInfo()
            
            pop_count = stats.get('population', np.nan)
            
            # Calculate Density (people per km2)
            if pop_count is not None and not np.isnan(pop_count):
                density = pop_count / buffer_areas_km2[radius]
            else:
                density = np.nan
                
        except Exception as e:
            print(f"Error processing {station} at {radius}m: {e}")
            density = np.nan
            
        station_data[f'worldpop_density_{radius}m'] = density
        
    results.append(station_data)
    print(f"Processed: {station} ({idx+1}/{len(unique_stations)})")

# Convert station-level results to DataFrame
stations_pop_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 6. Broadcast to Panel Format (1,615 rows)
# ---------------------------------------------------------
# We merge the static population data back to the master's shape
# preserving station, year, and month.
print("Broadcasting spatial features to the station-month panel...")
panel_keys = master_df[['station', 'year', 'month']].copy()

final_worldpop_df = pd.merge(
    panel_keys, 
    stations_pop_df, 
    on='station', 
    how='left', 
    validate='many_to_one' # Many panel rows to one station record
)

# ---------------------------------------------------------
# 7. Validation Checks
# ---------------------------------------------------------
print("Running validations...")
val_records = []

# 1. Row count check
row_count_pass = len(final_worldpop_df) == len(master_df)
val_records.append({'metric': 'row_count_match', 'status': 'PASS' if row_count_pass else 'FAIL', 'value': len(final_worldpop_df)})

# 2. Station count check
station_count_pass = final_worldpop_df['station'].nunique() == 35
val_records.append({'metric': 'station_count_match', 'status': 'PASS' if station_count_pass else 'FAIL', 'value': final_worldpop_df['station'].nunique()})

# 3. Missing values check
missing_250 = final_worldpop_df['worldpop_density_250m'].isna().sum()
missing_500 = final_worldpop_df['worldpop_density_500m'].isna().sum()
missing_1000 = final_worldpop_df['worldpop_density_1000m'].isna().sum()

val_records.append({'metric': 'missing_250m', 'status': 'PASS' if missing_250 == 0 else 'WARNING', 'value': missing_250})
val_records.append({'metric': 'missing_500m', 'status': 'PASS' if missing_500 == 0 else 'WARNING', 'value': missing_500})
val_records.append({'metric': 'missing_1000m', 'status': 'PASS' if missing_1000 == 0 else 'WARNING', 'value': missing_1000})

# 4. Negative values check
min_density = final_worldpop_df[['worldpop_density_250m', 'worldpop_density_500m', 'worldpop_density_1000m']].min().min()
no_negatives = min_density >= 0
val_records.append({'metric': 'no_negative_density', 'status': 'PASS' if no_negatives else 'FAIL', 'value': min_density})

val_df = pd.DataFrame(val_records)

# ---------------------------------------------------------
# 8. Save Outputs
# ---------------------------------------------------------
final_worldpop_df.to_csv(OUTPUT_FEAT_PATH, index=False)
val_df.to_csv(OUTPUT_VAL_PATH, index=False)

print(f"\nSuccess! Files saved:")
print(f"1. Features: {OUTPUT_FEAT_PATH} ({final_worldpop_df.shape[0]} rows, {final_worldpop_df.shape[1]} columns)")
print(f"2. Validation: {OUTPUT_VAL_PATH}")
print("\nValidation Summary:")
print(val_df)