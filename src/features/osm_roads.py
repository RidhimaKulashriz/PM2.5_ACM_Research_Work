import os
import sys
import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point
import osmium
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PBF_PATH = "data/01_raw/osm/delhi_ncr_osm_snapshot.osm.pbf"
MASTER_PATH = "data/ml_ready/master_modeling_dataset.csv"
OUTPUT_FEAT_PATH = "data/03_features/feat_osm_roads.csv"
OUTPUT_VAL_PATH = "data/05_validation/osm_validation.csv"

# Directories
os.makedirs("data/03_features", exist_ok=True)
os.makedirs("data/05_validation", exist_ok=True)

# CRS Settings
SRC_CRS = "EPSG:4326"
PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N (Delhi NCR metric)

# Road Classifications
EXCLUDED_ROADS = ['footway', 'cycleway', 'steps', 'path', 'pedestrian', 'track', 'raceway', 'elevator']
MAJOR_ROADS = ['motorway', 'trunk', 'primary', 'secondary', 
               'motorway_link', 'trunk_link', 'primary_link', 'secondary_link']
BUFFERS = [100, 250, 500, 1000]

# ==============================================================================
# 1. PBF EXISTENCE CHECK
# ==============================================================================
if not os.path.exists(PBF_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] OSM PBF file not found!\n"
        f"Expected location: {PBF_PATH}\n"
        f"Please download the Delhi NCR OSM PBF snapshot and place it in the specified directory."
    )

PBF_SIZE_MB = os.path.getsize(PBF_PATH) / (1024 * 1024)
print(f"Verified PBF file exists: {PBF_PATH} ({PBF_SIZE_MB:.2f} MB)")

# ==============================================================================
# 2. LOAD MASTER DATASSET
# ==============================================================================
print(f"Loading master dataset from {MASTER_PATH}...")
master_df = pd.read_csv(MASTER_PATH)
unique_stations = master_df[['station', 'latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
print(f"Master rows: {len(master_df)} | Unique stations: {len(unique_stations)}")

# ==============================================================================
# 3. PYOSMIUM EXTRACTION
# ==============================================================================

print(
    "Streaming OSM PBF with PyOsmium "
    "(this may take a minute depending on PBF size)..."
)


class HighwayHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()

        self.wkbfab = osmium.geom.WKBFactory()

        self.roads = []

        self.skipped_geometries = 0

    def way(self, w):

        if "highway" not in w.tags:
            return

        highway_type = w.tags["highway"]

        if highway_type in EXCLUDED_ROADS:
            return

        try:

            # IMPORTANT:
            # PyOsmium 4.3.1 expects the WayNodeList here.
            # The official geometry examples use:
            #     create_linestring(o.nodes)
            wkb_data = self.wkbfab.create_linestring(
                w.nodes
            )

            self.roads.append(
                {
                    "osmid": w.id,
                    "highway": highway_type,
                    "geometry": wkb_data,
                }
            )

        except (
            RuntimeError,
            osmium.InvalidLocationError,
        ):

            self.skipped_geometries += 1


handler = HighwayHandler()

handler.apply_file(
    PBF_PATH,
    locations=True,
)

print(
    f"Successfully extracted "
    f"{len(handler.roads)} road geometries."
)

print(
    f"Skipped invalid/incomplete geometries: "
    f"{handler.skipped_geometries}"
)


# ==============================================================================
# 4. GEOMETRY PROCESSING
# ==============================================================================

print(
    "Constructing GeoDataFrame and projecting "
    "to metric CRS..."
)

if not handler.roads:
    raise RuntimeError(
        "No highway geometries were successfully extracted "
        "from the OSM PBF."
    )


# PyOsmium WKBFactory returns WKB geometry data.
try:

    geometries = [
        wkb.loads(
            record["geometry"]
        )
        for record in handler.roads
    ]

except Exception:

    # Some builds may expose the WKB as hexadecimal text.
    geometries = [
        wkb.loads(
            record["geometry"],
            hex=True,
        )
        for record in handler.roads
    ]


roads_gdf = gpd.GeoDataFrame(
    handler.roads,
    geometry=geometries,
    crs=SRC_CRS,
)

# Remove invalid/empty geometry records.
roads_gdf = roads_gdf[
    roads_gdf.geometry.notna()
    & ~roads_gdf.geometry.is_empty
].copy()

if roads_gdf.empty:
    raise RuntimeError(
        "No valid road geometries remain after WKB conversion."
    )


# Project to UTM Zone 43N.
roads_gdf = roads_gdf.to_crs(
    PROJECT_CRS
)

print(
    f"Valid road geometries after conversion: "
    f"{len(roads_gdf)}"
)

print(
    f"Projected CRS: {roads_gdf.crs}"
)


# Major roads.
major_roads_gdf = roads_gdf[
    roads_gdf["highway"].isin(
        MAJOR_ROADS
    )
].copy()

print(
    f"Major-road geometries: "
    f"{len(major_roads_gdf)}"
)
# ==============================================================================
# 5. SPATIAL INTERSECTION (STATION BUFFERS)
# ==============================================================================
print("Processing spatial intersections for 35 stations...")
station_results = []

# Project stations to metric
stations_gdf = gpd.GeoDataFrame(
    unique_stations, 
    geometry=gpd.points_from_xy(unique_stations.longitude, unique_stations.latitude),
    crs=SRC_CRS
).to_crs(PROJECT_CRS)

# Use spatial index on roads for faster clipping
sindex_total = roads_gdf.sindex
sindex_major = major_roads_gdf.sindex

for idx, row in stations_gdf.iterrows():
    station_name = row['station']
    station_data = {'station': station_name}
    point_geom = row.geometry
    
    try:
        for radius in BUFFERS:
            buffer_geom = point_geom.buffer(radius)
            buffer_area_km2 = buffer_geom.area / 1e6
            
            # Clip total roads
            possible_matches_idx = list(sindex_total.intersection(buffer_geom.bounds))
            possible_matches = roads_gdf.iloc[possible_matches_idx]
            clipped_total = possible_matches.clip(buffer_geom)
            
            if clipped_total.empty:
                density = 0.0
            else:
                length_km = clipped_total.geometry.length.sum() / 1000.0
                density = length_km / buffer_area_km2
                
            station_data[f'road_density_{radius}m'] = density
            
            # Clip major roads (1000m only)
            if radius == 1000:
                possible_major_idx = list(sindex_major.intersection(buffer_geom.bounds))
                possible_major = major_roads_gdf.iloc[possible_major_idx]
                clipped_major = possible_major.clip(buffer_geom)
                
                if clipped_major.empty:
                    major_density = 0.0
                else:
                    major_length_km = clipped_major.geometry.length.sum() / 1000.0
                    major_density = major_length_km / buffer_area_km2
                    
                station_data['major_road_density_1000m'] = major_density
                
    except Exception as e:
        print(f"Warning: Failed processing {station_name}: {str(e)}")
        for r in BUFFERS:
            station_data[f'road_density_{r}m'] = float('nan')
        station_data['major_road_density_1000m'] = float('nan')
        
    station_results.append(station_data)
    sys.stdout.write(f"\rProcessed {idx + 1}/{len(stations_gdf)} stations")
    sys.stdout.flush()

print("\nSpatial processing complete.")
features_df = pd.DataFrame(station_results)

# ==============================================================================
# 6. BROADCAST TO PANEL
# ==============================================================================
print("Broadcasting static features to station-month panel...")
panel_keys = master_df[['station', 'year', 'month']].copy()
final_out_df = pd.merge(
    panel_keys, 
    features_df, 
    on='station', 
    how='left', 
    validate='many_to_one'
)

# ==============================================================================
# 7. VALIDATION
# ==============================================================================
print("Running validations...")
val_records = []

# 1. Master row count
val_records.append({'metric': 'master_row_count', 'value': len(master_df)})
# 2. Output row count
val_records.append({'metric': 'output_row_count', 'value': len(final_out_df)})
# 3. Master station count
val_records.append({'metric': 'master_station_count', 'value': master_df['station'].nunique()})
# 4. Output station count
val_records.append({'metric': 'output_station_count', 'value': final_out_df['station'].nunique()})

# 5. Duplicate station-year-month check
dupes = final_out_df.duplicated(subset=['station', 'year', 'month']).sum()
val_records.append({'metric': 'duplicate_station_year_month_count', 'value': dupes})

# 6. Missing feature count
missing_feats = final_out_df.isna().sum().sum()
val_records.append({'metric': 'missing_feature_count', 'value': missing_feats})

# 7-10. Min/Max Densities
densities = final_out_df[['road_density_100m', 'road_density_250m', 'road_density_500m', 'road_density_1000m']]
val_records.append({'metric': 'minimum_road_density', 'value': densities.min().min()})
val_records.append({'metric': 'maximum_road_density', 'value': densities.max().max()})
val_records.append({'metric': 'minimum_major_road_density', 'value': final_out_df['major_road_density_1000m'].min()})
val_records.append({'metric': 'maximum_major_road_density', 'value': final_out_df['major_road_density_1000m'].max()})

# 11. OSM Geometries Extracted
val_records.append({'metric': 'number_of_osm_highway_geometries', 'value': len(roads_gdf)})

# 12. CRS Used
val_records.append({'metric': 'crs_used', 'value': PROJECT_CRS})

# 13. Stations with all required features
complete_stations = features_df.dropna().shape[0]
val_records.append({'metric': 'number_of_stations_with_all_required_features', 'value': complete_stations})

# 14. PBF Input Filename
val_records.append({'metric': 'PBF_input_filename', 'value': os.path.basename(PBF_PATH)})

# 15. PBF Input File Size
val_records.append({'metric': 'PBF_input_file_size_MB', 'value': round(PBF_SIZE_MB, 2)})

# 16. OSM Snapshot Note
val_records.append({'metric': 'OSM_source_snapshot_note', 
                    'value': 'Static cross-sectional proxy for 2022-2025. Measures transportation infrastructure intensity, NOT temporal traffic volume.'})

# 17. Major road density <= Total road density (at 1000m)
# Allow tiny floating point tolerance
major_le_total = (features_df['major_road_density_1000m'] <= features_df['road_density_1000m'] + 0.0001).all()
val_records.append({'metric': 'major_road_density_le_total_road_density', 'value': str(major_le_total)})

val_df = pd.DataFrame(val_records)

# ==============================================================================
# 8. INTEGRITY ASSERTS & SAVE
# ==============================================================================
assert len(final_out_df) == len(master_df), "Row count mismatch!"
assert final_out_df['station'].nunique() == master_df['station'].nunique(), "Station count mismatch!"
assert dupes == 0, "Duplicates found in keys!"

final_out_df.to_csv(OUTPUT_FEAT_PATH, index=False)
val_df.to_csv(OUTPUT_VAL_PATH, index=False)

print("\nSUCCESS! Files saved:")
print(f"1. {OUTPUT_FEAT_PATH} ({final_out_df.shape[0]} rows)")
print(f"2. {OUTPUT_VAL_PATH}")
print("\nValidation Summary:")
print(val_df.to_string(index=False))