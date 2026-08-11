import os
import sys
import pandas as pd
import ee
from datetime import datetime
from pathlib import Path

# Dynamic import resolution for project root and src dirs
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
base_dir = src_dir.parent
sys.path.extend([str(script_dir), str(src_dir)])

from gee_config import initialize_ee, load_config, get_delhi_geometry

def build_station_features(df_stations, buffers):
    features = []
    for _, row in df_stations.iterrows():
        pt = ee.Geometry.Point([row['longitude'], row['latitude']])
        for b in buffers:
            features.append(ee.Feature(pt.buffer(b), {
                'station': row['station'],
                'buffer_m': b,
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }))
    return ee.FeatureCollection(features)

def extract_sentinel5p_features():
    initialize_ee()
    config = load_config()
    
    # Path resolution with fallback
    ref_station_file = base_dir / 'data' / 'reference' / 'stations' / 'cpcb_stations.csv'
    master_station_file = base_dir / 'data' / 'processed' / 'satellite' / 'station_master.csv'
    out_dir = base_dir / 'data' / 'processed' / 'satellite' / 'sentinel5p'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if ref_station_file.exists():
        station_file = ref_station_file
    elif master_station_file.exists():
        station_file = master_station_file
    else:
        raise FileNotFoundError("Station reference file missing. Run preparation script first.")
        
    df_stations = pd.read_csv(station_file)
    if 'match_status' in df_stations.columns:
        df_stations = df_stations[df_stations['match_status'] == 'MATCHED']
    df_stations = df_stations.dropna(subset=['latitude', 'longitude']).drop_duplicates(subset=['latitude', 'longitude'])

    fc_stations = build_station_features(df_stations, config['spatial']['buffers_m'])
    delhi_geom = get_delhi_geometry()
    
    years = config['time_period']['years']
    months = config['time_period']['months']
    
    master_records = []
    log_records = []
    
    print("\n=======================================================")
    print("STARTING PHASE 4: SENTINEL-5P NO2 EXTRACTION")
    print("=======================================================")
    
    for year in years:
        for month in months:
            print(f"Processing Sentinel-5P NO2: {year}-{month:02d}...")
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')
            
            # Sentinel-5P Offline L3 NO2 Collection
            s5p = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")\
                    .filterBounds(delhi_geom)\
                    .filterDate(start_date, end_date)
                    
            if s5p.size().getInfo() == 0:
                print(f"  -> No Sentinel-5P imagery for {year}-{month:02d}. Skipping.")
                log_records.append({'year': year, 'month': month, 'status': 'NO_DATA'})
                continue
                
            def process_no2(img):
                # Standard L3 filtering: cloud fraction <= 0.30 (30% cloud mask)
                cloud_mask = img.select('cloud_fraction').lte(0.30)
                
                trop_no2 = img.select('tropospheric_NO2_column_number_density').updateMask(cloud_mask).rename('no2_tropospheric')
                tot_no2 = img.select('NO2_column_number_density').updateMask(cloud_mask).rename('no2_total')
                valid_obs = cloud_mask.rename('valid_obs')
                
                return ee.Image.cat([trop_no2, tot_no2, valid_obs])

            processed = s5p.map(process_no2)
            monthly_composite = processed.median()
            
            reducers = (ee.Reducer.mean()
                        .combine(ee.Reducer.median(), sharedInputs=True)
                        .combine(ee.Reducer.stdDev(), sharedInputs=True)
                        .combine(ee.Reducer.count(), sharedInputs=True))
            
            extracted = monthly_composite.reduceRegions(
                collection=fc_stations,
                reducer=reducers,
                scale=1113.2  # ~1km scale matching native S5P product sampling
            )
            
            results = extracted.getInfo()
            
            for feat in results['features']:
                props = feat['properties']
                record = {
                    'station': props.get('station'),
                    'latitude': props.get('latitude'),
                    'longitude': props.get('longitude'),
                    'year': year,
                    'month': month,
                    'buffer_m': props.get('buffer_m'),
                    
                    's5p_no2_trop_mean': props.get('no2_tropospheric_mean'),
                    's5p_no2_trop_median': props.get('no2_tropospheric_median'),
                    's5p_no2_trop_std': props.get('no2_tropospheric_stdDev'),
                    
                    's5p_no2_total_mean': props.get('no2_total_mean'),
                    's5p_no2_total_median': props.get('no2_total_median'),
                    's5p_no2_total_std': props.get('no2_total_stdDev'),
                    
                    's5p_valid_obs_count': props.get('valid_obs_count'),
                    'source_dataset': 'COPERNICUS/S5P/OFFL/L3_NO2'
                }
                master_records.append(record)
                
            log_records.append({'year': year, 'month': month, 'status': 'SUCCESS', 'timestamp': datetime.now().isoformat()})

    df_out = pd.DataFrame(master_records)
    df_out.to_csv(out_dir / 'sentinel5p_no2_station_monthly_features.csv', index=False)
    
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(out_dir / 'sentinel5p_extraction_log.csv', index=False)
    
    print(f"[SUCCESS] Extracted {len(df_out)} Sentinel-5P records.")

if __name__ == "__main__":
    extract_sentinel5p_features()