import os
import sys
import pandas as pd
import ee
from datetime import datetime
from pathlib import Path

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

def extract_modis_veg_features():
    initialize_ee()
    config = load_config()
    
    ref_station_file = base_dir / 'data' / 'reference' / 'stations' / 'cpcb_stations.csv'
    master_station_file = base_dir / 'data' / 'processed' / 'satellite' / 'station_master.csv'
    out_dir = base_dir / 'data' / 'processed' / 'satellite' / 'modis_veg'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if ref_station_file.exists():
        station_file = ref_station_file
    elif master_station_file.exists():
        station_file = master_station_file
    else:
        raise FileNotFoundError("Station reference file missing.")
        
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
    print("STARTING PHASE 5: MODIS VEGETATION (MOD13Q1) EXTRACTION")
    print("=======================================================")
    
    for year in years:
        for month in months:
            print(f"Processing MODIS Vegetation: {year}-{month:02d}...")
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')
            
            modis = ee.ImageCollection("MODIS/061/MOD13Q1")\
                      .filterBounds(delhi_geom)\
                      .filterDate(start_date, end_date)
                      
            if modis.size().getInfo() == 0:
                print(f"  -> No MODIS Vegetation imagery for {year}-{month:02d}. Skipping.")
                log_records.append({'year': year, 'month': month, 'status': 'NO_DATA'})
                continue
                
            def process_veg(img):
                # SummaryQA: 0 = Good, 1 = Marginal
                qa_mask = img.select('SummaryQA').lte(1)
                ndvi = img.select('NDVI').multiply(0.0001).updateMask(qa_mask).rename('modis_ndvi')
                evi = img.select('EVI').multiply(0.0001).updateMask(qa_mask).rename('modis_evi')
                return ee.Image.cat([ndvi, evi])

            processed = modis.map(process_veg)
            monthly_composite = processed.mean()
            
            reducers = (ee.Reducer.mean()
                        .combine(ee.Reducer.median(), sharedInputs=True)
                        .combine(ee.Reducer.stdDev(), sharedInputs=True))
            
            extracted = monthly_composite.reduceRegions(
                collection=fc_stations,
                reducer=reducers,
                scale=250
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
                    
                    'modis_ndvi_mean': props.get('modis_ndvi_mean'),
                    'modis_ndvi_median': props.get('modis_ndvi_median'),
                    'modis_ndvi_std': props.get('modis_ndvi_stdDev'),
                    
                    'modis_evi_mean': props.get('modis_evi_mean'),
                    'modis_evi_median': props.get('modis_evi_median'),
                    'modis_evi_std': props.get('modis_evi_stdDev'),
                    
                    'source_dataset': 'MODIS/061/MOD13Q1'
                }
                master_records.append(record)
                
            log_records.append({'year': year, 'month': month, 'status': 'SUCCESS', 'timestamp': datetime.now().isoformat()})

    df_out = pd.DataFrame(master_records)
    df_out.to_csv(out_dir / 'modis_veg_station_monthly_features.csv', index=False)
    
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(out_dir / 'modis_veg_extraction_log.csv', index=False)
    
    print(f"[SUCCESS] Extracted {len(df_out)} MODIS Vegetation records.")

if __name__ == "__main__":
    extract_modis_veg_features()