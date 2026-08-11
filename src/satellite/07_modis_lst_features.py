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

def extract_modis_lst_features():
    initialize_ee()
    config = load_config()
    
    ref_station_file = base_dir / 'data' / 'reference' / 'stations' / 'cpcb_stations.csv'
    master_station_file = base_dir / 'data' / 'processed' / 'satellite' / 'station_master.csv'
    out_dir = base_dir / 'data' / 'processed' / 'satellite' / 'modis_lst'
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
    print("STARTING PHASE 6: MODIS LST (MOD11A2) EXTRACTION")
    print("=======================================================")
    
    for year in years:
        for month in months:
            print(f"Processing MODIS LST: {year}-{month:02d}...")
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')
            
            modis_lst = ee.ImageCollection("MODIS/061/MOD11A2")\
                          .filterBounds(delhi_geom)\
                          .filterDate(start_date, end_date)
                          
            if modis_lst.size().getInfo() == 0:
                print(f"  -> No MODIS LST imagery for {year}-{month:02d}. Skipping.")
                log_records.append({'year': year, 'month': month, 'status': 'NO_DATA'})
                continue
                
            def process_lst(img):
                # Convert raw Kelvin to Celsius: (Raw * 0.02) - 273.15
                lst_day = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('lst_day_c')
                lst_night = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('lst_night_c')
                diurnal = lst_day.subtract(lst_night).rename('lst_diurnal_range')
                
                return ee.Image.cat([lst_day, lst_night, diurnal])

            processed = modis_lst.map(process_lst)
            monthly_composite = processed.mean()
            
            reducers = (ee.Reducer.mean()
                        .combine(ee.Reducer.median(), sharedInputs=True)
                        .combine(ee.Reducer.stdDev(), sharedInputs=True))
            
            extracted = monthly_composite.reduceRegions(
                collection=fc_stations,
                reducer=reducers,
                scale=1000
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
                    
                    'modis_lst_day_mean_c': props.get('lst_day_c_mean'),
                    'modis_lst_day_median_c': props.get('lst_day_c_median'),
                    'modis_lst_day_std_c': props.get('lst_day_c_stdDev'),
                    
                    'modis_lst_night_mean_c': props.get('lst_night_c_mean'),
                    'modis_lst_night_median_c': props.get('lst_night_c_median'),
                    'modis_lst_night_std_c': props.get('lst_night_c_stdDev'),
                    
                    'modis_lst_diurnal_range_mean_c': props.get('lst_diurnal_range_mean'),
                    'modis_lst_diurnal_range_median_c': props.get('lst_diurnal_range_median'),
                    
                    'source_dataset': 'MODIS/061/MOD11A2'
                }
                master_records.append(record)
                
            log_records.append({'year': year, 'month': month, 'status': 'SUCCESS', 'timestamp': datetime.now().isoformat()})

    df_out = pd.DataFrame(master_records)
    df_out.to_csv(out_dir / 'modis_lst_station_monthly_features.csv', index=False)
    
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(out_dir / 'modis_lst_extraction_log.csv', index=False)
    
    print(f"[SUCCESS] Extracted {len(df_out)} MODIS LST records.")

if __name__ == "__main__":
    extract_modis_lst_features()