import os
import pandas as pd
import ee
from datetime import datetime
from pathlib import Path
from gee_config import initialize_ee, load_config, get_delhi_geometry

def build_station_features(df_stations, buffers):
    """Creates a flat ee.FeatureCollection of all stations x all buffers."""
    features = []
    for _, row in df_stations.iterrows():
        pt = ee.Geometry.Point([row['longitude'], row['latitude']])
        for b in buffers:
            buffered_geom = pt.buffer(b)
            # Store metadata to retain lineage
            features.append(ee.Feature(buffered_geom, {
                'station': row['station'],
                'buffer_m': b,
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }))
    return ee.FeatureCollection(features)

def extract_sentinel2_features():
    initialize_ee()
    config = load_config()
    
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    ref_station_file = base_dir / 'data' / 'reference' / 'stations' / 'cpcb_stations.csv'
    master_station_file = base_dir / 'data' / 'processed' / 'satellite' / 'station_master.csv'
    out_dir = base_dir / 'data' / 'processed' / 'satellite' / 'sentinel2'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not ref_station_file.exists():
        raise FileNotFoundError("Station reference file missing. Run Phase 1 first.")
        
    df_stations = pd.read_csv(ref_station_file)
    fc_stations = build_station_features(df_stations, config['spatial']['buffers_m'])
    delhi_geom = get_delhi_geometry()
    
    years = config['time_period']['years']
    months = config['time_period']['months']
    cs_threshold = config['satellite']['sentinel2']['cloud_score_plus_threshold']
    
    master_records = []
    log_records = []
    
    print("Starting Sentinel-2 Extraction (Server-side aggregation)...")
    
    for year in years:
        for month in months:
            print(f"Processing {year}-{month:02d}...")
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')
            
            # Load S2 and Cloud Score+
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                   .filterBounds(delhi_geom)\
                   .filterDate(start_date, end_date)
                   
            csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")\
                       .filterBounds(delhi_geom)\
                       .filterDate(start_date, end_date)
            
            try:
                # Check coverage
                if s2.size().getInfo() == 0:
                    print(f"  -> No imagery for {year}-{month:02d}. Skipping.")
                    log_records.append({'year': year, 'month': month, 'status': 'NO_DATA'})
                    continue
                
                # Link QA band and apply mask
                s2_linked = s2.linkCollection(csPlus, ['cs_cdf'])
                
                def process_indices(img):
                    mask = img.select('cs_cdf').gte(cs_threshold)
                    img_masked = img.updateMask(mask)
                    # Convert SR (scaled by 10000) to 0-1
                    opt = img_masked.select(['B2', 'B3', 'B4', 'B8']).divide(10000)
                    
                    ndvi = opt.normalizedDifference(['B8', 'B4']).rename('sentinel2_ndvi')
                    ndwi = opt.normalizedDifference(['B3', 'B8']).rename('sentinel2_ndwi')
                    evi = opt.expression(
                        '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
                        {'B8': opt.select('B8'), 'B4': opt.select('B4'), 'B2': opt.select('B2')}
                    ).rename('sentinel2_evi')
                    
                    # Valid pixel mask (1 where clear, 0 elsewhere)
                    valid_px = mask.rename('valid_pixels')
                    
                    return ee.Image.cat([ndvi, evi, ndwi, valid_px])
                
                s2_processed = s2_linked.map(process_indices)
                monthly_composite = s2_processed.median()
                
                # Combine reducers for spatial stats
                reducers = (ee.Reducer.mean()
                            .combine(ee.Reducer.median(), sharedInputs=True)
                            .combine(ee.Reducer.stdDev(), sharedInputs=True)
                            .combine(ee.Reducer.min(), sharedInputs=True)
                            .combine(ee.Reducer.max(), sharedInputs=True)
                            .combine(ee.Reducer.count(), sharedInputs=True))
                
                # Extract for all stations & buffers in one API call
                extracted = monthly_composite.reduceRegions(
                    collection=fc_stations,
                    reducer=reducers,
                    scale=config['satellite']['sentinel2']['scale_m']
                )
                
                results = extracted.getInfo()
                
                for feat in results['features']:
                    props = feat['properties']
                    # Clean up property naming to match research requirements
                    record = {
                        'station': props.get('station'),
                        'latitude': props.get('latitude'),
                        'longitude': props.get('longitude'),
                        'year': year,
                        'month': month,
                        'buffer_m': props.get('buffer_m'),
                        
                        'sentinel2_ndvi_mean': props.get('sentinel2_ndvi_mean'),
                        'sentinel2_ndvi_median': props.get('sentinel2_ndvi_median'),
                        'sentinel2_ndvi_std': props.get('sentinel2_ndvi_stdDev'),
                        'sentinel2_ndvi_min': props.get('sentinel2_ndvi_min'),
                        'sentinel2_ndvi_max': props.get('sentinel2_ndvi_max'),
                        # Using the count of valid_pixels band as valid pixel count
                        'sentinel2_valid_pixels': props.get('valid_pixels_count'),
                        
                        'sentinel2_evi_mean': props.get('sentinel2_evi_mean'),
                        'sentinel2_evi_median': props.get('sentinel2_evi_median'),
                        'sentinel2_evi_std': props.get('sentinel2_evi_stdDev'),
                        
                        'sentinel2_ndwi_mean': props.get('sentinel2_ndwi_mean'),
                        'sentinel2_ndwi_median': props.get('sentinel2_ndwi_median'),
                        'sentinel2_ndwi_std': props.get('sentinel2_ndwi_stdDev'),
                        
                        'spatial_scale_m': 10,
                        'temporal_aggregation': 'monthly_median',
                        'source_dataset': 'COPERNICUS/S2_SR_HARMONIZED'
                    }
                    master_records.append(record)
                    
                log_records.append({'year': year, 'month': month, 'status': 'SUCCESS', 'extraction_timestamp': datetime.now().isoformat()})
                
            except Exception as e:
                print(f"  [ERROR] {year}-{month:02d} failed: {e}")
                log_records.append({'year': year, 'month': month, 'status': f'ERROR: {str(e)}'})

    # Save Outputs
    print("\nSaving files...")
    df_out = pd.DataFrame(master_records)
    df_out.to_csv(out_dir / 'sentinel2_station_monthly_features.csv', index=False)
    
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(out_dir / 'sentinel2_extraction_log.csv', index=False)
    
    # Generate quick coverage summary
    coverage = df_out.groupby(['station', 'buffer_m']).size().reset_index(name='months_covered')
    coverage.to_csv(out_dir / 'sentinel2_monthly_coverage.csv', index=False)
    print(f"Done. Extracted {len(df_out)} station-month-buffer records.")

if __name__ == "__main__":
    extract_sentinel2_features()