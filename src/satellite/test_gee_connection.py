import ee
from gee_config import initialize_ee

def run_diagnostics():
    print("Testing Earth Engine Connection and Dataset Availability...")
    initialize_ee()
    
    # 1. Trivial Calculation
    try:
        num = ee.Number(10).add(32).getInfo()
        print(f"[PASS] Basic EE Calculation (10+32) = {num}")
    except Exception as e:
        print(f"[FAIL] Basic EE Calculation: {e}")
        
    # Datasets to verify
    datasets = {
        "Sentinel-2 Harmonized": "COPERNICUS/S2_SR_HARMONIZED",
        "Sentinel-2 Cloud Score+": "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED",
        "Sentinel-5P NO2": "COPERNICUS/S5P/OFFL/L3_NO2",
        "MODIS Vegetation": "MODIS/061/MOD13Q1"
    }
    
    for name, asset_id in datasets.items():
        try:
            # Check if collection exists by grabbing size of a small temporal subset
            collection = ee.ImageCollection(asset_id).filterDate('2023-01-01', '2023-01-05')
            count = collection.size().getInfo()
            print(f"[PASS] {name} accessible (Found {count} images in test window)")
        except Exception as e:
            print(f"[FAIL] {name} inaccessible: {e}")

if __name__ == "__main__":
    run_diagnostics()