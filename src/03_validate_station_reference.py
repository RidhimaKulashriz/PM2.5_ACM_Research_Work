import os
import pandas as pd
from pathlib import Path

def validate_station_reference():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    ref_dir = base_dir / 'data' / 'processed' / 'satellite'
    report_dir = base_dir / 'data' / 'processed' / 'satellite' / 'reports'
    
    ref_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    station_file = ref_dir / 'station_master.csv'
    template_file = base_dir / 'data' / 'reference' / 'stations' / 'station_coordinates_template.csv'
    report_file = report_dir / 'station_reference_validation.csv'

    # 1. Check if file exists
    if not station_file.exists():
        print(f"[ERROR] Station reference file not found at: {station_file}")
        # Create template
        df_template = pd.DataFrame(columns=['station', 'latitude', 'longitude', 'source', 'coordinate_status'])
        df_template.to_csv(template_file, index=False)
        print(f"[ACTION REQUIRED] A template has been created at: {template_file}")
        print("Please populate this template with the 45 CPCB stations and rename it to 'cpcb_stations.csv'. Do NOT invent coordinates.")
        return False

    # 2. Load and Validate
    df = pd.read_csv(station_file)
    required_cols = {'station', 'latitude', 'longitude'}
    
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] Missing required columns. Expected: {required_cols}")
        return False

    validation_logs = []
    
    # 3. Validation Logic
    # Check for duplicates
    dup_names = df['station'].duplicated().sum()
    dup_coords = df[['latitude', 'longitude']].duplicated().sum()
    
    validation_logs.append({'check': 'Duplicate Station Names', 'count': dup_names, 'status': 'PASS' if dup_names == 0 else 'FAIL'})
    validation_logs.append({'check': 'Duplicate Coordinates', 'count': dup_coords, 'status': 'PASS' if dup_coords == 0 else 'FAIL'})

    # Check bounds (Delhi NCT roughly 28.4 to 28.9 N, 76.8 to 77.4 E)
    lat_out_of_bounds = df[(df['latitude'] < 28.0) | (df['latitude'] > 29.5)].shape[0]
    lon_out_of_bounds = df[(df['longitude'] < 76.0) | (df['longitude'] > 78.0)].shape[0]
    missing_coords = df['latitude'].isna().sum() + df['longitude'].isna().sum()

    validation_logs.append({'check': 'Latitudes out of Delhi bounds', 'count': lat_out_of_bounds, 'status': 'WARN' if lat_out_of_bounds > 0 else 'PASS'})
    validation_logs.append({'check': 'Longitudes out of Delhi bounds', 'count': lon_out_of_bounds, 'status': 'WARN' if lon_out_of_bounds > 0 else 'PASS'})
    validation_logs.append({'check': 'Missing Coordinates', 'count': missing_coords, 'status': 'PASS' if missing_coords == 0 else 'FAIL'})

    # 4. Generate Report
    report_df = pd.DataFrame(validation_logs)
    report_df.to_csv(report_file, index=False)
    
    print("\n=== Station Validation Report ===")
    print(report_df.to_string(index=False))
    
    if 'FAIL' in report_df['status'].values:
        print("\n[CRITICAL] Station validation failed. Please fix the errors in cpcb_stations.csv before proceeding.")
        return False
        
    print("\n[SUCCESS] Station coordinates validated successfully.")
    return True

if __name__ == "__main__":
    validate_station_reference()