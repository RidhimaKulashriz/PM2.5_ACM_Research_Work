import pandas as pd
from pathlib import Path

def aggregate_cpcb_to_monthly():
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # 1. Locate your hourly CPCB CSV file
    # Replace 'cpcb_hourly.csv' below with the actual filename of your hourly CSV if different
    possible_hourly_paths = [
        base_dir / 'data' / 'processed' / 'master' / 'cpcb_pm25_master.csv',
    ]
    
    input_file = None
    for p in possible_hourly_paths:
        if p.exists():
            input_file = p
            break
            
    if not input_file:
        # Prompt user to check path if auto-detect fails
        print("\n[!] Please specify the exact path to your hourly CPCB CSV file.")
        path_str = input("Enter path to your hourly CSV file: ").strip('"').strip("'")
        input_file = Path(path_str)

    print(f"\nReading hourly CPCB data from: {input_file}")
    df = pd.read_csv(input_file)

    # 2. Parse Timestamps & Extract Month
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month

    # 3. Rename & Clean PM2.5
    pm25_col = [c for c in df.columns if 'pm2.5' in c.lower() or 'pm25' in c.lower()][0]
    df['pm25'] = pd.to_numeric(df[pm25_col], errors='coerce')

    # Remove negative or invalid PM2.5 readings
    df = df[df['pm25'] > 0]

    # 4. Aggregate to Monthly Means by Station
    print("Calculating monthly PM2.5 averages per station...")
    monthly_df = df.groupby(['station', 'year', 'month'], as_index=False)['pm25'].mean()
    monthly_df['pm25'] = monthly_df['pm25'].round(2)

    # 5. Save to the path expected by Phase 10
    out_dir = base_dir / 'data' / 'processed' / 'cpcb'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'cleaned_cpcb_monthly.csv'

    monthly_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Aggregated {len(df)} hourly rows into {len(monthly_df)} monthly station records.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    aggregate_cpcb_to_monthly()