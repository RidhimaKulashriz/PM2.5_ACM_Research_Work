import pandas as pd
from pathlib import Path

# Paths
base_dir = Path(__file__).resolve().parent.parent
master_file = base_dir / 'data' / 'processed' / 'satellite' / 'station_master.csv'
ref_file = base_dir / 'data' / 'reference' / 'stations' / 'cpcb_stations.csv'

if not master_file.exists():
    raise FileNotFoundError(f"Master file not found at {master_file}. Run 01_prepare_stations.py first.")

df = pd.read_csv(master_file)

print(f"Original stations in master: {len(df)}")

# 1. Filter only MATCHED status
df_clean = df[df['match_status'] == 'MATCHED'].copy()

# 2. Drop any missing coordinates
df_clean = df_clean.dropna(subset=['latitude', 'longitude'])

# 3. Drop duplicate coordinates (keep first occurrence)
df_clean = df_clean.drop_duplicates(subset=['latitude', 'longitude'], keep='first')

# 4. Save clean reference file
ref_file.parent.mkdir(parents=True, exist_ok=True)
df_clean[['station', 'latitude', 'longitude']].to_csv(ref_file, index=False)

print(f"[SUCCESS] Saved {len(df_clean)} valid, unique stations to: {ref_file}")