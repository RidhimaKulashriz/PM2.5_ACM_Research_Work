from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CHANGE THIS ONLY if your OGD file has a different name/path
OGD_FILE = PROJECT_ROOT / "data" / "raw" / "station_metadata" / "cpcb_realtime_aqi.csv"

TARGETS = [
    "CPRI Mathura Road",
    "Cantonment Area",
    "Lodhi Road",
    "NSIT Dwarka",
    "Pusa",
]

def normalize(text):
    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    text = text.replace("_", " ")

    return text


df = pd.read_csv(
    OGD_FILE,
    engine="python"
)

print("\nColumns:")
print(df.columns.tolist())

# Make sure these exist
required = ["station", "latitude", "longitude"]

for col in required:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

df["_station_norm"] = df["station"].map(normalize)

for target in TARGETS:

    print("\n" + "=" * 80)
    print(f"SEARCHING: {target}")
    print("=" * 80)

    target_norm = normalize(target)

    # Search by important words rather than exact match
    words = [
        w for w in target_norm.split()
        if len(w) > 2
    ]

    mask = df["_station_norm"].apply(
        lambda x: all(word in x for word in words)
    )

    matches = df.loc[
        mask,
        [
            "station",
            "latitude",
            "longitude"
        ]
    ].drop_duplicates()

    if matches.empty:
        print("NO MATCH FOUND")
    else:
        print(matches.to_string(index=False))