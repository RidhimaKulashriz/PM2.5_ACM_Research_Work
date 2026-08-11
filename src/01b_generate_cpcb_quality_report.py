"""
Generate a human-readable HTML quality report
from the CSV reports produced by 01_audit_cpcb.py.

Input:
    data/processed/reports/
        cpcb_file_inventory.csv
        cpcb_column_audit.csv
        cpcb_missing_data_report.csv
        cpcb_station_year_summary.csv

Output:
    data/processed/reports/cpcb_quality_report.html
    data/processed/reports/cpcb_station_year_coverage_matrix.csv
"""

from pathlib import Path
import pandas as pd
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "reports"

FILE_INVENTORY = REPORT_DIR / "cpcb_file_inventory.csv"
COLUMN_AUDIT = REPORT_DIR / "cpcb_column_audit.csv"
MISSING_REPORT = REPORT_DIR / "cpcb_missing_data_report.csv"
SUMMARY = REPORT_DIR / "cpcb_station_year_summary.csv"

HTML_OUTPUT = REPORT_DIR / "cpcb_quality_report.html"
MATRIX_OUTPUT = REPORT_DIR / "cpcb_station_year_coverage_matrix.csv"


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = [
    FILE_INVENTORY,
    COLUMN_AUDIT,
    MISSING_REPORT,
    SUMMARY,
]

missing_files = [str(f) for f in required_files if not f.exists()]

if missing_files:
    print("\nERROR: The following report files are missing:\n")

    for f in missing_files:
        print(f"  - {f}")

    raise SystemExit(1)


# ============================================================
# LOAD REPORTS
# ============================================================

inventory = pd.read_csv(FILE_INVENTORY)
column_audit = pd.read_csv(COLUMN_AUDIT)
missing_data = pd.read_csv(MISSING_REPORT)
summary = pd.read_csv(SUMMARY)


# ============================================================
# BASIC METRICS
# ============================================================

total_station_years = len(summary)

available = (
    summary["status"]
    .astype(str)
    .str.upper()
    .eq("AVAILABLE")
).sum()

missing = total_station_years - available

availability_pct = (
    available / total_station_years * 100
    if total_station_years > 0
    else 0
)

pm25_available = (
    column_audit["pm25_found"]
    .fillna(False)
    .astype(bool)
    .sum()
)

pm10_available = (
    column_audit["pm10_found"]
    .fillna(False)
    .astype(bool)
    .sum()
)

duplicate_rows = (
    pd.to_numeric(
        inventory.get("duplicate_rows", pd.Series(dtype=float)),
        errors="coerce"
    )
    .fillna(0)
    .sum()
)

# ============================================================
# COVERAGE MATRIX
# ============================================================

coverage = summary.copy()

coverage["available"] = (
    coverage["status"]
    .astype(str)
    .str.upper()
    .eq("AVAILABLE")
)

matrix = coverage.pivot_table(
    index="station",
    columns="year",
    values="available",
    aggfunc="first",
    fill_value=False
)

matrix = matrix.sort_index()

# Convert True/False to readable symbols
matrix_display = matrix.map(lambda x: "AVAILABLE" if x else "MISSING")

matrix_display.to_csv(MATRIX_OUTPUT)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def dataframe_to_html(df, max_rows=None):
    """Convert dataframe to readable HTML."""

    if max_rows is not None:
        df = df.head(max_rows)

    if df.empty:
        return "<p><em>No records available.</em></p>"

    return df.to_html(
        index=False,
        classes="data-table",
        border=0,
        justify="left"
    )


def coverage_matrix_html(matrix):
    """Generate HTML for station-year coverage matrix."""

    html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Station</th>
    """

    for year in matrix.columns:
        html += f"<th>{year}</th>"

    html += """
            </tr>
        </thead>
        <tbody>
    """

    for station, row in matrix.iterrows():

        html += f"<tr><td><strong>{station}</strong></td>"

        for value in row:
            if value == "AVAILABLE":
                html += '<td class="available">✓ AVAILABLE</td>'
            else:
                html += '<td class="missing">✗ MISSING</td>'

        html += "</tr>"

    html += """
        </tbody>
    </table>
    """

    return html


# ============================================================
# PREPARE SUMMARY TABLES
# ============================================================

missing_station_years = summary[
    ~summary["status"]
    .astype(str)
    .str.upper()
    .eq("AVAILABLE")
].copy()

# Internal completeness
completeness_columns = [
    "station",
    "year",
    "status",
    "row_count",
    "first_timestamp",
    "last_timestamp",
    "pm25_valid_obs",
    "pm25_completeness_pct",
    "pm10_valid_obs",
    "pm10_completeness_pct",
]

available_summary = summary[
    summary["status"]
    .astype(str)
    .str.upper()
    .eq("AVAILABLE")
].copy()

available_summary = available_summary[
    [c for c in completeness_columns if c in available_summary.columns]
]


# ============================================================
# HTML REPORT
# ============================================================

generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<title>CPCB Dataset Quality Report</title>

<style>

body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 40px;
    background: #f5f7fa;
    color: #222;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 40px;
    border-bottom: 2px solid #ddd;
    padding-bottom: 8px;
}}

.subtitle {{
    color: #666;
    margin-bottom: 30px;
}}

.cards {{
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin: 25px 0;
}}

.card {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    min-width: 180px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.card .number {{
    font-size: 28px;
    font-weight: bold;
}}

.card .label {{
    color: #666;
    margin-top: 5px;
}}

.data-table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    margin-top: 15px;
    font-size: 13px;
}}

.data-table th {{
    background: #e9edf2;
    font-weight: bold;
}}

.data-table th,
.data-table td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}}

.data-table tr:nth-child(even) {{
    background: #fafafa;
}}

.available {{
    background: #e8f5e9;
    color: #176b2c;
    font-weight: bold;
}}

.missing {{
    background: #ffebee;
    color: #a32121;
    font-weight: bold;
}}

.note {{
    background: white;
    border-left: 4px solid #555;
    padding: 15px;
    margin: 20px 0;
}}

.warning {{
    background: #fff8e1;
    border-left: 4px solid #f0ad00;
    padding: 15px;
    margin: 20px 0;
}}

.small {{
    color: #777;
    font-size: 12px;
}}

</style>

</head>


<body>

<h1>CPCB Dataset Quality Audit Report</h1>

<div class="subtitle">
Generated on {generated_at}
</div>


<div class="note">

<strong>Purpose:</strong>

This report summarizes the structural availability and
observation-level completeness of the CPCB monitoring dataset
before cleaning and downstream analysis.

The raw CPCB data has not been modified by this report.

</div>


<h2>1. Dataset Overview</h2>

<div class="cards">

<div class="card">
    <div class="number">{total_station_years}</div>
    <div class="label">Expected station-years</div>
</div>

<div class="card">
    <div class="number">{available}</div>
    <div class="label">Available station-years</div>
</div>

<div class="card">
    <div class="number">{missing}</div>
    <div class="label">Missing station-years</div>
</div>

<div class="card">
    <div class="number">{availability_pct:.2f}%</div>
    <div class="label">Station-year availability</div>
</div>

<div class="card">
    <div class="number">{pm25_available}</div>
    <div class="label">PM2.5 available</div>
</div>

<div class="card">
    <div class="number">{pm10_available}</div>
    <div class="label">PM10 available</div>
</div>

<div class="card">
    <div class="number">{int(duplicate_rows)}</div>
    <div class="label">Duplicate rows detected</div>
</div>

</div>


<h2>2. Station-Year Coverage</h2>

<p>
The following matrix shows whether each expected station-year
dataset is available.
</p>

{coverage_matrix_html(matrix_display)}


<h2>3. Missing Station-Years</h2>

<p>
These station-year combinations were expected from the
45-station × 4-year audit scope but were not available.
</p>

{dataframe_to_html(
    missing_station_years[
        [c for c in [
            "station",
            "year",
            "status",
            "row_count"
        ] if c in missing_station_years.columns]
    ]
)}


<h2>4. Observation-Level Completeness</h2>

<p>
For station-years where a CSV exists, this section reports
the number of valid PM2.5 and PM10 observations and their
completeness percentages.
</p>

{dataframe_to_html(available_summary)}


<h2>5. Missing Data Details</h2>

<p>
The following report identifies missing observations inside
available station-year datasets.
</p>

{dataframe_to_html(missing_data, max_rows=500)}


<h2>6. Column / Schema Audit</h2>

<p>
This section records whether expected timestamp, PM2.5 and PM10
fields were detected and identifies missing or extra columns.
</p>

{dataframe_to_html(column_audit, max_rows=500)}


<h2>7. File Inventory</h2>

<p>
The file inventory provides structural information about every
station-year dataset.
</p>

{dataframe_to_html(inventory, max_rows=500)}


<h2>8. Interpretation</h2>

<div class="note">

<strong>Current audit finding:</strong>

The dataset contains {total_station_years} expected station-year
combinations across the audited stations and years.

Of these, {available} ({availability_pct:.2f}%) are currently
available and {missing} are unavailable.

The missing station-years are retained as missing rather than
being artificially reconstructed at this stage.

</div>


<div class="warning">

<strong>Important:</strong>

Availability of a CSV file does not automatically imply that
the dataset is complete or suitable for modeling.

The next stage should investigate timestamp coverage,
duplicate timestamps, missing PM2.5/PM10 observations,
invalid values, schema differences and temporal gaps before
cleaning or imputation.

</div>


<p class="small">
Generated automatically from the CPCB audit CSV reports.
</p>

</body>

</html>
"""


# ============================================================
# WRITE HTML
# ============================================================

with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)


print("\n" + "=" * 70)
print("CPCB QUALITY REPORT GENERATED")
print("=" * 70)

print(f"\nHTML report:")
print(HTML_OUTPUT)

print(f"\nCoverage matrix:")
print(MATRIX_OUTPUT)

print("\nSummary:")
print(f"  Expected station-years : {total_station_years}")
print(f"  Available              : {available}")
print(f"  Missing                : {missing}")
print(f"  Availability           : {availability_pct:.2f}%")
print(f"  PM2.5 available        : {pm25_available}")
print(f"  PM10 available         : {pm10_available}")
print(f"  Duplicate rows         : {int(duplicate_rows)}")

print("\nDone.")