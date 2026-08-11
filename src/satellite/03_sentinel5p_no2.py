import ee
import geemap

ee.Initialize(project="delhi-pm25-research")

# Delhi
delhi = ee.Geometry.Rectangle([
    76.84, 28.40,
    77.35, 28.90
])

# Sentinel-5P NO2
no2 = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
    .filterBounds(delhi)
    .filterDate("2024-01-01", "2024-02-01")
    .select("tropospheric_NO2_column_number_density")
)

print(
    "Number of NO2 images:",
    no2.size().getInfo()
)

# Monthly median
no2_monthly = no2.median()

Map = geemap.Map()

Map.centerObject(delhi, 9)

Map.addLayer(
    no2_monthly,
    {
        "min": 0,
        "max": 0.0002
    },
    "Sentinel-5P NO2"
)

Map.addLayer(
    delhi,
    {},
    "Delhi"
)

Map.to_html(
    "delhi_no2_test.html"
)

print("Saved delhi_no2_test.html")