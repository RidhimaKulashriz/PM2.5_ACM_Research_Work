import ee
import geemap
from pathlib import Path

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------

ee.Initialize(project="delhi-pm25-research")

print("GEE initialized.")

# --------------------------------------------------
# DELHI TEST REGION
# --------------------------------------------------

delhi = ee.Geometry.Rectangle([
    76.84, 28.40,
    77.35, 28.90
])

# --------------------------------------------------
# SENTINEL-2
# --------------------------------------------------

s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(delhi)
    .filterDate("2024-01-01", "2024-02-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
)

print("Number of Sentinel-2 images:")
print(s2.size().getInfo())

# --------------------------------------------------
# CLOUD MASK
# --------------------------------------------------

def mask_clouds(image):

    qa = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )

    return image.updateMask(mask).divide(10000)


s2_clean = s2.map(mask_clouds)

# --------------------------------------------------
# NDVI
# --------------------------------------------------

def add_ndvi(image):

    ndvi = image.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")

    return image.addBands(ndvi)


s2_ndvi = s2_clean.map(add_ndvi)

# --------------------------------------------------
# MONTHLY COMPOSITE
# --------------------------------------------------

ndvi = s2_ndvi.select("NDVI").median()

print("NDVI composite created.")

# --------------------------------------------------
# VISUALIZE
# --------------------------------------------------

Map = geemap.Map()

Map.centerObject(delhi, 10)

Map.addLayer(
    ndvi,
    {
        "min": -0.2,
        "max": 0.8,
    },
    "Delhi NDVI"
)

Map.addLayer(
    delhi,
    {},
    "Delhi boundary"
)

Map.to_html(
    "delhi_ndvi_test.html"
)

print("Saved: delhi_ndvi_test.html")