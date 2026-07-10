import sys
from pathlib import Path
import json
from datetime import datetime

# ---------------------------------------------------------
# INITIALISE EARTH ENGINE
#
# Google Earth Engine's pip package is called `earthengine-api`,
# but the module is imported as `ee`.
# app loop is added to file as this is loaded in a popup
# ---------------------------------------------------------
try:
    import ee
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
                                "Google Earth Engine is not installed in this Python environment.\n"
                                "Install it with:\n"
                                "    python -m pip install earthengine-api --upgrade\n"
                                "Then authenticate once with:\n"
                                "    earthengine authenticate\n"
                                "or run ee.Authenticate() from Python."
    ) from exc

EE_PROJECT_ID = "msc-artificial-intelligence"

# ==================================================================
# Authenticate/initialise Earth Engine for this script.
# ==================================================================
def initialise_earth_engine(project_id: str = EE_PROJECT_ID) -> None:

    try:
        ee.Initialize(project=project_id)
        
    except Exception:
        # First run on a new machine usually needs an OAuth token.
        ee.Authenticate()
        ee.Initialize(project=project_id)

# ---------------------------------------------------------
# BOUNDING BOXES
# ---------------------------------------------------------
BOUNDING_BOXES = [
                    [-1.875, 51.145, -1.775, 51.205],
                    [-1.895, 51.405, -1.815, 51.455],
                    [-1.895, 51.31,  -1.85,  51.338],
                    [-1.12,  51.335, -1.05,  51.38],
                    [1.255,  52.565, 1.325,  52.6],
                    [1.275,  52.608, 1.322,  52.635],
]

DATASET_NAME = "custom_s2_s1"
folder_name = f"{DATASET_NAME}_dataset"

# ---------------------------------------------------------
# LOAD SENTINEL-2 (OPTICAL)
# ---------------------------------------------------------
s2_bands = ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]

def mask_s2_scl(image):
    scl = image.select("SCL")
    mask = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
    )
    return image.updateMask(mask)

def load_s2(aoi):
    s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate("2018-01-01", "2024-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(s2_bands)
    )
    s2_clean = s2.map(mask_s2_scl)
    return s2_clean.median().clip(aoi)

# ---------------------------------------------------------
# LOAD SENTINEL-1 (SAR)
# ---------------------------------------------------------
def load_s1(aoi):
    s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate("2018-01-01", "2024-12-31")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .select(["VV", "VH"])
    )
    return s1.median().clip(aoi)

# ---------------------------------------------------------
# BUILD STACK FOR A GIVEN AOI
# ---------------------------------------------------------
def build_stack(aoi):
    s2 = load_s2(aoi)
    s1 = load_s1(aoi)

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

    bsi = s2.expression(
        "(SWIR + RED - NIR - BLUE) / (SWIR + RED + NIR + BLUE)",
        {
            "SWIR": s2.select("B11"),
            "RED": s2.select("B4"),
            "NIR": s2.select("B8"),
            "BLUE": s2.select("B2"),
        },
    ).rename("BSI")

    return s2.addBands(s1).addBands(ndvi).addBands(bsi).toFloat()

# ---------------------------------------------------------
# EXPORT LOOP
# ---------------------------------------------------------
def main():
    initialise_earth_engine()

    for i, bbox in enumerate(BOUNDING_BOXES):
        print(f"\nProcessing tile {i} with bbox {bbox}")

        region = ee.Geometry.Rectangle(bbox)
        stack = build_stack(region)

        # -------------------------------------------------
        # CHECK IF TILE HAS ANY VALID PIXELS
        # -------------------------------------------------
        pixel_count = stack.reduceRegion(reducer=ee.Reducer.count(),
                                         geometry=region,
                                         scale=10,
                                         maxPixels=1e13).get("B2")

        if ee.Number(pixel_count).eq(0).getInfo():
            print(f"Skipping empty tile {i}")
            continue

        # -------------------------------------------------
        # METADATA OBJECT
        # -------------------------------------------------
        metadata = {
                    "tile_id": i,
                    "bbox": bbox,
                    "projection": stack.projection().crs().getInfo(),
                    "scale_m": stack.projection().nominalScale().getInfo(),
                    "bands": stack.bandNames().getInfo(),
                    "timestamp": datetime.utcnow().isoformat(),
        }

        meta_fc = ee.FeatureCollection([
            ee.Feature(None, metadata)
        ])

        # -------------------------------------------------
        # EXPORT METADATA
        # -------------------------------------------------
        meta_task = ee.batch.Export.table.toDrive(collection=meta_fc,
                                                  description=f"{DATASET_NAME}_tile_{i}_metadata",
                                                  folder=folder_name,
                                                  fileNamePrefix=f"{DATASET_NAME}_tile_{i}_metadata",
                                                  fileFormat="GeoJSON",
        )

        # -------------------------------------------------
        # EXPORT IMAGE TILE
        # -------------------------------------------------
        img_task = ee.batch.Export.image.toDrive(image=stack,
                                                 description=f"{DATASET_NAME}_tile_{i}",
                                                 folder=folder_name,
                                                 fileNamePrefix=f"{DATASET_NAME}_tile_{i}",
                                                 region=region,
                                                 scale=10,
                                                 maxPixels=1e13,
        )

        meta_task.start()
        img_task.start()

        print(f"Started export for tile {i}")

    print("\nAll bounding boxes processed.")


if __name__ == "__main__":
    main()
