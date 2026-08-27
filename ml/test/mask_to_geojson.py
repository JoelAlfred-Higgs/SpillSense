import os
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_DIR = r"predictions"
OUTPUT_DIR = r"geojson"

# Remove tiny predicted regions
MIN_AREA_M2 = 10000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND SCENES
# ============================================================

scenes = sorted([

    name

    for name in os.listdir(PREDICTION_DIR)

    if os.path.isdir(
        os.path.join(
            PREDICTION_DIR,
            name
        )
    )

])


print(
    f"Found {len(scenes)} prediction scenes."
)


# ============================================================
# PROCESS EACH SCENE
# ============================================================

for scene in scenes:

    print("\n" + "=" * 60)

    print(
        f"Processing: {scene}"
    )

    mask_path = os.path.join(
        PREDICTION_DIR,
        scene,
        "prediction_mask.tif"
    )


    if not os.path.exists(mask_path):

        print(
            "Mask not found."
        )

        continue


    # ========================================================
    # READ MASK
    # ========================================================

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

        raster_transform = src.transform

        crs = src.crs


    # ========================================================
    # BINARY MASK
    # ========================================================

    binary_mask = (
        mask > 0
    ).astype(
        np.uint8
    )


    # ========================================================
    # EXTRACT POLYGONS
    # ========================================================

    polygon_features = []


    for geom, value in shapes(

        binary_mask,

        mask=binary_mask,

        transform=raster_transform

    ):

        if value != 1:
            continue


        polygon = shape(
            geom
        )


        # ----------------------------------------------------
        # Remove tiny regions
        # ----------------------------------------------------

        area_m2 = polygon.area


        if area_m2 < MIN_AREA_M2:
            continue


        polygon_features.append(
            polygon
        )


    print(
        f"Detected regions: "
        f"{len(polygon_features)}"
    )


    # ========================================================
    # CONVERT UTM → WGS84
    # ========================================================

    transformer = Transformer.from_crs(

        crs,

        "EPSG:4326",

        always_xy=True

    )


    def convert_coordinates(
        x,
        y,
        z=None
    ):

        return transformer.transform(
            x,
            y
        )


    # ========================================================
    # CREATE GEOJSON FEATURES
    # ========================================================

    features = []


    for index, polygon in enumerate(

        polygon_features,

        start=1

    ):

        polygon_wgs84 = shapely_transform(

            convert_coordinates,

            polygon

        )


        # Area before converting CRS
        # is in square metres.

        area_m2 = polygon.area


        features.append({

            "type": "Feature",

            "properties": {

                "scene": scene,

                "region_id": index,

                "area_m2": round(
                    area_m2,
                    2
                ),

                "area_km2": round(
                    area_m2 / 1_000_000,
                    6
                )

            },

            "geometry": mapping(
                polygon_wgs84
            )

        })


    # ========================================================
    # CREATE GEOJSON
    # ========================================================

    geojson = {

        "type": "FeatureCollection",

        "features": features

    }


    # ========================================================
    # SAVE GEOJSON
    # ========================================================

    output_path = os.path.join(

        OUTPUT_DIR,

        f"{scene}.geojson"

    )


    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            geojson,

            file,

            indent=2

        )


    print(
        f"Saved: {output_path}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "GEOJSON GENERATION COMPLETE"
)

print("=" * 60)

print(
    f"Output folder: "
    f"{os.path.abspath(OUTPUT_DIR)}"
)